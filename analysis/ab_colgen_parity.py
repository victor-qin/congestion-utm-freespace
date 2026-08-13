"""Byte-identity gate for column-generation accelerations.

Every phase of the compiled-pricing work claims to change *work*, never *answers*. This
harness is what makes that claim checkable: it runs one or more fixed-work colgen solves
and fingerprints the result, then compares the working tree against an arbitrary git ref.

Two design points that are not optional:

* **Fixed work, never a wall clock.** Each arm pins ``max_iterations`` and sets an
  effectively infinite ``time_limit_s``. A time-limited solve prices as many flights as it
  can afford, so a faster tree reaches *different* subproblems and the comparison measures
  nothing. See ``ColGenParams``' own tuning tables for the same discipline.
* **The child asserts which tree it loaded.** ``sys.path[0]`` is the script's directory, so
  a script run from outside a source tree silently imports the *workspace* rather than the
  extract under test. The child therefore refuses to proceed unless
  ``freespace_sim.__file__`` sits under the root it was told to use.

The fingerprint covers the objective, the selected-flight count, the denial set, and a sha
over every column's ``(flight_id, departure_step, delay_s, cell_path, sorted claims)``. It
deliberately excludes wall time and anything else a legitimate acceleration may move.

**``--ladder`` defaults to 0, and that is the contract, not a convenience.** A departure
ladder changes the master's *pool*, hence its duals, hence the subproblems pricing is handed
-- so a laddered tree compared against an unladdered ref diverges for a reason that has
nothing to do with the kernel, and the mismatch would read as a kernel regression. This
harness exists to say "the compiled search reproduces the reference search", which is a
statement about ``price_flight``'s internals; ``seed_ladder_steps`` is upstream of that fork
and is pinned off so the claim stays the one being tested.

To test the ladder itself, compare the two searches *inside one tree* with
``--reference-baseline``, which reruns the same arm with the compiled kernel disabled.
That is the comparison that answers "does the kernel still reproduce the reference once
the ladder moves the duals", and it needs no git ref at all.

**``--objective`` is pinned for the same reason, and it is the one whose DEFAULT MOVED.**
``ColGenParams.objective`` shipped ``total_delay`` (ground and air weighted 1:1) and now
ships ``total_cost`` (the config's 1:3). Those are different cost currencies, so a ref from
before the flip prices a different problem and every column diverges -- which would read as
a catastrophic kernel regression. To compare against such a ref, pass
``--objective total_delay`` and get the currency the ref speaks; the effective value is
fingerprinted and the parent refuses to compare two arms that disagree on it.

Examples:

    # the kernel-identity gate: working tree vs origin/main, ladder pinned off
    uv run python analysis/ab_colgen_parity.py --ref origin/main

    # ...against a ref predating the objective flip, in the currency that ref speaks
    uv run python analysis/ab_colgen_parity.py --ref <old-sha> --objective total_delay

    # the ladder gate: compiled vs reference within this tree, ladder on
    uv run python analysis/ab_colgen_parity.py --reference-baseline --ladder 20

    # one arm, more iterations
    uv run python analysis/ab_colgen_parity.py --ref origin/main --arm colgen_test

    # fingerprint the working tree only (no comparison)
    uv run python analysis/ab_colgen_parity.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only single-level scenarios are colgen-eligible -- `run_batch` refuses anything else.
# Arms B and C are not reseeds of A: different demand model, ~105-hop routes instead of ~7,
# and TERMINAL endpoints, which is the only way to exercise the `terminal_claim_steps`
# branch of `_endpoint_claims`. Their iteration counts are lower purely because those
# subproblems are much larger; parity needs identical output on identical work, not
# convergence.
#
# ``gap_metric`` is pinned to ``cost`` on every arm and is NOT a stylistic choice. The
# shipped default is ``revenue``, whose gap is diluted by ``n * M`` -- on density it reads
# 2.67e-05 against a 1e-4 threshold at iteration 1, so the solve terminates before column
# generation has done anything and the arm silently degenerates to a seeding test. Measured:
# ``density_faa_wing_zipline`` x8 stops at ``iterations=1`` under ``revenue`` and runs all 6
# under ``cost``, adding 6-7 columns per iteration throughout.
ARMS: dict[str, dict] = {
    "colgen_test": {
        "scenario": "colgen_test",
        "flights": 50,
        "iterations": 3,
        "solver": "highs",
        "gap_metric": "cost",
    },
    "density_faa": {
        "scenario": "density_faa_wing_zipline",
        "flights": 50,
        "iterations": 2,
        "solver": "highs",
        "gap_metric": "cost",
    },
    "density_future": {
        "scenario": "density_future_wing_zipline",
        "flights": 50,
        "iterations": 2,
        "solver": "highs",
        "gap_metric": "cost",
    },
}

# Runs in a child interpreter, against whichever tree `cwd` selects.
_CHILD = r'''
import dataclasses, hashlib, json, resource as _rusage, sys, time
from pathlib import Path

# ru_maxrss is bytes on macOS, kilobytes on Linux.
_RSS_SCALE = 2**20 if sys.platform == "darwin" else 1024

import numpy as np

import freespace_sim

root = Path(sys.argv[1]).resolve()
loaded = Path(freespace_sim.__file__).resolve()
if root not in loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {loaded} is not under {root}")

from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.solver import ColGenSolver
from freespace_sim.scenarios import get_scenario

spec_name, n_flights, iterations = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
backend, gap_metric = sys.argv[5], sys.argv[6]
ladder, kernel, workers = int(sys.argv[7]), sys.argv[8], int(sys.argv[9])
greedy_budget = float(sys.argv[10])
bootstrap_roots = int(sys.argv[11])
objective = sys.argv[12]
bootstrap_ranking = sys.argv[13]
spec = get_scenario(spec_name)
cfg = spec.config()
demand = spec.demand_model()
requests = demand.generate(cfg, np.random.default_rng(cfg.seed))
requests = sorted(requests, key=lambda r: r.flight_id)[:n_flights]
static_terms = list(demand.terminals(cfg))

# `seed_ladder_steps` does not exist on every ref this harness compares against, so it is
# applied by feature test rather than assumed -- passing it as a kwarg to an older
# `ColGenParams` is a TypeError, and defaulting it silently would be worse: a baseline that
# quietly ran with no ladder diverges on every column and reads as a kernel regression.
# The EFFECTIVE value goes into the fingerprint and the parent refuses to compare two arms
# that did not get the same one.
#
# `greedy_budget_s_per_flight` is the same shape of problem and a sharper one: the greedy
# stage is bounded by a WALL CLOCK (`solver.py`'s `greedy_budget_s_per_flight * n_flights`),
# so a faster tree reaches more candidates, `best_heuristic` differs, the duals differ, and
# every column diverges for a reason that has nothing to do with what is under test. Pinning
# it high enough that the stage always finishes is what makes this harness able to gate a
# change that legitimately makes pricing faster. `origin/main` has no such field at all, so
# the same feature test applies.
_fields = {f.name for f in dataclasses.fields(ColGenParams)}
_ladder_kwargs = {"seed_ladder_steps": ladder} if "seed_ladder_steps" in _fields else {}
_greedy_kwargs = (
    {"greedy_budget_s_per_flight": greedy_budget}
    if "greedy_budget_s_per_flight" in _fields else {}
)
# The pricing bootstrap. Answer-AFFECTING (a tighter cutoff can return a different, equally
# optimal column), so like the ladder it must be pinned on both arms or the comparison
# measures the knob rather than the search. Its real gate is `--reference-baseline`: the
# bootstrap picks its roots ONCE and hands the allowlist to whichever search runs, so an
# arm with the kernel disabled must reach the identical column.
_bootstrap_kwargs = (
    {"bootstrap_roots": bootstrap_roots} if "bootstrap_roots" in _fields else {}
)
# `objective` decides the WEIGHTS, and it is the sharpest pin of the lot because its default
# MOVED: `total_delay` -> `total_cost`. An unpinned arm takes its own tree's default, so
# comparing this tree against a ref from before the flip would price 1:1 on one side and 1:3
# on the other and report a total regression that is really a change of currency. The field
# has existed since the cost model landed, so no feature test is needed -- but the EFFECTIVE
# value is fingerprinted, and the parent refuses to compare two arms that disagree.
_objective_kwargs = {"objective": objective}
# Ordering-only, but answer-AFFECTING through the cutoff it produces, so pinned like the
# rest.  Feature-tested: a ref predating the field would TypeError on the kwarg.
_ranking_kwargs = (
    {"bootstrap_ranking": bootstrap_ranking}
    if "bootstrap_ranking" in _fields else {}
)
params = ColGenParams(
    solver=backend, max_iterations=iterations, time_limit_s=86400.0, gap_metric=gap_metric,
    **_ladder_kwargs, **_greedy_kwargs, **_bootstrap_kwargs, **_objective_kwargs, **_ranking_kwargs,
)
effective_ladder = getattr(params, "seed_ladder_steps", 0)
effective_greedy_budget = getattr(params, "greedy_budget_s_per_flight", None)
effective_bootstrap = getattr(params, "bootstrap_roots", None)
effective_objective = getattr(params, "objective", None)
effective_ranking = getattr(params, "bootstrap_ranking", None)

if kernel == "off":
    # Force the pure-Python reference search.  Same seam the kernel tests use
    # (`tests/test_colgen_dp_kernel.py`), and the same one a numba-less install takes on its
    # own, so this exercises a shipped path rather than a test-only one.
    #
    # This is a COMPLETE disable only because both compiled entry points consult
    # `_dp_kernel()` and fall back when it returns None: `_best_column_compiled` (exact
    # pricing) and `_feasible_compiled` (the greedy's feasible search).  A third entry point
    # that skipped that check would leave this arm silently half-compiled, which would look
    # like the reference agreeing with itself.
    from freespace_sim.planner.colgen import pricing
    pricing._dp_kernel = lambda: None

# Same feature test as the ladder, for the same reason: older refs have no `parallel`
# kwarg.  Unlike the ladder this one is NOT required to match across arms -- worker count
# is forbidden to change the answer, so an arm at 0 and an arm at 8 SHOULD agree, and that
# disagreement-is-a-bug property is exactly what the comparison is for.
import inspect
_solve_params = inspect.signature(ColGenSolver.solve).parameters
_param_fields = {f.name for f in dataclasses.fields(ColGenParams)}
_pool_kwargs = {}
effective_workers = workers
# TWO MECHANISMS, because this harness compares across a change that replaced one with the
# other.  Newer trees read `params.n_pricing_workers`; older ones ignore that field entirely
# and take a `parallel=ParallelPricingConfig(...)` keyword instead.  Test for the keyword
# FIRST: a ref can have both -- `origin/main` carries the params field but its `solve` does
# not consult it -- so preferring the field would silently run that arm sequentially while
# the header claimed workers.
if "parallel" in _solve_params:
    # Old tree.  Omitting the keyword IS the sequential request there, because that tree's
    # `ParallelPricingConfig` still defaults to 0.
    if workers:
        from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig
        _pool_kwargs["parallel"] = ParallelPricingConfig(n_workers=workers)
elif "n_pricing_workers" in _param_fields:
    # New tree, and this MUST run for `workers == 0` too -- which is why it is not inside a
    # `if workers:` guard.  The shipped default is now 4, so leaving `params` alone no
    # longer means "sequential", it means "four".  Guarded, the 0-worker baseline arm of
    # `--sequential-baseline` silently ran four workers and compared a pool against itself.
    # The assertion below caught exactly that, on the first run after the default moved.
    params = dataclasses.replace(params, n_pricing_workers=workers)
else:
    effective_workers = 0

started = time.perf_counter()
result = ColGenSolver().solve(requests, cfg, static_terms, params, **_pool_kwargs)
wall = time.perf_counter() - started

# The knob is only useful if it took effect.  `n_pricing_workers` was accepted-and-ignored
# on the params object for the whole life of the `parallel=` keyword, and an arm that
# silently ran sequential does not fail -- it AGREES, which reads as a passing parity run
# that tested nothing.  The stat is derived from what the solver actually used.
_got_workers = result.stats.get("n_pricing_workers")
if _got_workers != effective_workers:
    raise SystemExit(
        f"asked for {effective_workers} pricing workers but the solver used "
        f"{_got_workers}; this arm did not test what it claims"
    )

rows = []
for flight_id, column in sorted(result.columns.items()):
    rows.append((
        int(column.flight_id),
        int(column.departure_step),
        int(column.level),
        column.origin_lane_idx,
        column.dest_lane_idx,
        repr(column.delay_s),
        tuple(tuple(cell) for cell in column.cell_path),
        tuple(sorted(tuple(row) for row in column.claims)),
    ))
stats = result.stats
print("@@FINGERPRINT@@" + json.dumps({
    "objective": repr(stats.get("objective")),
    "selected_flights": stats.get("selected_flights"),
    "n_columns": stats.get("n_columns"),
    "iterations": stats.get("iterations"),
    "termination_reason": stats.get("termination_reason"),
    "denied_flight_ids": sorted(stats.get("denied_flight_ids", ())),
    "column_sha": hashlib.sha256(repr(rows).encode()).hexdigest()[:16],
    "wall_s": round(wall, 3),
    "pricing_wall_s": round(float(stats.get("pricing_wall_s", 0.0)), 3),
    "seed_ladder_steps": effective_ladder,
    "ladder_columns": stats.get("ladder_columns", 0),
    "kernel": kernel,
    "workers": effective_workers,
    # `_greedy_feasible_selection` is bounded by a WALL CLOCK, not by a flight count, so at
    # scale it stops wherever the clock caught it -- and it produces `best_heuristic`, which
    # is the `known_column` cutoff handed to every pricing call.  If this is False for the
    # CLOCK's sake the arms did not start from the same incumbent and no fingerprint
    # comparison is valid, whatever else differs.  `--greedy-budget` exists to lift it, and
    # the effective value is fingerprinted so two arms that got different ones cannot be
    # compared.
    #
    # It is ALSO False for a structural reason no budget can fix: the stage caps itself at
    # `max(64, n_heuristic_tries * 16)` candidates before consulting any deadline, so above
    # that flight count it never "completes".  `greedy_candidate_capped` tells the two apart,
    # and only the clock case invalidates a comparison.
    "greedy_completed": stats.get("initial_greedy_completed"),
    "greedy_budget_s_per_flight": effective_greedy_budget,
    "bootstrap_roots": effective_bootstrap,
    "objective_mode": effective_objective,
    "bootstrap_ranking": effective_ranking,
    "greedy_candidate_capped": n_flights > max(64, params.n_heuristic_tries * 16),
    "greedy_elapsed_s": round(float(stats.get("initial_greedy_elapsed_s", 0.0) or 0.0), 1),
    "heuristic_strategy": stats.get("initial_heuristic_strategy"),
    # Label pools are per-process, so a parallel arm's memory is the thing most likely to
    # scale badly and the thing a wall-clock number hides.  SELF is the parent; CHILDREN is
    # the high-water mark of the pool's workers, which is where that risk actually lands.
    "rss_self_mb": round(_rusage.getrusage(_rusage.RUSAGE_SELF).ru_maxrss / _RSS_SCALE, 1),
    "rss_children_mb": round(
        _rusage.getrusage(_rusage.RUSAGE_CHILDREN).ru_maxrss / _RSS_SCALE, 1
    ),
    "tree": str(loaded.parent.parent),
}))
'''


def _run_arm(
    root: Path, arm: dict, ladder: int, kernel: str, workers: int, greedy_budget: float,
    bootstrap_roots: int,
    objective: str,
    bootstrap_ranking: str,
) -> dict:
    """Fingerprint one arm in a child interpreter rooted at ``root``."""

    proc = subprocess.run(
        [
            sys.executable, "-c", _CHILD, str(root),
            arm["scenario"], str(arm["flights"]), str(arm["iterations"]),
            arm["solver"], arm["gap_metric"], str(ladder), kernel, str(workers),
            str(greedy_budget), str(bootstrap_roots), objective, bootstrap_ranking,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"arm failed in {root}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@FINGERPRINT@@"):
            return json.loads(line[len("@@FINGERPRINT@@"):])
    sys.stderr.write(proc.stdout + proc.stderr)
    raise SystemExit(f"no fingerprint emitted from {root}")


def _extract(ref: str, into: Path) -> Path:
    """Materialize ``ref`` as a plain source tree (no worktree, no index churn)."""

    into.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", ref], cwd=REPO_ROOT, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive.stdout, check=True)
    return into


# Fields a legitimate acceleration must never move.
_COMPARED = (
    "objective", "selected_flights", "n_columns", "iterations",
    "termination_reason", "denied_flight_ids", "column_sha",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref",
        help="git ref to compare the working tree against (e.g. origin/main). "
             "Omit to fingerprint the working tree only.",
    )
    parser.add_argument(
        "--reference-baseline", action="store_true",
        help="compare the compiled search against the pure-Python reference INSIDE this "
             "tree, instead of against a git ref. Needs no ref and is the right gate for "
             "anything that changes the master's pool rather than the search.",
    )
    parser.add_argument(
        "--sequential-baseline", action="store_true",
        help="compare an N-worker sweep against the SEQUENTIAL sweep in this same tree. "
             "The speedup arm: worker count is forbidden to change the answer, so this "
             "reports a speedup and proves the fingerprint held at the same time.",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N",
        help="n_workers for the tree arm (default 0 = sequential). With "
             "--sequential-baseline the baseline arm is pinned to 0; otherwise both arms "
             "get N.",
    )
    parser.add_argument(
        "--ladder", type=int, default=0, metavar="N",
        help="pin seed_ladder_steps on BOTH arms (default 0). See the module docstring: a "
             "ladder changes the duals, so a laddered tree vs an unladdered ref diverges "
             "for reasons unrelated to the kernel.",
    )
    parser.add_argument(
        "--greedy-budget", type=float, default=1e6, metavar="S",
        help="pin greedy_budget_s_per_flight on BOTH arms (default 1e6, i.e. effectively "
             "unbounded). The greedy stage is bounded by a WALL CLOCK, so at the shipped "
             "default a faster tree reaches more candidates, gets a different "
             "best_heuristic, and therefore different duals -- every column then diverges "
             "for a reason that is not the change under test. Lifting the clock is what "
             "makes this harness able to gate an acceleration at all. Ignored on refs whose "
             "ColGenParams has no such field.",
    )
    parser.add_argument(
        "--objective", default="total_cost", choices=("total_delay", "total_cost"),
        help="pin the objective on BOTH arms (default total_cost, the shipped value). This "
             "default MOVED off total_delay, so a ref from before the flip would otherwise "
             "price 1:1 while this tree prices 1:3 and every column would diverge for a "
             "reason that is not an acceleration. Pass --objective total_delay to compare "
             "against such a ref.",
    )
    parser.add_argument(
        "--bootstrap-ranking", default="score", choices=("score", "bound"),
        help="pin the bootstrap's root ordering on BOTH arms. Ordering only, but it "
             "changes the cutoff and hence which equally-optimal column returns.",
    )
    parser.add_argument(
        "--bootstrap-roots", type=int, default=0, metavar="K",
        help="pin bootstrap_roots on BOTH arms (default 0). Answer-affecting, so a tree with "
             "it on compared against a ref without it diverges for a reason that is not the "
             "kernel. Its own gate is --reference-baseline, which reruns this tree with the "
             "compiled search disabled: the bootstrap ranks roots once and hands the same "
             "allowlist to whichever search runs, so the two must agree exactly.",
    )
    parser.add_argument(
        "--arm", action="append", choices=sorted(ARMS),
        help="restrict to one arm; repeatable. Default: all three.",
    )
    parser.add_argument(
        "--flights", type=int, default=None, metavar="N",
        help="override every selected arm's flight count. Both arms still get the same "
             "number, so the comparison stays fixed-work.",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, metavar="N",
        help="override every selected arm's iteration count.",
    )
    args = parser.parse_args()
    chosen = [
        name for name, on in (
            ("--ref", bool(args.ref)),
            ("--reference-baseline", args.reference_baseline),
            ("--sequential-baseline", args.sequential_baseline),
        ) if on
    ]
    if len(chosen) > 1:
        parser.error(f"{', '.join(chosen)} are alternative baselines; pick one")
    if args.ladder < 0:
        parser.error("--ladder must be non-negative")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.sequential_baseline and args.workers == 0:
        parser.error("--sequential-baseline compares against 0 workers; set --workers > 0")
    names = args.arm or sorted(ARMS)

    def say(text: str) -> None:
        """Flushed: an arm is minutes to tens of minutes, and this is routinely run
        redirected or in the background, where block buffering shows nothing until the
        whole comparison has finished."""
        print(text, flush=True)

    baseline_root = None
    baseline_kernel = "on"
    baseline_workers = args.workers
    baseline_label = args.ref or ""
    tmp = None
    if args.ref:
        tmp = tempfile.TemporaryDirectory(prefix="colgen_parity_")
        baseline_root = _extract(args.ref, Path(tmp.name) / "tree")
        say(f"baseline: {args.ref} -> {baseline_root}")
    elif args.reference_baseline:
        baseline_root = REPO_ROOT
        baseline_kernel = "off"
        baseline_label = "reference"
        say("baseline: this tree with the compiled kernel disabled")
    elif args.sequential_baseline:
        baseline_root = REPO_ROOT
        baseline_workers = 0
        baseline_label = "seq"
        say("baseline: this tree with the sequential (0-worker) sweep")
    say(f"seed_ladder_steps pinned to {args.ladder} on every arm")
    say(f"workers: tree={args.workers} baseline={baseline_workers}")

    failures = 0
    for name in names:
        arm = dict(ARMS[name])
        if args.flights is not None:
            arm["flights"] = args.flights
        if args.iterations is not None:
            arm["iterations"] = args.iterations
        say(f"\n=== {name}: {arm['scenario']} x{arm['flights']} "
            f"iters={arm['iterations']} {arm['solver']} gap={arm['gap_metric']} "
            f"ladder={args.ladder} ===")
        current = _run_arm(
            REPO_ROOT, arm, args.ladder, "on", args.workers, args.greedy_budget,
            args.bootstrap_roots, args.objective, args.bootstrap_ranking,
        )
        say(f"  tree     {current['wall_s']:8.2f}s pricing={current['pricing_wall_s']:8.2f}s "
            f"obj={current['objective']} sel={current['selected_flights']} "
            f"cols={current['n_columns']} sha={current['column_sha']} "
            f"w={current['workers']} rss={current['rss_self_mb']:.0f}"
            f"+{current['rss_children_mb']:.0f}MB "
            f"greedy_done={current['greedy_completed']}/{current['greedy_elapsed_s']}s")
        if baseline_root is None:
            continue
        base = _run_arm(
            baseline_root, arm, args.ladder, baseline_kernel, baseline_workers,
            args.greedy_budget, args.bootstrap_roots, args.objective, args.bootstrap_ranking,
        )
        say(f"  {baseline_label:<8.8} {base['wall_s']:8.2f}s "
            f"pricing={base['pricing_wall_s']:8.2f}s "
            f"obj={base['objective']} sel={base['selected_flights']} "
            f"cols={base['n_columns']} sha={base['column_sha']} "
            f"w={base['workers']} rss={base['rss_self_mb']:.0f}"
            f"+{base['rss_children_mb']:.0f}MB "
            f"greedy_done={base['greedy_completed']}/{base['greedy_elapsed_s']}s")
        # A ref predating `seed_ladder_steps` accepts no such kwarg and runs unladdered.
        # Every column would then differ, which looks exactly like a kernel divergence --
        # so refuse the comparison outright rather than report a mismatch it cannot explain.
        if current["seed_ladder_steps"] != base["seed_ladder_steps"]:
            failures += 1
            say(f"  UNCOMPARABLE: tree ran seed_ladder_steps="
                f"{current['seed_ladder_steps']} but {baseline_label} ran "
                f"{base['seed_ladder_steps']} -- the baseline predates the parameter, so "
                f"rerun with --ladder 0 or pick a newer ref")
            continue
        # The greedy's budget, which a ref predating the field ignores.  Unlike the ladder
        # this is NOT refused on inequality alone: the budget is a clock, and a clock that
        # never bound changed nothing.  `greedy_completed` on both arms is exactly that
        # evidence, so the guard is on the budget differing AND one of them having stopped
        # early -- which is the only case where the two started from different incumbents.
        budgets_differ = (
            current["greedy_budget_s_per_flight"] != base["greedy_budget_s_per_flight"]
        )
        if budgets_differ and not (current["greedy_completed"] and base["greedy_completed"]):
            failures += 1
            say(f"  UNCOMPARABLE: tree ran greedy_budget_s_per_flight="
                f"{current['greedy_budget_s_per_flight']} but {baseline_label} ran "
                f"{base['greedy_budget_s_per_flight']}, and at least one greedy stopped "
                f"early -- so the two started from different incumbents")
            continue
        # Two arms in different cost CURRENCIES price different problems, so every column
        # diverges -- the same trap as the ladder, refused rather than reported as a
        # regression.  Compared by value, not feature-tested: unlike `bootstrap_roots`
        # below, `objective` exists on every ref this harness can reach.
        if current.get("objective_mode") != base.get("objective_mode"):
            failures += 1
            say(f"  UNCOMPARABLE: tree ran objective={current.get('objective_mode')} but "
                f"{baseline_label} ran {base.get('objective_mode')} -- these are different "
                f"cost currencies (1:3 vs 1:1), not a faster search. Rerun both with "
                f"--objective {base.get('objective_mode')}")
            continue
        # `None` means the ref PREDATES the field, which is the same behaviour as 0 --
        # no bootstrap either way -- so normalise before comparing.  Comparing raw made a
        # ref older than `bootstrap_roots` UNCOMPARABLE against an unbootstrapped tree,
        # while telling the caller to "rerun with --bootstrap-roots 0", which is what they
        # had already done.  Only a tree that actually bootstraps against a ref that cannot
        # is a real mismatch.
        if (current["bootstrap_roots"] or 0) != (base["bootstrap_roots"] or 0):
            failures += 1
            say(f"  UNCOMPARABLE: tree ran bootstrap_roots={current['bootstrap_roots']} but "
                f"{baseline_label} ran {base['bootstrap_roots']} -- a ref predating the "
                f"parameter cannot bootstrap, so compare at --bootstrap-roots 0 or pick a "
                f"newer ref")
            continue
        # The greedy's stop reason, not merely whether it stopped.  A stage capped by its
        # CANDIDATE limit stops at the same place in both arms and is fine; one stopped by
        # the CLOCK stops wherever the machine happened to be, so the two arms started from
        # different `best_heuristic` incumbents and nothing downstream is comparable.
        clock_stopped = [
            label for label, arm_fp in ((baseline_label, base), ("tree", current))
            if not arm_fp["greedy_completed"] and not arm_fp["greedy_candidate_capped"]
        ]
        if clock_stopped:
            failures += 1
            say(f"  UNCOMPARABLE: the greedy ran out of CLOCK in {', '.join(clock_stopped)}"
                f" -- raise --greedy-budget (currently {args.greedy_budget:g} s/flight)")
            continue
        diffs = [f for f in _COMPARED if current[f] != base[f]]
        if diffs:
            failures += 1
            say(f"  MISMATCH on {', '.join(diffs)}")
            for field in diffs:
                say(f"    {field}: tree={current[field]!r} {baseline_label}={base[field]!r}")
        else:
            speedup = base["wall_s"] / current["wall_s"] if current["wall_s"] else float("nan")
            say(f"  IDENTICAL on all {len(_COMPARED)} fields -- {speedup:.2f}x")

    if tmp is not None:
        tmp.cleanup()
    if failures:
        say(f"\n{failures} arm(s) DIVERGED")
        return 1
    say(
        f"\nall arms identical vs {baseline_label}"
        if baseline_root is not None
        else "\nfingerprinted (no baseline requested)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
