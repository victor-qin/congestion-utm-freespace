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

Examples:

    # compare the working tree against origin/main on all three default arms
    uv run python analysis/ab_colgen_parity.py --ref origin/main

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
import hashlib, json, sys, time
from pathlib import Path

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
spec = get_scenario(spec_name)
cfg = spec.config()
demand = spec.demand_model()
requests = demand.generate(cfg, np.random.default_rng(cfg.seed))
requests = sorted(requests, key=lambda r: r.flight_id)[:n_flights]
static_terms = list(demand.terminals(cfg))

params = ColGenParams(
    solver=backend, max_iterations=iterations, time_limit_s=86400.0, gap_metric=gap_metric
)
started = time.perf_counter()
result = ColGenSolver().solve(requests, cfg, static_terms, params)
wall = time.perf_counter() - started

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
    "tree": str(loaded.parent.parent),
}))
'''


def _run_arm(root: Path, arm: dict) -> dict:
    """Fingerprint one arm in a child interpreter rooted at ``root``."""

    proc = subprocess.run(
        [
            sys.executable, "-c", _CHILD, str(root),
            arm["scenario"], str(arm["flights"]), str(arm["iterations"]),
            arm["solver"], arm["gap_metric"],
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
        "--arm", action="append", choices=sorted(ARMS),
        help="restrict to one arm; repeatable. Default: all three.",
    )
    args = parser.parse_args()
    names = args.arm or sorted(ARMS)

    def say(text: str) -> None:
        """Flushed: an arm is minutes to tens of minutes, and this is routinely run
        redirected or in the background, where block buffering shows nothing until the
        whole comparison has finished."""
        print(text, flush=True)

    baseline_root = None
    tmp = None
    if args.ref:
        tmp = tempfile.TemporaryDirectory(prefix="colgen_parity_")
        baseline_root = _extract(args.ref, Path(tmp.name) / "tree")
        say(f"baseline: {args.ref} -> {baseline_root}")

    failures = 0
    for name in names:
        arm = ARMS[name]
        say(f"\n=== {name}: {arm['scenario']} x{arm['flights']} "
            f"iters={arm['iterations']} {arm['solver']} gap={arm['gap_metric']} ===")
        current = _run_arm(REPO_ROOT, arm)
        say(f"  tree     {current['wall_s']:8.2f}s pricing={current['pricing_wall_s']:8.2f}s "
            f"obj={current['objective']} sel={current['selected_flights']} "
            f"cols={current['n_columns']} sha={current['column_sha']}")
        if baseline_root is None:
            continue
        base = _run_arm(baseline_root, arm)
        say(f"  {args.ref:<8.8} {base['wall_s']:8.2f}s pricing={base['pricing_wall_s']:8.2f}s "
            f"obj={base['objective']} sel={base['selected_flights']} "
            f"cols={base['n_columns']} sha={base['column_sha']}")
        diffs = [f for f in _COMPARED if current[f] != base[f]]
        if diffs:
            failures += 1
            say(f"  MISMATCH on {', '.join(diffs)}")
            for field in diffs:
                say(f"    {field}: tree={current[field]!r} {args.ref}={base[field]!r}")
        else:
            speedup = base["wall_s"] / current["wall_s"] if current["wall_s"] else float("nan")
            say(f"  IDENTICAL on all {len(_COMPARED)} fields -- {speedup:.2f}x")

    if tmp is not None:
        tmp.cleanup()
    if failures:
        say(f"\n{failures} arm(s) DIVERGED")
        return 1
    say("\nall arms identical" if args.ref else "\nfingerprinted (no baseline requested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
