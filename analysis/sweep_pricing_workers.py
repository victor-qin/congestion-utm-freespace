"""Is the pricing pool worth using again, now that the straggler is gone?

The pool's problem was never that it lost -- ``[[colgen-parallel-pricing-pool]]`` measured
4 workers at 2.62x back on 2026-08-04.  It was that ONE FLIGHT was 87.6% of the sweep,
which caps any worker pool at 1.14x however many cores it gets, and that peak RSS went
3.5 GB sequential to 7.9 GB at 4 workers.  Two things have moved since:

* ``objective=total_cost`` took the straggler's share to 20.6-28.3%, so the Amdahl cap is
  now **3.5-4.9x**;
* the label arena is flat and lazily mapped, so a worker no longer climbs the label ladder
  from 2^16 on a cold ``dag_budget`` -- though the STATE ladder still does, and that is
  per worker per sweep because ``_init_worker`` rebuilds every graph.

Do NOT reach for ``[[parallel-loses-on-density-scenarios]]`` here, as an earlier draft of
this docstring did: its 0.66-0.74x is the A* SPECULATIVE PARALLEL RUNNER, where the serial
commit floor dominates a compiled per-flight plan.  Different mechanism, different
bottleneck, and citing it made the pricing pool look like a reopened question when the
open question was only ever memory.

Against those, the pool got more expensive in one way: ``MAX_LABEL_CAPACITY`` is now 2^26,
so each worker maps 2.68 GB of arena.  Lazily backed, so virtual rather than resident --
and whether that holds in practice is the number that decides if this is usable on a
4 GB/core node, because an OOM-killed worker does not fail the sweep, it HANGS it.

**That number is ``rss_peak``, sampled across the process TREE, and it is not the one this
script used to print.**  ``getrusage(RUSAGE_CHILDREN).ru_maxrss`` is the largest SINGLE
child by definition, never the sum, so it reads flat however many workers run and cannot
distinguish "the arena costs nothing" from "aggregate memory is linear in workers".  It
reported flat here and the conclusion drawn from it -- that the memory objection to
defaulting the pool on had lifted -- was an artifact.  See ``_tree_rss_mib``.

Not `prof_colgen_cutoff`: its probes hook module-level functions and would instrument only
the parent, silently reporting the workers' searches as missing.  This measures the solve
from outside instead, and checks the invariant that makes a worker count safe to change --
the SCHEDULE must not move, fingerprinted the way the parity harness does rather than by
objective and a flight COUNT, which a different flight set passes unchanged.

    uv run python analysis/sweep_pricing_workers.py --flights 50
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

def _tree_rss_mib() -> float:
    """Summed RSS of this process and every descendant, right now, in MiB.

    NOT `getrusage(RUSAGE_CHILDREN).ru_maxrss`, which this script used to report and which
    cannot answer the question the pool poses.  POSIX defines that field as the largest
    SINGLE child, never the sum across the tree, so it reads FLAT however many workers run
    -- measured directly: 1, 2 and 4 concurrent children each touching 150 MiB all report
    172 MiB.  Reporting it as "rss_kids" made a pool whose aggregate scales linearly look
    like one that costs nothing, which is exactly backwards for the decision it informed.

    `ps` rather than a dependency, and a full descendant walk rather than direct children,
    because `mp.Pool` workers are direct children but anything they spawn is not.
    """

    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return float("nan")
    children: dict[int, list[int]] = {}
    rss_kib: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss_kib[pid] = rss
    total, stack = 0, [os.getpid()]
    while stack:
        pid = stack.pop()
        total += rss_kib.get(pid, 0)
        stack.extend(children.get(pid, ()))
    return total / 1024.0          # `ps` reports KiB on both macOS and Linux


class _RssSampler:
    """Poll the process tree's summed RSS and keep the high-water mark."""

    def __init__(self, period_s: float = 0.25) -> None:
        self.peak_mib = 0.0
        self._period = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = _tree_rss_mib()
            if now == now and now > self.peak_mib:   # NaN-safe
                self.peak_mib = now
            self._stop.wait(self._period)

    def __enter__(self) -> "_RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _schedule_sha(result) -> str:
    """A fingerprint of the SCHEDULE, not merely of its price.

    `objective` plus a selected-flight COUNT is not an identity check: a different set of
    flights, or a different equally-priced column for the same flight, passes it unchanged.
    Same fields the parity harness hashes, so a divergence here means the same thing there.
    """

    h = hashlib.sha1()
    for flight_id, column in sorted(result.columns.items()):
        h.update(repr((
            int(flight_id),
            int(column.departure_step),
            int(column.level),
            column.origin_lane_idx,
            column.dest_lane_idx,
            repr(column.delay_s),
            tuple(column.cell_path),
            tuple(sorted(repr(row) for row in column.claims)),
        )).encode())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+",
                        default=["density_faa_wing_zipline", "density_future_wing_zipline"])
    parser.add_argument("--flights", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--bootstrap-roots", type=int, default=2)
    parser.add_argument("--bootstrap-ranking", default="score", choices=("score", "bound"))
    parser.add_argument("--objective", default="total_cost")
    # MUST be pinned, and "cost" not the ColGenParams default of "revenue": every
    # `prof_colgen_cutoff` arm in this investigation used "cost", and the two metrics
    # terminate the loop differently -- so leaving it defaulted silently produced a
    # DIFFERENT solve (objective 5703.06 against 5663.06) that was not comparable to
    # anything else measured.
    parser.add_argument("--gap-metric", default="cost", choices=("cost", "revenue"))
    # The post-first-LP greedy is bounded by a WALL CLOCK -- `greedy_budget_s_per_flight *
    # n_flights` -- and it produces `best_heuristic`, the `known_column` cutoff every
    # pricing call is handed.  So a machine that is busier or hotter reaches fewer
    # candidates, the duals move, and every column changes for a reason that has NOTHING to
    # do with the worker count under test.  `[[greedy-clock-defeats-parity]]`.  Pin it high
    # enough that the stage always finishes, exactly as `ab_colgen_parity --greedy-budget`
    # does, or the comparison measures the clock.
    parser.add_argument("--greedy-budget", type=float, default=None, metavar="S")
    args = parser.parse_args()

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  x{args.flights} iters={args.iterations} "
          f"objective={args.objective} bootstrap_roots={args.bootstrap_roots} "
          f"ranking={args.bootstrap_ranking}")
    print(f"{'scenario':<30} {'w':>2} {'WALL':>9} {'pricing':>9} {'speedup':>8} "
          f"{'rss_peak':>9} {'per_wkr':>8} {'sha':>16} {'objective':>20} {'sel':>4} "
          f"{'cols':>6} {'it':>3} {'termination':>18} {'greedy':>7} {'g_done':>7}")

    for scenario in args.scenarios:
        spec = get_scenario(scenario)
        cfg = spec.config()
        demand = spec.demand_model()
        requests = sorted(
            demand.generate(cfg, np.random.default_rng(cfg.seed)),
            key=lambda r: r.flight_id,
        )[: args.flights]
        static_terms = list(demand.terminals(cfg))

        base_wall = None
        base_rss = None
        baseline_key = None
        for workers in args.workers:
            params = ColGenParams(
                max_iterations=args.iterations,
                time_limit_s=86400.0,
                objective=args.objective,
                gap_metric=args.gap_metric,
                bootstrap_roots=args.bootstrap_roots,
                bootstrap_ranking=args.bootstrap_ranking,
                # The worker count IS a `ColGenParams` field, and setting it here is the
                # whole mechanism -- `price_sweep` reads `params.n_pricing_workers`
                # directly.  This used to be the opposite: `solve` took a separate
                # `parallel=ParallelPricingConfig(...)` keyword and `n_pricing_workers=`
                # on the params object was accepted and SILENTLY IGNORED, which is how the
                # first version of this script measured six sequential runs and reported
                # them as a worker sweep (`rss_kids = 0` was the tell).  The assertion
                # below survives that history deliberately: it is cheap, and it is what
                # turns "the knob did not take effect" into a failed row instead of a
                # plausible number.
                n_pricing_workers=workers,
                **({} if args.greedy_budget is None
                   else {'greedy_budget_s_per_flight': args.greedy_budget}),
            )
            # One line per column-generation iteration, flushed, so a long gap-terminated
            # run is legible WHILE it runs rather than only in its final row.  `lp_gap_cost`
            # against `params.lp_gap` is the thing actually deciding when this stops.
            t_iter = [time.perf_counter()]

            def _on_iteration(state, _t=t_iter):
                now = time.perf_counter()
                print(
                    f"    iter {state.get('iteration'):>3}  "
                    f"{now - _t[0]:>7.1f}s  "
                    f"lp_obj={state.get('lp_objective'):>18.4f}  "
                    f"gap_cost={state.get('lp_gap_cost'):>10.3e}  "
                    f"gap_rev={state.get('lp_gap_revenue'):>10.3e}  "
                    f"cost_ub={state.get('cost_upper_bound'):>14.4f}",
                    flush=True,
                )
                _t[0] = now

            started = time.perf_counter()
            with _RssSampler() as sampler:
                result = ColGenSolver().solve(
                    requests, cfg, static_terms, params,
                    on_iteration=_on_iteration,
                )
            wall = time.perf_counter() - started
            stats = result.stats
            rss_peak = sampler.peak_mib
            # Charged against the SEQUENTIAL arm, so the column answers the question the
            # default flip actually turns on: what does adding a worker cost in resident
            # memory.  Linear here means the pool is bounded by RAM, not by cores.
            if base_rss is None:
                base_rss = rss_peak
            per_worker = ((rss_peak - base_rss) / workers) if workers else 0.0
            if base_wall is None:
                base_wall = wall
            # The invariant that makes a worker count safe to change at all.  `price_sweep`
            # reproduces the sequential loop's ACCEPTED PREFIX and index order, so these two
            # must be identical -- if they are not, the speedup is measuring a different
            # answer and is meaningless.
            sha = _schedule_sha(result)
            key = (repr(stats.get("objective")), stats.get("selected_flights"), sha)
            if baseline_key is None:
                baseline_key = key
            flag = "" if key == baseline_key else "  <-- ANSWER MOVED, speedup is void"
            # The stat is derived from what the solver ACTUALLY used, so a mismatch here
            # means the knob did not take effect and the row is not a parallel measurement.
            # Only when the stage actually RAN.  At `greedy_budget_s_per_flight == 0` it is
            # disabled, and `initial_greedy_completed` is False for the structural reason
            # rather than the clock -- flagging that reads as a warning about a stage that
            # did not happen.  Note the flag is imprecise even when it does fire: above
            # ~256 flights `solver.py` sets `completed = len(order) <= candidate_limit`
            # BEFORE consulting any deadline, so a large batch is marked incomplete by the
            # candidate cap, which lands identically in both arms and is fine.
            greedy_ran = (params.greedy_budget_s_per_flight > 0.0
                          and not stats.get("initial_greedy_completed", True))
            if greedy_ran:
                flag += ("  <-- GREEDY DID NOT COMPLETE ("
                         f"{stats.get('initial_greedy_elapsed_s', 0.0):.1f}s); if that was "
                         "the CLOCK rather than the 256-candidate cap, its cutoff differs "
                         "and this row is not comparable")
            got = stats.get("n_pricing_workers")
            if got != workers:
                flag += f"  <-- ASKED {workers} WORKERS, SOLVER USED {got}"
            print(
                f"{scenario:<30} {workers:>2} {wall:>9.2f} "
                f"{float(stats.get('pricing_wall_s', 0.0)):>9.2f} "
                f"{base_wall / wall:>7.2f}x "
                f"{rss_peak:>8.0f}M {per_worker:>7.0f}M {sha:>16} "
                f"{stats.get('objective'):>20} {stats.get('selected_flights'):>4} "
                f"{stats.get('n_columns'):>6} "
                f"{stats.get('iterations'):>3} "
                f"{str(stats.get('termination_reason')):>18} "
                f"{float(stats.get('initial_greedy_elapsed_s', 0.0)):>7.1f} "
                f"{str(stats.get('initial_greedy_completed')):>7}{flag}"
            )
    print("WORKER SWEEP DONE")


if __name__ == "__main__":
    main()
