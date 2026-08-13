"""Is the pricing pool worth using again, now that the straggler is gone?

``[[parallel-loses-on-density-scenarios]]`` measured the pool LOSING to sequential on
exactly these two scenarios -- 0.66-0.74x -- and the reason was Amdahl, not overhead: one
flight was 87.6% of the sweep, which caps ANY worker pool at 1.14x however many cores it
gets.  Two things have moved since:

* ``objective=total_cost`` took the straggler's share to 20.6-28.3%, so the cap is now
  **3.5-4.9x**;
* the label arena is flat and lazily mapped, so a worker no longer climbs the label ladder
  from 2^16 on a cold ``dag_budget`` -- though the STATE ladder still does, and that is
  per worker per sweep because ``_init_worker`` rebuilds every graph.

Against those, the pool got more expensive in one way: ``MAX_LABEL_CAPACITY`` is now 2^26,
so each worker maps 2.68 GB of arena.  Lazily backed, so virtual rather than resident, but
``rss_children`` is the number that says whether that is true in practice, and it is the
one that decides whether this is usable on a 4 GB/core cluster node.

Not `prof_colgen_cutoff`: its probes hook module-level functions and would instrument only
the parent, silently reporting the workers' searches as missing.  This measures the solve
from outside instead, and checks the one invariant that makes a worker count safe to
change -- the OBJECTIVE and the selected-flight set must not move.

    uv run python analysis/sweep_pricing_workers.py --flights 50
"""
from __future__ import annotations

import argparse
import resource
import sys
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

# macOS reports ru_maxrss in bytes, Linux in kibibytes.
_RSS_SCALE = 1024 * 1024 if sys.platform == "darwin" else 1024


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
          f"{'rss_self':>9} {'rss_kids':>9} {'objective':>20} {'sel':>4} {'cols':>6} {'it':>3} {'termination':>18} {'greedy':>7} {'g_done':>7}")

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
            result = ColGenSolver().solve(
                requests, cfg, static_terms, params,
                on_iteration=_on_iteration,
            )
            wall = time.perf_counter() - started
            stats = result.stats
            rss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_SCALE
            rss_kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / _RSS_SCALE
            if base_wall is None:
                base_wall = wall
            # The invariant that makes a worker count safe to change at all.  `price_sweep`
            # reproduces the sequential loop's ACCEPTED PREFIX and index order, so these two
            # must be identical -- if they are not, the speedup is measuring a different
            # answer and is meaningless.
            key = (repr(stats.get("objective")), stats.get("selected_flights"))
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
                f"{rss_self:>8.0f}M {rss_kids:>8.0f}M "
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
