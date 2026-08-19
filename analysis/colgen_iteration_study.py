"""What each column-generation iteration is actually worth: LP, gap, and the IP from it.

The question this answers is "could we have stopped here, and what would we have got".
Three numbers per iteration, none of which a normal run reports together:

* the **LP objective** -- the relaxation's value at that iteration;
* **both gap metrics**, and whether either would have terminated the solve at the shipped
  thresholds.  ``gap_metric`` welds two independent choices together -- the reporting SCALE
  and the termination GATE -- so a run tells you what one of them says and hides the other.
  The revenue gap is normalized by an objective whose scale includes ``n * M``, and ``M`` is
  an artificial constant chosen to make cancellation unattractive, so it largely measures
  how big ``M`` was set; the cost gap is normalized by total cost and is far stricter.
* the **IP solved from that iteration's pool**, which is the thing an operator would
  actually fly.  The LP bound and the heuristic bracket it, and neither is it.

TWO THINGS THIS RUN IS NOT.

**It is not a production replica once ``--ip-every-iteration`` is on.** ``solve_ip``
separates violated claim rows and MATERIALIZES them (master.py), which is permanent: the
next LP sees rows a production run would not have had yet, so it is tighter and the
trajectory diverges from the untouched one.  The direction is "more constrained sooner",
not random, but it is a perturbation and the ``--no-ip-every-iteration`` arm exists to
measure against.

**It calls the solver directly rather than through ``run_batch``**, so there is no DSS
filing and no intent translation.  Everything that decides LP and IP behaviour -- params,
ladder, worker pool, gap metric -- is production-shaped; what is skipped is downstream of
the solve and cannot change it.

CHECKPOINTING.  ``--checkpoint-out`` writes the master's column pool as JSON and
``--checkpoint-in`` feeds it back through ``solve(seed_columns=...)``.  What that restores
is the POOL -- the expensive part, hours of pricing.  What it cannot restore is the LP basis
or the duals, so iteration 6 of a restarted run is not iteration 6 of a continuous one: it
is iteration 1 of a solve that happens to start with a very good pool.  Say so when quoting
it.

Only the seven fields that IDENTIFY a column are stored, not its claims.  That is not a
space optimisation -- ``solve`` routes every seeded column through ``_canonical_column``,
which recomputes the claim set from the graph, so a stored one would be recomputed and
discarded.  Dropping it makes the checkpoint plain JSON rather than a pickle (no arbitrary
code execution when reading a file someone else produced), around thirty times smaller, and
diffable.

    uv run python analysis/colgen_iteration_study.py --flights 500 --iterations 5 \
        --workers 8 --checkpoint-out runs/study500/pool.json --out runs/study500/study.json
    # resume
    uv run python analysis/colgen_iteration_study.py --flights 500 --iterations 5 \
        --workers 8 --checkpoint-in runs/study500/pool.json
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.planner.colgen.translate import Column  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

_RSS_SCALE = 2**20 if sys.platform == "darwin" else 1024

# The thresholds a shipped run would have been judged against.  Reported as "would this
# have stopped here", never applied -- this study is bounded by its iteration cap alone.
_SHIPPED_LP_GAP = 1e-4
_SHIPPED_IP_GAP = 1e-3


def _finite(value):
    """JSON has no inf/nan; keep them as strings rather than silently emitting null."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _fmt(value, width: int = 11, digits: int = 5) -> str:
    """Format a `_finite`-processed value, which may already be the string 'nan'/'inf'.

    Needed because the values worth printing are exactly the ones that start out
    non-finite: `dual_l2` has no previous iteration to difference against on iteration 1,
    and an unbounded master reports `inf`.  A bare ``:g`` raises on those, which turned a
    diagnostic into a crash of the run it was diagnosing.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:<{width}.{digits}g}"
    return f"{value!s:<{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flights", type=int, default=500)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--solver", default="highs")
    ap.add_argument(
        "--gap-metric", default="revenue",
        help="the SHIPPED default; both scales are reported regardless.",
    )
    ap.add_argument("--ladder", type=int, default=None, help="default: the shipped value")
    ap.add_argument(
        "--ladder-stride", type=int, default=None,
        help="spacing between ladder rungs in lattice steps (default 1 = consecutive). "
             "stride 3 at dt_s=4 spans 240 s with the same 20 columns. ANSWER-AFFECTING.",
    )
    ap.add_argument(
        "--ip-gap", type=float, default=0.0,
        help="relative tolerance for the FINAL MILP. The study default of 0.0 forces a "
             "proof of exact optimality over the whole pool, which does not terminate at "
             "scale; 1e-3 is the shipped value and is what a production run would use.",
    )
    ap.add_argument(
        "--M", type=float, default=None, dest="big_m",
        help="per-flight benefit (default: the shipped 1e6). Only has to exceed the "
             "priciest column so denial is never attractive -- measured max_column_cost is "
             "891.6 at 2,000 density flights, so the shipped value is ~1,100x its floor. "
             "ANSWER-AFFECTING, and below the floor it makes denial CHEAPER THAN FLYING: "
             "watch `selected_flights` in the summary, not just the timings.",
    )
    ap.add_argument(
        "--lns", type=int, default=0,
        help="flights released per LNS try (0 = the shipped randomized rounding). LNS "
             "starts from the incumbent and swaps one flight at a time, so coverage is "
             "invariant and the result can never be worse than what it started with.",
    )
    ap.add_argument(
        "--contested-rows", type=int, default=0,
        help="separate capacity rows on the COLUMN POOL as well as the LP solution: "
             "materialize unmaterialized rows that at least cap+N-1 distinct flights claim "
             "(0 = off, the shipped behaviour). ANSWER-AFFECTING.",
    )
    ap.add_argument(
        "--contested-rows-limit", type=int, default=0,
        help="cap how many rows one pool-based pass may materialize, most-contested first "
             "(0 = unlimited).",
    )
    ap.add_argument(
        "--max-columns", type=int, default=0,
        help="bank at most this many priced columns per iteration, top-k by reduced cost "
             "(0 = all, the shipped behaviour). ANSWER-AFFECTING: a capped iteration banks "
             "a different pool, so the duals and the next subproblem differ.",
    )
    ap.add_argument(
        "--no-ip-every-iteration", action="store_true",
        help="skip the per-iteration IP. Leaves the master unperturbed, so this arm is the "
             "production-shaped control for the one that solves it.",
    )
    ap.add_argument("--ip-deadline-s", type=float, default=300.0)
    ap.add_argument("--checkpoint-out", default=None)
    ap.add_argument("--checkpoint-in", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))

    # Shipped defaults except the two gates, which are pinned OPEN so the iteration cap is
    # the only thing that stops the solve.  `ip_gap=0` also keeps the final IP from being
    # skipped, which the shipped value allows whenever the heuristic proves the gap.
    params = ColGenParams(
        solver=args.solver,
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        lp_gap=0.0,
        ip_gap=args.ip_gap,
        n_pricing_workers=args.workers,
        max_columns_per_iteration=args.max_columns,
        lns_destroy_flights=args.lns,
        contested_row_separation=args.contested_rows,
        contested_rows_limit=args.contested_rows_limit,
        **({} if args.big_m is None else {"M": args.big_m}),
        **({} if args.ladder is None else {"seed_ladder_steps": args.ladder}),
        **({} if args.ladder_stride is None else {"seed_ladder_stride": args.ladder_stride}),
    )

    seed_columns = None
    if args.checkpoint_in:
        loaded = json.loads(Path(args.checkpoint_in).read_text())
        known = {request.flight_id for request in requests}
        restored = {
            int(flight_id): [_column_from_record(record) for record in records]
            for flight_id, records in loaded["columns_by_flight"].items()
        }
        seed_columns = {
            flight_id: columns
            for flight_id, columns in restored.items()
            if flight_id in known
        }
        dropped = len(restored) - len(seed_columns)
        print(
            json.dumps({
                "resumed_from": args.checkpoint_in,
                "checkpoint_flights": len(restored),
                "checkpoint_columns": sum(len(v) for v in restored.values()),
                "restored_flights": len(seed_columns),
                "dropped_flights_not_in_batch": dropped,
                "note": "restores the POOL only; LP basis and duals start cold",
            }),
            flush=True,
        )

    header = {
        "scenario": args.scenario, "flights": len(requests), "solver": args.solver,
        "max_iterations": args.iterations, "workers": args.workers,
        "max_columns_per_iteration": params.max_columns_per_iteration,
        "lns_destroy_flights": params.lns_destroy_flights,
        "contested_row_separation": params.contested_row_separation,
        "contested_rows_limit": params.contested_rows_limit,
        # Recorded because `cost_upper_bound` is `n*M - lp_objective`: with zero denials it
        # equals total delay and compares across M, and with any denial it silently carries
        # M per denied flight and does not.  `selected_flights` in the summary is the check.
        "M": params.M,
        "gap_metric": params.gap_metric, "seed_ladder_steps": params.seed_ladder_steps,
        "seed_ladder_stride": params.seed_ladder_stride, "ip_gap": params.ip_gap,
        "greedy_budget_s_per_flight": params.greedy_budget_s_per_flight,
        "greedy_budget_s": params.greedy_budget_s_per_flight * len(requests),
        "ip_every_iteration": not args.no_ip_every_iteration,
        "resumed": bool(args.checkpoint_in),
    }
    print(json.dumps(header, indent=2), flush=True)

    rows: list[dict] = []
    master_ref: dict = {}
    # Index range in `master.columns` that the PREVIOUS iteration's pricing appended.
    # Safe to hold across iterations only because the pool is append-only (master.py:558);
    # if that ever changes these indices silently point at the wrong columns.
    prev_added: tuple[int, int] | None = None

    def on_iteration(state: dict) -> None:
        nonlocal prev_added
        master = state.get("master")
        master_ref["master"] = master
        # THE QUESTION BEHIND A FLAT LP: pricing accepts a column on the SIGN of its
        # reduced cost, which says it improves the basis -- not that the next LP gives it
        # x > 0.  When thousands of positive-rc columns land per iteration and the
        # objective does not move, those two have come apart, and only `lp_x` can tell
        # them apart from outside (the solver's own note at solver.py:1247).  Measured
        # here one iteration late, because "does the LP use it" needs the NEXT LP.
        used = added_span = None
        if prev_added is not None and state.get("lp_x") is not None:
            lo, hi = prev_added
            x = np.asarray(state["lp_x"], dtype=float)
            if hi <= len(x):
                added_span = hi - lo
                used = int((x[lo:hi] > 1e-9).sum())
        row_prev_used, row_prev_span = used, added_span

        # How much of what pricing just proposed was routed through capacity rows the
        # master has never materialized.  Those rows carry NO dual, so pricing values them
        # at zero -- they are free by construction, however contested they really are.  A
        # high, non-decaying fraction is the signature of column generation and row
        # separation chasing each other rather than converging.
        virgin_claims = virgin_total = None
        added_now = state.get("columns_added") or 0
        n_cols_now = state.get("columns") or 0
        if master is not None and added_now:
            materialized = master.materialized_rows          # frozenset, built once
            pool = master.columns                            # tuple copy, taken once
            fresh = pool[n_cols_now - added_now:n_cols_now]
            virgin_total = sum(len(c.claims) for c in fresh)
            virgin_claims = sum(
                1 for c in fresh for r in c.claims if r not in materialized
            )
        prev_added = (n_cols_now - added_now, n_cols_now) if added_now else None

        # WHERE THE BENEFIT WENT.  `n_uncovered=0` beside `n_rc_near_M=945` refutes the
        # explanation `_coverage_diagnostics` was written for -- these flights are covered,
        # so their rc of ~M is not a slack cover constraint.  Complementary slackness names
        # the alternative: for any column the LP uses, `pi_f = M - cost - sum(row duals)`,
        # so a flight prices at ~M exactly when its ROW duals already consumed the benefit
        # and left the cover constraint worth nothing.  That distinguishes a big-M scaling
        # artefact (a few rows priced at ~M) from genuine congestion pricing (the mass
        # spread over many rows), and those want different fixes.
        M = float(params.M)
        duals = state.get("capacity_duals") or {}
        dual_values = np.fromiter(
            (abs(float(v)) for v in duals.values()), dtype=float, count=len(duals)
        ) if duals else np.zeros(0)
        n_dual_near_M = int((dual_values >= 0.5 * M).sum())
        dual_sum = float(dual_values.sum())
        dual_max = float(dual_values.max()) if dual_values.size else 0.0
        # Implied cover duals over the LP's own support, which is where complementary
        # slackness holds.  Costed at ~1M dict lookups an iteration; the alternative is
        # plumbing cover duals out of the backend, which changes the solver to answer an
        # analysis question.
        pi_used: list[float] = []
        x_now = state.get("lp_x")
        if master is not None and x_now is not None:
            get = duals.get
            for value, column in zip(np.asarray(x_now, dtype=float), master.columns,
                                     strict=False):
                if value <= 1e-9:
                    continue
                pi_used.append(
                    M - float(column.delay_s)
                    - math.fsum(float(get(r, 0.0)) for r in column.claims)
                )
        pi_arr = np.asarray(pi_used, dtype=float) if pi_used else np.zeros(0)

        row = {
            "iteration": state["iteration"],
            "lp_objective": _finite(state.get("lp_objective")),
            "upper_bound": _finite(state.get("upper_bound")),
            # This iteration's OWN bound, against the running minimum kept above it, and
            # how many iterations that minimum has stood.  A gap moves when either end
            # moves; only these say which.  `raw` degrading away from `upper_bound` while
            # `bound_frozen_for` climbs is the signature of duals wandering rather than
            # converging -- and it is invisible in the gap, which the improving primal
            # drags in the opposite direction.
            "raw_upper_bound": _finite(state.get("raw_upper_bound")),
            "bound_frozen_for": state.get("bound_frozen_for"),
            "cost_lower_bound": _finite(state.get("cost_lower_bound")),
            "cost_upper_bound": _finite(state.get("cost_upper_bound")),
            # BOTH scales, always.  A run configured for one silently hides the other, and
            # they disagree by orders of magnitude on this instance.
            "lp_gap_revenue": _finite(state.get("lp_gap_revenue")),
            "lp_gap_cost": _finite(state.get("lp_gap_cost")),
            "heuristic_gap_revenue": _finite(state.get("heuristic_gap_revenue")),
            "heuristic_gap_cost": _finite(state.get("heuristic_gap_cost")),
            "heuristic_cost": _finite(state.get("heuristic_cost")),
            "columns": state.get("columns"),
            "columns_added": state.get("columns_added"),
            # What pricing OFFERED against what the cap let in.  Equal when uncapped.
            "columns_priced": state.get("columns_priced"),
            "columns_deferred": state.get("columns_deferred"),
            # Of the previous iteration's priced columns, how many this LP actually uses.
            "prev_added_span": row_prev_span,
            "prev_added_used": row_prev_used,
            # Claims on rows the master had not materialized when this column was priced.
            "virgin_claims": virgin_claims,
            "virgin_claims_total": virgin_total,
            "rc_n_positive": state.get("rc_n_positive"),
            "rc_sum": _finite(state.get("rc_sum")),
            # WHERE `rc_sum` comes from, which decides whether a stuck bound is targetable.
            # The bound is `LP + sum(max(0, rc))`, so a handful of flights at rc ~ M swamp
            # thousands at rc ~ 1.  `rc_max` against `rc_p50` separates those two worlds, and
            # `n_rc_near_M` / `n_overlap` name the mechanism: a flight the LP leaves
            # fractionally uncovered has a zero cover dual, so its rc is roughly the whole
            # benefit (see `_coverage_diagnostics` in solver.py).
            "rc_max": _finite(state.get("rc_max")),
            "rc_p50": _finite(state.get("rc_p50")),
            "rc_p90": _finite(state.get("rc_p90")),
            "n_uncovered": state.get("n_uncovered"),
            "n_rc_near_M": state.get("n_rc_near_M"),
            "n_overlap": state.get("n_overlap"),
            "max_column_cost": _finite(state.get("max_column_cost")),
            # Tailing-off with a frozen bound has two causes needing opposite fixes -- duals
            # converging slowly (run more iterations) or duals oscillating (stabilize).  The
            # solver computes the separator and nothing recorded it.
            "dual_l2": _finite(state.get("dual_l2")),
            "dual_linf": _finite(state.get("dual_linf")),
            "dual_nonzero": state.get("dual_nonzero"),
            # What the LP wanted before the box, and how many rows it refused.
            # Row-dual mass, and how much of it sits at big-M scale.  `n_dual_near_M > 0`
            # means the LP is pricing individual (cell, step) rows at an entire flight's
            # benefit, which is what leaves a covered flight's implied cover dual at ~0.
            "n_dual_near_M": n_dual_near_M,
            "dual_sum": dual_sum,
            "dual_max": dual_max,
            # Implied cover duals over the LP's support: pi_f = M - cost - sum(row duals).
            # Near zero means the row duals took the benefit; near M means the flight is
            # cheap to fly and pricing has nothing to gain from it.
            "pi_used_n": int(pi_arr.size),
            "pi_used_min": float(pi_arr.min()) if pi_arr.size else None,
            "pi_used_p50": float(np.median(pi_arr)) if pi_arr.size else None,
            "pi_used_max": float(pi_arr.max()) if pi_arr.size else None,
            "pi_used_near_zero": int((np.abs(pi_arr) < 0.01 * M).sum()),
            # Whole-iteration wall, and what the named stages fail to explain.  Reported
            # because a stage table that does not sum to the clock is a table nobody can
            # act on -- the residual has been the largest single block in a solve before.
            "iteration_wall_s": round(float(state.get("iteration_wall_s") or 0.0), 2),
            "unattributed_s": round(
                float(state.get("iteration_wall_s") or 0.0)
                - float(state.get("sweep_s") or 0.0)
                - sum(float(v) for v in (state.get("stage_s") or {}).values()),
                2,
            ),
            "elapsed_s": round(float(state.get("elapsed_s") or 0.0), 2),
            "sweep_s": round(float(state.get("sweep_s") or 0.0), 2),
            "sweep_task_total_s": round(float(state.get("sweep_task_total_s") or 0.0), 2),
            # Rounding-heuristic autopsy: best/worst try, how many flights each covered
            # against the full batch, and how much LP guidance there was to begin with.
            "round_n_forced": (state.get("round_stats") or {}).get("n_forced"),
            "round_n_swapped_best": (state.get("round_stats") or {}).get("n_swapped_best"),
            "round_n_swapped_max": (state.get("round_stats") or {}).get("n_swapped_max"),
            "round_mode": (state.get("round_stats") or {}).get("mode", "round"),
            "round_try_covered_min": min(
                (state.get("round_stats") or {}).get("try_covered") or [0], default=0
            ),
            "round_try_covered_max": max(
                (state.get("round_stats") or {}).get("try_covered") or [0], default=0
            ),
            "round_try_cost_best": _finite(
                min(
                    (
                        (state.get("round_stats") or {}).get("n_flights", 0) * params.M - o
                        for o in ((state.get("round_stats") or {}).get("try_objectives") or ())
                    ),
                    default=float("nan"),
                )
            ),
            "round_try_cost_worst": _finite(
                max(
                    (
                        (state.get("round_stats") or {}).get("n_flights", 0) * params.M - o
                        for o in ((state.get("round_stats") or {}).get("try_objectives") or ())
                    ),
                    default=float("nan"),
                )
            ),
            "lazy_rows_added": state.get("lazy_rows_added"),
            # Rows this iteration got from the POOL rather than from the LP solution.
            "contested_rows_added": state.get("contested_rows_added"),
            # The serial master block, per iteration rather than summed over the solve.  A
            # total cannot answer "does this grow as the pool grows", and that is the whole
            # question about the tail: `solve_lp` is the LP, everything else beside it is
            # Python walking the pool once (`add_violated_rows`) or n_tries times
            # (`round_heuristic`), so which of them grows decides what is worth fixing.
            "stage_s": {k: round(float(v), 3) for k, v in (state.get("stage_s") or {}).items()},
            "stage_n": dict(state.get("stage_n") or {}),
            "rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_SCALE, 1
            ),
        }
        sweep_s = float(state.get("sweep_s") or 0.0)
        lanes = max(1, args.workers)
        row["worker_efficiency"] = (
            round(row["sweep_task_total_s"] / (sweep_s * lanes), 4) if sweep_s > 0 else None
        )
        # Would the shipped thresholds have stopped here?  Reported, never acted on.
        for scale in ("revenue", "cost"):
            gap = state.get(f"lp_gap_{scale}")
            row[f"would_stop_lp_{scale}"] = bool(
                isinstance(gap, float) and math.isfinite(gap) and gap <= _SHIPPED_LP_GAP
            )
            hgap = state.get(f"heuristic_gap_{scale}")
            row[f"would_stop_heuristic_{scale}"] = bool(
                isinstance(hgap, float) and math.isfinite(hgap) and hgap <= _SHIPPED_IP_GAP
            )

        if not args.no_ip_every_iteration and master is not None:
            started = time.perf_counter()
            selection = master.solve_ip(deadline=time.monotonic() + args.ip_deadline_s)
            row["ip_s"] = round(time.perf_counter() - started, 2)
            row["ip_objective"] = _finite(master.last_ip_objective)
            row["ip_bound"] = _finite(master.last_ip_bound)
            row["ip_status"] = master.last_ip_status
            row["ip_optimal"] = master.last_ip_optimal
            row["ip_selected"] = len(selection)
            row["ip_total_delay_s"] = round(
                sum(column.delay_s for column in selection.values()), 6
            )
            # Rows the IP's separation added.  Nonzero means this measurement changed the
            # solve it was measuring -- the next LP is tighter than production's would be.
            row["rows_after_ip"] = len(master.materialized_rows)

        rows.append(row)
        print(
            f"  it {row['iteration']:>3} lp={_fmt(row['lp_objective'], 14, 8)} "
            f"gap_rev={_fmt(row['lp_gap_revenue'], 11, 4)} "
            f"gap_cost={_fmt(row['lp_gap_cost'], 11, 4)} "
            f"stop_rev={row['would_stop_lp_revenue']!s:<5} "
            f"ip={row.get('ip_objective', 'n/a')} "
            f"ip_delay={row.get('ip_total_delay_s', 'n/a')} "
            f"cols={row['columns']:>6} sweep={row['sweep_s']:>8.1f}s "
            f"eff={row['worker_efficiency']}\n"
            f"        cost_ub={_fmt(row['cost_upper_bound'], 12, 6)} "
            f"cost_lb={_fmt(row['cost_lower_bound'], 12, 6)} "
            f"frozen={row['bound_frozen_for']!s:<3} "
            f"rc_sum={_fmt(row['rc_sum'], 12, 6)} rc_n+={row['rc_n_positive']:<6} "
            f"rc_max={_fmt(row['rc_max'])} rc_p50={_fmt(row['rc_p50'], 10, 4)} "
            f"nearM={row['n_rc_near_M']:<4} uncov={row['n_uncovered']:<4} "
            f"dual_l2={_fmt(row['dual_l2'])}\n"
            f"        dual_nearM={row['n_dual_near_M']:<5} "
            f"dual_max={_fmt(row['dual_max'], 11, 6)} "
            f"pi_used p50={_fmt(row['pi_used_p50'], 11, 5)} "
            f"~0={row['pi_used_near_zero']}/{row['pi_used_n']}\n"
            f"        prev_added_used={row['prev_added_used']}/{row['prev_added_span']}  "
            f"virgin_claims={row['virgin_claims']}/{row['virgin_claims_total']}"
            + (
                f" ({100.0 * row['virgin_claims'] / row['virgin_claims_total']:.1f}%)"
                if row["virgin_claims_total"] else ""
            ) + "\n"
            f"        {row['round_mode']}: forced={row['round_n_forced']} "
            f"swapped={row['round_n_swapped_best']}/{row['round_n_swapped_max']} "
            f"covered={row['round_try_covered_min']}-{row['round_try_covered_max']}"
            f"/{len(requests)} "
            f"try_cost={_fmt(row['round_try_cost_best'], 10, 6)}..{_fmt(row['round_try_cost_worst'], 10, 6)} "
            f"incumbent={_fmt(row['heuristic_cost'], 10, 6)}\n"
            f"        wall={row['iteration_wall_s']}s sweep={row['sweep_s']}s "
            f"master={sum(row['stage_s'].values()):.1f}s unattributed={row['unattributed_s']}s\n"
            "        master " + " ".join(
                f"{k}={v:.2f}" for k, v in sorted(row["stage_s"].items())
            ),
            flush=True,
        )
        if args.out:  # flushed per iteration: an hours-long run must survive being killed
            Path(args.out).write_text(
                json.dumps({"header": header, "iterations": rows}, indent=2, default=str)
            )

    started = time.perf_counter()
    result = ColGenSolver().solve(
        requests, cfg, static_terms, params,
        on_iteration=on_iteration,
        **({} if seed_columns is None else {"seed_columns": seed_columns}),
    )
    wall = time.perf_counter() - started
    stats = dict(result.stats)

    if args.checkpoint_out:
        master = master_ref.get("master")
        if master is None:
            print("no iteration completed; nothing to checkpoint", flush=True)
        else:
            by_flight: dict[int, list] = defaultdict(list)
            for column in master.columns:
                by_flight[column.flight_id].append(_column_to_record(column))
            path = Path(args.checkpoint_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "scenario": args.scenario,
                        "flights": len(requests),
                        "seed": cfg.seed,
                        "iterations_run": stats.get("iterations"),
                        "columns_by_flight": {str(k): v for k, v in by_flight.items()},
                    }
                )
            )
            print(
                f"checkpoint: {sum(len(v) for v in by_flight.values())} columns over "
                f"{len(by_flight)} flights -> {path} ({path.stat().st_size / 2**20:.1f} MB)",
                flush=True,
            )

    summary = {
        **header,
        "wall_s": round(wall, 2),
        "termination_reason": stats.get("termination_reason"),
        "iterations_run": stats.get("iterations"),
        "final_lp_objective": _finite(stats.get("final_lp_objective")),
        "objective": _finite(stats.get("objective")),
        "selected_flights": stats.get("selected_flights"),
        "n_columns": stats.get("n_columns"),
        "ladder_columns": stats.get("ladder_columns"),
        "ip_status": stats.get("ip_status"),
        "ip_skipped": stats.get("ip_skipped"),
        "ip_objective": _finite(stats.get("ip_objective")),
        "ip_elapsed_s": stats.get("ip_elapsed_s"),
        "pricing_wall_s": stats.get("pricing_wall_s"),
        "pricing_task_total_s": stats.get("pricing_task_total_s"),
        "n_pricing_workers": stats.get("n_pricing_workers"),
        "kernel_priced": stats.get("kernel_priced"),
        "kernel_fell_back": stats.get("kernel_fell_back"),
        "initial_greedy_completed": stats.get("initial_greedy_completed"),
        "initial_greedy_elapsed_s": stats.get("initial_greedy_elapsed_s"),
        "rss_self_mb": round(_rss(resource.RUSAGE_SELF), 1),
        # LARGEST SINGLE CHILD, not the sum across the tree -- `getrusage` defines
        # `ru_maxrss` that way for RUSAGE_CHILDREN, so this reads flat however many
        # workers ran and says NOTHING about aggregate pool memory.  For that use
        # `analysis/sweep_pricing_workers.py`'s tree sampler.
        "rss_largest_child_mb": round(_rss(resource.RUSAGE_CHILDREN), 1),
    }
    print("\n" + json.dumps(summary, indent=2, default=str), flush=True)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"header": header, "iterations": rows, "summary": summary},
                       indent=2, default=str)
        )
    return 0


def _rss(who) -> float:
    return resource.getrusage(who).ru_maxrss / _RSS_SCALE


def _column_to_record(column) -> dict:
    """The fields that identify a column. Claims are omitted -- see the module docstring."""

    return {
        "flight_id": int(column.flight_id),
        "departure_step": int(column.departure_step),
        "level": int(column.level),
        "origin_lane_idx": column.origin_lane_idx,
        "dest_lane_idx": column.dest_lane_idx,
        "cell_path": [[int(q), int(r)] for q, r in column.cell_path],
        "delay_s": float(column.delay_s),
    }


def _column_from_record(record: dict) -> Column:
    """Rebuild a column with EMPTY claims; ``solve`` re-derives them canonically.

    ``delay_s`` is restored rather than recomputed because it is the objective's own value
    for this column under the cost model that produced it -- recomputing it here would need
    that model and would be a second place for the objective to live.
    """

    return Column(
        flight_id=record["flight_id"],
        departure_step=record["departure_step"],
        level=record["level"],
        origin_lane_idx=record["origin_lane_idx"],
        dest_lane_idx=record["dest_lane_idx"],
        cell_path=tuple((q, r) for q, r in record["cell_path"]),
        delay_s=record["delay_s"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
