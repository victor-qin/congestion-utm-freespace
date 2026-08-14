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
        ip_gap=0.0,
        n_pricing_workers=args.workers,
        **({} if args.ladder is None else {"seed_ladder_steps": args.ladder}),
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
        "gap_metric": params.gap_metric, "seed_ladder_steps": params.seed_ladder_steps,
        "greedy_budget_s_per_flight": params.greedy_budget_s_per_flight,
        "greedy_budget_s": params.greedy_budget_s_per_flight * len(requests),
        "ip_every_iteration": not args.no_ip_every_iteration,
        "resumed": bool(args.checkpoint_in),
    }
    print(json.dumps(header, indent=2), flush=True)

    rows: list[dict] = []
    master_ref: dict = {}

    def on_iteration(state: dict) -> None:
        master = state.get("master")
        master_ref["master"] = master
        row = {
            "iteration": state["iteration"],
            "lp_objective": _finite(state.get("lp_objective")),
            "upper_bound": _finite(state.get("upper_bound")),
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
            "rc_n_positive": state.get("rc_n_positive"),
            "rc_sum": _finite(state.get("rc_sum")),
            "dual_nonzero": state.get("dual_nonzero"),
            "sweep_s": round(float(state.get("sweep_s") or 0.0), 2),
            "sweep_task_total_s": round(float(state.get("sweep_task_total_s") or 0.0), 2),
            "lazy_rows_added": state.get("lazy_rows_added"),
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
            f"  it {row['iteration']:>3} lp={row['lp_objective']:<14.8g} "
            f"gap_rev={row['lp_gap_revenue']:<11.4g} gap_cost={row['lp_gap_cost']:<11.4g} "
            f"stop_rev={row['would_stop_lp_revenue']!s:<5} "
            f"ip={row.get('ip_objective', 'n/a')} "
            f"ip_delay={row.get('ip_total_delay_s', 'n/a')} "
            f"cols={row['columns']:>6} sweep={row['sweep_s']:>8.1f}s "
            f"eff={row['worker_efficiency']}",
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
