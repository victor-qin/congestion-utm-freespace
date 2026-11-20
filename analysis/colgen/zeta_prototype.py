"""Compare bounded Zeta search under column generation and FCFS filing."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from freespace_sim.planner.colgen.params import ColGenParams

PROCESS_STARTED = time.monotonic()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("COLGEN_GUROBI_THREADS", "1")

import numpy as np  # noqa: E402

from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.planner import hexgrid as hg  # noqa: E402
from freespace_sim.planner.route_cost import cost_model  # noqa: E402
from freespace_sim.planner.zeta import Route, ZetaOptions  # noqa: E402
from freespace_sim.planner.zeta.fcfs import run_fcfs  # noqa: E402
from freespace_sim.planner.zeta.planner import ZetaPlannerParams  # noqa: E402
from freespace_sim.types import FlightRequest, Terminal, vec  # noqa: E402
from freespace_sim.verify import find_interflight_conflict  # noqa: E402


def workload(
    name: str, count: int, distance_cells: int, ground_delay_s: float
) -> tuple[SimConfig, list[FlightRequest], tuple]:
    """Make deterministic crossing or shared-hub requests with unchanged simulator rules.

    Parameters
    ------------
    - name (str): ``crossing`` or ``hub``.
    - count (int): number of requests to retain, including any ultimately denied flights.
    - distance_cells (int): radius from the common centre, in lattice pitches.
    - ground_delay_s (float): legal ground-hold budget, shared by every arm.

    Return
    --------
    - output (tuple[SimConfig, list[FlightRequest], tuple]): configuration, requests and
      permanent terminal metadata.
    """
    cfg = SimConfig(
        planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        region_size_m=(12000.0, 12000.0), max_ground_delay_s=ground_delay_s,
        terminal_airspace_always_active=True, fixed_exit_lanes=True,
    )
    pitch = cfg.nominal_speed_mps * cfg.dt_s
    center = np.asarray(hg.hex_center(30, 20, hg.circumradius(cfg)))
    distance = distance_cells * pitch
    terminal = Terminal("prototype-hub", capacity=2, radius=180.0)
    requests = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count + 0.17
        direction = np.asarray((math.cos(angle), math.sin(angle)))
        origin = center - distance * direction if name == "crossing" else center
        destination = center + distance * direction
        requests.append(FlightRequest(
            index, vec(*origin, 0.0), vec(*destination, 0.0), 0.0, 0.0,
            origin_terminal=terminal if name == "hub" else None,
        ))
    terms = ((vec(*center, 0.0), terminal),) if name == "hub" else ()
    return cfg, requests, terms


def _json_safe(value: Any) -> Any:
    """Convert optimizer diagnostics to portable JSON without non-finite numeric literals."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_arm(
    arm: str,
    cfg: SimConfig,
    requests: list[FlightRequest],
    terms: tuple,
    params: ColGenParams,
    options: ZetaOptions,
    *,
    fcfs_leg_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure one complete solve and independently check its filed reservation schedule.

    Parameters
    ------------
    - arm (str): production ``dp``, full-volume column generation or standalone ``zeta_fcfs``.
    - cfg (SimConfig): identical physical simulation rules across arms.
    - requests (list[FlightRequest]): identical flight requests across arms.
    - terms (tuple): permanent terminal catalogue.
    - params (ColGenParams): identical master objective and final-IP budgets.
    - options (ZetaOptions): search radius and work caps for experimental pricing.
    - fcfs_leg_seconds (float | None): wall budget per FCFS leg. Defaults to ``params.time_limit_s``.

    Return
    --------
    - output (dict[str, Any]): timing, objective, certificate, geometry and route evidence.
    """
    started = time.monotonic()
    if arm == "dp":
        from freespace_sim.planner.colgen.master import RestrictedMaster
        from freespace_sim.planner.colgen.solver import ColGenSolver
        from freespace_sim.planner.colgen.translate import column_to_intent

        masters: list[RestrictedMaster] = []

        def observe(state: dict[str, Any]) -> None:
            """Retain the live master solely to inspect its final generated pool."""
            if not masters:
                masters.append(state["master"])

        result = ColGenSolver().solve(requests, cfg, terms, params, on_iteration=observe)
        by_id = {request.flight_id: request for request in requests}
        columns = result.columns
        intents = {
            flight_id: column_to_intent(column, by_id[flight_id], cfg)
            for flight_id, column in columns.items()
        }
        stats = dict(result.stats)
        master = masters[0] if masters else None
        stats["row_rule"] = "production DP visit-window claims (not experimental full-box cover)"
    elif arm == "zeta_fcfs":
        if params.n_pricing_workers != 0:
            raise ValueError("zeta_fcfs runs serially and requires --workers 0")
        standalone_params = ZetaPlannerParams(
            options=options, objective=params.objective,
            max_air_overrun_hops=params.max_air_overrun_hops,
            time_limit_s=params.time_limit_s if fcfs_leg_seconds is None else fcfs_leg_seconds,
        )
        columns, intents, stats = run_fcfs(cfg, requests, terms, standalone_params)
        master = None
    else:
        from freespace_sim.planner.colgen.zeta_solver import solve_zeta

        seed_result = None
        seed_seconds = 0.0
        supplied = None
        if arm == "zeta_seeded":
            seed_started = time.monotonic()
            seed_result = solve_zeta(
                requests, cfg, terms, params,
                options=replace(options, connection_radius_m=cfg.nominal_speed_mps * cfg.dt_s + 1e-8),
            )
            seed_seconds = time.monotonic() - seed_started
            supplied = {}
            for request in requests:
                flight_id = request.flight_id
                primary = seed_result.columns.get(flight_id)
                pool = [c for c in seed_result.master.columns if c.flight_id == flight_id and c != primary]
                supplied[flight_id] = ([primary] if primary is not None else []) + pool
            remaining = max(0.001, params.time_limit_s - seed_seconds)
            params = replace(params, time_limit_s=remaining,
                             ip_reserve_s=min(params.ip_reserve_s or 0.0, remaining / 2))
        if arm == "adjacent":
            options = replace(options,
                connection_radius_m=cfg.nominal_speed_mps * cfg.dt_s + 1e-8,
            )
        result = solve_zeta(requests, cfg, terms, params, options=options, seed_columns=supplied)
        columns, intents, stats, master = result.columns, result.intents, dict(result.stats), result.master
        if seed_result is not None:
            stats["initial_schedule_stats"] = seed_result.stats
            stats["initial_schedule_s"] = seed_seconds
            stats["initial_schedule_served"] = len(seed_result.columns)
            stats["initial_schedule_cost"] = seed_result.stats["objective_cost"]
            stats["initial_schedule_pricing_s"] = seed_result.stats["pricing_wall_s"]
            stats["pricing_wall_s"] += seed_result.stats["pricing_wall_s"]
            stats["elapsed_s"] += seed_seconds
    model = cost_model(cfg, params)
    served = len(columns)
    objective = math.fsum(model.intent_cost(intent, cfg) for intent in intents.values())
    objective += (len(requests) - served) * params.M
    conflict = find_interflight_conflict(list(intents.values()), cfg, terms)
    violation_count = None
    unique_rows = nonzeros = None
    if master is not None:
        violation_count = len(master.violated_claim_rows(columns))
        unique_rows = len({row for column in master.columns for row in column.claims})
        nonzeros = sum(len(column.claims) for column in master.columns)
    elif arm == "zeta_fcfs":
        violation_count = stats["violated_rows"]
        unique_rows = stats["unique_rows"]
        nonzeros = stats["nonzeros"]
    if conflict is not None or violation_count not in (0, None):
        raise RuntimeError(f"{arm} audit failed: collision={conflict}, overloaded rows={violation_count}")
    evidence = {
        "arm": arm,
        "served": served,
        "denied": len(requests) - served,
        "objective_cost": objective,
        "total_solve_and_audit_s": time.monotonic() - started,
        "pricing_s": stats.get("pricing_wall_s"),
        "solver_reported_s": stats.get("elapsed_s"),
        "global_cost_lower_bound": stats.get("global_cost_lower_bound", stats.get("cost_lower_bound")),
        "global_integer_gap": stats.get("global_integer_gap"),
        "restricted_ip_cost_lower_bound": stats.get("restricted_ip_cost_lower_bound", stats.get("ip_cost_lower_bound")),
        "collision_pair": conflict,
        "violated_rows": violation_count,
        "columns": None if master is None else len(master.columns),
        "unique_rows": unique_rows,
        "nonzeros": nonzeros,
        "selected": [
            {
                "flight_id": flight_id,
                "departure_step": column.departure_step,
                "cell_path": column.cell_path,
                "origin_lane_idx": column.origin_lane_idx,
                "dest_lane_idx": column.dest_lane_idx,
                "cost": column.physical_cost if isinstance(column, Route) else column.delay_s,
                "claims": sorted(column.claims, key=repr),
                "centerline": [(point.tolist(), timestamp) for point, timestamp in intents[flight_id].centerline],
            }
            for flight_id, column in columns.items()
        ],
        "stats": stats,
    }
    return _json_safe(evidence)


def main() -> None:
    """Run explicitly selected bounded arms and persist all outcomes, including failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("dp", "adjacent", "zeta", "zeta_seeded", "zeta_fcfs", "all"),
        default="zeta", help="all runs dp, adjacent and zeta",
    )
    parser.add_argument("--case", choices=("crossing", "hub"), default="crossing")
    parser.add_argument("--flights", type=int, default=2)
    parser.add_argument("--distance-cells", type=int, default=4)
    parser.add_argument("--ground-delay", type=float, default=64.0)
    parser.add_argument("--overrun", type=int, default=2)
    parser.add_argument("--radius", type=float, default=480.0)
    parser.add_argument("--connection-mode", choices=("full", "hex12", "hex12_long"),
                        default="full", help="permitted waypoint offsets within --radius")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="whole column-generation solve budget and default FCFS per-leg budget")
    parser.add_argument("--fcfs-leg-seconds", type=float, default=None,
                        help="override the wall budget for each leg under --arm zeta_fcfs")
    parser.add_argument("--ip-seconds", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--max-labels", type=int, default=50000)
    parser.add_argument("--no-search-caps", action="store_true",
                        help="disable expansion and generated-label limits; retain wall deadline")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--backend", choices=("reference", "compiled"), default="reference")
    parser.add_argument("--traversal", choices=("best_first", "forward_dp"), default="best_first")
    parser.add_argument("--no-compiled-search", action="store_true",
                        help="use Python successor scans, expiry and label writes for comparison")
    parser.add_argument("--compiled-loop", action="store_true",
                        help="compile the compact pricing loop with Cython; requires compiled backend")
    parser.add_argument("--compiled-history", action="store_true",
                        help="compile exact history comparisons; requires --compiled-loop")
    parser.add_argument("--native-state-index", action="store_true",
                        help="use native exact continuation keys in every Zeta connection mode")
    parser.add_argument("--timed-cache-bytes", type=int, default=16 * 1024 * 1024,
                        help="timed-footprint cache accounting budget per pricing call; zero disables")
    parser.add_argument("--leg-cache", action="store_true",
                        help="share bounded price-independent leg geometry within each worker")
    parser.add_argument("--persistent-timing-cache", action="store_true",
                        help="reuse exact timing windows across compiled pricing calls")
    parser.add_argument("--lazy-successors", action="store_true")
    parser.add_argument("--early-return", action="store_true",
                        help="stop on a useful new column or a bound proving no useful improvement")
    parser.add_argument("--expire-history", action="store_true")
    parser.add_argument("--history-dominance", action="store_true")
    parser.add_argument("--strong-history-dominance", action="store_true",
                        help="compare different length histories with certified rounding guards")
    parser.add_argument("--history-group-skips", action="store_true",
                        help="skip entire history comparison groups when their charge range rules out replacement")
    parser.add_argument("--merge-equivalent-histories", action="store_true",
                        help="merge guarded equal-price histories; requires --strong-history-dominance")
    parser.add_argument("--merge-shorter-histories", action="store_true",
                        help="merge guarded histories no longer in both lengths; requires --strong-history-dominance")
    parser.add_argument("--forward-cruise", action="store_true",
                        help="require positive origin-to-destination progress on cruise waypoint links")
    parser.add_argument("--turn-below-90", action="store_true",
                        help="require every interior cruise waypoint to turn strictly below 90 degrees")
    parser.add_argument("--lattice-distance-bound", action="store_true",
                        help="use a guarded twelve-direction remaining-distance lower bound where supported")
    parser.add_argument("--route-coverage-cache", action="store_true",
                        help="reuse complete route coverage projected onto currently priced rows")
    parser.add_argument("--compiled-route-coverage", action="store_true",
                        help="compile coverage repricing; requires --compiled-loop and --route-coverage-cache")
    parser.add_argument("--completion-bounds", nargs="*", default=[],
                        choices=("arrival", "resource", "selected", "crossing"),
                        help="optional certified completion relaxations, combined by maximum")
    parser.add_argument("--arrival-snap", type=float, default=0.0, help="arrival grid in seconds; zero disables")
    parser.add_argument("--speed-fraction", type=float, default=0.05, help="fractional horizontal speed allowance")
    parser.add_argument("--workers", type=int, default=0, help="persistent pricing processes; zero is serial")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.flights < 1 or args.distance_cells < 2 or args.repeats < 1:
        parser.error("flights/repeats must be positive and distance-cells at least two")
    if args.arm == "zeta_fcfs" and args.workers != 0:
        parser.error("--arm zeta_fcfs requires --workers 0")
    if args.fcfs_leg_seconds is not None:
        if args.arm != "zeta_fcfs":
            parser.error("--fcfs-leg-seconds requires --arm zeta_fcfs")
        if not math.isfinite(args.fcfs_leg_seconds) or args.fcfs_leg_seconds <= 0:
            parser.error("--fcfs-leg-seconds must be finite and positive")
    cfg, requests, terms = workload(args.case, args.flights, args.distance_cells, args.ground_delay)
    if args.arm == "zeta_fcfs":
        cfg = replace(cfg, planner="zeta")
    from freespace_sim.planner.colgen.params import ColGenParams

    params = ColGenParams(
        solver=args.solver, max_air_overrun_hops=args.overrun, turn_below_90=args.turn_below_90,
        time_limit_s=args.seconds, ip_time_limit_s=args.ip_seconds,
        ip_reserve_s=min(args.ip_seconds + 1.0, args.seconds / 2),
        max_iterations=args.iterations, n_pricing_workers=args.workers,
        iteration_ip_time_limit_s=0.0, lp_gap=0.0, ip_gap=0.0,
    )
    options = ZetaOptions(connection_radius_m=args.radius,
                          max_expansions=None if args.no_search_caps else args.max_expansions,
                          max_labels=None if args.no_search_caps else args.max_labels,
                          merge_identical_states=not args.no_merge,
                          backend=args.backend, lazy_successors=args.lazy_successors,
                          expire_history=args.expire_history,
                          history_dominance=args.history_dominance,
                          arrival_snap_s=args.arrival_snap, speed_fraction=args.speed_fraction,
                          compiled_search=not args.no_compiled_search,
                          timed_cache_bytes=args.timed_cache_bytes, early_return=args.early_return,
                          compiled_loop=args.compiled_loop,
                          completion_bounds=tuple(args.completion_bounds),
                          leg_cache=args.leg_cache,
                          persistent_timing_cache=args.persistent_timing_cache,
                          strong_history_dominance=args.strong_history_dominance,
                          compiled_history=args.compiled_history,
                          native_state_index=args.native_state_index,
                          history_group_skips=args.history_group_skips,
                          merge_equivalent_histories=args.merge_equivalent_histories,
                          merge_shorter_histories=args.merge_shorter_histories,
                          forward_cruise=args.forward_cruise,
                          connection_mode=args.connection_mode,
                          traversal=args.traversal,
                          lattice_distance_bound=args.lattice_distance_bound,
                          route_coverage_cache=args.route_coverage_cache,
                          compiled_route_coverage=args.compiled_route_coverage,
                          turn_below_90=args.turn_below_90)
    arms = ("dp", "adjacent", "zeta") if args.arm == "all" else (args.arm,)
    source_root = Path(__file__).resolve().parents[2]
    planner_root = source_root / "freespace_sim" / "planner"
    source_files = sorted([
        Path(__file__).resolve(), *planner_root.glob("*.py"),
        *(path for package in ("colgen", "zeta") for path in (planner_root / package).iterdir()
          if path.suffix in {".py", ".pxd", ".h"}),
    ])
    limitations = [
        "First process calls include cold JIT work; caches are retained and are not a dedicated-machine timing claim.",
    ]
    if args.arm == "zeta_fcfs":
        limitations.insert(0, "Bounds apply per leg with earlier filings fixed.")
    else:
        limitations[:0] = [
            "Production DP retains its legacy visit-window rows; adjacent and Zeta use full simulator volume cover.",
            "Equal configured pricing-worker counts; same final-IP cap, no intermediate IP in any arm.",
        ]
    evidence: dict[str, Any] = {
        "case": args.case,
        "flights": len(requests),
        "cfg": asdict(cfg),
        "params": asdict(params),
        "options": asdict(options),
        "requests": [
            {"flight_id": request.flight_id, "origin": request.origin.tolist(),
             "dest": request.dest.tolist(), "t_departure": request.t_departure}
            for request in requests
        ],
        "source_hashes": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "limitations": limitations,
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            for arm in (arms if repeat % 2 == 0 else tuple(reversed(arms))):
                print(f"starting {arm} {args.case} flights={args.flights} repeat={repeat}", flush=True)
                result = run_arm(
                    arm, cfg, requests, terms, params, options, fcfs_leg_seconds=args.fcfs_leg_seconds,
                )
                result["repeat"] = repeat
                evidence["results"].append(result)
                args.out.write_text(json.dumps(_json_safe(evidence), indent=2) + "\n")
                print(json.dumps({key: result[key] for key in ("arm", "served", "objective_cost", "pricing_s", "total_solve_and_audit_s")}), flush=True)
    except Exception as exc:
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        evidence["process_elapsed_s"] = time.monotonic() - PROCESS_STARTED
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        evidence["process_peak_rss_bytes"] = int(raw_rss if sys.platform == "darwin" else raw_rss * 1024)
        args.out.write_text(json.dumps(_json_safe(evidence), indent=2) + "\n")


if __name__ == "__main__":
    main()
