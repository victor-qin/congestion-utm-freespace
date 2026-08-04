"""EXECUTE — run ONE scenario and persist it. The single "experiment box".

This is the entire execute stage: pick a named scenario (optionally override fields), run the
simulation, and write a complete, reloadable run folder + one index row. It produces **no plots and
no replay** — those are readout scripts (``experiments/readouts/``). It runs **one** scenario; sweeps
and planner comparisons are pure-shell loops over this script (see ``experiments/batch/``), joined by
a shared ``--tag``.

    # one run, captured to results/<stamp>_<tag>_<hash>/ ; the folder path is the last stdout line
    uv run python -m experiments.run --scenario dallas_hub_2uss --planner astar_shortcut

    # override any field; capture the folder in a shell variable for a readout
    FOLDER=$(uv run python -m experiments.run --scenario metro_2uss --lam 240 --tag demo | tail -1)

Scenario identity (``--scenario`` / ``--tag`` / demand pattern) is written into the index so cross-run
readouts can filter to exactly the runs a batch produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from freespace_sim import metrics, runs
from freespace_sim.scenarios import SCENARIOS, get_scenario, with_overrides
from freespace_sim.sim import RETURN_ANCHORS, run

log = logging.getLogger("experiments.run")


def _scaled_lam_per_uss(spec, total_lam: float) -> dict[str, float]:
    """Scale an explicit per-USS rate map while preserving its operator proportions."""
    rates = spec.demand.lam_per_uss
    if rates is None:
        raise ValueError("scenario has no explicit lam_per_uss rates to scale")
    current_total = sum(float(rate) for rate in rates.values())
    if current_total <= 0.0:
        raise ValueError("cannot scale lam_per_uss rates whose total is not positive")
    scale = float(total_lam) / current_total
    return {uss_id: float(rate) * scale for uss_id, rate in rates.items()}


def _kernel_status(planner_name: str) -> str:
    """One-line compiled-kernel status for the startup INFO block. Mirrors AStarPlanner's own import
    probe; the module lands in ``sys.modules`` so the sim's later import is free. Only the astar
    family has a kernel — anything else reports n/a rather than paying the numba import."""
    if "astar" not in planner_name:
        return "n/a (planner has no compiled kernel)"
    if planner_name == "astar_ref":
        return "pure-Python reference (explicitly requested via astar_ref)"
    try:
        from freespace_sim.planner import astar_kernel  # noqa: F401
        return "compiled (numba kernel active)"
    except ImportError:
        return ("REFERENCE FALLBACK — numba unavailable, ~5-7x slower search. "
                "Run via plain `uv run` (numba is in tool.uv default-groups) or `uv sync`.")


def spec_from_args(args):
    """Layer CLI overrides on top of the chosen registry scenario (frozen → copies, never mutates)."""
    spec = get_scenario(args.scenario)
    top: dict = {}
    if args.region is not None:
        top["region_m"] = (float(args.region[0]), float(args.region[1]))
    if args.lam is not None:
        top["lam_per_hour"] = args.lam
    if args.horizon is not None:
        top["horizon_s"] = args.horizon
    if args.demand_duration is not None:
        top["demand_duration_s"] = args.demand_duration
    if args.seed is not None:
        top["seed"] = args.seed
    if args.planner is not None:
        top["planner"] = args.planner
    if args.terminal_airspace_always_active is not None:
        top["terminal_airspace_always_active"] = args.terminal_airspace_always_active

    demand: dict = {}
    if args.lam is not None and spec.demand.lam_per_uss is not None:
        demand["lam_per_uss"] = _scaled_lam_per_uss(spec, args.lam)
    if args.demand is not None:
        demand["pattern"] = args.demand
    if args.uss is not None:
        demand["uss"] = tuple(args.uss)
    if args.hubs is not None:
        demand["hubs"] = tuple(int(h) for h in args.hubs)
    if args.direction is not None:
        demand["direction"] = args.direction
    if args.radius is not None:
        demand["radius_m"] = args.radius
    if args.pads_per_hub is not None:
        demand["pads_per_hub"] = args.pads_per_hub
    if args.terminal_radius is not None:
        demand["terminal_radius_m"] = args.terminal_radius
    if args.corridor_overlap is not None:
        demand["corridor_overlap_m"] = args.corridor_overlap
    if args.return_flights is not None:
        demand["return_flights"] = args.return_flights
    if args.turnaround is not None:
        demand["turnaround_s"] = args.turnaround

    return with_overrides(spec, demand_overrides=demand or None, **top)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from ``main`` so tests can construct a real args namespace.

    Hand-mirroring this flag set in a test fixture silently rots the moment a flag is added —
    ``spec_from_args`` reads attributes that the fixture doesn't have. Parse from here instead.
    """
    p = argparse.ArgumentParser(description="Run one scenario and persist it (the execute box).")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="metro_uniform",
                   help="named world from the registry (override individual fields with the flags below)")
    p.add_argument("--region", type=float, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--lam", type=float, default=None, help="arrival rate (req/h)")
    p.add_argument("--horizon", type=float, default=None, help="sim horizon (s)")
    p.add_argument("--demand-duration", type=float, default=None,
                   help="offered-load window (s); demand is generated over this, the run continues to "
                        "--horizon so the return tail is never clipped. Must be <= --horizon, so shrinking "
                        "a density_* scenario for a smoke test needs BOTH (e.g. --horizon 900 "
                        "--demand-duration 120). Unset keeps the scenario's own value.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--planner", default=None, help="override planner")
    p.add_argument("--terminal-airspace-always-active", action=argparse.BooleanOptionalAction,
                   default=None, dest="terminal_airspace_always_active",
                   help="permanently wall each hub's column+lanes off from foreign traffic (foreign "
                        "transit → air detour instead of ground-block); A* only")
    p.add_argument("--demand", choices=("uniform", "hub", "hub_radius"), default=None,
                   help="demand pattern")
    p.add_argument("--uss", nargs="+", default=None, help="USS labels (multi-operator demand)")
    p.add_argument("--hubs", nargs="+", type=int, default=None, help="per-USS hub counts (hub patterns)")
    p.add_argument("--direction", choices=("delivery", "pickup"), default=None)
    # hub_radius knobs
    p.add_argument("--radius", type=float, default=None, help="customer demand radius around a hub (m)")
    p.add_argument("--pads-per-hub", type=int, default=None, help="terminal capacity N per hub")
    p.add_argument("--terminal-radius", type=float, default=None,
                   help="shared terminal column radius (m); default = hover footprint")
    p.add_argument("--corridor-overlap", type=float, default=None,
                   help="how far the first corridor box penetrates the terminal (m); default corridor_width/2")
    p.add_argument("--return-flights", action=argparse.BooleanOptionalAction, default=None,
                   help="emit a return flight to the origin pad for each delivery")
    p.add_argument("--turnaround", type=float, default=None, help="return-flight turnaround (s)")
    p.add_argument("--return-anchor", choices=RETURN_ANCHORS, default="nominal",
                   dest="return_anchor",
                   help="what a round-trip return's desired departure waits on. nominal (default): a "
                        "straight-line undelayed estimate of the outbound's arrival, fixed when demand "
                        "is generated — under congestion the return is scheduled before its aircraft "
                        "lands. realized: plan the outbound, then anchor its return to the arrival that "
                        "actually happened (+ turnaround). Same cost; sequential mode only. NOT a "
                        "column in results/index.parquet (only experiment.json records it), so give "
                        "the two anchors distinct --tag values or a cross-run readout cannot tell "
                        "them apart")
    p.add_argument("--tag", default=None, help="run-folder label + index join key (default: scenario name)")
    p.add_argument("--window-frac", type=float, default=0.9,
                   help="steady-state plateau threshold: measure where airborne density ≥ frac×peak "
                        "(issue #25). summary.json always reports both whole-run and this windowed twin")
    p.add_argument("--telemetry", action="store_true",
                   help="capture observer-only congestion telemetry (filed-but-rejected corridors, "
                        "conflict_filed culprits, per-hub metadata, end-of-run walls) into extra parquets")
    p.add_argument("--no-progress", action="store_true", help="silence the live progress line")
    p.add_argument("--mode", choices=("sequential", "exact", "relaxed"), default="sequential",
                   help="execution strategy for the whole simulation (issue #8 Track A). "
                        "sequential (default): the classic serial FCFS loop. exact: speculative "
                        "worker-pool planning, byte-identical to the serial run and faster. relaxed: "
                        "keep any still-feasible speculation — a valid FCFS-class allocation, faster "
                        "and scalable to more workers, at a small delay cost (deterministic via pinned "
                        "prefixes; result depends on workers+window). Parallel modes currently support "
                        "only astar, astar_ref, and astar_shortcut; every other planner runs sequential.")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="worker processes for explicit exact/relaxed mode only (default min(8, cores-2); the "
                        "benchmark sweet spot is ~4 workers for exact, ~8 for relaxed)")
    p.add_argument("--parallel-window", type=int, default=None,
                   help="speculation window for explicit exact/relaxed mode only (default 4×workers); "
                        "result-affecting in relaxed mode")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the execute CLI's cross-argument execution-mode contract."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "sequential" and (
        args.workers is not None or args.parallel_window is not None
    ):
        parser.error("--workers and --parallel-window require --mode exact or --mode relaxed")
    return args


def main() -> None:
    args = parse_args()

    spec = spec_from_args(args)
    # to_json_dict, not asdict: the latter loses every tuple to a JSON list and leaves `demand` a
    # plain dict, so the archived recipe could not be rebuilt. See ScenarioSpec.from_json_dict.
    scenario_payload = spec.to_json_dict()
    cfg = spec.config()
    demand = spec.demand_model()
    tag = args.tag or spec.name
    # everything human-facing goes to stderr; stdout is reserved for the folder path (shell capture)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    log.info("invocation: python -m experiments.run %s", " ".join(sys.argv[1:]) or "(no arguments)")
    log.info("scenario=%s tag=%s planner=%s demand=%s region=%s λ=%s/h planner-envelope=%ss seed=%s",
             spec.name, tag, cfg.planner, spec.demand.pattern, cfg.region_size_m,
             cfg.lam_per_hour, cfg.horizon_s, cfg.seed)
    if spec.description:
        log.info("description: %s", spec.description)
    log.info("active demand duration=%ss", cfg.effective_demand_duration_s)
    log.info("A* kernel: %s", _kernel_status(cfg.planner))

    pcfg = None
    if args.mode != "sequential":
        from freespace_sim.parallel import PARALLEL_PLANNERS, ParallelConfig
        if cfg.planner in PARALLEL_PLANNERS:
            kw = {"mode": args.mode, "window": args.parallel_window}
            if args.workers is not None:                 # else ParallelConfig's own capped default
                kw["n_workers"] = args.workers
            pcfg = ParallelConfig(**kw)
            log.info("mode=%s: %d workers, window=%d", pcfg.mode, pcfg.n_workers, pcfg.resolved_window)
        else:                                            # MILP/straight/etc. have no envelope-recording
            log.info("planner %r has no parallel kernel — running sequential (--mode %s ignored)",
                     cfg.planner, args.mode)
    else:
        log.info("mode=sequential: serial FCFS planning")

    if args.return_anchor == "realized":
        # The coupling keys off FlightRequest.paired_outbound_id, which ONLY hub_radius sets. Testing
        # `return_flights` alone is not enough: it defaults True on DemandSpec but the uniform and hub
        # patterns ignore it entirely, so metro_2uss/dallas_hub_2uss would sail past the guard and the
        # flag would be a silent no-op — the failure it exists to prevent.
        if demand is None or spec.demand.pattern != "hub_radius" or not spec.demand.return_flights:
            raise SystemExit(
                "--return-anchor realized needs round-trip returns, which only the hub_radius demand "
                f"pattern emits (scenario {spec.name!r} is pattern={spec.demand.pattern!r}, "
                f"return_flights={spec.demand.return_flights}); drop the flag or pick a hub_radius "
                "scenario such as a density_* world")
        if pcfg is not None:
            # run() raises on this too; catching it here keeps the CLI failure a one-line message
            # rather than a traceback, and does it before the world is built.
            raise SystemExit(
                f"--return-anchor realized is sequential-only (got --mode {args.mode}): a speculative "
                "worker may plan a return before its outbound has committed, and the exact-mode "
                "envelope check cannot detect that. Use --mode sequential or --return-anchor nominal.")
        log.info("return anchor: realized — each return departs on its outbound's actual arrival "
                 "+ %.0fs turnaround, not a nominal estimate", spec.demand.turnaround_s)

    t0 = time.time()
    res = run(cfg, demand=demand, progress=not args.no_progress, telemetry=args.telemetry,
              parallel=pcfg, return_anchor=args.return_anchor)
    wall = time.time() - t0
    sim_lo, sim_hi = metrics.simulation_window(res)
    log.info(
        "realized simulation: first activity %.1fs, final landing %.1fs, duration %.1fs",
        sim_lo,
        sim_hi,
        sim_hi - sim_lo,
    )
    # SimConfig only validates demand_duration_s <= horizon_s, but what actually has to fit under the
    # horizon is preroll + demand window + lead spread + trip. The preroll is a max-order statistic over
    # the departure-lead draws (~2300 s for amazon_uss's N(1800, 300)), so a shrunken --horizon can pass
    # validation and still push most departures past it — where the compiled A* box guard silently
    # dispatches to the ~5-7x slower pure-Python reference. On a cluster that is the difference between
    # a 6-hour job and a 30-hour one, so say it out loud rather than let the allocation absorb it.
    late = sum(1 for i in res.intents if i.request.t_departure > cfg.horizon_s)
    if late:
        log.warning("%d/%d departures (%.0f%%) are past horizon_s=%.0fs — those flights fall back to "
                    "the slow reference A* (box guard). Raise --horizon or lower --demand-duration.",
                    late, len(res.intents), 100.0 * late / len(res.intents), cfg.horizon_s)
    if pcfg is not None and pcfg.stats:
        log.info("parallel stats: %s", pcfg.stats)

    folder = runs.save_run(
        res, label=tag, experiment="run", scenario=spec.name, demand=spec.demand.pattern,
        experiment_args={"scenario": spec.name, "tag": tag, "overrides": vars(args)},
        scenario_spec=scenario_payload,
        wall_seconds=wall, write_replay=False,   # execute persists data only; replay is a readout
        window_frac=args.window_frac,
    )
    s = res.summary()
    log.info("n=%s acc=%s den=%s verified=%s (%.1fs) → %s",
             s["n_requests"], s["n_accepted"], s["n_denied"], res.verified, wall, folder)
    # the steady-state twin vs the whole-run number (read back from the summary save_run just wrote,
    # so it's the exact persisted value — no recompute). window == full horizon when no plateau exists.
    import json
    summ = json.loads((folder / "summary.json").read_text())
    st = summ.get("steady_state", {})
    log.info("steady window [%.0f,%.0f]s · mean delay %.1fs (whole-run %.1fs)",
             st.get("window_lo", 0), st.get("window_hi", 0),
             st.get("mean_total_delay_s", 0), summ.get("mean_total_delay_s", 0))
    print(folder)   # LAST stdout line: the run folder, for `FOLDER=$(... | tail -1)`


if __name__ == "__main__":
    main()
