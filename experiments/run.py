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
import sys
import time

from freespace_sim import runs
from freespace_sim.scenarios import SCENARIOS, get_scenario, with_overrides
from freespace_sim.sim import run


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
    if args.seed is not None:
        top["seed"] = args.seed
    if args.planner is not None:
        top["planner"] = args.planner
    if args.heuristic_weight is not None:
        top["heuristic_weight"] = args.heuristic_weight
    if args.terminal_airspace_always_active is not None:
        top["terminal_airspace_always_active"] = args.terminal_airspace_always_active

    demand: dict = {}
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


def main() -> None:
    p = argparse.ArgumentParser(description="Run one scenario and persist it (the execute box).")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="metro_uniform",
                   help="named world from the registry (override individual fields with the flags below)")
    p.add_argument("--region", type=float, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--lam", type=float, default=None, help="arrival rate (req/h)")
    p.add_argument("--horizon", type=float, default=None, help="sim horizon (s)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--planner", default=None, help="override planner")
    p.add_argument("--heuristic-weight", type=float, default=None, dest="heuristic_weight",
                   help="weighted A*: f = g + w*h (1.0=optimal; ~1.25 ≈ 5-7x faster, ~2%% cost)")
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
    p.add_argument("--tag", default=None, help="run-folder label + index join key (default: scenario name)")
    p.add_argument("--no-progress", action="store_true", help="silence the live progress line")
    args = p.parse_args()

    spec = spec_from_args(args)
    cfg = spec.config()
    demand = spec.demand_model()
    tag = args.tag or spec.name
    # everything human-facing goes to stderr; stdout is reserved for the folder path (shell capture)
    print(f"scenario={spec.name} tag={tag} planner={cfg.planner} w={cfg.heuristic_weight} "
          f"demand={spec.demand.pattern} region={cfg.region_size_m} λ={cfg.lam_per_hour}/h "
          f"horizon={cfg.horizon_s}s seed={cfg.seed}",
          file=sys.stderr)

    t0 = time.time()
    res = run(cfg, demand=demand, progress=not args.no_progress)
    wall = time.time() - t0

    folder = runs.save_run(
        res, label=tag, experiment="run", scenario=spec.name, demand=spec.demand.pattern,
        experiment_args={"scenario": spec.name, "tag": tag, "overrides": vars(args)},
        wall_seconds=wall, write_replay=False,   # execute persists data only; replay is a readout
    )
    s = res.summary()
    print(f"  n={s['n_requests']} acc={s['n_accepted']} den={s['n_denied']} "
          f"verified={res.verified} ({wall:.1f}s) → {folder}", file=sys.stderr)
    print(folder)   # LAST stdout line: the run folder, for `FOLDER=$(... | tail -1)`


if __name__ == "__main__":
    main()
