"""Does asymmetric reservation filing survive the ACTUAL ledger, end to end?

Runs the real whole-schedule path -- ``run_batch`` -> ``column_to_intent`` ->
``DSS.commit`` against a live :class:`ReservationLedger` with permanent hub walls --
under both filings, then re-derives the core ASTM invariant independently with
``verify.find_interflight_conflict`` (fresh-ledger FCFS replay of every accepted
intent, walls included).

Asymmetric (leading-only) filing is now what production ships, so ``--filing asym`` is
UNPATCHED -- it measures the tree as-is and is the default. The historical comparison
survives as ``--filing legacy-sym``, which re-installs the symmetric pads. That arm has
to patch BOTH halves of the product, unlike ``probe_dag_vs_labels`` which only needed
pricing:

* ``corridor_segment_volume`` in every module that bound it (row measurement, wall
  probes, and the reservation builder itself), and
* ``translate._retime_lattice_reservation``, the commit-path re-stamp that would
  otherwise silently leave the shipped leading-only pads on every transit sub-box.

    uv run python analysis/probe_ledger_e2e.py --flights 50 --iterations 2
    uv run python analysis/probe_ledger_e2e.py --flights 50 --iterations 2 --filing legacy-sym
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim import verify  # noqa: E402
from freespace_sim import volumes as volumes_mod  # noqa: E402
from freespace_sim.dss import DSS  # noqa: E402
from freespace_sim.ledger import ReservationLedger  # noqa: E402
from freespace_sim.mechanism import FCFSMechanism  # noqa: E402
from freespace_sim.planner import terminal_capacity as terminal_capacity_mod  # noqa: E402
from freespace_sim.planner.colgen import network as network_mod  # noqa: E402
from freespace_sim.planner.colgen import translate as translate_mod  # noqa: E402
from freespace_sim.planner.colgen import windows as windows_mod  # noqa: E402
from freespace_sim.planner.colgen.batch import run_batch  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.scenario import scenario_from_requests  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402


def install_legacy_sym_filing() -> None:
    """Re-install the pre-2026-08-14 symmetric pads, for the historical A/B arm."""

    real_builder = volumes_mod.corridor_segment_volume

    def sym_corridor_segment_volume(p0, t0, p1, t1, cfg, *, terminal_id=None):
        vol = real_builder(p0, t0, p1, t1, cfg, terminal_id=terminal_id)
        return replace(vol, t_start=vol.t_start - cfg.time_buffer_s)

    for mod in (volumes_mod, windows_mod, network_mod, terminal_capacity_mod):
        mod.corridor_segment_volume = sym_corridor_segment_volume
    windows_mod.derive_cell_window.cache_clear()
    windows_mod.validate_edge_locality.cache_clear()

    real_retime = translate_mod._retime_lattice_reservation

    def sym_retime(volumes, centerline, corners, corridor_t0, origin_t0, origin_dwell_s,
                   destination_dwell_s, cfg):
        retimed, timed_centerline = real_retime(
            volumes, centerline, corners, corridor_t0, origin_t0, origin_dwell_s,
            destination_dwell_s, cfg,
        )
        # Endpoint cylinders (first/last) keep their exact dwell windows; every transit
        # sub-box regrows its trailing pad, exactly as the patched builder files it.
        shifted = [retimed[0]]
        shifted.extend(
            replace(volume, t_start=volume.t_start - cfg.time_buffer_s)
            for volume in retimed[1:-1]
        )
        shifted.append(retimed[-1])
        return shifted, timed_centerline

    translate_mod._retime_lattice_reservation = sym_retime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--filing", default="asym", choices=("asym", "legacy-sym"))
    parser.add_argument("--gap-metric", default="cost", choices=("cost", "revenue"))
    args = parser.parse_args()

    if args.filing == "legacy-sym":
        install_legacy_sym_filing()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    offsets = windows_mod.derive_cell_window(cfg)
    print(f"filing    {args.filing}  cell window {offsets}")

    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    scenario = scenario_from_requests(requests)

    ledger = ReservationLedger(cfg)
    for center, term in static_terms:
        ledger.register_static_terminal(center, term)
    dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
    params = ColGenParams(
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        n_pricing_workers=0,
    )

    started = time.perf_counter()
    intents, stats = run_batch(
        scenario, cfg, ledger, dss, static_terms,
        lambda done, request, intent: None,
        lambda done, total, intent: None,
        collector=object(),
        params=params,
    )
    wall = time.perf_counter() - started

    accepted = [i for i in intents if i.accepted]
    denials = Counter(
        (i.denial_reason.name if i.denial_reason is not None else "?")
        for i in intents if not i.accepted
    )
    ground = sum(i.ground_delay_s for i in accepted)
    detour = sum(i.air_detour_m for i in accepted)
    n_volumes = sum(len(i.volumes) for i in accepted)
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"gap={args.gap_metric}")
    print(f"WALL {wall:.1f}s  iters={stats.get('iterations')} "
          f"termination={stats.get('termination_reason')!r} "
          f"objective={stats.get('objective')!r}")
    print(f"accepted  {len(accepted)}/{len(intents)}  denials {dict(denials) or '{}'}")
    print(f"ground_delay_s sum {ground:.1f}  air_detour_m sum {detour:.1f}  "
          f"volumes committed {n_volumes}")

    conflict = verify.find_interflight_conflict(intents, cfg, static_terminals=static_terms)
    print(f"LEDGER REPLAY AUDIT: "
          f"{'CLEAN (no inter-flight or wall conflict)' if conflict is None else f'CONFLICT {conflict}'}")

    # Own-reservation contiguity: the moving aircraft must sit inside a live own volume
    # at every centerline timestamp (trailing-pad removal must not open a coverage gap).
    uncovered = 0
    for intent in accepted:
        for _point, t in intent.centerline:
            if not any(v.t_start <= t <= v.t_end for v in intent.volumes):
                uncovered += 1
    print(f"centerline timestamps outside every own volume window: {uncovered}")


if __name__ == "__main__":
    main()
