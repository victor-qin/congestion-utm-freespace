"""Do the schedules a planner emits describe flights a real drone could actually fly?

Runs the REAL production sim (``freespace_sim.sim.run`` -> per-flight FCFS for A*, ``run_batch``
for colgen) on N density flights and audits the committed result along four independent axes:

1. **Kinematics** -- implied ground speed and climb rate between consecutive centerline
   timestamps, against ``nominal_speed_mps`` / ``climb_rate_mps``.  A lattice hop is
   ``nominal_speed_mps * dt_s`` by construction, so this is really a test of everything that
   is NOT a plain cruise hop: terminal lane folds, hex snapping, climb/descent rungs, and the
   resampled sub-boxes ``_retime_lattice_reservation`` interpolates.
2. **Separation** -- minimum pairwise distance between airborne aircraft on a time grid, split
   into all-pairs and en-route-only (both aircraft clear of every hub's terminal radius, where
   proximity is pad-capacity business rather than a corridor question).
3. **Ledger replay** -- ``verify.find_interflight_conflict`` on a fresh ledger, walls included.
4. **Own-reservation coverage** -- every centerline instant inside a live own volume.

``--filing legacy-sym`` restores the pre-2026-08-14 symmetric pads so the same audit can be run
against the construction this replaced.  Both planners share one setup (same truncated request
list, same full hub-wall set) so the arms are comparable to each other and across filings.

    uv run python analysis/probe_physics_500.py --planner astar  --flights 500
    uv run python analysis/probe_physics_500.py --planner colgen --flights 500
    uv run python analysis/probe_physics_500.py --planner astar  --flights 500 --filing legacy-sym
"""
from __future__ import annotations

import argparse
import math
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
from freespace_sim.planner import terminal_capacity as terminal_capacity_mod  # noqa: E402
from freespace_sim.planner.colgen import translate as translate_mod  # noqa: E402
from freespace_sim.planner.colgen import windows as windows_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.scenario import scenario_from_requests  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402
from freespace_sim.sim import run as sim_run  # noqa: E402


def install_legacy_sym_filing() -> None:
    """Restore the pre-2026-08-14 symmetric time padding on corridor transit volumes.

    Patches the builder for EVERY module that bound it by name plus ``volumes`` itself (A* reaches
    it through that module's own globals via ``build_corridor`` /
    ``build_reservation_from_corners``), and the colgen commit-path re-stamp.
    """

    real_builder = volumes_mod.corridor_segment_volume

    def sym_corridor_segment_volume(p0, t0, p1, t1, cfg, *, terminal_id=None):
        vol = real_builder(p0, t0, p1, t1, cfg, terminal_id=terminal_id)
        return replace(vol, t_start=vol.t_start - cfg.time_buffer_s)

    for mod in (volumes_mod, windows_mod, terminal_capacity_mod):
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
        shifted = [retimed[0]]
        shifted.extend(
            replace(volume, t_start=volume.t_start - cfg.time_buffer_s)
            for volume in retimed[1:-1]
        )
        shifted.append(retimed[-1])
        return shifted, timed_centerline

    translate_mod._retime_lattice_reservation = sym_retime


# --------------------------------------------------------------------------- audits

def audit_kinematics(accepted, cfg, tol=1e-6):
    """Implied speed per centerline leg vs the configured envelope.

    Horizontal and vertical are checked SEPARATELY because the sim models them as independent
    rates (``nominal_speed_mps`` cruise, ``climb_rate_mps`` climb/descent) -- a diagonal leg is
    legal iff each component is within its own limit, which is also how ``climb_time_to`` and
    ``corridor_segment_len_m`` derive their timings.
    """
    v_max, c_max = cfg.nominal_speed_mps, cfg.climb_rate_mps
    worst_h = (0.0, None)
    worst_v = (0.0, None)
    n_legs = n_bad_h = n_bad_v = n_zero_dt = 0
    for intent in accepted:
        cl = intent.centerline
        for (p0, t0), (p1, t1) in zip(cl, cl[1:]):
            dt = float(t1) - float(t0)
            if dt <= 0.0:
                # A zero-duration leg is only sane if it also has zero displacement.
                if math.dist(tuple(map(float, p0[:3])), tuple(map(float, p1[:3]))) > 1e-6:
                    n_zero_dt += 1
                continue
            n_legs += 1
            dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
            dz = float(p1[2]) - float(p0[2])
            sh, sv = math.hypot(dx, dy) / dt, abs(dz) / dt
            if sh > worst_h[0]:
                worst_h = (sh, (intent.request.flight_id, float(t0), float(t1), math.hypot(dx, dy)))
            if sv > worst_v[0]:
                worst_v = (sv, (intent.request.flight_id, float(t0), float(t1), dz))
            if sh > v_max * (1.0 + tol):
                n_bad_h += 1
            if sv > c_max * (1.0 + tol):
                n_bad_v += 1
    return {
        "legs": n_legs, "bad_horizontal": n_bad_h, "bad_vertical": n_bad_v,
        "zero_dt_moves": n_zero_dt,
        "max_horizontal_mps": worst_h[0], "worst_horizontal": worst_h[1],
        "max_vertical_mps": worst_v[0], "worst_vertical": worst_v[1],
        "limit_horizontal_mps": v_max, "limit_vertical_mps": c_max,
    }


def _tracks(accepted):
    """(flight_id, times[n], xyz[n,3]) per accepted intent, for time-sliced sampling."""
    out = []
    for intent in accepted:
        cl = intent.centerline
        if len(cl) < 2:
            continue
        times = np.asarray([float(t) for _p, t in cl], dtype=float)
        xyz = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p, _t in cl], dtype=float)
        out.append((intent.request.flight_id, times, xyz))
    return out


def _sample_positions(tracks, t):
    """Linear interpolation of each centerline at absolute time ``t`` (airborne cruise only)."""
    out_xyz, out_id = [], []
    for fid, times, xyz in tracks:
        if t < times[0] or t > times[-1]:
            continue
        i = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side="right")) - 1))
        span = times[i + 1] - times[i]
        u = 0.0 if span <= 0 else (t - times[i]) / span
        out_xyz.append(xyz[i] + (xyz[i + 1] - xyz[i]) * u)
        out_id.append(fid)
    return np.asarray(out_xyz, dtype=float), out_id


def audit_separation(accepted, cfg, hub_xy, hub_radius, step_s):
    """Minimum pairwise distance between airborne aircraft, sampled on a time grid.

    Reported twice: over all airborne pairs, and restricted to pairs where BOTH aircraft are
    clear of every hub's terminal radius.  Near a hub, aircraft are legitimately close (separate
    pads, serialised by terminal capacity), so the en-route figure is the one that speaks to
    corridor separation and therefore to the headway change.
    """
    from scipy.spatial import cKDTree

    tracks = _tracks(accepted)
    if len(tracks) < 2:
        return {"min_all_m": (math.inf, None, None), "min_enroute_m": (math.inf, None, None),
                "peak_airborne": 0, "grid_s": step_s}
    t_lo = min(times[0] for _f, times, _x in tracks)
    t_hi = max(times[-1] for _f, times, _x in tracks)
    best_all = (math.inf, None, None)
    best_enr = (math.inf, None, None)
    peak_airborne = 0
    hubs = np.asarray(hub_xy, dtype=float) if len(hub_xy) else np.zeros((0, 2))
    hub_tree = cKDTree(hubs) if len(hubs) else None

    for t in np.arange(float(t_lo), float(t_hi) + step_s, step_s):
        xyz, ids = _sample_positions(tracks, float(t))
        if len(xyz) < 2:
            continue
        peak_airborne = max(peak_airborne, len(xyz))
        # k=2 nearest-neighbour gives the TRUE minimum every slice -- a fixed query radius
        # silently reports `inf` on sparse slices, which reads like "no violation" when it
        # actually means "not measured".
        dist, idx = cKDTree(xyz).query(xyz, k=2)
        j = int(np.argmin(dist[:, 1]))
        if dist[j, 1] < best_all[0]:
            best_all = (float(dist[j, 1]), (ids[j], ids[int(idx[j, 1])]), float(t))
        if hub_tree is not None:
            free = hub_tree.query(xyz[:, :2])[0] > hub_radius      # clear of every hub
            if int(free.sum()) >= 2:
                sub_xyz = xyz[free]
                sub_ids = [i for i, keep in zip(ids, free) if keep]
                d2, i2 = cKDTree(sub_xyz).query(sub_xyz, k=2)
                j = int(np.argmin(d2[:, 1]))
                if d2[j, 1] < best_enr[0]:
                    best_enr = (float(d2[j, 1]), (sub_ids[j], sub_ids[int(i2[j, 1])]), float(t))
    return {"min_all_m": best_all, "min_enroute_m": best_enr,
            "peak_airborne": peak_airborne, "grid_s": step_s}


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--flights", type=int, default=500)
    ap.add_argument("--planner", default="astar", choices=("astar", "colgen"))
    ap.add_argument("--filing", default="asym", choices=("asym", "legacy-sym"))
    ap.add_argument("--colgen-iterations", type=int, default=2)
    ap.add_argument("--pricing-workers", type=int, default=8)
    ap.add_argument("--sep-grid-s", type=float, default=1.0,
                    help="separation sampling step; 0 disables the separation audit")
    ap.add_argument("--time-buffer", type=float, default=None,
                    help="override time_buffer_s. 0 removes corridor time padding entirely, which "
                         "is the decisive test of whether corridor headway BINDS at all for a "
                         "planner on a scenario (vs terminal/pad capacity, which carries no pad)")
    args = ap.parse_args()

    if args.filing == "legacy-sym":
        install_legacy_sym_filing()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    if args.time_buffer is not None:
        cfg = replace(cfg, time_buffer_s=args.time_buffer)
    print(f"tree      {REPO_ROOT}")
    print(f"buffer    time_buffer_s={cfg.time_buffer_s}  dt_s={cfg.dt_s}")
    print(f"filing    {args.filing}  cell window {windows_mod.derive_cell_window(cfg)}")

    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    scenario = scenario_from_requests(requests)
    params = None
    if args.planner == "colgen":
        params = ColGenParams(
            max_iterations=args.colgen_iterations, time_limit_s=86400.0,
            gap_metric="cost", n_pricing_workers=args.pricing_workers,
        )
    print(f"workload  {args.scenario} x{len(requests)} planner={args.planner}"
          + (f" iters={args.colgen_iterations} workers={args.pricing_workers}"
             if params else ""))

    started = time.perf_counter()
    result = sim_run(cfg, scenario=scenario, demand=demand,
                     planner_name=args.planner, planner_params=params)
    wall = time.perf_counter() - started

    intents = list(result.intents)
    accepted = [i for i in intents if i.accepted]
    denials = Counter((i.denial_reason.name if i.denial_reason is not None else "?")
                      for i in intents if not i.accepted)
    ground = sum(i.ground_delay_s for i in accepted)
    detour = sum(i.air_detour_m for i in accepted)
    print(f"WALL {wall:.1f}s")
    print(f"accepted  {len(accepted)}/{len(intents)}  denials {dict(denials) or '{}'}")
    print(f"ground_delay_s sum {ground:.1f}  air_detour_m sum {detour:.1f}  "
          f"volumes {sum(len(i.volumes) for i in accepted)}")

    # Exact plan fingerprint: every committed volume window + every timed centerline point, so
    # "the two filings agree" can be a bit-level claim rather than an inference from summary
    # totals that could coincide while individual plans differ.
    import hashlib

    h = hashlib.sha256()
    for intent in sorted(accepted, key=lambda i: i.request.flight_id):
        h.update(f"|{intent.request.flight_id}".encode())
        for v in intent.volumes:
            h.update(f";{v.t_start!r},{v.t_end!r},{v.terminal_id!r}".encode())
        for p, t in intent.centerline:
            h.update(f";{float(p[0])!r},{float(p[1])!r},{float(p[2])!r},{float(t)!r}".encode())
    print(f"plan fingerprint (volumes+centerlines): {h.hexdigest()[:32]}")
    air_hold = sum(i.air_hold_s for i in accepted)
    print(f"delay split: ground {ground:.1f}s  air_hold {air_hold:.1f}s  "
          f"lattice_overhead {sum(i.lattice_overhead_m for i in accepted):.1f}m")

    kin = audit_kinematics(accepted, cfg)
    print(f"\nKINEMATICS  legs={kin['legs']:,}  limits {kin['limit_horizontal_mps']:.1f} m/s "
          f"horizontal / {kin['limit_vertical_mps']:.1f} m/s vertical")
    print(f"  max horizontal {kin['max_horizontal_mps']:.6f} m/s   violations {kin['bad_horizontal']}")
    print(f"  max vertical   {kin['max_vertical_mps']:.6f} m/s   violations {kin['bad_vertical']}")
    print(f"  zero-dt moves with nonzero displacement: {kin['zero_dt_moves']}")
    if kin["bad_horizontal"] or kin["bad_vertical"]:
        print(f"  WORST horizontal leg {kin['worst_horizontal']}")
        print(f"  WORST vertical   leg {kin['worst_vertical']}")
    print(f"  KINEMATIC VERDICT: "
          f"{'FEASIBLE' if not (kin['bad_horizontal'] or kin['bad_vertical'] or kin['zero_dt_moves']) else 'INFEASIBLE'}")

    if args.sep_grid_s > 0 and len(accepted) >= 2:
        hub_xy = [(float(c[0]), float(c[1])) for c, _t in demand.terminals(cfg)]
        sep = audit_separation(accepted, cfg, hub_xy, cfg.terminal_radius_m, args.sep_grid_s)
        d, pair, t = sep["min_all_m"]
        print(f"\nSEPARATION  grid={sep['grid_s']}s  peak airborne={sep['peak_airborne']}  "
              f"hubs={len(hub_xy)}")
        print(f"  min distance, ALL airborne pairs : {d:8.2f} m  flights {pair} at t={t}")
        d, pair, t = sep["min_enroute_m"]
        print(f"  min distance, EN ROUTE pairs     : {d:8.2f} m  flights {pair} at t={t}"
              f"   (both > {cfg.terminal_radius_m:.0f} m from every hub)")
        print(f"  reference: corridor is {cfg.corridor_width_m:.0f} m wide, "
              f"{cfg.corridor_height_m:.0f} m tall, hop pitch {cfg.corridor_segment_len_m:.0f} m")

    conflict = verify.find_interflight_conflict(
        intents, cfg, static_terminals=list(demand.terminals(cfg))
    )
    print(f"\nLEDGER REPLAY AUDIT: "
          f"{'CLEAN' if conflict is None else f'CONFLICT {conflict}'}")
    uncovered = sum(
        1 for i in accepted for _p, t in i.centerline
        if not any(v.t_start <= t <= v.t_end for v in i.volumes)
    )
    print(f"centerline timestamps outside every own volume window: {uncovered}")


if __name__ == "__main__":
    main()
