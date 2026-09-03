"""Where does SIPP's LNS wall-clock actually go? Per-structure, per-operation attribution.

#124 rewrote A*'s occupancy (claim arena + dense window) and left SIPP's structures untouched, so
SIPP went from ~1.1x FASTER than A* under LNS to 1.50x slower. This script splits the ledger side
into the individual structures and the individual operations inside each, so the fix targets the
term that dominates instead of the one that is easiest to see.

Measures, per structure, over a warm ledger at density_faa scale:
  * commit  (`on_commit`) and release (`on_release`) wall, per flight;
  * for the interval pool, the RE-APPLY AMPLIFICATION -- how many survivor claims a release has to
    replay per claim it actually removes (A*'s predecessor measured 12.2x here, which is why #124
    replaced it with a swap-remove);
  * for the hex service, the cost attributable to the `blocked` map alone. SIPP USED to force this
    on via `needs_blocked_map`; Phase 1 (`f4be4f0`) removed that, so the "blocked ON" row is now the
    HISTORICAL cost, not what a compiled SIPP pays. Both rows are kept because the delta between
    them is what Phase 1 banked (-1.807 ms/flight, measured -10.0 s of a 124.1 s LNS ledger).

    uv run python analysis/prof_sipp_ledger.py [n_warm] [n_timed]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim import sim                                   # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402
from freespace_sim.planner.astar.occupancy import HexOccupancyService  # noqa: E402
from freespace_sim.planner.astar.compiled_hex_occupancy import CompiledHexOccupancy  # noqa: E402
from freespace_sim.planner.compiled_occupancy import CompiledOccupancy  # noqa: E402
from freespace_sim.planner.sipp import SafeIntervalIndex        # noqa: E402

N_WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 600
N_TIMED = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
HOR = float(sys.argv[4]) if len(sys.argv) > 4 else 6000.0


def _build(spec, cfg, demand):
    res = sim.run(cfg, demand=demand, planner_name="astar", progress=False, return_anchor="nominal")
    by_fid = {}
    for fid, v in res.ledger.iter_committed():
        by_fid.setdefault(fid, []).append(v)
    return res, by_fid


def _absorb(struct, groups, statics):
    for c, t in statics:
        struct._on_static(c, t)
    for fid, vols in groups:
        struct.on_commit(fid, vols)


def main() -> int:
    spec = with_overrides(get_scenario("density_faa_wing_zipline"),
                          demand_duration_s=DUR, horizon_s=HOR)
    cfg = spec.config()
    demand = spec.demand_model()
    t0 = time.monotonic()
    res, by_fid = _build(spec, cfg, demand)
    print(f"baseline: {len(by_fid)} flights, {sum(len(v) for v in by_fid.values())} volumes, "
          f"{time.monotonic() - t0:.0f}s", flush=True)

    statics = list(res.ledger._static_terms)   # (center, term) pairs; the subscribe_static replay

    items = list(by_fid.items())
    warm, timed = items[:N_WARM], items[N_WARM:N_WARM + N_TIMED]
    n_vol_timed = sum(len(v) for _, v in timed)
    print(f"warm {len(warm)} flights / timed {len(timed)} flights, {n_vol_timed} volumes\n", flush=True)

    variants = [
        ("pre-#Ph1 _svc (hex dicts, blocked ON )", lambda: HexOccupancyService(
            cfg, track_removal=True, maintain_blocked=True)),
        ("both     _svc (hex dicts, blocked OFF)", lambda: HexOccupancyService(
            cfg, track_removal=True, maintain_blocked=False)),
        ("sipp  _scocc (interval POOL)        ", lambda: CompiledOccupancy(cfg, track_removal=True)),
        ("astar _cocc  (claim ARENA)          ", lambda: CompiledHexOccupancy(cfg, track_removal=True)),
        ("sipp  _sidx  (step dicts)           ", lambda: SafeIntervalIndex(cfg, track_removal=True)),
    ]

    print(f"{'structure':38} {'commit ms/fl':>13} {'release ms/fl':>14} {'total':>8}")
    rows = {}
    for label, mk in variants:
        st = mk()
        _absorb(st, warm, statics)
        t0 = time.perf_counter()
        for fid, vols in timed:
            st.on_commit(fid, vols)
        t_commit = time.perf_counter() - t0
        t0 = time.perf_counter()
        for fid, vols in reversed(timed):
            st.on_release(fid, vols)
        t_release = time.perf_counter() - t0
        c, r = t_commit / len(timed) * 1e3, t_release / len(timed) * 1e3
        rows[label.strip()] = (c, r)
        print(f"{label:38} {c:13.3f} {r:14.3f} {c + r:8.3f}", flush=True)

    # The re-apply amplification is the whole reason #124 exists on the A* side, and it GROWS with
    # congestion (the multiplier is how many OTHER flights share the released cells). One number at
    # one schedule size would under-state the fix, so sweep the warm set.
    print("\n--- interval-pool release: re-apply amplification vs congestion ---", flush=True)
    print(f"  {'warm flights':>13} {'own claims':>11} {'cells':>8} {'re-applied':>11} "
          f"{'amplif':>8} {'rel ms/fl':>10}")
    for n_warm in [w for w in (150, 300, 600, 1200, 2400, 4800) if w <= len(items) - N_TIMED] or [N_WARM]:
        st = CompiledOccupancy(cfg, track_removal=True)
        _absorb(st, items[:n_warm], statics)
        tset = items[n_warm:n_warm + N_TIMED]
        for fid, vols in tset:
            st.on_commit(fid, vols)
        n_own = n_survivor = n_cells = n_static = 0
        for fid, _vols in tset:
            rows_j = st._rows.get(fid)
            if not rows_j:
                continue
            cells = {rows_j[i] for i in range(0, len(rows_j), 2)}
            n_own += len(rows_j) // 2
            n_cells += len(cells)
            for c in cells:
                if c in st._static_cells:
                    n_static += 1
                else:
                    n_survivor += len(st._claims.get(c, ()))
        t0 = time.perf_counter()
        for fid, vols in reversed(tset):
            st.on_release(fid, vols)
        rel = (time.perf_counter() - t0) / len(tset) * 1e3
        print(f"  {n_warm:>13} {n_own:>11} {n_cells:>8} {n_survivor:>11} "
              f"{n_survivor / max(1, n_own):>7.2f}x {rel:>10.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
