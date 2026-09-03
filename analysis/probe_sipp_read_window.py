"""How much of SIPP's GLOBAL interval pool does one plan actually read?

``#124`'s argument for A* was that a plan reads a small axial box over a bounded step range, so a
global derived structure is maintained for nothing. This measured the same ratio for SIPP and is what
sized the window in `context/sipp_runtime_plan.md` Phase 2: a plan read 1.37% of the pool's cells
over 28.5% of its steps.

Phase 3 acted on that — SIPP now derives its safe intervals per plan and the global pool is deleted —
so this script is now a WINDOW-SIZING instrument rather than an argument for the rewrite: the box it
reports against is the claim arena's, and the read bbox is what `_SWINDOW_MARGIN_HEX` has to cover.

    uv run python analysis/probe_sipp_read_window.py [demand_duration_s] [horizon_s]
"""
from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim import sim                                     # noqa: E402
from freespace_sim.planner import sipp as sipp_mod                # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
HOR = float(sys.argv[2]) if len(sys.argv) > 2 else 6000.0


def main() -> int:
    spec = with_overrides(get_scenario("density_faa_wing_zipline"),
                          demand_duration_s=DUR, horizon_s=HOR)
    cfg = spec.config()
    demand = spec.demand_model()

    samples: list[tuple] = []
    orig = sipp_mod.SIPPPlanner._splan_compiled

    def probe(self, req, ledger, cfg):
        out = orig(self, req, ledger, cfg)
        rb, cocc = self._k_read_bbox, self._cocc
        # min > max is `_note_cell`'s "never probed" sentinel; a plan that fell to the reference
        # before its first probe has no read set and must not be counted as a tiny one.
        if rb is not None and cocc is not None and rb[0] <= rb[1]:
            samples.append((int(rb[1] - rb[0] + 1), int(rb[3] - rb[2] + 1), int(rb[5] - rb[4] + 1),
                            int(rb[7] - rb[6] + 1), cocc.qspan, cocc.rspan, cocc.n_levels,
                            cocc.MAXS, cocc._arena.n_claims, cocc.NC))
        return out

    sipp_mod.SIPPPlanner._splan_compiled = probe
    t0 = time.monotonic()
    sim.run(cfg, demand=demand, planner_name="sipp", progress=False, return_anchor="nominal")
    print(f"sipp run: {len(samples)} compiled plans in {time.monotonic() - t0:.0f}s")
    if not samples:
        print("  no compiled plans — every flight took the reference; nothing to report")
        return 1

    qs, rs, ls, ss, QS, RS, NL, MAXS, nslots, NC = zip(*samples)
    box_cells = QS[0] * RS[0] * NL[0]
    rd = sorted(q * r * lv for q, r, lv in zip(qs, rs, ls))
    sss = sorted(ss)
    p90 = lambda a: a[int(0.9 * len(a))]           # noqa: E731 — one-line percentile, read once
    print(f"  global box     : {QS[0]} x {RS[0]} q,r x {NL[0]} lev = {box_cells:,} cells, "
          f"MAXS {MAXS[0]}, arena claims {max(nslots):,} (NC {NC[0]:,})")
    print(f"  read bbox cells: p50 {st.median(rd):,.0f}  p90 {p90(rd):,}  max {max(rd):,}")
    print(f"  read steps     : p50 {st.median(sss):,.0f}  p90 {p90(sss):,}  max {max(sss):,} "
          f"(of {MAXS[0]})")
    print(f"  COVERAGE       : p50 {st.median(rd) / box_cells * 100:.3f}% of cells, "
          f"{st.median(sss) / MAXS[0] * 100:.1f}% of steps => "
          f"{st.median(rd) / box_cells * st.median(sss) / MAXS[0] * 100:.4f}% of the (cell,step) box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
