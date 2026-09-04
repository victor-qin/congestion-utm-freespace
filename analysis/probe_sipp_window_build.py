"""What does building SIPP's safe intervals per plan actually cost?

This is the gate on Phase 3 of `context/sipp_runtime_plan.md`. Phases 2-3 trade a global, always-
maintained free-interval pool (9.572 ms/flight of commit+release, GROWING with congestion) for a
per-plan build that lands in `t_plan_s`. That trade pays only if the build is cheap, and "cheap" has
reference point: `#124` measured A*'s dense-window paint at **0.514 ms per window** over the same
boxes. Gate: **under ~1.5 ms p90**.

Measured on the real thing, deliberately:

* on `density_faa`, not a synthetic box — interval count is data-dependent, so a fixture with three
  walls in it measures nothing;
* against the same window ANCHORS A* uses (same cells, same `_WINDOW_MARGIN_HEX`), by wrapping
  `AStarPlanner._build_window` and rebuilding SIPP's intervals over its box. The lateral extent is
  therefore identical; the STEP extent deliberately is not — A* clips to `s0 + n_gsteps +
  tail_steps` and this spans `[base, max_step]` (see `sipp.window.window_bounds`), so SIPP's box is
  the WIDER one and the 0.514 ms reference below is measured over a narrower span. The comparison is
  indicative, not like-for-like, and it is the absolute gate that decides Phase 3;
* with the njit **warm** (the first call compiles), and the compile discarded from the sample. A
  number taken inside pytest on a cold dispatcher would be dominated by LLVM.

    uv run python analysis/probe_sipp_window_build.py [demand_duration_s] [horizon_s]
"""
from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim import sim                                     # noqa: E402
from freespace_sim.planner.sipp import window as SW               # noqa: E402
from freespace_sim.planner.astar import planner as AP             # noqa: E402
from freespace_sim.planner.astar.compiled_hex_occupancy import (  # noqa: E402
    _FIELD_MASK,
    _S0_SHIFT,
    _SPAN_BITS,
)
from freespace_sim.planner.astar.window import W_Q0, W_Q1, W_R0, W_R1  # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0
HOR = float(sys.argv[2]) if len(sys.argv) > 2 else 12000.0
GATE_P90_MS = 1.5


class _Buffers:
    """One set of work arrays for the whole run, grown on demand — what the planner will hold.

    Sizing them per call would measure `np.zeros`; sizing them for the worst plan would tax every
    plan; `#124` makes the same argument for `ks["win"]`.
    """

    def __init__(self):
        self.n = 0
        self.iv_lo = self.iv_hi = self.iv_nxt = self.scratch = None

    def ensure(self, n: int) -> None:
        if n <= self.n:
            return
        self.n = n
        self.iv_lo = np.zeros(n, np.int32)
        self.iv_hi = np.zeros(n, np.int32)
        self.iv_nxt = np.full(n, -1, np.int32)
        # `build_window_intervals` requires scratch >= iv_lo: its capacity pass bounds the window's
        # TOTAL claim count, and one cell cannot exceed the total, so a single sizing rule covers
        # both buffers and there is only ever one shortfall to report.
        self.scratch = np.zeros(n, np.int64)


def main() -> int:
    spec = with_overrides(get_scenario("density_faa_wing_zipline"),
                          demand_duration_s=DUR, horizon_s=HOR)
    cfg = spec.config()
    demand = spec.demand_model()

    buf = _Buffers()
    wbox = SW.empty_wbox()
    samples: list[float] = []
    cells: list[int] = []
    slots: list[int] = []
    n_grow = 0
    orig = AP.AStarPlanner._build_window

    def probe(self, cocc, ks, gen, oq, orr, lane_q, lane_r, lane_stp, goal_q, goal_r,
              base, max_step, n_gsteps, tks, climb_span, n_hops, widen=0):
        nonlocal n_grow
        ok = orig(self, cocc, ks, gen, oq, orr, lane_q, lane_r, lane_stp, goal_q, goal_r,
                  base, max_step, n_gsteps, tks, climb_span, n_hops, widen=widen)
        # A*'s own anchors, verbatim (see `_build_window`), so the box is the one Phase 3 asks for.
        n = SW.window_bounds(
            cocc, wbox, lateral_margin=AP._WINDOW_MARGIN_HEX * (1 << widen), base=base,
            max_step=max_step,
            q_cells=(oq, int(lane_q.min()), int(lane_q.max()),
                     int(goal_q.min()), int(goal_q.max())),
            r_cells=(orr, int(lane_r.min()), int(lane_r.max()),
                     int(goal_r.min()), int(goal_r.max())),
        )
        if n <= 0:
            return ok
        buf.ensure(n + 16)
        for _ in range(6):
            t0 = time.perf_counter()
            tail = SW.build_window_intervals(
                cocc._arena.arena, cocc._arena.start, cocc._arena.length, cocc.static_col,
                ks["ov_own_gen"], gen, cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox,
                buf.iv_lo, buf.iv_hi, buf.iv_nxt, buf.scratch, _S0_SHIFT, _SPAN_BITS, _FIELD_MASK)
            dt = time.perf_counter() - t0
            if tail >= 0:
                samples.append(dt * 1e3)
                cells.append((int(wbox[W_Q1]) - int(wbox[W_Q0]) + 1)
                             * (int(wbox[W_R1]) - int(wbox[W_R0]) + 1) * cocc.n_levels)
                slots.append(tail)
                break
            n_grow += 1
            buf.ensure(-tail)          # a shortfall costs a rebuild; count them, they are not free
        return ok

    AP.AStarPlanner._build_window = probe
    t0 = time.monotonic()
    try:
        sim.run(cfg, demand=demand, planner_name="astar", progress=False, return_anchor="nominal")
    finally:
        AP.AStarPlanner._build_window = orig      # a raising run must not leave the class patched
    wall = time.monotonic() - t0

    # Drop the first sample from ALL THREE lists together: it pays the njit compile, and reporting
    # that as a build cost would be dishonest in the direction that kills the phase — but trimming
    # only the timings would report shape statistics over a different sample set than the ms line.
    if len(samples) < 2:
        print(f"only {len(samples)} window(s) built — too few to report past the compile call")
        return 1
    compile_ms = samples[0]
    samples, cells, slots = samples[1:], cells[1:], slots[1:]
    s = sorted(samples)
    p = lambda f: s[min(len(s) - 1, int(f * len(s)))]           # noqa: E731
    print(f"{len(s)} windows over {wall:.0f}s  (first call {compile_ms:.1f} ms = compile, dropped)")
    c9 = sorted(cells)[min(len(cells) - 1, int(.9 * len(cells)))]
    s9 = sorted(slots)[min(len(slots) - 1, int(.9 * len(slots)))]
    print(f"  window cells : p50 {st.median(cells):,.0f}  p90 {c9:,}")
    print(f"  slots used   : p50 {st.median(slots):,.0f}  p90 {s9:,}")
    print(f"  build ms     : p50 {st.median(s):.3f}  p90 {p(0.9):.3f}  p99 {p(0.99):.3f}  "
          f"max {s[-1]:.3f}  mean {st.mean(s):.3f}")
    print(f"  buffer grows : {n_grow}")
    print("\n  A* dense-window paint, for reference (#124): 0.514 ms/window")
    verdict = "PASS" if p(0.9) < GATE_P90_MS else "FAIL"
    print(f"  GATE p90 < {GATE_P90_MS} ms: {verdict} ({p(0.9):.3f} ms)")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
