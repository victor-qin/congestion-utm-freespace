"""Dense per-plan occupancy window (``planner.astar.window``) — truth and end-to-end parity.

The window is a cache of the interval pools ``kernel._blocked`` otherwise walks, so it has exactly
one contract: **every bit equals what the pools would have answered**. That is checked twice, at two
different levels, because the two failure modes are different:

* :func:`test_window_bit_matches_pool_truth` compares the bitmap cell-by-cell, step-by-step against
  ``CompiledHexOccupancy.blocked_py`` — the same oracle ``test_compiled_occupancy_matches_is_blocked``
  uses for the pools themselves. This is what catches a build bug (an off-by-one in the interval
  merge, a mis-folded ``static_col``, a bit-packing slip) directly, at the bit.
* the ``_assert_window_exact`` tests compare whole PLANS with the window on and off. This is what
  catches a wiring bug — a window built under a stale ``gen``, or one whose bounds miss probes the
  kernel then answers from the wrong place.

The second kind needs the window to actually be in force, so every one of them asserts a non-zero
hit count. A test whose window was silently disabled would pass while proving nothing.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar import window as W
from freespace_sim.planner.astar.compiled_hex_occupancy import CompiledHexOccupancy
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import Volume4D


def test_window_module_imports_without_numba():
    """``planner`` imports ``window`` at module level, but its numba fallback is an ImportError guard
    around ``.kernel`` inside ``AStarPlanner.__init__``. So a hard ``from numba import njit`` in
    ``window`` turns the documented "degrade to the pure-Python reference" into "the package will not
    import" — which it did, until ``window`` grew its own guard. Reproduces a numba-less install the
    way ``test_compiled_absent_falls_back_to_reference`` does, in a subprocess because the import has
    to happen from cold."""
    src = textwrap.dedent("""
        import sys
        sys.modules["numba"] = None                  # what a numba-less install looks like
        import warnings; warnings.filterwarnings("ignore")
        from freespace_sim.config import SimConfig
        from freespace_sim.ledger import ReservationLedger
        from freespace_sim.planner.astar import AStarPlanner
        from freespace_sim.types import FlightRequest, vec
        p = AStarPlanner()
        assert p.compiled is False, "should have degraded to the reference"
        cfg = SimConfig()
        it = p.plan(FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
                    ReservationLedger(cfg), cfg)
        assert it.accepted
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


# Deliberately BELOW the test above, which needs the repo rather than numba: the skip is
# whole-module, and in a numba-less environment that test is redundant anyway (the environment is
# the proof). In the normal environment — the one a regression would be introduced in — it runs.
pytest.importorskip("numba", reason="the rest of this file needs the compiled kernel path")

CFG = SimConfig()
_ON = 2 << 20


def _req(fid=1, dx=2000.0, **kw):
    return FlightRequest(fid, vec(0, 0, 0), vec(dx, 0, 0), 0.0, **kw)


def _clkey(intent):
    return [(round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 3), round(float(t), 6))
            for p, t in (intent.centerline or [])]


def test_window_jit_warms_under_the_compiled_fallback_guard(monkeypatch):
    """The separately-decorated window dispatcher must fail closed during planner construction."""
    from freespace_sim.planner.astar import kernel as K

    calls = []
    monkeypatch.setattr(K, "_search", lambda *_args: calls.append("search"))

    def fail_window(*_args):
        calls.append("window")
        raise RuntimeError("broken window cache")

    monkeypatch.setattr(W, "build_window", fail_window)
    with pytest.warns(RuntimeWarning, match="dense-window kernel failed"):
        planner = AStarPlanner(compiled=True)

    assert calls == ["search", "window"]
    assert planner.compiled is False and planner._kernel is None


def test_window_jit_runtime_failure_replans_and_stays_disabled(monkeypatch):
    """A post-warm dispatcher failure must re-run this flight in Python and disable later JIT use."""
    req = _req()
    reference = AStarPlanner(compiled=False)
    expected = reference.plan(req, ReservationLedger(CFG), CFG)
    planner = AStarPlanner(compiled=True)
    calls = []

    def fail_window(*_args):
        calls.append(True)
        raise RuntimeError("window dispatcher failed after warm-up")

    monkeypatch.setattr(W, "build_window", fail_window)
    with pytest.warns(RuntimeWarning, match="dense-window kernel failed"):
        got = planner.plan(req, ReservationLedger(CFG), CFG)

    assert calls == [True], "the test never reached the window dispatcher"
    assert planner.compiled is False and planner._kernel is None
    assert planner._ref_dispatch["window-jit"] == 1
    assert got.status is expected.status and got.cost == expected.cost
    assert planner.last_expansions == reference.last_expansions
    assert _clkey(got) == _clkey(expected)

    again = planner.plan(req, ReservationLedger(CFG), CFG)
    assert calls == [True], "a disabled planner called the failing JIT again"
    assert again.status is expected.status and again.cost == expected.cost
    assert _clkey(again) == _clkey(expected)


def test_fanout_benchmark_rejects_window_divergence(monkeypatch):
    from analysis import ab_dense_window as ab

    results = iter([
        (1.0, [("same", 11)]),
        (0.5, [("different", 11)]),
    ])
    monkeypatch.setattr(ab, "_pass", lambda *_args: next(results))

    with pytest.raises(RuntimeError, match=r"DIVERGENCE: window changed 1 of 1 plans"):
        ab._paired_pass({"off": object(), "on": object()}, (), None, None)


def test_ab_benchmark_disables_monotone_eviction(monkeypatch):
    from analysis import ab_dense_window as ab

    class FakePlanner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.evict_floor = None

    monkeypatch.setattr(ab, "AStarPlanner", FakePlanner)
    arms = ab._make_arms(window_bytes=1234, kernel_log2=18)

    assert arms["off"].kwargs == {"window_bytes": 0, "kernel_log2_min": 18}
    assert arms["on"].kwargs == {"window_bytes": 1234, "kernel_log2_min": 18}
    assert all(planner.evict_floor == 0.0 for planner in arms.values())


def test_fanout_benchmark_synchronizes_each_arm_and_uses_medians(monkeypatch):
    from analysis import ab_dense_window as ab

    rows = [("same", 11)]
    results = iter((duration, rows) for duration in (9.0, 5.0, 3.0, 7.0, 6.0, 1.0))
    monkeypatch.setattr(ab, "_pass", lambda *_args: next(results))
    barriers = []

    elapsed = ab._timed_passes(
        {"off": object(), "on": object()}, (), None, None, 3,
        before_arm=lambda: barriers.append(True),
    )

    assert elapsed == {"off": 6.0, "on": 5.0}
    assert len(barriers) == 6


def _assert_window_exact(reqs, commits, cfg=CFG, statics=(), window_bytes=_ON, expect_window=True):
    """Plan every request twice against identically-built ledgers — window off, then on — and assert
    the two runs are indistinguishable: same accept/deny, same cost, same centerline, and the same
    node-expansion count (two different searches essentially never expand the same number of nodes).

    Returns the window-on planner so a caller can assert on its counters."""
    out = {}
    planners = {}
    for name, wb in (("off", 0), ("on", window_bytes)):
        p = AStarPlanner(window_bytes=wb)
        assert p.compiled, "numba kernel inactive — this parity guard would compare reference to itself"
        led = ReservationLedger(cfg)
        for c, t in statics:
            led.register_static_terminal(c, t)
        for fid, vols in commits:
            led.commit(fid, vols)
        rows = []
        for req in reqs:
            it = p.plan(req, led, cfg)
            rows.append((it.status, it.denial_reason, None if not it.accepted else round(it.cost, 12),
                         _clkey(it), p.last_expansions))
        out[name] = rows
        planners[name] = p
    assert out["off"] == out["on"], "the window changed a plan"
    hits = planners["on"]._ks["win_stats"][W.WS_HIT]
    if expect_window:
        assert hits > 0, "no probe was answered from the window — this test proves nothing"
    else:
        assert hits == 0, "expected the window to be disabled"
    return planners["on"]


# ------------------------------------------------------------------ bit-level truth

def test_window_bit_matches_pool_truth():
    """Every bit of a built window equals ``blocked_py`` on the pools it was built from.

    Uses a real committed schedule so the pools are FRAGMENTED — a cell whose free list is one
    interval would exercise none of the interval merge. The window is deliberately placed over the
    busiest part of the box, and the own-column overlay is stamped on a stripe of cells so the
    ``ov_own_gen`` fold is covered as well as the plain corridor/column paths.
    """
    from freespace_sim.demand import HubRadiusDemand
    from freespace_sim.dss import DSS
    from freespace_sim.mechanism import FCFSMechanism
    from freespace_sim.uss import USS

    cfg = SimConfig(region_size_m=(20000.0, 15000.0), lam_per_hour=3000.0, horizon_s=300.0,
                    planner="astar", seed=0)
    reqs = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 6, "stripmall_uss": 30},
                           radius_m={"walmart_uss": 6000.0, "stripmall_uss": 3000.0},
                           terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                           pads_per_hub=8, return_flights=True).generate(
        cfg, np.random.default_rng(cfg.seed))
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar_ref")
    usses = {u: USS(u, dss, cfg, astar) for u in {r.uss_id for r in reqs}}
    for ev in reqs[:400]:
        usses[ev.uss_id].handle_request(ev)

    cocc = CompiledHexOccupancy(cfg)
    for fid, grp in itertools.groupby(led.iter_committed(), key=lambda fv: fv[0]):
        cocc.on_commit(fid, [v for _, v in grp])
    assert cocc.corr.nslots > cocc.NC, "pools are unfragmented — the interval merge would be untested"

    ov = np.zeros(cocc.NC, np.int32)
    gen = 4
    # A stripe of own-marked cells crossing the window, so both branches of the `own` fold appear.
    for c in range(0, cocc.NC, 7):
        ov[c] = gen
    own_cells = {c for c in range(0, cocc.NC, 7)}

    wbox = W.empty_wbox()
    # Centre the window on the region and keep it small enough to check exhaustively.
    qmid = cocc.qmin + cocc.qspan // 2
    rmid = cocc.rmin + cocc.rspan // 2
    nbytes = W.window_bounds(cocc, wbox, q_cells=(qmid,), r_cells=(rmid,), base=40,
                             max_step=cocc.MAXS, n_gsteps=60, lateral_margin=9, tail_steps=0,
                             max_bytes=1 << 20)
    assert nbytes > 0
    win = np.zeros(nbytes, np.uint8)
    W.build_window(cocc.corr.iv, cocc.col.iv, cocc.static_col, ov, gen,
                   cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox, win)

    q0, q1, r0, r1, s0, s1 = (int(wbox[i]) for i in
                              (W.W_Q0, W.W_Q1, W.W_R0, W.W_R1, W.W_S0, W.W_S1))
    row_bytes = int(wbox[W.W_ROWB])
    wrspan = int(wbox[W.W_RSPAN])
    checked = blocked = 0
    for iq, q in enumerate(range(q0, q1 + 1)):
        for ir, r in enumerate(range(r0, r1 + 1)):
            for L in range(cocc.n_levels):
                wcell = (iq * wrspan + ir) * cocc.n_levels + L
                for k, s in enumerate(range(s0, s1 + 1)):
                    bit = bool(win[wcell * row_bytes + (k >> 3)] & (1 << (k & 7)))
                    ref = cocc.blocked_py(q, r, L, s, own_cells=own_cells)
                    assert bit == ref, f"window != pools at (q={q},r={r},L={L},s={s})"
                    checked += 1
                    blocked += ref
    assert checked > 10_000, "window too small to be a meaningful check"
    assert 0 < blocked < checked, "window is all-free or all-blocked — a constant would pass this"


def test_window_merges_two_interleaved_free_lists():
    """The one place the build relies on the pools' ascending-sort invariant: when a cell has claims
    in BOTH pools, the free steps are the INTERSECTION of two free lists and ``build_window`` walks
    them as a two-pointer merge. The real-schedule truth test rarely produces such a cell (column
    claims sit on hub cells, corridor claims on route cells), so construct one deliberately, with the
    two lists interleaved so neither is a prefix or a superset of the other.

    ``blocked_py`` is the oracle, and it computes the same answer the opposite way — two independent
    ``blocked_at`` walks folded with OR — so agreeing with it is a real check on the merge, not a
    restatement of it."""
    cocc = CompiledHexOccupancy(CFG)
    q, r, L = cocc.qmin + 5, cocc.rmin + 5, 0
    cell = cocc.cell_id(q, r, L)
    assert cell >= 0
    for lo, hi in ((3, 9), (14, 17), (30, 33), (41, 44)):        # corridor blocks
        cocc.corr.block_range(cell, lo, hi)
    for lo, hi in ((6, 12), (20, 24), (31, 40), (50, 55)):       # column blocks, deliberately offset
        cocc.col.block_range(cell, lo, hi)

    wbox = W.empty_wbox()
    nbytes = W.window_bounds(cocc, wbox, q_cells=(q,), r_cells=(r,), base=0, max_step=cocc.MAXS,
                             n_gsteps=70, tail_steps=0, lateral_margin=1, max_bytes=1 << 20)
    assert nbytes > 0
    win = np.zeros(nbytes, np.uint8)
    W.build_window(cocc.corr.iv, cocc.col.iv, cocc.static_col, np.zeros(cocc.NC, np.int32), 1,
                   cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox, win)

    q0, r0, s0, s1 = (int(wbox[i]) for i in (W.W_Q0, W.W_R0, W.W_S0, W.W_S1))
    row_bytes, wrspan = int(wbox[W.W_ROWB]), int(wbox[W.W_RSPAN])
    wcell = ((q - q0) * wrspan + (r - r0)) * cocc.n_levels + L
    got = [bool(win[wcell * row_bytes + (k >> 3)] & (1 << (k & 7))) for k in range(s1 - s0 + 1)]
    want = [cocc.blocked_py(q, r, L, s, own_cells=None) for s in range(s0, s1 + 1)]
    assert got == want, [i for i, (a, b) in enumerate(zip(got, want)) if a != b][:20]
    # A cell whose two lists interleave must produce BOTH answers, or the comparison is vacuous.
    assert any(want) and not all(want)


def test_window_bits_past_the_step_span_are_untouched():
    """A row's padding bits (the tail of the last byte past ``W_STEPS``) stay 0. Nothing reads them,
    but a build that ran off the end of a row would corrupt the NEXT cell's first steps, which is a
    silent wrong answer rather than a crash."""
    cocc = CompiledHexOccupancy(CFG)
    cocc.corr.block_range(0, 0, 10_000)                  # cell 0 fully blocked → row is all 1s
    wbox = W.empty_wbox()
    nbytes = W.window_bounds(cocc, wbox, q_cells=(cocc.qmin + 3,), r_cells=(cocc.rmin + 3,),
                             base=0, max_step=cocc.MAXS, n_gsteps=11, tail_steps=0,
                             lateral_margin=4, max_bytes=1 << 20)
    assert int(wbox[W.W_STEPS]) % 8 != 0, "pick a span that does not land on a byte boundary"
    win = np.full(nbytes, 0xAA, np.uint8)                # poison: a skipped write would show up
    W.build_window(cocc.corr.iv, cocc.col.iv, cocc.static_col, np.zeros(cocc.NC, np.int32), 1,
                   cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels, wbox, win)
    steps, row_bytes = int(wbox[W.W_STEPS]), int(wbox[W.W_ROWB])
    pad = row_bytes * 8 - steps
    for wcell in range(nbytes // row_bytes):
        tail = win[wcell * row_bytes + row_bytes - 1]
        assert int(tail) >> (8 - pad) == 0, f"row {wcell} wrote into its padding bits"


# ------------------------------------------------------------------ end-to-end parity

def test_window_parity_empty_airspace():
    _assert_window_exact([_req()], [])


def test_window_parity_reroute_and_ground_delay():
    wall = Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)
    pad = Volume4D(CylinderSpec(2000, 0, 60, 0, 150), 0.0, 200.0)
    _assert_window_exact([_req()], [(98, [wall]), (99, [pad])])


def test_window_parity_static_terminal_wall_and_own_hub():
    """The two per-cell terms the window folds in at build time: an always-active FOREIGN wall
    (``static_col``) must stay a wall, and the flight's OWN hub must stay transparent."""
    cfg = SimConfig(terminal_airspace_always_active=True)
    own, foreign = Terminal("own_uss#0", 8, 180.0), Terminal("foreign_uss#0", 8, 180.0)
    reqs = [FlightRequest(1, vec(0, 0, 0), vec(5000, 0, 0), 0.0, uss_id="own_uss",
                          origin_terminal=own),
            FlightRequest(2, vec(0, 3000, 0), vec(5000, 3000, 0), 0.0, uss_id="uss_a")]
    _assert_window_exact(reqs, [], cfg=cfg,
                         statics=[((0.0, 0.0), own), ((2500.0, 0.0), foreign)])


def test_window_parity_after_release_and_recommit():
    """The LNS destroy/repair shape, which is where the pools fragment worst: commit a schedule,
    release part of it, replan against the hole. ``block_range`` leaves empty ``lo > hi`` slots and
    splits intervals in place, so this is the case the interval merge has to survive."""
    cfg = SimConfig()
    reqs = [FlightRequest(i, vec(0, 400.0 * i, 0), vec(4000, 400.0 * i, 0), 10.0 * i)
            for i in range(1, 7)]
    seed = AStarPlanner()
    led = ReservationLedger(cfg)
    committed = []
    for req in reqs:
        it = seed.plan(req, led, cfg)
        if it.accepted:
            led.commit(req.flight_id, it.volumes)
            committed.append((req.flight_id, it.volumes))
    assert len(committed) >= 4

    # Both arms: commit everything, release half, then replan the released ones.
    victims = [fid for fid, _ in committed[:2]]
    out = {}
    for name, wb in (("off", 0), ("on", _ON)):
        p = AStarPlanner(window_bytes=wb, incremental_release=True)
        led2 = ReservationLedger(cfg)
        for fid, vols in committed:
            led2.commit(fid, vols)
        p.plan(reqs[-1], led2, cfg)          # one plan first, to bind + absorb before the release
        led2.release_many(victims)
        rows = []
        for fid in victims:
            req = next(r for r in reqs if r.flight_id == fid)
            it = p.plan(req, led2, cfg)
            rows.append((it.status, None if not it.accepted else round(it.cost, 12),
                         _clkey(it), p.last_expansions))
        out[name] = rows
        if name == "on":
            assert p._ks["win_stats"][W.WS_HIT] > 0
    assert out["off"] == out["on"], "the window changed a plan after a release"


def test_window_over_cap_disables_itself_and_stays_exact():
    """A window bigger than ``window_bytes`` is skipped, not truncated: the plan runs the original
    pool walk and the answer is unchanged. This is the escape hatch that makes the bounds a tuning
    knob rather than a correctness argument."""
    p = _assert_window_exact([_req(dx=6000.0)], [], window_bytes=64, expect_window=False)
    assert p._win_off > 0, "expected the cap to reject the window"


def test_window_bytes_zero_is_the_pre_window_path():
    p = AStarPlanner(window_bytes=0)
    p.plan(_req(), ReservationLedger(CFG), CFG)
    st = p._ks["win_stats"]
    assert st[W.WS_HIT] == 0 and st[W.WS_MISS] > 0, "window_bytes=0 must take the pool walk"


def test_window_survives_the_mask_widen_rerun():
    """The FB_MASK widen re-runs the search under a FRESH ``gen`` and a wider ``n_gsteps``. The window
    folds ``ov_own_gen == gen`` and is sized from ``n_gsteps``, so it must be rebuilt inside that
    loop; a hoisted build would leave the re-run reading a window stamped for the previous
    generation. Forced with a long ground-delay allowance, as ``test_compiled_mask_widen_re_run_exact``
    does."""
    import dataclasses as dc
    cfg = dc.replace(CFG, max_ground_delay_s=3600.0)
    blocker = Volume4D(CylinderSpec(0, 0, 120, 0, 200), 0.0, 1500.0)
    p = _assert_window_exact([_req(dx=1500.0)], [(99, [blocker])], cfg=cfg)
    assert p._remask > 0, "the search never reached past the tight mask — this test proves nothing"
