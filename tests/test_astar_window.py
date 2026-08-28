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

import subprocess
import sys
import textwrap

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar import window as W
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

    monkeypatch.setattr(W, "build_window_claims", fail_window)
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

    monkeypatch.setattr(W, "build_window_claims", fail_window)
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
    """Plan every request on the COMPILED path and on the pure-Python reference, against identically
    built ledgers, and assert the two are indistinguishable: same accept/deny, same cost, same
    centerline, same node-expansion count.

    This used to compare window-on against window-off. That comparison no longer exists — the
    interval pools are gone, so a disabled window means the kernel cannot answer a probe at all and
    ``window_bytes=0`` raises. Comparing against the reference search is what replaces it, and it is
    strictly stronger: the reference shares no occupancy structure with the window, where the old
    off-arm shared the pools the window was built from.

    Returns the compiled planner so a caller can assert on its counters."""
    ref, com = AStarPlanner(compiled=False), AStarPlanner(window_bytes=window_bytes)
    assert com.compiled, "numba kernel inactive — this parity guard would compare reference to itself"
    out = {}
    for name, p in (("ref", ref), ("com", com)):
        led = ReservationLedger(cfg)
        for c, t in statics:
            led.register_static_terminal(c, t)
        for fid, vols in commits:
            led.commit(fid, vols)
        rows = []
        for req in reqs:
            it = p.plan(req, led, cfg)
            rows.append((it.status, it.denial_reason, None if not it.accepted else round(it.cost, 9),
                         _clkey(it), p.last_expansions))
        out[name] = rows
    assert out["ref"] == out["com"], "the compiled window path diverged from the reference search"
    hits = com._ks["win_stats"][W.WS_HIT] if com._ks is not None else 0
    if expect_window:
        assert hits > 0, "no probe was answered from the window — this test proves nothing"
    return com


# ------------------------------------------------------------------ bit-level truth

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
    """The LNS destroy/repair shape, which is where the claim slabs get their swap-removed holes:
    commit a schedule, release part of it, replan against the hole. The compiled window path must
    still match the reference search exactly."""
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

    victims = [fid for fid, _ in committed[:2]]
    out = {}
    for name, planner in (("ref", AStarPlanner(compiled=False, incremental_release=True)),
                          ("com", AStarPlanner(incremental_release=True))):
        led2 = ReservationLedger(cfg)
        for fid, vols in committed:
            led2.commit(fid, vols)
        planner.plan(reqs[-1], led2, cfg)        # one plan first, to bind + absorb before the release
        led2.release_many(victims)
        rows = []
        for fid in victims:
            req = next(r for r in reqs if r.flight_id == fid)
            it = planner.plan(req, led2, cfg)
            rows.append((it.status, None if not it.accepted else round(it.cost, 9),
                         _clkey(it), planner.last_expansions))
        out[name] = rows
        if name == "com":
            assert planner._ks["win_stats"][W.WS_HIT] > 0
    assert out["ref"] == out["com"], "compiled diverged from the reference after a release"


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
