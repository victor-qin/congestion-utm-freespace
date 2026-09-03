"""Compiled (numba) SIPP kernel: exact equivalence with the pure-Python reference.

The pure-Python ``SIPPPlanner`` (``sipp_ref``) is the oracle — already proven cost-equivalent to A*
(see ``test_sipp.py``). The compiled ``sipp`` must reproduce it **exactly**, including fixed-terminal
lane choice. These tests assert
``compiled == reference`` (cost + accept + centerline), that the kernel actually runs (low fallback),
and that absent numba degrades to the reference. The pool-vs-index parity test that used to live here
went with the global interval pool (`context/sipp_runtime_plan.md` Phase 3); its successor is
``tests/test_sipp_window.py``, which compares the per-plan build against ``blocked_py``. If numba is unavailable every plan falls back, so equivalence still holds trivially.
"""
import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.dss import DSS
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar.compiled_hex_occupancy import (
    CompiledHexOccupancy,
    schedulable_horizon_steps,
)
from freespace_sim.planner.sipp import SIPPPlanner
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run
from freespace_sim.types import DenialReason, FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()
_COMPILED = get_planner("sipp").sipp_compiled   # False if numba is unavailable → SIPP uses its reference


def _req(fid=1, dx=2000.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(dx, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)


def _plan_cc(req, committed=()):
    """Plan with astar, the pure-Python reference, and the compiled kernel on identical ledgers."""
    out = {}
    for name in ("astar", "sipp_ref", "sipp"):
        led = ReservationLedger(CFG)
        for fid, vols in committed:
            led.commit(fid, vols)
        out[name] = get_planner(name).plan(req, led, CFG)
    return out


# ---- isolated, exact ----

def test_compiled_empty_matches_reference():
    o = _plan_cc(_req())
    assert o["sipp"].accepted and o["sipp_ref"].accepted
    assert not ReservationLedger(CFG).any_conflict(o["sipp"].volumes)   # self-consistent
    assert abs(o["sipp"].cost - o["sipp_ref"].cost) < 1e-9              # exact vs reference
    assert abs(o["sipp"].cost - o["astar"].cost) < 1e-6                 # ⇒ exact vs A*


def test_compiled_reroutes_around_wall_exactly():
    o = _plan_cc(_req(2), committed=[(99, [_wall()])])
    assert o["sipp"].accepted and o["sipp_ref"].accepted
    assert o["sipp"].air_detour_m > 0
    assert abs(o["sipp"].cost - o["sipp_ref"].cost) < 1e-9
    assert len(o["sipp"].centerline) == len(o["sipp_ref"].centerline)


@pytest.mark.skipif(not _COMPILED, reason="requires the compiled SIPP kernel")
def test_compiled_long_route_is_not_truncated_by_occupancy_horizon():
    """A short demand horizon is not a flight-duration cap: the pool must cover the route tail."""
    cfg = SimConfig(
        region_size_m=(30_000.0, 20_000.0),
        horizon_s=100.0,
        max_ground_delay_s=0.0,
        flight_levels_m=(30.0,),
    )
    req = FlightRequest(1, vec(7_000, 10_000, 0), vec(21_400, 10_000, 0), 0.0)
    compiled = get_planner("sipp")
    reference = get_planner("sipp_ref")
    got = compiled.plan(req, ReservationLedger(cfg), cfg)
    want = reference.plan(req, ReservationLedger(cfg), cfg)

    assert CompiledHexOccupancy(cfg).MAXS == schedulable_horizon_steps(cfg)
    assert got.accepted and want.accepted
    assert got.cost == pytest.approx(want.cost, abs=1e-9)
    assert len(got.centerline) == len(want.centerline)
    assert compiled._sfb == 0


@pytest.mark.skipif(not _COMPILED, reason="requires the compiled SIPP kernel")
def test_compiled_late_departure_past_pool_horizon_uses_reference():
    cfg = SimConfig(region_size_m=(1_000.0, 1_000.0), horizon_s=100.0)
    req = FlightRequest(
        1,
        vec(200, 500, 0),
        vec(800, 500, 0),
        0.0,
        t_departure=(CompiledHexOccupancy(cfg).MAXS + 20) * cfg.dt_s,
    )
    compiled = get_planner("sipp")
    got = compiled.plan(req, ReservationLedger(cfg), cfg)
    want = get_planner("sipp_ref").plan(req, ReservationLedger(cfg), cfg)
    assert got.accepted and want.accepted
    assert got.cost == pytest.approx(want.cost, abs=1e-9)
    assert len(got.centerline) == len(want.centerline)
    assert all(np.allclose(gp, wp) and gt == pytest.approx(wt)
               for (gp, gt), (wp, wt) in zip(got.centerline, want.centerline))


def test_compiled_deterministic():
    a = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    b = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    assert abs(a.cost - b.cost) < 1e-12 and len(a.centerline) == len(b.centerline)


def test_numba_absent_falls_back_to_reference():
    """compiled=False is byte-identical to the reference (the optional-dependency contract)."""
    off = SIPPPlanner(compiled=False)
    assert off.sipp_compiled is False and off.compiled is False
    ref = get_planner("sipp_ref")
    for fid, dx in ((1, 2000.0), (2, 3500.0)):
        a = off.plan(_req(fid, dx), ReservationLedger(CFG), CFG)
        b = ref.plan(_req(fid, dx), ReservationLedger(CFG), CFG)
        assert abs(a.cost - b.cost) < 1e-12 and a.accepted == b.accepted


def test_astar_warm_failure_keeps_sipp_kernel_fallback_on_astar_reference(monkeypatch):
    """The SIPP flag must not erase an A* warm-up failure before the FB_* safety path runs."""
    def fail_warm(planner):
        planner.compiled = False
        planner._kernel = None

    monkeypatch.setattr(AStarPlanner, "_warm_jit", fail_warm)
    planner = SIPPPlanner(compiled=True)
    assert planner.compiled is False and planner._kernel is None
    assert planner.sipp_compiled is (planner._skernel is not None)

    reference = type("FallbackIntent", (), {"planner": "astar"})()
    monkeypatch.setattr(planner, "_plan_reference", lambda *_: reference)

    def compiled_must_not_run(*_):
        pytest.fail("SIPP fallback selected the unavailable compiled A* kernel")

    monkeypatch.setattr(planner, "_plan_compiled", compiled_must_not_run)
    got = planner._fallback(_req(), ReservationLedger(CFG), CFG)
    assert got is reference and got.planner == "sipp"


def test_sipp_warm_compiles_the_window_builder_too():
    """`build_window_intervals` owns a numba cache separate from `_search`, so it needs its own warm
    call — measured ~0.92 s cold and ~132 ms off a warm on-disk cache, against ~5 us hot.

    Without it every spawned DROP worker meets an uncompiled builder on its FIRST repair,
    simultaneously: the compile stampede `_swarm_jit` exists to prevent, and one that would be billed
    to SIPP in any parallel measurement. Asserting the CALL rather than a timing keeps this a gate
    rather than a flake.
    """
    from freespace_sim.planner import sipp as sipp_mod
    from freespace_sim.planner import sipp_window

    calls = []
    real = sipp_window.build_window_intervals
    try:
        sipp_mod.SW.build_window_intervals = lambda *a, **k: calls.append(a)
        planner = SIPPPlanner(compiled=False)
        planner._skernel = lambda *args: None
        planner._swarm_jit()
    finally:
        sipp_mod.SW.build_window_intervals = real
    assert len(calls) == 1, "the warm-up never compiled the window builder"
    iv_lo, iv_hi, iv_nxt, scratch = calls[0][11:15]
    assert (iv_lo.dtype, iv_hi.dtype, iv_nxt.dtype) == (np.dtype(np.int32),) * 3
    assert scratch.dtype == np.dtype(np.int64), "warmed a scratch dtype no real plan uses"


def test_sipp_warm_uses_the_production_kernel_signature():
    """The warm-up must compile the signature a real plan uses — and this is the ONLY thing standing
    between a signature change and a silent all-reference run.

    `_swarm_jit` calls `_search` POSITIONALLY. Change the kernel's parameter list and the call raises
    `TypeError`, which its own handler would otherwise swallow into `sipp_compiled = False`: every
    plan then routes to the pure-Python reference, the run stays exact, every parity gate passes, and
    the only symptom is that it got much slower. `_swarm_jit` re-raises `TypeError` for that reason;
    this test pins the other half, that warm and production agree on ARITY and DTYPE.
    """
    planner = SIPPPlanner(compiled=False)
    calls = []
    planner._skernel = lambda *args: calls.append(args)
    planner._swarm_jit()
    assert len(calls) == 1
    warm = calls[0]
    warm_dtypes = tuple(a.dtype for a in warm[:3])          # the window pool: iv_lo / iv_hi / iv_nxt

    planner._k_lab_cell = np.empty(0, np.int64)   # skip the unrelated fixed 1<<21 label/hash tables
    planner._skernel_state(CompiledHexOccupancy(CFG), 9)
    production_dtypes = tuple(
        a.dtype for a in (planner._k_iv_lo, planner._k_iv_hi, planner._k_iv_nxt))
    assert warm_dtypes == production_dtypes == (np.dtype(np.int32),) * 3

    # Arity, against the live kernel rather than a hand-maintained number.
    from freespace_sim.planner import sipp_kernel

    n_params = sipp_kernel._search.py_func.__code__.co_argcount
    assert len(warm) == n_params, f"warm-up passes {len(warm)} args, `_search` takes {n_params}"


@pytest.mark.parametrize("planner_name", ("astar_ref", "astar", "sipp_ref", "sipp"))
def test_ground_delay_budget_is_binding_for_every_lattice_path(planner_name):
    """The route horizon bounds arrival; it must not extend the legal ground-delay domain."""
    cfg = SimConfig(
        region_size_m=(1_000.0, 1_000.0),
        max_ground_delay_s=8.0,
        flight_levels_m=(30.0,),
    )
    req = FlightRequest(1, vec(300, 500, 0), vec(700, 500, 0), 0.0)
    ledger = ReservationLedger(cfg)
    ledger.commit(99, [Volume4D(CylinderSpec(300, 500, 100, 0, 150), 0.0, 12.0)])

    intent = get_planner(planner_name).plan(req, ledger, cfg)
    assert not intent.accepted
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED


@pytest.mark.parametrize("planner_name", ("astar_ref", "astar", "sipp_ref", "sipp"))
def test_non_integral_ground_delay_cap_rounds_down_for_every_lattice_path(planner_name):
    """A discrete delay may not round the configured seconds cap upward to the next timestep."""
    cfg = SimConfig(
        region_size_m=(1_000.0, 1_000.0),
        dt_s=4.0,
        time_buffer_s=0.0,
        max_ground_delay_s=5.0,
        flight_levels_m=(30.0,),
    )
    req = FlightRequest(1, vec(300, 500, 0), vec(700, 500, 0), 0.0)
    ledger = ReservationLedger(cfg)
    # The pad is blocked at departure steps 0 and 1, then free at step 2. Rounding 5/4 upward therefore
    # accepts with an illegal 8 s delay; rounding down allows only steps 0..1 and correctly denies.
    ledger.commit(99, [Volume4D(CylinderSpec(300, 500, 100, 0, 150), 0.0, 0.0)])

    intent = get_planner(planner_name).plan(req, ledger, cfg)
    assert not intent.accepted
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED


@pytest.mark.parametrize("planner_name", ("sipp_ref", "sipp"))
def test_sipp_bounded_infeasibility_is_budget_exceeded(planner_name):
    cfg = SimConfig(
        region_size_m=(1_000.0, 1_000.0),
        horizon_s=100.0,
        max_ground_delay_s=0.0,
        flight_levels_m=(30.0,),
    )
    req = FlightRequest(1, vec(300, 500, 0), vec(700, 500, 0), 0.0)
    ledger = ReservationLedger(cfg)
    ledger.commit(99, [Volume4D(CylinderSpec(700, 500, 100, 0, 150), 0.0, 1e5)])

    intent = get_planner(planner_name).plan(req, ledger, cfg)
    assert not intent.accepted
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED


@pytest.mark.skipif(not _COMPILED, reason="requires the compiled SIPP kernel")
def test_compiled_kernel_no_path_is_budget_exceeded(monkeypatch):
    from freespace_sim.planner.sipp_kernel import NO_PATH

    planner = SIPPPlanner(compiled=True)
    monkeypatch.setattr(planner, "_skernel", lambda *_: (-1, 0.0, 17, NO_PATH))
    intent = planner.plan(_req(), ReservationLedger(CFG), CFG)

    assert not intent.accepted
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED
    assert planner.last_expansions == planner._n_expansions == 17
    assert planner._air == []


@pytest.mark.skipif(not _COMPILED, reason="requires the compiled SIPP kernel")
def test_compiled_diagnostics_use_sipp_fallback_counter_and_clear_old_path(monkeypatch):
    from freespace_sim.planner.sipp_kernel import FB_CAP

    planner = SIPPPlanner(compiled=True)
    first = planner.plan(_req(), ReservationLedger(CFG), CFG)
    assert first.accepted
    assert planner._air
    assert planner.last_expansions == planner._n_expansions > 0

    monkeypatch.setattr(planner, "_skernel", lambda *_: (-1, 0.0, 23, FB_CAP))
    second = planner.plan(_req(2), ReservationLedger(CFG), CFG)

    assert second.accepted and second.planner == "sipp"
    assert planner._sfb == 1 and planner._sfb_cap == 1
    assert planner._fb == 0                    # inherited counter is only for a secondary A* kernel fallback
    assert planner._n_expansions == 23         # the failed SIPP attempt remains available for diagnosis
    assert planner.last_expansions > 0         # final A* fallback search owns the public per-plan telemetry
    assert planner._air == []                  # never expose the first flight's compiled path as the second's


def test_sipp_reference_compute_truncation_stays_search_exhausted():
    intent = SIPPPlanner(max_expansions=0, compiled=False).plan(
        _req(), ReservationLedger(CFG), CFG
    )
    assert not intent.accepted
    assert intent.denial_reason is DenialReason.SEARCH_EXHAUSTED


# ---- dense interval pool == SafeIntervalIndex oracle ----

def test_shared_sipp_occupancy_preserves_nonzero_ledger_epoch():
    """A worker must reuse the master's frozen caches instead of rebuilding them after a handoff."""
    ledger = ReservationLedger(CFG)
    ledger.detach_subscribers()                  # make the regression observable: epoch is now non-zero
    req = _req()
    master = SIPPPlanner(compiled=False)
    worker = SIPPPlanner(compiled=False)
    svc = master._occupancy(req, ledger, CFG)
    sidx = master._sipp_index(req, ledger, CFG)
    cocc = master._compiled_occ(req, ledger, CFG)

    worker.share_occupancy_from(master)

    assert worker._svc_epoch == worker._sidx_epoch == worker._cocc_epoch == ledger.epoch
    assert worker._occupancy(req, ledger, CFG) is svc
    assert worker._sipp_index(req, ledger, CFG) is sidx
    assert worker._compiled_occ(req, ledger, CFG) is cocc


# ---- replay equivalence (headline): compiled vs reference against the SAME A*-committed ledger ----

def _replay_cc(scenario, lam, H, seed, region=None):
    if scenario == "uniform":
        cfg = SimConfig(region_size_m=(region, region), lam_per_hour=lam, horizon_s=H, seed=seed)
        demand = UniformPoissonDemand()
    else:
        spec = with_overrides(get_scenario(scenario), lam_per_hour=lam, horizon_s=H, seed=seed)
        cfg = spec.config()
        demand = spec.demand_model() or UniformPoissonDemand()
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    from freespace_sim.uss import USS
    astar = get_planner("astar")
    sipp, sref = get_planner("sipp"), get_planner("sipp_ref")
    usses = {u: USS(u, dss, cfg, astar) for u in sc.uss_ids}
    rows = []
    for ev in sc.events:
        rq = ev.request
        c = sipp.plan(rq, led, cfg)
        r = sref.plan(rq, led, cfg)
        usses[rq.uss_id].handle_request(rq)
        rows.append((c.accepted, r.accepted, c.cost, r.cost,
                     len(c.centerline) if c.centerline is not None else -1,
                     len(r.centerline) if r.centerline is not None else -1))
    return rows, sipp._sfb


@pytest.mark.parametrize("lam", [120.0, 400.0])
def test_compiled_replay_exact_metro(lam):
    rows, fb = _replay_cc("metro_uniform", lam, 600.0, 0)
    assert rows
    assert all(ca == ra for ca, ra, _, _, _, _ in rows), "accept-set mismatch"
    assert all(abs(cc - rc) < 1e-9 for ca, _, cc, rc, _, _ in rows if ca), "cost mismatch vs reference"
    assert all(lc == lr for ca, _, _, _, lc, lr in rows if ca), "centerline length mismatch"
    if _COMPILED:
        assert fb < 0.10 * len(rows), f"kernel fell back too often ({fb}/{len(rows)})"


def _short_reqs(W, n, rmin, rmax, horizon, seed):
    """Dallas-shaped demand: short hub→customer flights (``rmin``..``rmax`` m) in a big ``W`` box."""
    from freespace_sim.types import vec
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        o = rng.uniform([0, 0], [W, W])
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(rmin, rmax)
        d = np.clip(o + rad * np.array([np.cos(ang), np.sin(ang)]), 0, W)
        out.append(FlightRequest(i, vec(o[0], o[1], 0), vec(d[0], d[1], 0), float(rng.uniform(0, horizon))))
    return sorted(out, key=lambda r: (r.t_request, r.flight_id))


@pytest.mark.slow
def test_compiled_replay_exact_big_dense_short_flights():
    """The Dallas regime: short (4-8 km) flights in a big DENSE 24 km box. Region size only sizes the
    kernel box; the search depth is per-flight, so this is the winning regime — and exact."""
    from freespace_sim.uss import USS
    W = 24000.0
    cfg = SimConfig(region_size_m=(W, W), lam_per_hour=600.0, horizon_s=1800.0, seed=0)
    reqs = _short_reqs(W, 700, 4000.0, 8000.0, 1800.0, 0)
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar"); sipp, sref = get_planner("sipp"), get_planner("sipp_ref")
    usses = {u: USS(u, dss, cfg, astar) for u in sc.uss_ids}
    rows = []
    for ev in sc.events:
        rq = ev.request
        c = sipp.plan(rq, led, cfg)
        r = sref.plan(rq, led, cfg)
        usses[rq.uss_id].handle_request(rq)
        rows.append((c.accepted, r.accepted, c.cost, r.cost))
    assert rows
    assert all(ca == ra for ca, ra, _, _ in rows), "accept-set mismatch"
    assert all(abs(cc - rc) < 1e-9 for ca, _, cc, rc in rows if ca), "cost mismatch vs reference"
    if _COMPILED:
        assert sipp._sfb < 0.10 * len(rows), "kernel fell back too often"


# ---- the ground-state fold's exit-lane path: compiled == reference on terminal flights ----


def test_compiled_replay_dallas_terminal_accept_set():
    """Terminal (exit-lane) flights: the compiled kernel and the pure-Python reference must agree on
    WHO FLIES and must not lean on the A* fallback. Cost equality is asserted separately."""
    rows, fb = _replay_cc("dallas_hub_2uss_large", 150.0, 1200.0, 0)
    assert rows
    assert all(ca == ra for ca, ra, _, _, _, _ in rows), "accept-set mismatch vs reference"
    assert sum(1 for r in rows if r[0]) > 20, "too few accepted flights to exercise the exit-lane fold"
    if _COMPILED:
        assert fb < 0.15 * len(rows), f"kernel fell back too often ({fb}/{len(rows)})"


@pytest.mark.slow
def test_compiled_replay_exact_dallas_terminal():
    """Exact destination-lane scoring keeps compiled and reference terminal costs identical."""
    rows, _fb = _replay_cc("dallas_hub_2uss_large", 150.0, 1200.0, 0)
    assert rows
    assert all(abs(cc - rc) < 1e-9 for ca, _, cc, rc, _, _ in rows if ca), "cost mismatch vs reference"


# ---- saturation regression: kernel must respect the own-lane overlay intervals ----

@pytest.mark.slow
def test_compiled_terminal_path_never_routes_through_blocked():
    """Regression for the overlay-chain-walk OOB (``_search`` skip-ahead used the global pool's
    ``iv_nxt`` for OVERLAY slots ``sj >= cap``, reading out of bounds and fabricating "free" space
    across blocked steps). At saturation the own-lane overlays fragment, so the bug made the kernel
    hover/route a flight THROUGH a foreign corridor occupying its own landing lane. Guard the exact
    invariant the bug broke: no accepted compiled terminal path visits an ``is_blocked`` cell-step.

    This test asserts occupancy validity directly because that is precisely the invariant the overlay
    OOB violated; exact terminal cost equivalence is covered by the Dallas replay above.
    """
    if not _COMPILED:
        pytest.skip("numba unavailable; every plan falls back to the reference")
    from freespace_sim.uss import USS
    spec = with_overrides(
        get_scenario("dallas_hub_2uss_large"), lam_per_hour=12000.0, horizon_s=1200.0, seed=0,
        demand_overrides={"pads_per_hub": {"walmart_uss": 40, "stripmall_uss": 16},
                          "terminal_radius_m": {"walmart_uss": 135.0, "stripmall_uss": 90.0},
                          "radius_m": 6000.0})
    cfg = spec.config()
    reqs = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    sipp = get_planner("sipp")
    usses = {u: USS(u, dss, cfg, sipp) for u in sc.uss_ids}
    WARM = 1200
    for ev in sc.events[:WARM]:                       # warm to saturation → own-lane overlays fragment
        usses[ev.request.uss_id].handle_request(ev.request)
    violations, checked = [], 0
    for ev in sc.events[WARM:WARM + 200]:
        rq = ev.request
        fb0 = sipp._sfb
        c = sipp.plan(rq, led, cfg)
        if not c.accepted or sipp._sfb != fb0:        # skip denied / fell-back-to-A* plans
            continue
        own, svc = sipp._own, sipp._svc
        # The compiled path builds `_svc` with `maintain_blocked=False` (the map's only
        # reader is the pure-Python reference), so this oracle has to arm it. Explicitly,
        # and NOT relying on a stray fallback in the warm loop having armed it stickily:
        # `enable_blocked` is idempotent, and without this the test either raises deep
        # inside `is_blocked` or passes for a reason it does not state. Outside the loop
        # would be wrong too — the map must be current for THIS commit.
        svc.enable_blocked(led)
        for (q, r, L, s) in sipp._air:                 # the per-step compiled search path (per flight level)
            if svc.is_blocked(q, r, L, s, own):
                violations.append((rq.flight_id, q, r, L, s))
        checked += 1
    assert checked > 20, f"too few terminal plans exercised the kernel ({checked})"
    assert not violations, f"compiled path routed through {len(violations)} blocked cell-steps: {violations[:5]}"


# ---- end-to-end ASTM conflict-freeness ----

def test_compiled_full_run_verified_and_matches_reference():
    cfg = dict(region_size_m=(8000.0, 8000.0), lam_per_hour=400.0, horizon_s=600.0, seed=1)
    rc = run(SimConfig(planner="sipp", **cfg))
    rr = run(SimConfig(planner="sipp_ref", **cfg))
    assert rc.verified and rr.verified
    assert rc.summary()["n_accepted"] == rr.summary()["n_accepted"]


# ---- issue #114: the range-blocked commit path ----


def test_commit_hook_shares_one_geometry_sweep_across_every_subscriber(monkeypatch):
    """A flight longer than the base LRU still gets one geometry sweep shared by all subscribers.

    Two structures now, not three: SIPP's global interval pool is gone. They still share the sweep
    because `CompiledHexOccupancy` and `HexOccupancyService` derive identical `infl_blocked` /
    `infl_pad` radii, which is what `rasterize_ranges`' memo keys on."""
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.planner.astar.occupancy import HexOccupancyService

    cfg = SimConfig(region_size_m=(2000.0, 2000.0), horizon_s=300.0, seed=0)
    svc, cocc = HexOccupancyService(cfg), CompiledHexOccupancy(cfg)
    nvol = hg._RANGE_CACHE_MIN_CAP + 1
    shape = CylinderSpec(500.0, 500.0, 30.0, 0.0, cfg.airspace_ceiling_m)
    volumes = [Volume4D(shape, 8.0, 12.0, terminal_id="hub") for _ in range(nvol)]

    calls = 0
    # `_sweep_kept` is "the single place the compiled/reference choice is made", so counting here is
    # backend-independent. It replaced `_candidate_slack` as the sweep entry point in #115, which
    # silently zeroed this counter (0/1025) rather than changing the answer — the assert below is what
    # catches that class of drift, so keep it counting the funnel, not either backend's leaf.
    real = hg._sweep_kept

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hg, "_sweep_kept", counting)
    monkeypatch.setattr(hg, "_RANGE_CACHE_CAP", hg._RANGE_CACHE_MIN_CAP)
    hg._RANGE_CACHE.clear()
    try:
        ledger = ReservationLedger(cfg)
        for structure in (svc, cocc):
            ledger.subscribe(structure.on_commit)
        ledger.commit(1, volumes)
        assert calls == nvol, f"expected one geometry sweep per volume, got {calls}/{nvol}"
        assert hg._RANGE_CACHE_CAP >= nvol
    finally:
        hg._RANGE_CACHE.clear()
