"""Compiled (numba) SIPP kernel: exact equivalence with the pure-Python reference.

The pure-Python ``SIPPPlanner`` (``sipp_ref``) is the oracle — already proven cost-equivalent to A*
(see ``test_sipp.py``). The compiled ``sipp`` must reproduce it **exactly** on non-terminal flights
(the air-cruise kernel; terminal flights fall back to the reference in Phase 1). These tests assert
``compiled == reference`` (cost + accept + centerline), that the kernel actually runs (low fallback),
that the dense interval pool matches ``SafeIntervalIndex``, and that absent numba degrades to the
reference. If numba is unavailable every plan falls back, so equivalence still holds trivially.
"""
import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.dss import DSS
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import _absorb
from freespace_sim.planner.compiled_occupancy import CompiledOccupancy
from freespace_sim.planner.sipp import SafeIntervalIndex
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()
_COMPILED = get_planner("sipp").compiled        # False if numba is unavailable → everything falls back


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


def test_compiled_deterministic():
    a = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    b = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    assert abs(a.cost - b.cost) < 1e-12 and len(a.centerline) == len(b.centerline)


def test_numba_absent_falls_back_to_reference():
    """compiled=False is byte-identical to the reference (the optional-dependency contract)."""
    from freespace_sim.planner.sipp import SIPPPlanner
    off = SIPPPlanner(compiled=False)
    ref = get_planner("sipp_ref")
    for fid, dx in ((1, 2000.0), (2, 3500.0)):
        a = off.plan(_req(fid, dx), ReservationLedger(CFG), CFG)
        b = ref.plan(_req(fid, dx), ReservationLedger(CFG), CFG)
        assert abs(a.cost - b.cost) < 1e-12 and a.accepted == b.accepted


# ---- dense interval pool == SafeIntervalIndex oracle ----

def test_compiled_occupancy_matches_safe_interval_index():
    spec = with_overrides(get_scenario("metro_uniform"), lam_per_hour=400.0, horizon_s=600.0, seed=0)
    cfg = spec.config()
    reqs = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed)) \
        if spec.demand_model() else UniformPoissonDemand().generate(cfg, np.random.default_rng(0))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    from freespace_sim.uss import USS
    usses = {u: USS(u, dss, cfg, get_planner("astar")) for u in sc.uss_ids}
    for ev in sc.events:
        usses[ev.request.uss_id].handle_request(ev.request)

    sidx = SafeIntervalIndex(cfg); _absorb(sidx, led)
    cocc = CompiledOccupancy(cfg); _absorb(cocc, led)
    own = frozenset()
    checked = 0
    for (q, r) in list(sidx.corr.keys())[:1500]:       # every committed (non-terminal) cell
        ref = sidx.free_intervals(q, r, own, 0, cocc.MAXS, False)
        got = cocc.free_intervals_py(q, r, 0, cocc.MAXS)
        assert got is not None and ref == got, f"interval mismatch at ({q},{r}): {ref} vs {got}"
        checked += 1
    assert checked > 50


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
    return rows, getattr(sipp, "_fb", 0)


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
        assert getattr(sipp, "_fb", 0) < 0.10 * len(rows), "kernel fell back too often"


# ---- saturation regression: kernel must respect the own-lane overlay intervals ----

@pytest.mark.slow
def test_compiled_terminal_path_never_routes_through_blocked():
    """Regression for the overlay-chain-walk OOB (``_search`` skip-ahead used the global pool's
    ``iv_nxt`` for OVERLAY slots ``sj >= cap``, reading out of bounds and fabricating "free" space
    across blocked steps). At saturation the own-lane overlays fragment, so the bug made the kernel
    hover/route a flight THROUGH a foreign corridor occupying its own landing lane. Guard the exact
    invariant the bug broke: no accepted compiled terminal path visits an ``is_blocked`` cell-step.

    A strict ``compiled == reference`` assertion can't be used here — at this density the two settle
    on different (occupancy-VALID) optima via the bounded fixed-lanes tie — so we assert occupancy
    validity directly, which is robust to that tie and is precisely what the OOB violated.
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
        fb0 = sipp._fb
        c = sipp.plan(rq, led, cfg)
        if not c.accepted or sipp._fb != fb0:         # skip denied / fell-back-to-reference plans
            continue
        own, svc = sipp._own, sipp._svc
        for (q, r, s) in sipp._air:                    # the per-step compiled search path
            if svc.is_blocked(q, r, s, own):
                violations.append((rq.flight_id, q, r, s))
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
