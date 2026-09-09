"""Paired round trips must be flyable: a return cannot depart before its own aircraft has landed.

This is a PRECEDENCE property, not a separation one, and the distinction is the whole point of the
check. The two legs are independent flights with independently timed reservations, so a return that
lifts off early holds a *disjoint* window at the same pad — no 4D overlap, ledger accepts it, and
`find_interflight_conflict` reports the schedule clean. Only an explicit check sees it.
"""
from __future__ import annotations

import numpy as np
import pytest

from freespace_sim import verify
from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _leg(fid, origin, dest, t_takeoff, t_land, *, paired=None, dwell=40.0, cost=0.0):
    """An accepted intent holding a takeoff column at `origin` and a landing column at `dest`."""
    req = FlightRequest(fid, origin, dest, 0.0, t_departure=t_takeoff, paired_outbound_id=paired)
    vols = [
        Volume4D(CylinderSpec(float(origin[0]), float(origin[1]), 60.0, 0.0, 125.0),
                 t_takeoff, t_takeoff + dwell),
        Volume4D(CylinderSpec(float(dest[0]), float(dest[1]), 60.0, 0.0, 125.0),
                 t_land, t_land + dwell),
    ]
    # The corridor starts at the column EDGE, `dwell` after liftoff — the gap `realized_takeoff_s`
    # exists to measure. A fixture that started it at `t_takeoff` would make the two definitions
    # identical and every test of the difference vacuous.
    return OperationalIntent(request=req, status=IntentStatus.ACCEPTED, volumes=vols,
                             centerline=[(np.asarray(origin, float), t_takeoff + dwell),
                                         (np.asarray(dest, float), t_land)], cost=cost)


HUB, CUST = vec(0, 0, 0), vec(3000, 0, 0)


def test_a_return_that_waits_for_its_aircraft_is_clean():
    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                 # lands 500, pad clears 540
    ret = _leg(2, CUST, HUB, 540.0, 1040.0, paired=1)         # departs exactly when it clears
    assert verify.find_paired_precedence_violation([outbound, ret], CFG) is None
    assert verify.count_paired_precedence_violations([outbound, ret], CFG) == (0, 0.0)


def test_a_return_that_departs_before_its_aircraft_lands_is_caught():
    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                 # pad clears at 540
    ret = _leg(2, CUST, HUB, 300.0, 800.0, paired=1)          # departs 240s early
    bad = verify.find_paired_precedence_violation([outbound, ret], CFG)
    assert bad is not None
    ret_id, out_id, short = bad
    assert (ret_id, out_id) == (2, 1)
    assert short == pytest.approx(240.0)
    assert verify.count_paired_precedence_violations([outbound, ret], CFG)[0] == 1
    with pytest.raises(AssertionError, match="before outbound 1 releases"):
        verify.assert_no_paired_precedence_violation([outbound, ret], CFG)


def test_this_is_invisible_to_the_separation_check():
    """The reason the check has to exist: the early return is not a conflict."""
    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                 # customer column 500-540
    ret = _leg(2, CUST, HUB, 300.0, 800.0, paired=1)          # customer column 300-340: DISJOINT
    assert verify.find_interflight_conflict([outbound, ret], CFG) is None
    assert verify.find_paired_precedence_violation([outbound, ret], CFG) is not None


def test_turnaround_is_part_of_availability():
    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                 # pad clears at 540
    ret = _leg(2, CUST, HUB, 560.0, 1060.0, paired=1)
    assert verify.find_paired_precedence_violation([outbound, ret], CFG, turnaround_s=0.0) is None
    bad = verify.find_paired_precedence_violation([outbound, ret], CFG, turnaround_s=60.0)
    assert bad is not None and bad[2] == pytest.approx(40.0)


def test_unpaired_and_denied_legs_are_ignored():
    solo = _leg(1, HUB, CUST, 0.0, 500.0)                     # no paired_outbound_id
    assert verify.find_paired_precedence_violation([solo], CFG) is None
    orphan = _leg(2, CUST, HUB, 300.0, 800.0, paired=99)      # outbound absent from the list
    assert verify.find_paired_precedence_violation([orphan], CFG) is None
    denied = OperationalIntent(request=FlightRequest(1, HUB, CUST, 0.0, t_departure=0.0),
                               status=IntentStatus.REJECTED, volumes=[], centerline=[])
    ret = _leg(2, CUST, HUB, 300.0, 800.0, paired=1)
    assert verify.find_paired_precedence_violation([denied, ret], CFG) is None


def test_realized_takeoff_is_the_column_start_not_the_first_waypoint():
    """Under fixed exit lanes the corridor begins at the column EDGE, so the first centerline point
    follows liftoff by the climb dwell — measuring there would understate the hold."""
    leg = _leg(1, HUB, CUST, 100.0, 500.0, dwell=40.0)
    assert leg.centerline[0][1] == pytest.approx(140.0)          # the corridor starts 40 s later
    assert verify.realized_takeoff_s(leg) == pytest.approx(100.0)
    assert verify.realized_takeoff_s(leg) < leg.centerline[0][1] - 1e-9


# --------------------------------------------------------------- the LNS anchor guard

def _state_with_pair(monkeypatch, *, turnaround_s=0.0):
    """An LNSState over one round trip, with the unimpeded ruler stubbed out (it would plan)."""
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import state as state_mod
    from freespace_sim.planner.lns.state import LNSState

    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                 # pad clears at 540
    ret = _leg(2, CUST, HUB, 540.0, 1040.0, paired=1, cost=9.0)   # departs exactly when it clears
    led = ReservationLedger(CFG)
    for it in (outbound, ret):
        led.commit(it.request.flight_id, it.volumes)
    monkeypatch.setattr(state_mod, "unimpeded_costs",
                        lambda cfg, st, reqs, *, n_workers: [(r.flight_id, 1.0, None) for r in reqs])
    st = LNSState(CFG, led, [outbound, ret], turnaround_s=turnaround_s)
    return st, outbound, ret


class _FixedPlanner:
    """A repair planner that always returns one prepared intent, so `try_repair` is exercised for
    real (guard included) instead of a test re-implementing the guard's arithmetic."""

    def __init__(self, out):
        self._out = out

    def plan(self, request, ledger, cfg):
        return self._out


def test_a_repaired_return_that_departs_early_is_rejected(monkeypatch):
    """End-to-end through `try_repair`: a return re-planned to lift off before its outbound's pad
    clears is a strict cost improvement, the ledger accepts it, and `find_interflight_conflict`
    sees nothing — only the guard can refuse it. Before the fix this returned "improved"."""
    st, outbound, ret = _state_with_pair(monkeypatch)
    early = _leg(2, CUST, HUB, 400.0, 900.0, paired=1, cost=1.0)   # 140 s before the pad clears
    st.repair_planner = _FixedPlanner(early)
    rng = np.random.default_rng(0)

    out = st.try_repair([2], rng)
    assert out.reason == "anchor", "the early return must be refused by the guard"
    assert not out.accepted
    # `cost_new` is inf on any non-"improved" exit, so the improvement it gave up is the plan's
    # own cost: 1.0 against an incumbent 9.0. The guard refused a strict gain, not a wash.
    assert early.cost < out.cost_old, "and refused DESPITE being a strict improvement"

    # Nothing else in the stack would have caught it.
    assert verify.find_interflight_conflict([outbound, early], CFG) is None
    assert verify.find_paired_precedence_violation([outbound, early], CFG) is not None

    # The incumbent is untouched by the rejected repair.
    assert st.incumbent[2] is ret


def test_a_repaired_return_that_waits_is_still_accepted(monkeypatch):
    """The guard must not be a blanket veto on returns: same repair, departing on time, accepted."""
    st, outbound, ret = _state_with_pair(monkeypatch)
    ok = _leg(2, CUST, HUB, 600.0, 1100.0, paired=1, cost=1.0)     # 60 s AFTER the pad clears
    st.repair_planner = _FixedPlanner(ok)

    out = st.try_repair([2], np.random.default_rng(0))
    assert out.reason == "improved" and out.accepted
    assert verify.find_paired_precedence_violation([outbound, ok], CFG) is None


def _state_with_headroom(monkeypatch):
    """A round trip whose return departs 60 s AFTER its pad clears — so the outbound has slack."""
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import state as state_mod
    from freespace_sim.planner.lns.state import LNSState

    outbound = _leg(1, HUB, CUST, 0.0, 500.0, cost=9.0)        # lands 500, pad clears at 540
    ret = _leg(2, CUST, HUB, 600.0, 1100.0, paired=1)          # departs 600: 60 s of headroom
    led = ReservationLedger(CFG)
    for it in (outbound, ret):
        led.commit(it.request.flight_id, it.volumes)
    monkeypatch.setattr(state_mod, "unimpeded_costs",
                        lambda cfg, st, reqs, *, n_workers: [(r.flight_id, 1.0, None) for r in reqs])
    return LNSState(CFG, led, [outbound, ret], turnaround_s=0.0), outbound, ret


def test_an_outbound_slip_inside_the_pairs_headroom_is_allowed(monkeypatch):
    """The over-strictness this fixes. The old guard compared the outbound's release against the
    return's FILED `t_departure`, which on nominal every pair is already behind (p50 -58 s over
    2,318 pairs) — so ANY outbound slip was vetoed, 2,318 pairs blocked to protect the 85 that
    actually violate. The predicate now compares against the return's REALIZED takeoff, so a slip
    the return can absorb is accepted."""
    st, outbound, ret = _state_with_headroom(monkeypatch)
    slipped = _leg(1, HUB, CUST, 0.0, 550.0, cost=1.0)         # lands 50 s later; pad clears at 590
    st.repair_planner = _FixedPlanner(slipped)
    out = st.try_repair([1], np.random.default_rng(0))
    assert out.reason == "improved" and out.accepted, "60 s of headroom exists; do not veto it"
    assert verify.find_paired_precedence_violation([slipped, ret], CFG) is None


def test_an_outbound_slip_past_the_headroom_is_refused(monkeypatch):
    st, outbound, ret = _state_with_headroom(monkeypatch)
    late = _leg(1, HUB, CUST, 0.0, 600.0, cost=1.0)            # pad does not clear until 640 > 600
    st.repair_planner = _FixedPlanner(late)
    out = st.try_repair([1], np.random.default_rng(0))
    assert out.reason == "anchor" and not out.accepted
    assert verify.find_paired_precedence_violation([late, ret], CFG) is not None


def test_a_pre_existing_violation_is_grandfathered_but_not_worsened(monkeypatch):
    """A nominal-anchor baseline arrives WITH violations. LNS is answerable for not adding to them,
    not for the schedule it was handed — so the ratchet is per-pair 'no worse', not 'must be clean'."""
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import state as state_mod
    from freespace_sim.planner.lns.state import LNSState

    outbound = _leg(1, HUB, CUST, 0.0, 500.0)                  # pad clears at 540
    ret = _leg(2, CUST, HUB, 400.0, 900.0, paired=1, cost=9.0)  # ALREADY 140 s early
    led = ReservationLedger(CFG)
    for it in (outbound, ret):
        led.commit(it.request.flight_id, it.volumes)
    monkeypatch.setattr(state_mod, "unimpeded_costs",
                        lambda cfg, st, reqs, *, n_workers: [(r.flight_id, 1.0, None) for r in reqs])
    st = LNSState(CFG, led, [outbound, ret], turnaround_s=0.0)
    assert st._pair_shortfall[(1, 2)] == pytest.approx(140.0)
    assert st._precedence_baseline == 1

    # same shortfall, cheaper: allowed, because it does not make the pair worse
    same = _leg(2, CUST, HUB, 400.0, 880.0, paired=1, cost=1.0)
    st.repair_planner = _FixedPlanner(same)
    assert st.try_repair([2], np.random.default_rng(0)).reason == "improved"

    # one second earlier: refused
    worse = _leg(2, CUST, HUB, 399.0, 879.0, paired=1, cost=1.0)
    st.repair_planner = _FixedPlanner(worse)
    assert st.try_repair([2], np.random.default_rng(0)).reason == "anchor"


# ------------------------------------------------- the itinerary model: precedence by construction

def _itinerary_world(service_s=180.0, lam=900.0):
    """A congested hub world whose deliveries are round-trip itineraries."""
    from freespace_sim.demand import HubRadiusDemand

    cfg = SimConfig(planner="astar", lam_per_hour=lam, horizon_s=900.0,
                    region_size_m=(4000.0, 4000.0), seed=3, flight_levels_m=(75.0,),
                    airspace_ceiling_m=125.0, max_ground_delay_s=600.0)
    return cfg, HubRadiusDemand(n_hubs_per_uss={"a": 2}, return_flights=True,
                                turnaround_s=service_s)


def test_an_itinerarys_return_leg_never_precedes_its_own_arrival():
    """The property the whole model exists for, under congestion that used to break it.

    A two-request round trip filed its return on a straight-line estimate of the outbound's arrival,
    so any ground hold or detour put the return in the air before its aircraft was down. An itinerary
    reads the arrival that actually happened, so the second leg cannot start before the first ends —
    there is no filed time left to be wrong.
    """
    from freespace_sim.sim import run

    cfg, model = _itinerary_world()
    res = run(cfg, demand=model)
    trips = [i for i in res.intents if i.accepted and i.request.return_to_origin]
    assert trips, "the fixture must actually fly round trips"
    assert res.verified
    # The fixture has to be in the regime that broke the two-request scheme, or this passes for the
    # wrong reason: a return only departed early because its outbound ran over the estimate.
    assert all(_parked_s(i) > i.request.service_time_s + 1e-6 for i in trips), (
        "every return should be held past its service here; an uncongested fixture proves nothing")

    for it in trips:
        legs = _split_legs(it)
        assert len(legs) == 2
        out_land = max(v.t_end for v in legs[0])
        back_off = min(v.t_start for v in legs[1])
        assert back_off >= out_land - 1e-6, (
            f"flight {it.request.flight_id} leaves {out_land - back_off:.1f}s before it arrives")

    # And the check written for the two-request scheme agrees: nothing to find.
    assert verify.count_paired_precedence_violations(res.intents, cfg) == (0, 0.0)


def _parked_s(intent):
    """How long the aircraft sat on the customer pad, service plus any hold on the return."""
    box = [v for v in intent.volumes if hasattr(v.shape, "z_hi") and v.shape.z_hi < 100.0]
    return (box[0].t_end - box[0].t_start) if box else 0.0


def _split_legs(intent):
    """Volumes grouped by leg, split at the parked-aircraft ground box."""
    ground = [k for k, v in enumerate(intent.volumes)
              if hasattr(v.shape, "z_hi") and v.shape.z_hi < 100.0]
    if not ground:
        return [intent.volumes]
    k = ground[0]
    return [intent.volumes[:k], intent.volumes[k + 1:]]


def test_the_pad_is_held_continuously_while_the_aircraft_is_parked():
    """No gap between arriving and leaving: a held return is still on the pad, so the ground box
    spans arrival-column end -> departure-column start, not merely `service_time_s`."""
    from freespace_sim.sim import run

    cfg, model = _itinerary_world(service_s=180.0)
    res = run(cfg, demand=model)
    trips = [i for i in res.intents if i.accepted and i.request.return_to_origin]
    assert trips

    held = 0
    for it in trips:
        box = [v for v in it.volumes if hasattr(v.shape, "z_hi") and v.shape.z_hi < 100.0]
        if not box:
            continue
        held += 1
        (box,) = box
        assert box.shape.z_hi == pytest.approx(cfg.ground_level_m + cfg.ground_box_height_m)
        legs = _split_legs(it)
        assert box.t_start == pytest.approx(max(v.t_end for v in legs[0]))
        assert box.t_end == pytest.approx(min(v.t_start for v in legs[1]))
        assert box.t_end - box.t_start >= 180.0 - 1e-6      # service, plus any hold on top
    assert held == len(trips)


def test_an_itinerary_is_denied_whole_when_its_return_cannot_be_planned(monkeypatch):
    """A delivery whose aircraft cannot get home is not a half success, so `accepted` keeps meaning
    'this aircraft flew the trip' and nothing downstream has to handle a stranded leg."""
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.astar import AStarPlanner
    from freespace_sim.planner.itinerary import ItineraryPlanner
    from freespace_sim.types import DenialReason

    cfg = SimConfig(flight_levels_m=(75.0,), airspace_ceiling_m=125.0)
    inner = AStarPlanner()
    real = inner.plan
    calls = {"n": 0}

    def deny_the_second(req, ledger, c):
        calls["n"] += 1
        if calls["n"] == 2:
            return OperationalIntent(request=req, status=IntentStatus.REJECTED, volumes=[],
                                     denial_reason=DenialReason.BUDGET_EXCEEDED)
        return real(req, ledger, c)

    monkeypatch.setattr(inner, "plan", deny_the_second)
    req = FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, service_time_s=60.0)
    out = ItineraryPlanner(inner).plan(req, ReservationLedger(cfg), cfg)
    assert calls["n"] == 2                                  # it really did try the return
    assert not out.accepted and out.volumes == []
    assert out.request is req                               # the itinerary, not the leg it split off


def test_colgen_refuses_an_itinerary_rather_than_dropping_the_return():
    """Colgen prices one path per flight, so an itinerary would be solved as its outbound alone —
    indistinguishable in the output from a one-way delivery. It must fail loudly instead."""
    from freespace_sim.planner.colgen.batch import run_batch
    from freespace_sim.scenario import scenario_from_requests

    scen = scenario_from_requests([
        FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, service_time_s=30.0)])
    with pytest.raises(NotImplementedError, match="round-trip itineraries"):
        run_batch(scen, SimConfig(), None, None, (), None, None, None)
