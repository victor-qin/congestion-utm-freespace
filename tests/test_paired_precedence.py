"""A round trip is ONE flight, so its return leg cannot precede its own arrival.

The two legs are planned as one itinerary (`planner.itinerary.ItineraryPlanner`): the return departs
from the arrival the outbound actually achieved, so there is no filed time left to be wrong. This
replaced a scheme that filed the return separately, on a straight-line estimate of that arrival.
"""
from __future__ import annotations

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec

HUB, CUST = vec(0, 0, 0), vec(3000, 0, 0)


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
