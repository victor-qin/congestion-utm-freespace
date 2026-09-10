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


def _itinerary_world(dwell_s=180.0, lam=900.0):
    """A congested hub world whose deliveries are round-trip itineraries."""
    from freespace_sim.demand import HubRadiusDemand

    cfg = SimConfig(planner="astar", lam_per_hour=lam, horizon_s=900.0,
                    region_size_m=(4000.0, 4000.0), seed=3, flight_levels_m=(75.0,),
                    airspace_ceiling_m=125.0, max_ground_delay_s=600.0)
    return cfg, HubRadiusDemand(n_hubs_per_uss={"a": 2}, return_flights=True,
                                turnaround_s=dwell_s)


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
    assert all(_parked_s(i, cfg) > (cfg.turnaround_s if i.request.turnaround_s is None else i.request.turnaround_s) + 1e-6 for i in trips), (
        "every return should be held past its service here; an uncongested fixture proves nothing")

    for it in trips:
        legs = _split_legs(it, cfg)
        assert len(legs) == 2
        out_land = max(v.t_end for v in legs[0])
        back_off = min(v.t_start for v in legs[1])
        assert back_off >= out_land - 1e-6, (
            f"flight {it.request.flight_id} leaves {out_land - back_off:.1f}s before it arrives")


def _ground_boxes(intent, cfg):
    """The parked-aircraft boxes: the ones only `ground_box_height_m` tall, not full columns."""
    top = cfg.ground_level_m + cfg.ground_box_height_m
    return [v for v in intent.volumes
            if hasattr(v.shape, "z_hi") and v.shape.z_hi <= top + 1e-9]


def _parked_s(intent, cfg):
    """How long the aircraft sat on the customer pad, service plus any hold on the return."""
    box = _ground_boxes(intent, cfg)
    return (box[0].t_end - box[0].t_start) if box else 0.0


def _split_legs(intent, cfg):
    """Volumes grouped by leg, split at the parked-aircraft ground box."""
    boxes = set(id(v) for v in _ground_boxes(intent, cfg))
    ground = [k for k, v in enumerate(intent.volumes) if id(v) in boxes]
    if not ground:
        return [intent.volumes]
    k = ground[0]
    return [intent.volumes[:k], intent.volumes[k + 1:]]


def test_the_pad_is_held_continuously_while_the_aircraft_is_parked():
    """No gap between arriving and leaving: a held return is still on the pad, so the ground box
    spans arrival-column end -> departure-column start, not merely `turnaround_s`."""
    from freespace_sim.sim import run

    cfg, model = _itinerary_world(dwell_s=180.0)
    res = run(cfg, demand=model)
    trips = [i for i in res.intents if i.accepted and i.request.return_to_origin]
    assert trips

    held = 0
    for it in trips:
        box = _ground_boxes(it, cfg)
        if not box:
            continue
        held += 1
        (box,) = box
        assert box.shape.z_hi == pytest.approx(cfg.ground_level_m + cfg.ground_box_height_m)
        legs = _split_legs(it, cfg)
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
    req = FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, turnaround_s=60.0)
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
        FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, turnaround_s=30.0)])
    with pytest.raises(NotImplementedError, match="round-trip itineraries"):
        run_batch(scen, SimConfig(), None, None, (), None, None, None)


def test_the_pad_hold_is_deconflicted_before_the_itinerary_is_accepted():
    """The hold is built FROM both legs' results, so neither leg's search ever saw it.

    Checking it in `_compose` keeps `FCFSMechanism.commit`'s re-check the no-op its docstring
    promises, and matters most under LNS, whose commit path re-checks nothing at all.
    """
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner import get_planner
    from freespace_sim.types import DenialReason
    from freespace_sim.volumes import Volume4D

    cfg = SimConfig(flight_levels_m=(75.0,), airspace_ceiling_m=125.0)
    req = FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, turnaround_s=60.0)
    planner = get_planner("astar")

    clean = planner.plan(req, ReservationLedger(cfg), cfg)
    assert clean.accepted
    top = cfg.ground_level_m + cfg.ground_box_height_m
    hold = next(v for v in clean.volumes
                if isinstance(v.shape, CylinderSpec) and v.shape.z_hi <= top + 1e-9)
    assert hold.t_end - hold.t_start > 10.0          # room for a blocker that clears both columns

    # Strictly inside the turnaround gap: both columns stay clear, so both legs still plan exactly
    # as before and the hold between them is the only thing that conflicts.
    mid = 0.5 * (hold.t_start + hold.t_end)
    led = ReservationLedger(cfg)
    led.commit(2, [Volume4D(CylinderSpec(cx=hold.shape.cx, cy=hold.shape.cy,
                                         radius=hold.shape.radius, z_lo=cfg.ground_level_m,
                                         z_hi=cfg.airspace_ceiling_m), mid - 5.0, mid + 5.0)])
    out = planner.plan(req, led, cfg)
    assert not out.accepted
    assert out.denial_reason is DenialReason.CONFLICT_FILED


def test_an_outbound_accepted_without_volumes_raises_rather_than_stranding_the_return():
    """`realized_release_s` is None for an accepted intent holding no volumes, and returning the
    outbound there would be the silent half-trip `reject_itinerary` exists to prevent: a round trip
    at roughly half the cost, which a cost-comparing caller reads as an improvement."""
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.itinerary import ItineraryPlanner

    class _AcceptsWithNothing:
        def plan(self, req, ledger, cfg):
            return OperationalIntent(request=req, status=IntentStatus.ACCEPTED, volumes=[],
                                     centerline=[])

    cfg = SimConfig(flight_levels_m=(75.0,), airspace_ceiling_m=125.0)
    req = FlightRequest(1, HUB, CUST, 0.0, return_to_origin=True, turnaround_s=60.0)
    with pytest.raises(ValueError, match="no volumes"):
        ItineraryPlanner(_AcceptsWithNothing()).plan(req, ReservationLedger(cfg), cfg)


def test_the_itinerary_wrapper_neither_reorders_the_chain_nor_blocks_copying():
    """`iter_planner_chain` order is load-bearing — `_terminal_capacity_for` takes the FIRST match —
    and a wrapper that answers `warm_planner` on its child's behalf yields a grandchild at its own
    depth. The same missing guard let `__getattr__` recurse forever on the empty-dict instance that
    `copy`/`pickle` build before restoring state."""
    import copy
    import pickle

    from freespace_sim.planner import _get_planner, get_planner, iter_planner_chain

    for name in ("astar_milp", "milp", "astar_shortcut"):
        bare = [type(p).__name__ for p in iter_planner_chain(_get_planner(name))]
        wrapped = [type(p).__name__ for p in iter_planner_chain(get_planner(name))]
        assert wrapped[0] == "ItineraryPlanner"
        assert wrapped[1:] == bare               # the wrapper prepends itself; it must not reorder

    planner = get_planner("astar")
    assert type(copy.deepcopy(planner)) is type(planner)
    # Round-trips a planner this test just built — no external data is deserialised.
    assert type(pickle.loads(pickle.dumps(planner))) is type(planner)


@pytest.mark.slow
def test_lns_repair_keeps_both_legs_of_an_itinerary():
    """LNS must repair a round trip as a round trip.

    An unwrapped repair planner plans the outbound alone and drops the return; because that costs
    about half the trip, `try_repair`'s strict-improvement test then ADOPTS it. Measured before the
    fix: 36 of 36 round trips stranded, reported as a 66.93% improvement with verified=True.
    """
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.sim import run

    cfg, model = _itinerary_world(lam=300.0)      # the ruler replans every flight; keep it small
    res = run(cfg, demand=model)
    before = {i.request.flight_id: i for i in res.intents
              if i.accepted and i.request.return_to_origin and i.leg_starts}
    assert before, "the fixture must fly round trips with both legs"

    out = run_lns(cfg, res.ledger, res.intents,
                  LNSConfig(seed=7, max_iterations=60, neighborhood_size=2,
                            operators=("agent",), log_every=0, unimpeded_workers=1),
                  static_terms=res.ledger.static_terminals())
    after = {i.request.flight_id: i for i in out.intents}
    stranded = [f for f in before if not after[f].leg_starts]
    assert not stranded, f"{len(stranded)}/{len(before)} round trips lost their return leg"
    assert out.n_accepted > 0, "an LNS pass that accepted nothing cannot show the return survived"


def test_the_arrival_and_departure_clocks_are_the_columns_not_the_waypoints():
    """`ItineraryPlanner` derives the return's entire departure clock from `realized_release_s` and
    ends the pad hold at `realized_takeoff_s`. Both are extremes over the volume list, which is the
    landing/takeoff column only while nothing else reaches further — and the corridor stops at the
    column EDGE at cruise altitude, so the first and last waypoints sit INSIDE the columns by the
    climb. Pin both, or the first planner to file a volume past the landing column shifts every
    return late and over-reserves every customer pad with nothing to catch it."""
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner import get_planner
    from freespace_sim.sim import realized_release_s
    from freespace_sim.verify import realized_takeoff_s

    cfg = SimConfig(flight_levels_m=(75.0,), airspace_ceiling_m=125.0)
    leg = get_planner("astar").plan(FlightRequest(1, HUB, CUST, 0.0), ReservationLedger(cfg), cfg)
    assert leg.accepted

    columns = [v for v in leg.volumes if isinstance(v.shape, CylinderSpec)]
    assert len(columns) == 2            # a leg is [takeoff column, corridor boxes..., landing column]
    assert realized_takeoff_s(leg) == pytest.approx(columns[0].t_start)
    assert realized_release_s(leg) == pytest.approx(columns[-1].t_end)
    assert realized_takeoff_s(leg) < leg.centerline[0][1] - 1e-9
    assert realized_release_s(leg) > leg.centerline[-1][1] + 1e-9
