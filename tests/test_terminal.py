"""Shared terminal volumes — multi-pad vertiport airspace.

Phase A: the exemption + geometry + backward-compat. A flight with no terminal is byte-for-byte
today's behavior; a hub flight tags its column with the hub id (shared among that hub's flights,
opaque to cruise). Phase B: pad capacity. The drone climbs in the shared column and the strict
corridor (its "exit lane") begins a clear margin beyond the column edge — the centre→edge leg is
flown but unreserved (vertiport-tactical), so up to ``capacity`` same-hub flights launch concurrently
and the (N+1)th takes ground delay. Divergent launches go at once; same-lane launches contend.
"""

import warnings

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec, CylinderSpec
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, Terminal, vec


def _astar(**over):
    return SimConfig(planner="astar", region_size_m=(5000.0, 5000.0), **over)


def _terminal_req():
    return FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=("H", 4))


def test_non_terminal_flight_emits_untagged_volumes():
    # backward-compat: an ordinary flight (no terminal) → every volume's terminal_id is None
    res = run(_astar(), requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    assert res.verified and res.accepted
    assert all(v.terminal_id is None for v in res.accepted[0].volumes)


def test_delivery_tags_its_origin_hub_column():
    req = FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=("hubX", 4))
    res = run(_astar(), requests=[req])
    assert res.verified
    vols = res.accepted[0].volumes
    assert any(v.terminal_id == "hubX" for v in vols)     # the origin terminal column is tagged
    assert any(v.terminal_id is None for v in vols)       # the cruise corridor stays strict


def test_return_tags_its_dest_hub_column():
    req = FlightRequest(0, vec(3400, 1000, 0), vec(1000, 1000, 0), 0.0, dest_terminal=("hubX", 4))
    res = run(_astar(), requests=[req])
    assert res.verified
    assert any(v.terminal_id == "hubX" for v in res.accepted[0].volumes)   # the landing column is tagged


def test_two_same_hub_deliveries_coexist_and_verify():
    # two deliveries from one hub to different customers share its terminal → the run stays verified
    reqs = [
        FlightRequest(0, vec(2500, 2500, 0), vec(4200, 2500, 0), 0.0, origin_terminal=("H", 4)),
        FlightRequest(1, vec(2500, 2500, 0), vec(2500, 4200, 0), 0.0, origin_terminal=("H", 4)),
    ]
    res = run(_astar(), requests=reqs)
    assert res.verified
    assert len(res.accepted) == 2


def test_delivery_then_return_roundtrip_verifies():
    reqs = [
        FlightRequest(0, vec(2500, 2500, 0), vec(3800, 2500, 0), 0.0, origin_terminal=("H", 4)),
        FlightRequest(1, vec(3800, 2500, 0), vec(2500, 2500, 0), 300.0, dest_terminal=("H", 4)),
    ]
    res = run(_astar(), requests=reqs)
    assert res.verified and len(res.accepted) == 2
    assert any(v.terminal_id == "H" for v in res.accepted[1].volumes)


def test_corridor_starts_away_from_the_hub_centre():
    # the reserved corridor (exit lane) starts at the column edge, not on top of the hub: waypoints
    # inside the column are folded away and the centre→edge leg is left unreserved (vertiport-tactical)
    cfg = _astar()
    req = FlightRequest(0, vec(1000, 1000, 0), vec(4000, 1000, 0), 0.0, origin_terminal=("H", 4))
    res = run(cfg, requests=[req])
    first_air = np.array(res.accepted[0].centerline[0][0])[:2]
    # exit lane starts a clear corridor_width beyond the column edge (default terminal radius = hover)
    assert np.linalg.norm(first_air - np.array([1000.0, 1000.0])) >= cfg.effective_hover_radius_m - 1e-6


def test_terminal_radius_sizes_the_column():
    # a custom terminal radius makes the shared column that big (default is the hover footprint)
    req = FlightRequest(0, vec(1500, 1500, 0), vec(4000, 1500, 0), 0.0,
                        origin_terminal=Terminal("H", 4, radius=200.0))
    res = run(_astar(), requests=[req])
    cyl = [v for v in res.accepted[0].volumes
           if v.terminal_id == "H" and isinstance(v.shape, CylinderSpec)][0]
    assert abs(cyl.shape.radius - 200.0) < 1e-6


def test_only_the_column_is_shared_corridors_strict():
    # ONLY the hover column is tagged shared; every corridor box stays strict (conflicts with all,
    # including same-hub flights), even the bit that dips into the terminal by corridor_overlap
    req = FlightRequest(0, vec(1500, 1500, 0), vec(4000, 1500, 0), 0.0,
                        origin_terminal=Terminal("H", 4, corridor_overlap=40.0))
    res = run(_astar(), requests=[req])
    vols = res.accepted[0].volumes
    tagged = [v for v in vols if v.terminal_id == "H"]
    assert tagged and all(isinstance(v.shape, CylinderSpec) for v in tagged)         # only column(s)
    assert all(v.terminal_id is None for v in vols if isinstance(v.shape, BoxSpec))  # boxes strict


def test_corridor_overlap_controls_perimeter_start():
    # the overlap is a *geometry* knob now: bigger overlap → corridor starts deeper inside the terminal
    def start_dist(overlap):
        req = FlightRequest(0, vec(1500, 1500, 0), vec(4000, 1500, 0), 0.0,
                            origin_terminal=Terminal("H", 4, corridor_overlap=overlap))
        first = np.array(run(_astar(), requests=[req]).accepted[0].centerline[0][0])[:2]
        return float(np.linalg.norm(first - np.array([1500.0, 1500.0])))

    assert start_dist(40.0) < start_dist(0.0)   # more overlap → starts closer to the hub centre


# --- Phase B: pad capacity — N concurrent same-hub launches, then ground delay ----------------


def _radial_delivery(hub_xy, angle_deg, dist, capacity, fid, t=0.0):
    """A delivery from ``hub_xy`` (capacity N) to a customer ``dist`` m away at ``angle_deg`` — so a
    batch of them diverge from the shared column and contend ONLY for pads, not airspace."""
    a = np.radians(angle_deg)
    dest = vec(hub_xy[0] + dist * np.cos(a), hub_xy[1] + dist * np.sin(a), 0)
    return FlightRequest(fid, vec(hub_xy[0], hub_xy[1], 0), dest, t,
                         origin_terminal=Terminal("H", capacity))


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_pad_capacity_admits_n_concurrent_then_delays(cap):
    # N+1 deliveries leave one hub at the same instant: exactly N launch now (capacity), the extra
    # takes ground delay — admitted, not denied. This is the whole point of Phase B.
    hub = (3000.0, 3000.0)
    n = cap + 1
    reqs = [_radial_delivery(hub, i * 360.0 / n, 2000.0, cap, i) for i in range(n)]
    res = run(SimConfig(planner="astar", region_size_m=(6000.0, 6000.0)), requests=reqs)
    assert res.verified and len(res.accepted) == n                    # all admitted, none denied
    concurrent = [a for a in res.accepted if a.ground_delay_s == 0.0]
    assert len(concurrent) == cap                                     # exactly N share the column now
    assert any(a.ground_delay_s > 0.0 for a in res.accepted)          # the (N+1)th waits for a pad


def test_two_same_hub_flights_launch_concurrently_under_capacity():
    # the sharp before/after: with capacity 2, two same-hub launches BOTH go at t0 (pre-Phase-B the
    # first flight's column blocked the second into a ground delay)
    hub = (3000.0, 3000.0)
    reqs = [_radial_delivery(hub, 0.0, 2000.0, 2, 0), _radial_delivery(hub, 180.0, 2000.0, 2, 1)]
    res = run(SimConfig(planner="astar", region_size_m=(6000.0, 6000.0)), requests=reqs)
    assert res.verified and len(res.accepted) == 2
    assert all(a.ground_delay_s == 0.0 for a in res.accepted)         # concurrent, zero ground delay


def test_capacity_one_serializes_like_a_single_pad():
    # capacity 1 ⟺ the legacy single pad: the second same-hub launch must wait
    hub = (3000.0, 3000.0)
    reqs = [_radial_delivery(hub, 0.0, 2000.0, 1, 0), _radial_delivery(hub, 180.0, 2000.0, 1, 1)]
    res = run(SimConfig(planner="astar", region_size_m=(6000.0, 6000.0)), requests=reqs)
    assert res.verified and len(res.accepted) == 2
    assert sum(a.ground_delay_s == 0.0 for a in res.accepted) == 1    # one now, one delayed


def test_non_astar_planner_warns_on_terminal_flight():
    # a non-A* planner rebuilds corridors and drops the tag → loud RuntimeWarning, not silent wrongness
    with pytest.warns(RuntimeWarning, match="terminal airspace"):
        run(SimConfig(planner="straight", region_size_m=(5000.0, 5000.0)), requests=[_terminal_req()])


def test_astar_planner_does_not_warn_on_terminal_flight():
    # A* tags the column → no warning (turn RuntimeWarning into an error to prove silence)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = run(_astar(), requests=[_terminal_req()])
    assert res.verified and res.accepted


def test_astar_shortcut_preserves_terminal_tags_no_warning():
    # the refiner rebuilds the corridor but now keeps the inner A*'s terminal tags → no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = run(SimConfig(planner="astar_shortcut", region_size_m=(5000.0, 5000.0)),
                  requests=[_terminal_req()])
    assert res.verified
    assert any(v.terminal_id == "H" for v in res.accepted[0].volumes)   # column tag survived the rebuild
