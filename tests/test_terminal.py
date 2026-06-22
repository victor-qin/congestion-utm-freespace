"""Shared terminal volumes — multi-pad vertiport airspace.

Phase A: the exemption + perimeter geometry + backward-compat. A flight with no terminal is byte-for-
byte today's behavior; a hub flight tags its column with the hub id (shared among that hub's flights,
opaque to cruise). Capacity (N concurrent) is Phase B.
"""

import warnings

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


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
    # perimeter-start: the first cruise waypoint sits ~terminal_radius from the hub, not on top of it
    cfg = _astar()
    req = FlightRequest(0, vec(1000, 1000, 0), vec(4000, 1000, 0), 0.0, origin_terminal=("H", 4))
    res = run(cfg, requests=[req])
    cl = res.accepted[0].centerline
    first_air = np.array(cl[0][0])[:2]
    assert np.linalg.norm(first_air - np.array([1000.0, 1000.0])) >= cfg.terminal_radius_m - 1e-6


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
