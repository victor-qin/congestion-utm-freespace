"""Shared-terminal + pad-capacity parity for the MILP family (issue #36).

Mirrors the load-bearing A* properties in ``tests/test_terminal.py``: hub geometry is folded to the
column edge and TAGGED (so the same-tid cylinder exemption admits shared-column use), and pad
capacity is gated through ``TerminalCapacity.admits`` at plan/verify time (``any_conflict`` cannot
catch over-subscription — same-hub columns are conflict-exempt).
"""

import warnings

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.milp import MILPOptPlanner
from freespace_sim.planner.straight import StraightLineTimeShift
from freespace_sim.sim import _wall_aware, run
from freespace_sim.types import FlightRequest, IntentStatus, Terminal, vec
from freespace_sim.volumes import exit_radius

PLANNERS = ["milp", "astar_milp"]


def _cfg(planner, region=5000.0, **over):
    return SimConfig(planner=planner, region_size_m=(region, region), **over)


def _radial_delivery(hub_xy, angle_deg, dist, capacity, fid, t=0.0, radius=None):
    """A delivery from ``hub_xy`` to a customer ``dist`` m away at ``angle_deg`` — a batch of them
    diverges from the shared column and contends ONLY for pads, not airspace (same diagnostic shape
    as tests/test_terminal.py)."""
    a = np.radians(angle_deg)
    dest = vec(hub_xy[0] + dist * np.cos(a), hub_xy[1] + dist * np.sin(a), 0)
    return FlightRequest(fid, vec(hub_xy[0], hub_xy[1], 0), dest, t,
                         origin_terminal=Terminal("H", capacity, radius=radius))


def _max_concurrent(intervals):
    """Max number of half-open [t0, t1) windows overlapping at any instant."""
    evts = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    cur = mx = 0
    for _, d in evts:
        cur += d
        mx = max(mx, cur)
    return mx


# --- tags + fold -----------------------------------------------------------------------------------

@pytest.mark.parametrize("planner", PLANNERS)
def test_milp_delivery_tags_origin_hub_column(planner):
    req = FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=("hubX", 4))
    res = run(_cfg(planner), requests=[req])
    assert res.verified
    vols = res.accepted[0].volumes
    assert any(v.terminal_id == "hubX" for v in vols)     # the origin terminal column is tagged
    assert any(v.terminal_id is None for v in vols)       # the cruise corridor stays strict


@pytest.mark.parametrize("planner", PLANNERS)
def test_milp_no_terminal_warning(planner):
    req = FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=("H", 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)    # a dropped terminal would raise here
        res = run(_cfg(planner), requests=[req])
    assert res.verified and res.accepted


def test_milp_corridor_starts_at_column_edge():
    # the fold: the strict corridor is rooted at exit_radius, the centre→edge leg stays unreserved
    cfg = SimConfig()
    term = Terminal("H", 4)
    req = FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=term)
    intent = MILPOptPlanner().plan(req, ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    p0 = intent.centerline[0][0]
    d0 = float(np.hypot(p0[0] - 1000.0, p0[1] - 1000.0))
    assert d0 >= exit_radius(term, cfg) - 1e-6


# --- pad capacity ----------------------------------------------------------------------------------

@pytest.mark.parametrize("planner", PLANNERS)
def test_milp_divergent_same_hub_launches_concurrent(planner):
    hub = (3000.0, 3000.0)
    reqs = [_radial_delivery(hub, 0.0, 2000.0, 2, 0), _radial_delivery(hub, 180.0, 2000.0, 2, 1)]
    res = run(_cfg(planner, region=6000.0), requests=reqs)
    assert res.verified and len(res.accepted) == 2
    assert all(a.ground_delay_s == 0.0 for a in res.accepted)   # concurrent under capacity 2


def test_milp_pad_capacity_admits_n_then_delays():
    # capacity 2, three simultaneous launches: exactly 2 go now, the 3rd waits — admitted, not denied
    hub = (3000.0, 3000.0)
    reqs = [_radial_delivery(hub, i * 120.0, 2000.0, 2, i) for i in range(3)]
    res = run(_cfg("milp", region=6000.0), requests=reqs)
    assert res.verified and len(res.accepted) == 3
    assert sum(a.ground_delay_s == 0.0 for a in res.accepted) == 2
    assert any(a.ground_delay_s > 0.0 for a in res.accepted)


def test_milp_launch_waits_for_foreign_corridor_in_column():
    # a foreign cruise transits the hub; the launch must ground-delay until its column is clear
    cfg = SimConfig(region_size_m=(8000.0, 8000.0))
    led = ReservationLedger(cfg)
    foreign = FlightRequest(99, vec(1200, 4000, 0), vec(6800, 4000, 0), 0.0)
    led.commit(99, StraightLineTimeShift().plan(foreign, led, cfg).volumes)
    launch = FlightRequest(0, vec(4000, 4000, 0), vec(4000, 7000, 0), 80.0,
                           origin_terminal=Terminal("H", 4, radius=150.0))
    intent = MILPOptPlanner().plan(launch, led, cfg)
    assert intent.status is IntentStatus.ACCEPTED          # admitted, not denied
    assert intent.ground_delay_s > 0.0                     # waited for the airspace to free
    assert not led.any_conflict(intent.volumes)            # the activated column is genuinely clear


def test_milp_landing_capacity_not_oversubscribed():
    # three returns into a capacity-2 hub arriving in a cluster: peak concurrent landing dwells ≤ 2
    hub = (3000.0, 3000.0)
    term = Terminal("H", 2)
    reqs = [
        FlightRequest(i, vec(hub[0] + 2000 * np.cos(a), hub[1] + 2000 * np.sin(a), 0),
                      vec(hub[0], hub[1], 0), 5.0 * i, dest_terminal=term)
        for i, a in enumerate(np.radians([0.0, 120.0, 240.0]))
    ]
    res = run(_cfg("milp", region=6000.0), requests=reqs)
    assert res.verified and len(res.accepted) == 3
    dwells = [(v.t_start, v.t_end) for a in res.accepted for v in a.volumes
              if v.terminal_id == "H" and isinstance(v.shape, CylinderSpec)]
    assert dwells and _max_concurrent(dwells) <= 2


# --- always-active admission -----------------------------------------------------------------------

def test_wall_aware_admits_milp_family_refuses_plain():
    assert _wall_aware(get_planner("milp"))
    assert _wall_aware(get_planner("astar_milp"))
    assert _wall_aware(get_planner("astar_milp_shortcut"))
    assert not _wall_aware(get_planner("straight"))
    assert not _wall_aware(get_planner("decoupled"))


def test_always_active_runs_bare_milp():
    # pre-#36 this raised the untagged-near-hub ValueError; now the MILP is wall-aware end-to-end
    from freespace_sim.scenarios import get_scenario, with_overrides
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"), horizon_s=4.0)
    cfg = spec.config()
    assert cfg.terminal_airspace_always_active
    r = run(cfg, demand=spec.demand_model(), planner_name="milp")
    assert r.verified


def test_astar_milp_shortcut_preserves_terminal_tags():
    req = FlightRequest(0, vec(1000, 1000, 0), vec(3400, 1000, 0), 0.0, origin_terminal=("H", 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = run(_cfg("astar_milp_shortcut"), requests=[req])
    assert res.verified
    assert any(v.terminal_id == "H" for v in res.accepted[0].volumes)
