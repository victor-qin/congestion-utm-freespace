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


def _leg(fid, origin, dest, t_takeoff, t_land, *, paired=None, dwell=40.0):
    """An accepted intent holding a takeoff column at `origin` and a landing column at `dest`."""
    req = FlightRequest(fid, origin, dest, 0.0, t_departure=t_takeoff, paired_outbound_id=paired)
    vols = [
        Volume4D(CylinderSpec(float(origin[0]), float(origin[1]), 60.0, 0.0, 125.0),
                 t_takeoff, t_takeoff + dwell),
        Volume4D(CylinderSpec(float(dest[0]), float(dest[1]), 60.0, 0.0, 125.0),
                 t_land, t_land + dwell),
    ]
    return OperationalIntent(request=req, status=IntentStatus.ACCEPTED, volumes=vols,
                             centerline=[(np.asarray(origin, float), t_takeoff),
                                         (np.asarray(dest, float), t_land)])


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
    leg = _leg(1, HUB, CUST, 100.0, 500.0)
    assert verify.realized_takeoff_s(leg) == pytest.approx(100.0)
    assert verify.realized_takeoff_s(leg) < leg.centerline[0][1] + 1e-9
