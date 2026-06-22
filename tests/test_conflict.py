"""The core ASTM §3.2.8 predicate matrix: conflict iff space-overlap AND time-overlap."""

import numpy as np

from freespace_sim.conflict import volumes_conflict
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.volumes import Volume4D


def box_vol(cx, cy, cz, t0, t1, *, length=40, width=40, height=30):
    p0 = np.array([cx - length / 2, cy, cz], float)
    p1 = np.array([cx + length / 2, cy, cz], float)
    return Volume4D(box_from_segment(p0, p1, width, height), t0, t1)


def test_space_and_time_overlap_conflicts():
    a = box_vol(0, 0, 150, 0, 10)
    b = box_vol(0, 0, 150, 5, 15)
    assert volumes_conflict(a, b)


def test_space_overlap_time_disjoint_no_conflict():
    a = box_vol(0, 0, 150, 0, 10)
    b = box_vol(0, 0, 150, 20, 30)
    assert not volumes_conflict(a, b)


def test_time_overlap_space_disjoint_no_conflict():
    a = box_vol(0, 0, 150, 0, 10)
    b = box_vol(1000, 0, 150, 0, 10)
    assert not volumes_conflict(a, b)


def test_altitude_separation_no_conflict():
    # same xy & time, different altitude bands → ASTM 3D test must separate them
    low = box_vol(0, 0, 60, 0, 10)    # z ∈ [45, 75]
    high = box_vol(0, 0, 150, 0, 10)  # z ∈ [135, 165]
    assert not volumes_conflict(low, high)


def test_hover_cylinder_blocks_overlapping_corridor():
    hover = Volume4D(CylinderSpec(0, 0, 60, 0, 150), 0, 60)
    corridor = box_vol(0, 0, 150, 10, 20)  # cruise box passing over the pad while it's hovering
    assert volumes_conflict(hover, corridor)


def test_hover_cylinder_clears_when_time_disjoint():
    hover = Volume4D(CylinderSpec(0, 0, 60, 0, 150), 0, 60)
    corridor = box_vol(0, 0, 150, 100, 110)  # same place, much later
    assert not volumes_conflict(hover, corridor)


def test_security_margin_enforces_separation():
    a = box_vol(0, 0, 150, 0, 10, length=40, width=40)   # x ∈ [-20, 20]
    b = box_vol(60, 0, 150, 0, 10, length=40, width=40)  # x ∈ [40, 80] → 20 m gap
    assert not volumes_conflict(a, b)
    assert volumes_conflict(a, b, security_margin=25.0)  # require 25 m apart → now a conflict


# --- shared-terminal exemption (multi-pad vertiport airspace) --------------------------------

def _terminal(cx, cy, t0, t1, hub):
    return Volume4D(CylinderSpec(cx, cy, 60, 0, 150), t0, t1, terminal_id=hub)


def test_same_terminal_volumes_are_transparent():
    # two flights sharing one hub's terminal occupy the same column at the same time — NOT a conflict
    a = _terminal(0, 0, 0, 60, hub="walmart#3")
    b = _terminal(0, 0, 0, 60, hub="walmart#3")
    assert not volumes_conflict(a, b)


def test_different_terminals_still_conflict():
    a = _terminal(0, 0, 0, 60, hub="walmart#3")
    b = _terminal(0, 0, 0, 60, hub="walmart#7")   # different vertiport, same spot/time → conflict
    assert volumes_conflict(a, b)


def test_cruise_corridor_blocked_by_terminal():
    # a corridor (terminal_id=None) over a busy hub still conflicts → cruise must route around it
    terminal = _terminal(0, 0, 0, 60, hub="walmart#3")
    corridor = box_vol(0, 0, 150, 10, 20)
    assert corridor.terminal_id is None
    assert volumes_conflict(terminal, corridor)


def test_terminal_exemption_ignores_time_and_space_when_shared():
    # the exemption short-circuits before the time/space tests — same id ⇒ never a conflict
    a = _terminal(0, 0, 0, 10, hub="h")
    b = _terminal(0, 0, 0, 10, hub="h")
    assert not volumes_conflict(a, b, security_margin=100.0)
