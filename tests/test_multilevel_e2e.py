"""End-to-end: discrete altitude flight levels relieve lateral congestion (issue #2 acceptance).

A saturated metro — many crossing flights in a small region with a bounded ground-delay escape — fills
a single cruise plane and forces denials. Opening additional flight levels lets crossing traffic
deconflict by altitude instead, so strictly more flights are admitted, and the FCL replay confirms the
extra levels never phantom-collide.
"""

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.sim import run


def _saturated(levels):
    return SimConfig(
        planner="astar", cruise_level_m=75.0, flight_levels_m=levels, airspace_ceiling_m=125.0,
        z_min_m=75.0, z_max_m=75.0, lam_per_hour=3000.0, horizon_s=300.0,
        region_size_m=(900.0, 900.0), seed=1, max_ground_delay_s=120.0,
    )


@pytest.mark.slow
def test_multilevel_drops_denials_vs_single_level():
    single = run(_saturated((75.0,)))                      # one flight level (the baseline plane)
    multi = run(_saturated((30.0, 70.0, 110.0)))           # three levels
    assert single.verified and multi.verified              # no phantom cross-level conflicts (FCL replay)
    assert len(multi.accepted) > len(single.accepted)      # altitude is a real capacity lever
