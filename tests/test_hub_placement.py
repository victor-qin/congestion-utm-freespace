"""Hub placement keeps terminal airspaces from overlapping.

``HubRadiusDemand.place_hubs`` reject-samples hub centres so no two hubs' terminal columns overlap — the
headline, non-flaky invariant is a pairwise geometric one: for every pair of hubs the centre distance is
at least ``r_i + r_j`` (their radii), across both operators. Before this, an unconstrained uniform
scatter could drop two hubs within a radius of each other (e.g. ``stripmall_uss#11`` and ``#17`` landed
45 m apart with 90 m radii), which under ``terminal_airspace_always_active`` makes one hub's landing
approach infeasible. ``HubVoronoiDemand`` flights carry no terminals, so it keeps the plain scatter
(nothing to overlap). Placement is deterministic in ``hub_seed`` and independent of pad capacity; a
region too crowded to satisfy the separation fails loudly; and the clearance is reachable from a
scenario via ``DemandSpec.min_hub_gap_m``.
"""
from itertools import combinations

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand, HubVoronoiDemand
from freespace_sim.planner.hexgrid import SQRT3, circumradius
from freespace_sim.scenarios import get_scenario
from freespace_sim.types import Terminal
from freespace_sim.volumes import exit_radius


def _radius_of(dm, uss, cfg):
    tr = dm._terminal_radius_for(uss)          # already collapses a per-USS dict to a scalar (or None)
    if cfg.terminal_airspace_always_active:    # match place_hubs: taa reject-samples on the WIDER walled
        term = Terminal(f"{uss}#0", dm._pads_for(uss), tr, dm.corridor_overlap_m)   # extent (column+ring)
        return exit_radius(term, cfg) + SQRT3 * circumradius(cfg)
    return cfg.terminal_radius_m if tr is None else float(tr)


def _hubs_with_radii(dm, cfg):
    """[(id, centre(2,), radius)] for every hub the model places."""
    hubs = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
    out = []
    for uss, pts in hubs.items():
        r = _radius_of(dm, uss, cfg)
        for j in range(len(pts)):
            out.append((f"{uss}#{j}", np.asarray(pts[j], float), r))
    return out


def _worst_overlap(hubs):
    """Most-overlapping pair as (slack, id_i, id_j) where slack = dist - (r_i + r_j); slack < 0 ⇒ overlap."""
    worst = None
    for (i1, c1, r1), (i2, c2, r2) in combinations(hubs, 2):
        slack = float(np.linalg.norm(c1 - c2)) - (r1 + r2)
        if worst is None or slack < worst[0]:
            worst = (slack, i1, i2)
    return worst


# ---- headline: the shipped hub_radius scenario has no overlapping terminal airspaces ----

def test_dallas_large_hubs_have_no_overlapping_airspaces():
    # dallas_hub_2uss_large is the scenario where stripmall_uss#11/#17 engulfed each other (45 m apart,
    # 90 m radii) under the old unconstrained scatter — the worst-overlap assert below IS that pair's
    # regression. (dallas_hub_2uss uses HubVoronoiDemand, whose flights carry no terminals to overlap.)
    spec = get_scenario("dallas_hub_2uss_large")
    cfg, dm = spec.config(), spec.demand_model()
    hubs = _hubs_with_radii(dm, cfg)
    assert len(hubs) == 26
    slack, i, j = _worst_overlap(hubs)
    assert slack >= 0.0, f"{i} and {j} terminal airspaces overlap by {-slack:.0f} m"


# ---- the general placement contract, on both demand classes ----

def test_min_hub_gap_is_respected_across_operators():
    cfg = SimConfig(region_size_m=(12000.0, 12000.0))
    dm = HubRadiusDemand(n_hubs_per_uss={"big": 4, "small": 10},
                         terminal_radius_m={"big": 180.0, "small": 90.0}, min_hub_gap_m=150.0)
    hubs = _hubs_with_radii(dm, cfg)
    for (i1, c1, r1), (i2, c2, r2) in combinations(hubs, 2):
        assert float(np.linalg.norm(c1 - c2)) >= r1 + r2 + 150.0 - 1e-6, f"{i1}/{i2} closer than the gap"


def test_voronoi_keeps_plain_unconstrained_scatter():
    """HubVoronoiDemand flights have no terminals, so it must NOT reject-sample — its placement is the
    plain uniform scatter (guards against re-adding a min-separation that has no physical referent)."""
    cfg = SimConfig(region_size_m=(6000.0, 6000.0))
    dm = HubVoronoiDemand(n_hubs_per_uss={"a": 30})
    got = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))["a"]
    want = np.random.default_rng(dm.hub_seed).uniform([0.0, 0.0], [6000.0, 6000.0], size=(30, 2))
    assert np.allclose(got, want)


def test_demand_spec_threads_min_hub_gap():
    """min_hub_gap_m set on a DemandSpec reaches the built model AND its placement — before, the knob
    the crowded-region ValueError told you to tune was frozen at its default, unreachable from specs."""
    from freespace_sim.scenarios.spec import DemandSpec
    dm = DemandSpec(pattern="hub_radius", uss=("a", "b"), hubs=(3, 5),
                    terminal_radius_m=90.0, min_hub_gap_m=300.0).build()
    assert dm.min_hub_gap_m == 300.0
    cfg = SimConfig(region_size_m=(12000.0, 12000.0))
    hubs = _hubs_with_radii(dm, cfg)
    for (i1, c1, r1), (i2, c2, r2) in combinations(hubs, 2):
        assert float(np.linalg.norm(c1 - c2)) >= r1 + r2 + 300.0 - 1e-6, f"{i1}/{i2} ignores spec gap"


def test_placement_deterministic_in_hub_seed():
    cfg = SimConfig(region_size_m=(9000.0, 9000.0))
    for dm in (HubVoronoiDemand(), HubRadiusDemand()):
        a = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
        b = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
        assert set(a) == set(b)
        for uid in a:
            assert np.allclose(a[uid], b[uid])


def test_placement_independent_of_pad_capacity():
    """Pads are terminal capacity, not geometry — changing them must not move a hub."""
    cfg = SimConfig(region_size_m=(9000.0, 9000.0))
    ha = HubRadiusDemand(n_hubs_per_uss={"a": 4}, pads_per_hub=1).place_hubs(
        cfg, np.random.default_rng(0xA17F))["a"]
    hb = HubRadiusDemand(n_hubs_per_uss={"a": 4}, pads_per_hub=9).place_hubs(
        cfg, np.random.default_rng(0xA17F))["a"]
    assert np.allclose(ha, hb)


def test_hub_counts_preserved():
    cfg = SimConfig(region_size_m=(10000.0, 10000.0))
    hubs = HubRadiusDemand(n_hubs_per_uss={"a": 3, "b": 7}).place_hubs(
        cfg, np.random.default_rng(0xA17F))
    assert {uid: len(v) for uid, v in hubs.items()} == {"a": 3, "b": 7}


def test_place_hubs_raises_when_too_crowded():
    """A region that can't fit the hubs at their separation fails loudly, never silently overlaps."""
    cfg = SimConfig(region_size_m=(100.0, 100.0))                   # 280 m separation can't fit in 100 m
    dm = HubRadiusDemand(n_hubs_per_uss={"a": 5}, terminal_radius_m=90.0, min_hub_gap_m=100.0)
    with pytest.raises(ValueError, match="too crowded"):
        dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
