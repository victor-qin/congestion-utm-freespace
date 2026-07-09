"""Hub placement keeps terminal airspaces from overlapping.

``place_hubs`` reject-samples hub centres so no two hubs' terminal columns overlap — the headline,
non-flaky invariant is a pairwise geometric one: for every pair of hubs the centre distance is at least
``r_i + r_j`` (their radii), enforced across BOTH operators. Before this, an unconstrained uniform
scatter could drop two same-operator hubs within a radius of each other (e.g. ``stripmall_uss#11`` and
``#17`` landed 45 m apart with 90 m radii), which under ``terminal_airspace_always_active`` makes one
hub's landing approach infeasible. Placement stays deterministic in ``hub_seed`` and independent of pad
capacity, and a region too crowded to satisfy the separation fails loudly.
"""
from itertools import combinations

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand, HubVoronoiDemand
from freespace_sim.scenarios import get_scenario


def _radius_of(dm, uss, cfg):
    tr = dm._terminal_radius_for(uss) if hasattr(dm, "_terminal_radius_for") else None
    if isinstance(tr, dict):
        tr = tr.get(uss)
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


# ---- headline: shipped scenarios have no overlapping terminal airspaces ----

@pytest.mark.parametrize("name", ["dallas_hub_2uss", "dallas_hub_2uss_large"])
def test_scenario_hubs_have_no_overlapping_airspaces(name):
    spec = get_scenario(name)
    cfg, dm = spec.config(), spec.demand_model()
    hubs = _hubs_with_radii(dm, cfg)
    assert len(hubs) == 26
    slack, i, j = _worst_overlap(hubs)
    assert slack >= 0.0, f"{name}: {i} and {j} terminal airspaces overlap by {-slack:.0f} m"


def test_regression_stripmall_11_17_no_longer_engulfed():
    """The exact pair that broke always-active (45 m apart, 90 m radii) is now separated."""
    spec = get_scenario("dallas_hub_2uss_large")
    cfg, dm = spec.config(), spec.demand_model()
    hubs = dict(((hid, (c, r)) for hid, c, r in _hubs_with_radii(dm, cfg)))
    c11, r11 = hubs["stripmall_uss#11"]
    c17, r17 = hubs["stripmall_uss#17"]
    assert float(np.linalg.norm(c11 - c17)) >= r11 + r17


# ---- the general placement contract, on both demand classes ----

def test_min_hub_gap_is_respected_across_operators():
    cfg = SimConfig(region_size_m=(12000.0, 12000.0))
    dm = HubRadiusDemand(n_hubs_per_uss={"big": 4, "small": 10},
                         terminal_radius_m={"big": 180.0, "small": 90.0}, min_hub_gap_m=150.0)
    hubs = _hubs_with_radii(dm, cfg)
    for (i1, c1, r1), (i2, c2, r2) in combinations(hubs, 2):
        assert float(np.linalg.norm(c1 - c2)) >= r1 + r2 + 150.0 - 1e-6, f"{i1}/{i2} closer than the gap"


def test_voronoi_scatter_also_separated():
    cfg = SimConfig(region_size_m=(8000.0, 8000.0))                 # terminal_radius_m defaults to 90 m
    hubs = _hubs_with_radii(HubVoronoiDemand(), cfg)
    slack, i, j = _worst_overlap(hubs)
    assert slack >= 0.0, f"{i} and {j} overlap"


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
