"""The TIGHTNESS direction of colgen's capacity model — the one nothing else tests.

`windows._cross_check_conflicts` ties colgen's rows to the ledger's geometry in exactly one
direction, and says so: *"a geometric conflict must imply an intersecting row claim"*.  Its
scan opens with ``if not volumes_conflict(first, shifted): continue`` — every NON-conflicting
pair is discarded, so the converse (an intersecting row claim implies a real conflict) has
never been checked anywhere in the suite.

That asymmetry is deliberate and safe: a cover that over-claims refuses legal schedules but
never files a conflicting one.  It is not free, though.  Measured on `density_faa_wing_zipline`
at 600 flights, A*'s accepted schedule — every pair of which the ledger cleared — hits 89 of
colgen's cap-1 cell rows across 10 flight pairs, and **none of those 10 pairs conflicts** under
`conflict.volumes_conflict`: their minimum separations are 208–624 m against a 103.92 m floor.
Attributing all 178 over-cap claims to the function that emitted them gives origin dwell 44%,
destination dwell 40%, en-route visit 14%.  See GitHub issue #101.

These tests exist so that over-constraint is *measured and bounded* rather than invisible.
Each pairs a soundness assertion (must hold, and does) with a tightness characterisation whose
bound is the currently-measured over-reach.  A change that widens the gap fails; the fix for
#101 should let the bounds be tightened toward 1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.geometry import CylinderSpec
from freespace_sim.planner.colgen.windows import (
    _shift_volume,
    _template_arcs,
    derive_cell_window,
    endpoint_claim_cells,
    visit_rows,
)
from freespace_sim.planner.hexgrid import circumradius
from freespace_sim.volumes import Volume4D


def _cfg() -> SimConfig:
    return SimConfig()


def _hover(cfg: SimConfig, x: float) -> Volume4D:
    """One endpoint's hover cylinder, at the same time window as its twin."""
    return Volume4D(
        shape=CylinderSpec(
            cx=x, cy=0.0, radius=cfg.effective_hover_radius_m,
            z_lo=0.0, z_hi=cfg.flight_levels_m[-1],
        ),
        t_start=0.0,
        t_end=10.0 * cfg.dt_s,
    )


def _endpoint_claims_at(cfg: SimConfig, x: float) -> set:
    point = np.array([x, 0.0, 0.0], dtype=float)
    return set(map(tuple, endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)))


def _crossover(predicate, lo: float, hi: float) -> float:
    """Largest separation at which ``predicate`` still holds (it is monotone in distance)."""
    assert predicate(lo) and not predicate(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if predicate(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Endpoint dwell — 84% of the measured over-constraint
# --------------------------------------------------------------------------------------

def test_endpoint_claims_cover_every_real_cylinder_conflict():
    """SOUNDNESS: two endpoints that genuinely conflict must share a claimed cell.

    This is the direction the model depends on for its 0-denials guarantee, restated for
    endpoints (``_cross_check_conflicts`` covers arcs only). It must never regress.
    """
    cfg = _cfg()
    touching = 2.0 * cfg.effective_hover_radius_m
    for d in np.linspace(0.0, touching, 25):
        if not volumes_conflict(_hover(cfg, 0.0), _hover(cfg, float(d))):
            continue
        shared = _endpoint_claims_at(cfg, 0.0) & _endpoint_claims_at(cfg, float(d))
        assert shared, f"endpoints {d:.1f} m apart conflict but share no claimed cell"


def test_endpoint_claim_overlap_reaches_far_beyond_any_real_conflict():
    """TIGHTNESS: bound how far past a real conflict the endpoint claims keep colliding.

    Two hover cylinders conflict iff their centres are within ``2 * effective_hover_radius_m``
    (120 m shipped). Their CLAIM sets keep intersecting out to ~309 m, because
    ``endpoint_claim_cells`` inflates by ``hover + max(corridor_width, hover) + circumradius``
    = 189.28 m per endpoint — sized for a cylinder-vs-transit-BOX worst case, then applied
    uniformly at cap 1 to cylinder-vs-cylinder as well.

    Everything in the band (120 m, 309 m] is a pair the ledger accepts and colgen's LP refuses.
    The bound below is the measured ratio; #101 aims to drive it toward 1.0.
    """
    cfg = _cfg()
    real = _crossover(
        lambda d: volumes_conflict(_hover(cfg, 0.0), _hover(cfg, d)), 1.0, 2000.0)
    claimed = _crossover(
        lambda d: bool(_endpoint_claims_at(cfg, 0.0) & _endpoint_claims_at(cfg, d)), 1.0, 2000.0)

    assert real == pytest.approx(2.0 * cfg.effective_hover_radius_m, abs=1.0)
    assert claimed > real, "claims must at least cover real conflicts"
    # Measured 309.28 / 120.00 = 2.58x. A change that widens this refuses more legal schedules.
    assert claimed / real <= 2.60, (
        f"endpoint claim over-reach grew: real conflict ends at {real:.1f} m, claims still "
        f"collide at {claimed:.1f} m ({claimed / real:.2f}x)"
    )
    # And pin the arithmetic that produces it, so the cause stays visible if it changes.
    expected_claim_radius = (
        cfg.effective_hover_radius_m
        + max(cfg.corridor_width_m, cfg.effective_hover_radius_m)
        + circumradius(cfg)
    )
    assert expected_claim_radius == pytest.approx(189.28, abs=0.05)


def test_a_concrete_ledger_accepted_endpoint_pair_is_refused_by_the_row_model():
    """The defect in one assertion: 200 m apart, ledger-clear, colgen-blocked.

    200 m sits inside the measured (120, 309] band. This is the unit-scale twin of the
    600-flight finding, where the 10 contested pairs were 208-624 m apart with zero real
    conflicts.
    """
    cfg = _cfg()
    d = 200.0
    assert not volumes_conflict(_hover(cfg, 0.0), _hover(cfg, d)), "no real conflict at 200 m"
    shared = _endpoint_claims_at(cfg, 0.0) & _endpoint_claims_at(cfg, d)
    assert shared, "expected the row model to (over-)collide here"
    # Cap is 1 per cell row, so a shared claimed cell at a shared step is a refusal.
    assert len(shared) >= 1


# --------------------------------------------------------------------------------------
# En-route arcs — the remaining 14%
# --------------------------------------------------------------------------------------

def test_arc_claims_are_sound_but_not_tight():
    """Run ``_cross_check_conflicts``' own scan in BOTH directions and bound the slack.

    Soundness (conflict => claims intersect) is asserted, matching the shipped guard.
    Tightness (claims intersect => conflict) is only measured: the shipped model books a
    conflict for a share of pairs that do not actually conflict, and this pins that share.
    """
    cfg = _cfg()
    offsets = derive_cell_window(cfg)
    templates = _template_arcs(cfg, circumradius(cfg))
    claims = [
        frozenset(row for visit in visits for row in visit_rows(visit, offsets))
        for _volume, visits in templates
    ]
    max_extent = max(max(abs(v.t_start), abs(v.t_end)) for v, _ in templates)
    shift_limit = 2 * int(np.ceil(max_extent / cfg.dt_s)) + 2

    sound_failures = 0
    spurious = 0          # claims intersect, no real conflict
    blocked = 0           # claims intersect at all -- what colgen refuses
    considered = 0
    for i, (first, _fv) in enumerate(templates):
        for _j, (second, second_visits) in enumerate(templates):
            for shift in range(-shift_limit, shift_limit + 1):
                shifted = _shift_volume(second, shift, cfg.dt_s)
                shifted_claims = frozenset(
                    row for visit in second_visits for row in visit_rows(visit + shift, offsets))
                intersects = not claims[i].isdisjoint(shifted_claims)
                conflicts = volumes_conflict(first, shifted)
                considered += 1
                blocked += intersects
                sound_failures += conflicts and not intersects
                spurious += intersects and not conflicts

    assert sound_failures == 0, "the shipped coverage guarantee is broken"
    assert considered > 0 and blocked > 0

    # Measured on the shipped config: 8,228 pairs scanned, 2,772 real conflicts,
    # 1,408 spurious -> 17.1% of all pairs, and 33.7% of everything colgen blocks.
    # The second ratio is the one that matters: a third of en-route blocks are not conflicts.
    assert spurious / considered <= 0.18, (
        f"en-route claim over-reach grew: {spurious}/{considered} "
        f"({100 * spurious / considered:.1f}%) of template pairs collide without conflicting"
    )
    assert spurious / blocked <= 0.35, (
        f"spurious share of en-route blocks grew: {spurious}/{blocked} "
        f"({100 * spurious / blocked:.1f}%) of the pairs colgen refuses are not real conflicts"
    )
