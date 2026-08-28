"""Incremental hex-occupancy service: it must stay byte-identical to a from-scratch rasterization
(the property that lets A* trust the maintained cache instead of rebuilding every plan)."""

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar.occupancy import HexOccupancyService

CFG = SimConfig(planner="astar", region_size_m=(3000.0, 3000.0), lam_per_hour=400.0,
                horizon_s=300.0, seed=3)
R = hg.circumradius(CFG)
INF_B = CFG.corridor_width_m / 2.0 + R
INF_P = CFG.effective_hover_radius_m + R


def _batch(volumes, infl):
    out = set()
    for v in volumes:
        out.update(hg.rasterize_volume(v, CFG, R, infl=infl))
    return out


def _flatten(buckets):
    return {(q, r, L, s) for s, cells in buckets.items() for (q, r, L) in cells}


def _accepted_volumes():
    from freespace_sim.sim import run
    res = run(CFG)
    return [(i.request.flight_id, i.volumes) for i in res.accepted]


def test_incremental_add_matches_batch_rebuild():
    """After each committed flight, the service equals a from-scratch dual rasterization."""
    svc = HexOccupancyService(CFG)
    seen = []
    flights = _accepted_volumes()
    assert flights
    for fid, vols in flights:
        svc.on_commit(fid, vols)
        seen.extend(vols)
        assert _flatten(svc.blocked) == _batch(seen, INF_B)
        assert _flatten(svc.pad) == _batch(seen, INF_P)


def test_evict_drops_past_keeps_future():
    """Eviction removes exactly the steps below the watermark and nothing at or above it."""
    svc = HexOccupancyService(CFG)
    seen = []
    for fid, vols in _accepted_volumes():
        svc.on_commit(fid, vols)
        seen.extend(vols)
    full_b = _batch(seen, INF_B)
    steps = sorted({s for (_, _, _, s) in full_b})
    watermark = steps[len(steps) // 2]
    svc.evict_before(watermark)
    assert _flatten(svc.blocked) == {(q, r, L, s) for (q, r, L, s) in full_b if s >= watermark}
    assert min(svc.blocked) >= watermark
    # eviction is monotonic — an earlier watermark is a no-op
    before = _flatten(svc.blocked)
    svc.evict_before(watermark - 5)
    assert _flatten(svc.blocked) == before


# --- shared terminal columns — per-cell hub-id set for the cruise own-hub exemption ------------
# (pad capacity is NOT here anymore — it's gated temporally by TerminalCapacity; see test_terminal_capacity)

from freespace_sim.volumes import corridor_segment_volume, hover_reservation   # noqa: E402


def _hub_cell_and_step(svc):
    """A (cell, step) squarely inside whatever terminal column(s) have been added."""
    s = sorted(svc.term_cells)[len(svc.term_cells) // 2]
    cell = next(iter(svc.term_cells[s]))
    return cell, s


def test_terminal_column_stays_out_of_binary_maps():
    # a tagged column is recorded, not walled: blocked/pad (the binary obstacle maps) stay empty
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))
    assert svc.blocked == {} and svc.pad == {}
    assert svc.term_cells                                   # it landed in the per-cell hub set instead


def test_own_hub_column_transparent_foreign_blocks():
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))
    (q, r, L), s = _hub_cell_and_step(svc)
    assert svc.is_blocked(q, r, L, s)                       # default own=∅ → walls off cruise
    assert not svc.is_blocked(q, r, L, s, own={"H"})       # the hub's own flights pass through
    assert svc.is_blocked(q, r, L, s, own={"other"})       # a different hub still sees a wall


def test_evict_drops_terminal_cells_in_lockstep():
    # the per-cell hub set must be time-evicted like blocked/pad, or stale cells would wrongly mark
    # cruise obstacles (and leak memory) — eviction loops over term_cells too
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1500.0, 1500.0, 0.0), 80.0, CFG, terminal_id="H"))
    steps = sorted(svc.term_cells)
    assert steps                                              # the dwell occupies a band of steps
    watermark = steps[len(steps) // 2]
    svc.evict_before(watermark)
    assert svc.term_cells and min(svc.term_cells) >= watermark
    svc.reset()
    assert svc.term_cells == {}                               # reset clears term_cells too


def test_publish_hook_feeds_service_on_commit():
    """Subscribing to the ledger keeps the service in lockstep with commits (the Option-A wiring)."""
    led = ReservationLedger(CFG)
    svc = HexOccupancyService(CFG)
    led.subscribe(svc.on_commit)
    seen = []
    for fid, vols in _accepted_volumes():
        led.commit(fid, vols)            # fires the publish hook -> svc.on_commit
        seen.extend(vols)
    assert _flatten(svc.blocked) == _batch(seen, INF_B)
    assert _flatten(svc.pad) == _batch(seen, INF_P)
    assert svc.n_added == led.n_volumes


# --- multi-altitude: per-level keying -----------------------------------------------------------

def test_per_level_keying_separates_levels():
    """A corridor at level 0 occupies (q,r,0) cells only; the same hex at level 1 is clear."""
    from freespace_sim.types import vec
    svc = HexOccupancyService(CFG)
    z0 = CFG.level_z(0)
    svc.add_volume(corridor_segment_volume(vec(500, 0, z0), 0.0, vec(620, 0, z0), CFG.dt_s, CFG))
    s = next(iter(svc.blocked))
    assert {L for (_, _, L) in svc.blocked[s]} == {0}        # corridor lands on its own level only
    q, r, L = next(iter(svc.blocked[s]))
    assert svc.is_blocked(q, r, 0, s)                        # blocked at level 0
    assert not svc.is_blocked(q, r, 1, s)                    # clear one level up


def test_terminal_column_recorded_at_all_levels():
    """A [ground, ceiling] tagged column records its hub at every in-band level."""
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.volumes import Volume4D
    svc = HexOccupancyService(CFG)
    col = Volume4D(CylinderSpec(1000.0, 1000.0, 90.0, CFG.ground_level_m, CFG.airspace_ceiling_m),
                   0.0, 60.0, terminal_id="H")
    svc.add_volume(col)
    s = next(iter(svc.term_cells))
    assert {L for (_, _, L) in svc.term_cells[s]} == {0, 1, 2}
    q, r, L = next(iter(svc.term_cells[s]))
    assert svc.is_blocked(q, r, L, s)                        # foreign cruise walled at every level
    assert not svc.is_blocked(q, r, L, s, own={"H"})        # the hub's own flights pass through


def test_pad_clear_blocked_by_corridor_at_any_level():
    """The pad (full-tube column) is blocked by a committed corridor at ANY flight level."""
    from freespace_sim.types import vec
    svc = HexOccupancyService(CFG)
    z1 = CFG.level_z(1)
    svc.add_volume(corridor_segment_volume(vec(1000, 0, z1), 0.0, vec(1120, 0, z1), CFG.dt_s, CFG))
    s = next(iter(svc.pad))
    q, r, L = next(iter(svc.pad[s]))
    assert L == 1                                            # the corridor sits at level 1
    assert not svc.pad_clear(q, r, s, 0)                     # but the pad's column spans all levels


# ------------------------------------------------- the `blocked` map's deferred build (area 2)

@pytest.mark.parametrize("track_removal", [False, True])
def test_enable_blocked_matches_a_map_maintained_all_along(track_removal):
    """A service built with ``maintain_blocked=False`` and armed later must hold EXACTLY the map a
    service that maintained it throughout holds.

    Three things a from-scratch rebuild loop would get wrong, and which routing through
    ``add_volume`` inherits for free: the committing flight's own-column skip (so its 90 m terminal
    interior stays passable), the refcount-dict-vs-set branch on ``track_removal`` (a set where
    ``on_release`` expects counts raises ``TypeError`` on the next destroy), and the
    ``evicted_before`` clamp. Parameterised over both container shapes for the second of those.

    ``pad`` is asserted too: ``enable_blocked`` passes ``_pad=False``, so a slip there would
    double-count the wider footprint rather than leave it alone."""
    flights = _accepted_volumes()
    assert flights, "no committed flights — the comparison would be vacuous"

    eager = HexOccupancyService(CFG, track_removal=track_removal, maintain_blocked=True)
    lazy = HexOccupancyService(CFG, track_removal=track_removal, maintain_blocked=False)
    led = ReservationLedger(CFG)
    for fid, vols in flights:
        eager.on_commit(fid, vols)
        lazy.on_commit(fid, vols)
        led.commit(fid, vols)

    assert not lazy.blocked, "maintain_blocked=False still wrote the map"
    assert _flatten(lazy.pad) == _flatten(eager.pad), "pad must be unaffected either way"
    lazy.enable_blocked(led)
    assert _flatten(lazy.blocked) == _flatten(eager.blocked)
    assert lazy.blocked == eager.blocked, "refcounts/sets differ, not just membership"
    assert _flatten(lazy.pad) == _flatten(eager.pad), "enable_blocked touched pad"
    assert _flatten(eager.blocked), "the map is empty — a rebuild of nothing would pass this"


def test_enable_blocked_leaves_release_reversible():
    """The ordering that could desync the two maps: commit while the map is OFF, arm, then release.
    ``on_release`` guards on the CURRENT flag, and ``enable_blocked`` rebuilds from the ledger rather
    than from rows, so the decrement finds the entry the rebuild put there."""
    flights = _accepted_volumes()
    svc = HexOccupancyService(CFG, track_removal=True, maintain_blocked=False)
    led = ReservationLedger(CFG)
    for fid, vols in flights:
        svc.on_commit(fid, vols)
        led.commit(fid, vols)
    svc.enable_blocked(led)
    assert svc.blocked
    for fid, vols in flights:
        svc.on_release(fid, vols)          # would KeyError/TypeError on a mismatched rebuild
    assert not svc.blocked and not svc.pad, "release did not fully reverse the armed map"


def test_is_blocked_refuses_an_unmaintained_map():
    """Answering from a map nobody is maintaining would make the reference planner — the oracle every
    compiled-path parity gate compares against — silently wrong. Raise instead."""
    svc = HexOccupancyService(CFG, maintain_blocked=False)
    with pytest.raises(RuntimeError, match="enable_blocked"):
        svc.is_blocked(0, 0, 0, 0)
