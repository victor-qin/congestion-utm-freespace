"""Incremental hex-occupancy service: it must stay byte-identical to a from-scratch rasterization
(the property that lets A* trust the maintained cache instead of rebuilding every plan)."""

from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.occupancy import HexOccupancyService

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
    return {(q, r, s) for s, cells in buckets.items() for (q, r) in cells}


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
    steps = sorted({s for (_, _, s) in full_b})
    watermark = steps[len(steps) // 2]
    svc.evict_before(watermark)
    assert _flatten(svc.blocked) == {(q, r, s) for (q, r, s) in full_b if s >= watermark}
    assert min(svc.blocked) >= watermark
    # eviction is monotonic — an earlier watermark is a no-op
    before = _flatten(svc.blocked)
    svc.evict_before(watermark - 5)
    assert _flatten(svc.blocked) == before


# --- Phase B: shared terminal columns — per-hub dwell counter + capacity gate -----------------

from freespace_sim.volumes import hover_reservation   # noqa: E402


def _hub_cell_and_step(svc):
    """A (cell, step) squarely inside whatever terminal column(s) have been added."""
    s = sorted(svc.term_cells)[len(svc.term_cells) // 2]
    cell = next(iter(svc.term_cells[s]))
    return cell, s


def test_terminal_column_stays_out_of_binary_maps():
    # a tagged column is counted, not walled: blocked/pad (the binary obstacle maps) stay empty
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))
    assert svc.blocked == {} and svc.pad == {}
    assert svc.term_cells                                   # it landed in the per-hub counter instead


def test_own_hub_column_transparent_foreign_blocks():
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))
    (q, r), s = _hub_cell_and_step(svc)
    assert svc.is_blocked(q, r, s)                          # default own=∅ → walls off cruise
    assert not svc.is_blocked(q, r, s, own={"H"})          # the hub's own flights pass through
    assert svc.is_blocked(q, r, s, own={"other"})          # a different hub still sees a wall


def test_capacity_gate_counts_both_takeoff_and_landing_dwells():
    # a departure dwell AND an arrival dwell at the SAME hub both occupy a pad (Phase B counts both)
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))   # takeoff
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))   # landing
    (q, r), s = _hub_cell_and_step(svc)
    assert svc.term_cells[s][(q, r)]["H"] == 2             # two concurrent dwells counted
    assert not svc.pad_clear(q, r, s, 0, terminal_id="H", capacity=2)   # both pads busy → no slot
    assert svc.pad_clear(q, r, s, 0, terminal_id="H", capacity=3)       # room for a third


def test_capacity_one_terminal_matches_binary_pad():
    # capacity 1 ⟺ the old exclusive pad: one dwell already there ⇒ no slot
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1000.0, 1000.0, 0.0), 0.0, CFG, terminal_id="H"))
    (q, r), s = _hub_cell_and_step(svc)
    assert not svc.pad_clear(q, r, s, 0, terminal_id="H", capacity=1)


def test_evict_drops_terminal_counter_in_lockstep():
    # the per-hub dwell counter must be time-evicted like blocked/pad, or stale dwells would wrongly
    # gate future launches (and leak memory) — eviction loops over term_cells too
    svc = HexOccupancyService(CFG)
    svc.add_volume(hover_reservation((1500.0, 1500.0, 0.0), 80.0, CFG, terminal_id="H"))
    steps = sorted(svc.term_cells)
    assert steps                                              # the dwell occupies a band of steps
    watermark = steps[len(steps) // 2]
    svc.evict_before(watermark)
    assert svc.term_cells and min(svc.term_cells) >= watermark
    svc.reset()
    assert svc.term_cells == {}                               # reset clears the counter too


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
