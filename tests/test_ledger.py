from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.types import vec
from freespace_sim.volumes import build_corridor

CFG = SimConfig()


def _corridor(x0=0.0, t0=0.0):
    seg = CFG.corridor_segment_len_m
    cl = [
        (vec(x0 + 0 * seg, 0, CFG.cruise_level_m), t0 + 0 * CFG.dt_s),
        (vec(x0 + 1 * seg, 0, CFG.cruise_level_m), t0 + 1 * CFG.dt_s),
        (vec(x0 + 2 * seg, 0, CFG.cruise_level_m), t0 + 2 * CFG.dt_s),
    ]
    return build_corridor(cl, CFG)


def test_commit_then_identical_query_conflicts():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor())
    incoming = _corridor()
    assert led.any_conflict(incoming)
    assert led.conflicting_flights(incoming) == {1}


def test_time_shifted_query_is_clear():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor(t0=0.0))
    # same path but 1000 s later → no time overlap
    assert not led.any_conflict(_corridor(t0=1000.0))


def test_spatially_far_query_is_clear():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor(x0=0.0))
    assert not led.any_conflict(_corridor(x0=5000.0))


def test_release_clears_conflicts():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor())
    led.commit(2, _corridor(x0=5000.0))
    assert led.any_conflict(_corridor())
    led.release(1)
    assert not led.any_conflict(_corridor())
    assert led.any_conflict(_corridor(x0=5000.0))   # flight 2 still committed


def test_conflicts_reports_committed_volumes():
    led = ReservationLedger(CFG)
    led.commit(7, _corridor())
    hits = led.conflicts(_corridor())
    assert hits and all(fid == 7 for fid, _ in hits)
    assert led.n_volumes == 2
