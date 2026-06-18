from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner.straight import build_reservation
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec

CFG = SimConfig()


def _intent(fid):
    req = FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)
    vols, cl = build_reservation(req.origin, req.dest, 0.0, CFG)
    return OperationalIntent(req, IntentStatus.ACCEPTED, volumes=vols, centerline=cl)


def test_fcfs_commits_first_then_rejects_conflicting():
    led = ReservationLedger(CFG)
    mech = FCFSMechanism()
    a = _intent(1)
    assert mech.commit(led, a) is True
    assert led.n_volumes > 0

    b = _intent(2)   # identical path & time → conflicts with committed a
    assert mech.commit(led, b) is False
    assert b.status is IntentStatus.REJECTED


def test_mechanism_ignores_rejected_intent():
    led = ReservationLedger(CFG)
    rejected = OperationalIntent(
        FlightRequest(3, vec(0, 0, 0), vec(1, 0, 0), 0.0), IntentStatus.REJECTED
    )
    assert FCFSMechanism().commit(led, rejected) is False
    assert led.n_volumes == 0
