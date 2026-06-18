from freespace_sim.config import SimConfig
from freespace_sim.planner.straight import build_reservation
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec
from freespace_sim.verify import find_interflight_conflict

CFG = SimConfig()


def _intent(fid, t_dep):
    req = FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)
    vols, cl = build_reservation(req.origin, req.dest, t_dep, CFG)
    return OperationalIntent(req, IntentStatus.ACCEPTED, volumes=vols, centerline=cl)


def test_clean_when_separated_in_time():
    a = _intent(0, 0.0)
    b = _intent(1, 1000.0)
    assert find_interflight_conflict([a, b], CFG) is None


def test_detects_injected_inter_flight_overlap():
    a = _intent(0, 0.0)
    b = _intent(1, 0.0)   # identical reservation as a different flight → must be caught
    assert find_interflight_conflict([a, b], CFG) == (1, 0)
