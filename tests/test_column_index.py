"""Foreign-transit index for ``TerminalCapacity.column_clear``.

``column_clear`` no longer scans the ledger every step; it serves an O(log) lookup from a per-hub
index of the time intervals a foreign volume sits over the column, built lazily/incrementally from the
ledger. The headline guarantee is **behavior parity**: for every (hub, t0), the index answer must equal
the old ``not ledger.any_conflict([column at t0])`` exactly — asserted here over a real committed ledger.
Plus unit tests for the one piece of genuinely new logic (the sorted-interval merge + overlap query).
"""
import numpy as np
import pytest

from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.planner.terminal_capacity import TerminalCapacity, _merge_intervals, _overlaps
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.types import as_terminal
from freespace_sim.uss import USS
from freespace_sim.volumes import hover_reservation, terminal_radius


# ---- unit: the sorted-interval helpers (the only genuinely new logic) ----

def test_merge_intervals():
    assert _merge_intervals([]) == []
    assert _merge_intervals([(5, 10)]) == [(5, 10)]
    assert _merge_intervals([(1, 3), (2, 5)]) == [(1, 5)]            # overlap
    assert _merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]            # touching ⇒ coalesce
    assert _merge_intervals([(5, 8), (1, 3)]) == [(1, 3), (5, 8)]    # disjoint + unsorted input
    assert _merge_intervals([(1, 10), (3, 5)]) == [(1, 10)]          # fully contained
    assert _merge_intervals([(1, 2), (5, 6), (2, 5)]) == [(1, 6)]    # transitive chain merge


def test_overlaps():
    ivs = [(10, 20), (30, 40)]                                       # sorted, disjoint
    assert not _overlaps([], 5, 15)
    assert not _overlaps(None, 5, 15)                                # column_clear passes .get(tid) → None
    assert not _overlaps(ivs, 0, 5)                                  # before all
    assert not _overlaps(ivs, 22, 28)                                # in the gap
    assert not _overlaps(ivs, 45, 50)                                # after all
    assert _overlaps(ivs, 15, 18)                                    # inside first
    assert _overlaps(ivs, 5, 12)                                     # straddles first's start
    assert _overlaps(ivs, 18, 35)                                    # spans the gap, hits both
    assert _overlaps(ivs, 19, 25)                                    # positive overlap into first (19,20)
    assert not _overlaps(ivs, 20, 25)                                # STRICT: only touches first's end ⇒ no
    assert not _overlaps(ivs, 25, 30)                                # STRICT: only touches second's start ⇒ no


# ---- parity: index == the any_conflict it replaces, over a real committed ledger ----

@pytest.mark.slow
def test_column_index_matches_any_conflict():
    """Commit a dense terminal scenario (foreign flights cross hubs ⇒ real intrusions), then assert the
    index-backed column_clear equals the any_conflict reference at a grid of times for every hub."""
    spec = with_overrides(
        get_scenario("dallas_hub_2uss_large"), lam_per_hour=8000.0, horizon_s=300.0, seed=0,
        demand_overrides={"pads_per_hub": {"walmart_uss": 20, "stripmall_uss": 8}, "radius_m": 6000.0})
    cfg = spec.config()
    reqs = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar")
    usses = {u: USS(u, dss, cfg, astar) for u in sc.uss_ids}
    for ev in sc.events[:600]:
        usses[ev.request.uss_id].handle_request(ev.request)

    terms: dict = {}                                                 # tid -> (Terminal, a representative center)
    for ev in sc.events[:600]:
        rq = ev.request
        if rq.origin_terminal is not None:
            terms.setdefault(rq.origin_terminal.id, (rq.origin_terminal, np.asarray(rq.origin, float)))
        if rq.dest_terminal is not None:
            terms.setdefault(rq.dest_terminal.id, (rq.dest_terminal, np.asarray(rq.dest, float)))
    assert len(terms) >= 5

    tcap = TerminalCapacity(cfg, led)                                # FRESH: index built lazily from the ledger
    grid = list(np.arange(0.0, cfg.horizon_s, 8.0))                  # deterministic sweep (hits any intrusion)
    checked = blocked = 0
    for term, center in terms.values():
        tid = as_terminal(term).id
        r = terminal_radius(as_terminal(term), cfg)
        for t0 in grid:
            got = tcap.column_clear(term, center, float(t0))
            ref = not led.any_conflict(
                [hover_reservation(center, float(t0), cfg, terminal_id=tid, radius=r)])
            assert got == ref, f"hub {tid} t0={t0:.0f}: index={got} any_conflict={ref}"
            checked += 1
            blocked += (not got)
    assert checked > 200
    assert blocked > 0, "no foreign-transit intrusions exercised — raise density so the test is meaningful"


@pytest.mark.slow
def test_column_index_stable_across_repeated_queries():
    """The lazy top-up must be idempotent: re-querying the same (hub, t0) never changes the answer, and
    a fresh tcap agrees with one that has already been queried many times (no accumulation bug)."""
    spec = with_overrides(
        get_scenario("dallas_hub_2uss_large"), lam_per_hour=4000.0, horizon_s=200.0, seed=1,
        demand_overrides={"pads_per_hub": {"walmart_uss": 20, "stripmall_uss": 8}, "radius_m": 6000.0})
    cfg = spec.config()
    reqs = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar")
    usses = {u: USS(u, dss, cfg, astar) for u in sc.uss_ids}
    for ev in sc.events[:300]:
        usses[ev.request.uss_id].handle_request(ev.request)
    rq = next(ev.request for ev in sc.events[:300] if ev.request.origin_terminal is not None)
    term, center = rq.origin_terminal, np.asarray(rq.origin, float)
    warm = TerminalCapacity(cfg, led)
    for t0 in np.arange(0.0, cfg.horizon_s, 4.0):                    # exercise the top-up + cache repeatedly
        warm.column_clear(term, center, float(t0))
    fresh = TerminalCapacity(cfg, led)
    for t0 in np.arange(0.0, cfg.horizon_s, 4.0):
        a = warm.column_clear(term, center, float(t0))
        b = fresh.column_clear(term, center, float(t0))             # cold index, same ledger
        assert a == b, f"warmed vs fresh disagree at t0={t0:.0f}: {a} vs {b}"
