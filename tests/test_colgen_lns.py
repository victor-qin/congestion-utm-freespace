"""LNS coverage, rollback, input validation, and deadline regressions."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from freespace_sim.planner.colgen.master import RestrictedMaster
from freespace_sim.planner.colgen.network import RowIndex, RowKey
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.translate import Column


def _column(fid, slot, cost):
    return Column(fid, slot, 0, None, None, ((0, 0), (1, 0)), cost,
                  frozenset({RowKey.cell((slot, 0), 0, 0)}))


def _example():
    master = RestrictedMaster([1, 2, 3], RowIndex(), ColGenParams(solver="highs"))
    columns = [_column(1, 0, 2), _column(1, 1, 5), _column(2, 2, 10),
               _column(2, 0, 2), _column(3, 3, 4)]
    for column in columns:
        master.add_column(column)
    return master, columns, {1: columns[0], 2: columns[2], 3: columns[4]}


def test_lns_accepts_temporary_sacrifice_that_unlocks_better_complete_trial():
    master, columns, incumbent = _example()
    result = master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(1),
                                  incumbent, destroy=2, n_tries=1)
    assert result == {1: columns[1], 2: columns[3], 3: columns[4]}
    assert sum(c.delay_s for c in result.values()) == 11
    assert master.is_claim_feasible(result)
    assert master.last_round_stats["n_unique_trial_solutions"] == 1


@pytest.mark.parametrize("seed", range(12))
def test_lns_repeated_rejected_trials_restore_claims_and_preserve_coverage(seed):
    rng = np.random.default_rng(seed)
    master = RestrictedMaster(range(8), RowIndex(), ColGenParams(solver="highs"))
    incumbent = {}
    for fid in range(8):
        incumbent[fid] = _column(fid, fid, float(rng.integers(1, 10)))
        master.add_column(incumbent[fid])
        for slot in rng.choice(12, size=5, replace=False):
            master.add_column(_column(fid, int(slot), float(rng.integers(0, 20))))
    x = rng.random(len(master.columns))
    result = master.lns_heuristic(x, rng, incumbent, destroy=3, n_tries=40)
    assert set(result) == set(incumbent)
    assert master.is_claim_feasible(result)
    assert master.objective_of(result) >= master.objective_of(incumbent) - 1e-9
    assert master.is_claim_feasible(incumbent)


@pytest.mark.parametrize("invalid", ["outside", "key", "conflict"])
def test_lns_rejects_invalid_incumbent_instead_of_losing_coverage(invalid):
    master, columns, incumbent = _example()
    if invalid == "outside":
        incumbent[1] = replace(columns[0], delay_s=123)
    elif invalid == "key":
        incumbent[8] = incumbent.pop(1)
    else:
        incumbent[2] = columns[3]
    with pytest.raises(ValueError):
        master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(1), incumbent, 2)


def test_lns_expired_deadline_retains_incumbent_and_reports_no_trials():
    master, _, incumbent = _example()
    result = master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(1),
                                  incumbent, 2, deadline=0)
    assert result == incumbent
    assert master.last_round_stats["deadline_reached"]
    assert master.last_round_stats["try_objectives"] == ()
    assert master.last_round_stats["n_unique_trial_solutions"] == 0


def test_gurobi_ip_trajectory_records_current_candidate_instead_of_stale_best():
    pytest.importorskip("gurobipy")
    master = RestrictedMaster([1], RowIndex(), ColGenParams(solver="gurobi"))
    slow, fast = _column(1, 0, 6), _column(1, 1, 2)
    master.add_column(slow)
    master.add_column(fast)
    result = master.solve_ip({1: slow}, budget_s=3)
    assert result == {1: fast}
    trajectory = master.last_ip_trajectory
    assert trajectory
    objectives = [entry[1] for entry in trajectory]
    assert all(np.isfinite(value) and value >= master.objective_of({1: slow})
               for value in objectives)
    assert objectives == sorted(objectives)
    assert objectives[-1] == pytest.approx(master.objective_of(result))


def test_lns_empty_incumbent_clears_previous_telemetry():
    master, _, incumbent = _example()
    master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(1), incumbent, 2)
    assert master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(1), {}, 2) == {}
    assert master.last_round_stats["try_objectives"] == ()


@pytest.mark.parametrize("backend", ["highs", "gurobi"])
def test_joint_repair_exchanges_mutually_blocked_reservations(backend):
    if backend == "gurobi":
        pytest.importorskip("gurobipy")
    master = RestrictedMaster([1, 2], RowIndex(), ColGenParams(
        solver=backend, lns_joint_time_limit_s=5,
    ))
    columns = [_column(1, 0, 5), _column(1, 1, 1),
               _column(2, 1, 5), _column(2, 0, 1)]
    for column in columns:
        master.add_column(column)
    incumbent = {1: columns[0], 2: columns[2]}
    result = master.lns_heuristic([0, 1, 0, 1], np.random.default_rng(0),
                                  incumbent, destroy=2)
    assert result == {1: columns[1], 2: columns[3]}
    assert master.is_claim_feasible(result)
    assert master.last_round_stats["n_swapped_sequential"] == 0
    assert master.last_round_stats["n_swapped_best"] == 2
    assert master.last_round_stats["joint_repair"]["improved"]


def test_joint_repair_keeps_flights_outside_the_neighborhood_fixed():
    master = RestrictedMaster([1, 2], RowIndex(), ColGenParams(solver="highs"))
    columns = [_column(1, 0, 5), _column(1, 1, 1),
               _column(2, 1, 5), _column(2, 0, 1)]
    for column in columns:
        master.add_column(column)
    incumbent = {1: columns[0], 2: columns[2]}
    result = master.lns_heuristic([0, 1, 0, 1], np.random.default_rng(0),
                                  incumbent, destroy=1, n_tries=1)
    assert result == incumbent


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan")])
def test_joint_repair_budget_must_be_finite_and_nonnegative(budget):
    with pytest.raises(ValueError, match="lns_joint_time_limit_s"):
        ColGenParams(lns_joint_time_limit_s=budget)


@pytest.mark.parametrize("backend", ["highs", "gurobi"])
@pytest.mark.parametrize("fixed_load", [0, 1])
def test_joint_repair_matches_enumeration_with_fixed_loads_and_duplicate_rows(backend, fixed_load):
    if backend == "gurobi":
        pytest.importorskip("gurobipy")
    from freespace_sim.planner.colgen.lns import repair_neighborhood

    row_a, row_b = RowKey.term("pad", 0), RowKey.term("pad", 1)
    master = RestrictedMaster(range(3), RowIndex({"pad": 2}),
                              ColGenParams(solver=backend),
                              fixed_loads={row_a: fixed_load, row_b: fixed_load})
    alternatives, incumbent = {}, {}
    for fid in range(3):
        original = _column(fid, fid, 10 + fid)
        cheap = replace(original, departure_step=9, delay_s=float(fid),
                        claims=frozenset({row_a, row_b}))
        alternatives[fid] = [original, cheap]
        incumbent[fid] = original
        master.add_column(original)
        master.add_column(cheap)
    feasible = [dict(enumerate(columns)) for columns in product(*alternatives.values())
                if master.is_claim_feasible(columns)]
    expected = min(sum(c.delay_s for c in selection.values()) for selection in feasible)
    import time

    result, info = repair_neighborhood(master, np.ones(6), incumbent,
                                       master.claim_loads(incumbent), 3, time.monotonic() + 5)
    assert sum(c.delay_s for c in result.values()) == expected
    assert master.is_claim_feasible(result)
    assert info["rows"] == 1  # Identical incidence/capacity rows safely collapse.


def test_lns_skips_repair_when_incumbent_already_meets_pool_bound():
    master, _, incumbent = _example()
    result = master.lns_heuristic([1, 0, 1, 0, 1], np.random.default_rng(0),
                                  incumbent, destroy=2,
                                  lp_objective=master.objective_of(incumbent))
    assert result == incumbent
    assert master.last_round_stats["pool_gap_satisfied"]
    assert master.last_round_stats["try_objectives"] == ()
    assert master.last_round_stats["joint_repair"] == {}


def test_lns_does_not_repeat_identical_rejected_full_neighborhood():
    master = RestrictedMaster([1], RowIndex(), ColGenParams(
        solver="highs", lns_joint_time_limit_s=0,
    ))
    original, worse = _column(1, 0, 1), _column(1, 1, 2)
    master.add_column(original)
    master.add_column(worse)
    result = master.lns_heuristic([0, 1], np.random.default_rng(0),
                                  {1: original}, destroy=1, n_tries=20)
    assert result == {1: original}
    assert len(master.last_round_stats["try_objectives"]) == 1
    assert master.last_round_stats["exhausted_full_neighborhood"]


def test_lns_full_neighborhood_continues_after_an_improvement():
    master, columns, incumbent = _example()
    master.params = replace(master.params, lns_joint_time_limit_s=0)
    result = master.lns_heuristic([0, 1, 0, 1, 1], np.random.default_rng(0),
                                  incumbent, destroy=3, n_tries=20)
    assert result == {1: columns[1], 2: columns[3], 3: columns[4]}
    assert len(master.last_round_stats["try_objectives"]) == 2
