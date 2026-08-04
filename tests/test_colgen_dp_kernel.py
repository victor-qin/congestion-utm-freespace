"""Parity and safety-valve contracts for the compiled colgen pricing DP.

The pure-Python ``pricing._best_column`` is the oracle.  The kernel is validated by
asserting it produces the **same reduced cost and the same column** -- not merely an
equally-good one: five committed assertions elsewhere pin the exact column, so the
tie-break is part of the contract, not an implementation detail.

Two parameters matter more than they look:

* ``time_buffer_s`` -- at the default the revisit-ban width and the dominance-state
  width are both 3, so a default-geometry sweep cannot tell them apart.  At
  ``time_buffer_s=0`` they are 1 and 2, and keying the state on the wrong one
  silently drops labels the reference keeps.
* mixed-sign duals -- ``DualView`` deliberately preserves negative duals, and the
  kernel's bound needs ``max_negative_credit`` to stay an upper bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.network import (
    RowKey,
    build_flight_graph,
    column_claims,
)
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView, price_flight
from freespace_sim.types import FlightRequest, vec

dp_kernel = pytest.importorskip(
    "freespace_sim.planner.colgen.dp_kernel", reason="requires Numba pricing kernel"
)


def _cfg(**overrides) -> SimConfig:
    values = {
        "planner": "colgen",
        "flight_levels_m": (100.0,),
        "airspace_ceiling_m": 125.0,
        "region_size_m": (20_000.0, 20_000.0),
        "terminal_airspace_always_active": True,
        "max_ground_delay_s": 48.0,
        "max_detour_factor": 10.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _point(cell, cfg):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _graph(cfg, origin=(0, 0), dest=(4, -1), slack=4, flight_id=1):
    params = ColGenParams(solver="highs", detour_slack_hops=slack)
    request = FlightRequest(flight_id, _point(origin, cfg), _point(dest, cfg), 0.0, 0.0)
    return build_flight_graph(request, cfg, (), params), params


def _random_duals(rng, cells, base_step, count, *, signed=True):
    duals: dict[RowKey, float] = {}
    for _ in range(count):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(base_step - 3, base_step + 30))
        key = RowKey.cell(cell, 0, step)
        value = float(rng.normal(3.0, 6.0)) if signed else float(rng.gamma(2.0, 3.0))
        duals[key] = duals.get(key, 0.0) + value
    return duals


def _price_both(monkeypatch, graph, duals, pi_f, cfg, params):
    """Price once with the kernel and once with it disabled, on identical inputs."""

    view = DualView(duals, cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected = price_flight(graph, view, pi_f, cfg, params, require_improving=False)
    # A fresh graph: the reference run populates answer-neutral caches, and the
    # compiled run must not be handed a warmed topology by accident.
    got = price_flight(graph, view, pi_f, cfg, params, require_improving=False)
    return expected, got


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_kernel_matches_reference_column_on_random_mixed_sign_duals(
    monkeypatch, time_buffer_s
):
    assert dp_kernel is not None, "kernel inactive -- this parity guard would be vacuous"
    rng = np.random.default_rng(20260803)
    compared = 0
    for trial in range(12):
        cfg = _cfg(
            time_buffer_s=time_buffer_s,
            max_ground_delay_s=float(rng.choice([0.0, 48.0, 120.0])),
        )
        graph, params = _graph(
            cfg,
            dest=(int(rng.integers(2, 6)), int(rng.integers(-3, 3))),
            slack=int(rng.integers(0, 5)),
            flight_id=trial + 1,
        )
        cells = sorted(graph.corridor_cells)
        duals = _random_duals(rng, cells, graph.base_step, int(rng.integers(0, 40)))
        pi_f = float(rng.normal(0.0, 50.0))

        (expected_rc, expected), (got_rc, got) = _price_both(
            monkeypatch, graph, duals, pi_f, cfg, params
        )
        if expected is None:
            assert got is None
            continue
        compared += 1
        assert got is not None
        assert got_rc == pytest.approx(expected_rc, abs=1e-8)
        # The column itself, not just its value: the tie-break is contractual.
        assert got.cell_path == expected.cell_path
        assert got.departure_step == expected.departure_step
        assert got.origin_lane_idx == expected.origin_lane_idx
        assert got.dest_lane_idx == expected.dest_lane_idx
    assert compared >= 6, f"only {compared} graphs produced a column; sweep is too weak"


def test_kernel_column_is_claim_feasible(monkeypatch):
    """A cheap column that cannot be filed is worse than useless."""

    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -2), slack=4)
    rng = np.random.default_rng(7)
    duals = _random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 25)
    _reduced_cost, column = price_flight(
        graph, DualView(duals, cfg), 0.0, cfg, params, require_improving=False
    )
    assert column is not None
    claims = column_claims(column, graph, cfg)
    assert claims == column.claims


def test_kernel_result_is_deterministic():
    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -1), slack=4)
    rng = np.random.default_rng(11)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 30), cfg)
    first = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    second = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_kernel_absent_falls_back_to_an_identical_answer(monkeypatch):
    cfg = _cfg(max_ground_delay_s=48.0)
    graph, params = _graph(cfg, dest=(4, 0), slack=3)
    rng = np.random.default_rng(3)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 20), cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        without = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    with_kernel = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert with_kernel[0] == pytest.approx(without[0], abs=1e-8)
    assert with_kernel[1] == without[1]


# ------------------------------------------------------------------- safety valves


def test_label_overflow_regrows_and_stays_exact(monkeypatch):
    """A pool too small to hold the search must grow and re-run, not degrade."""

    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -1), slack=4)
    rng = np.random.default_rng(5)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 25), cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected = price_flight(graph, view, 0.0, cfg, params, require_improving=False)

    original = dp_kernel.search_dag
    seen: list[int] = []

    def tiny(*args, **kwargs):
        kwargs["label_limit"] = 8  # far too small; forces the grow-and-re-run path
        result = original(*args, **kwargs)
        seen.append(result.regrow)
        return result

    monkeypatch.setattr(dp_kernel, "search_dag", tiny)
    got = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert seen and max(seen) > 0, "pool never regrew -- the adaptive path is untested"
    assert got[0] == pytest.approx(expected[0], abs=1e-8)
    assert got[1] == expected[1]


def test_kernel_failure_status_still_returns_the_reference_answer(monkeypatch):
    """A kernel that gives up must be invisible except in the clock."""

    cfg = _cfg(max_ground_delay_s=48.0)
    graph, params = _graph(cfg, dest=(4, -1), slack=3)
    rng = np.random.default_rng(9)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 18), cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected = price_flight(graph, view, 0.0, cfg, params, require_improving=False)

    def hash_full(*_args, **_kwargs):
        return dp_kernel.DagSearchResult(
            status=dp_kernel.FB_HASH_FULL, remaining_rc_upper_bound=float("inf")
        )

    monkeypatch.setattr(dp_kernel, "search_dag", hash_full)
    got = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert got[0] == pytest.approx(expected[0], abs=1e-8)
    assert got[1] == expected[1]


def test_cancelled_kernel_raises_pricing_timeout(monkeypatch):
    cfg = _cfg(max_ground_delay_s=48.0)
    graph, params = _graph(cfg, dest=(4, -1), slack=3)
    view = DualView({}, cfg)

    def cancelled(*_args, **_kwargs):
        return dp_kernel.DagSearchResult(
            status=dp_kernel.FB_CANCELLED, remaining_rc_upper_bound=float("inf")
        )

    monkeypatch.setattr(dp_kernel, "search_dag", cancelled)
    # Reach the compiled path with a live (non-expired) deadline so cancellation,
    # not the pre-check, is what raises.
    with pytest.raises(pricing.PricingTimeout):
        pricing._best_column(
            graph,
            view,
            0.0,
            cfg,
            params.M,
            frozenset(),
            seed=False,
            incumbent=None,
            deadline=__import__("time").monotonic() + 30.0,
        )


def test_status_names_cover_every_status():
    codes = {
        dp_kernel.OK,
        dp_kernel.NO_PATH,
        dp_kernel.FB_LABEL_OVERFLOW,
        dp_kernel.FB_HASH_FULL,
        dp_kernel.FB_CANCELLED,
    }
    assert codes <= set(dp_kernel.STATUS_NAMES)
    assert dp_kernel.warm_kernel() is True


# ------------------------------------------------------------------ packing layer


def test_topology_mirrors_the_object_api_arc_for_arc():
    """Element-wise, not set-wise: arc ORDER decides insertion-order ties."""

    cfg = _cfg()
    graph, _params = _graph(cfg, dest=(5, 0), slack=4)
    topology = dp_prepare.prepare_topology(graph, cfg)
    assert topology.unsupported_reason is None
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    assert cells == sorted(cells), "cell ids must be sorted; _path_cmp relies on it"
    for index, cell in enumerate(cells):
        lo = int(topology.arc_start[index])
        hi = int(topology.arc_start[index + 1])
        targets = [cells[int(t)] for t in topology.arc_target[lo:hi]]
        assert targets == list(graph.outgoing_neighbors(cell))
        for arc, target in zip(range(lo, hi), targets):
            assert int(topology.arc_roles[arc]) == graph.hop_role_mask(cell, target)


def test_reverse_remaining_is_a_real_admissible_path_length():
    cfg = _cfg()
    graph, _params = _graph(cfg, dest=(5, -1), slack=4)
    topology = dp_prepare.prepare_topology(graph, cfg)
    destinations = frozenset(pricing._destination_options(graph))
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    for index, cell in enumerate(cells):
        remaining = int(topology.rev_remaining[index])
        if remaining == dp_prepare.UNREACHABLE:
            continue
        # Admissible: never below the straight-line bound the reference uses.
        assert remaining >= pricing._distance_lower_bound(cell, destinations)
        # And real: a greedy downhill walk reaches a destination in exactly that many hops.
        node, steps = index, 0
        while int(topology.rev_remaining[node]) > 0:
            lo = int(topology.arc_start[node])
            hi = int(topology.arc_start[node + 1])
            node = min(
                (int(topology.arc_target[a]) for a in range(lo, hi)),
                key=lambda t: int(topology.rev_remaining[t]),
            )
            steps += 1
        assert steps == remaining
        assert topology.dest_mask[node]


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_prepared_duals_match_dualview_bitwise(time_buffer_s):
    cfg = _cfg(time_buffer_s=time_buffer_s)
    graph, _params = _graph(cfg, dest=(4, -1), slack=4)
    topology = dp_prepare.prepare_topology(graph, cfg)
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    rng = np.random.default_rng(17)
    for _trial in range(6):
        view = DualView(_random_duals(rng, cells, graph.base_step, 30), cfg)
        prepared = dp_prepare.prepare_duals(view, topology)
        for index, cell in enumerate(cells):
            for step in range(graph.base_step - 4, graph.base_step + 24):
                # Bitwise, not approximate: the kernel's score arithmetic branches
                # on a 1e-12 tie band, so a drifting window price could flip a tie.
                assert prepared.visit_cost(index, step) == view.visit_cost(cell, 0, step)


def test_paid_class_partitions_variants_by_origin_paid_rows():
    """Variants sharing a paid-row set must share a class, and no class may mix sets.

    The reference's dominance key holds ``origin_paid_rows`` itself, so two roots
    from different departure steps are allowed to merge downstream when they paid
    the same rows.  Keying on the variant id instead would keep them apart -- still
    optimal, but a larger state space and a different tie-break.
    """

    cfg = _cfg(max_ground_delay_s=200.0)
    graph, _params = _graph(cfg, dest=(5, 0), slack=5)
    topology = dp_prepare.prepare_topology(graph, cfg)
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    view = DualView(_random_duals(np.random.default_rng(3), cells, graph.base_step, 60), cfg)
    variants = dp_prepare.prepare_variants(graph, cfg, view, topology, seed=False)
    assert variants.n_variants > 1

    offsets = view.offsets
    options = {
        (-1 if lane is None else lane): (cell, steps)
        for lane, cell, steps in pricing._origin_options(graph)
    }
    by_class: dict[int, set] = {}
    for index in range(variants.n_variants):
        departure = int(variants.departure_step[index])
        cell, lane_steps = options[int(variants.lane_idx[index])]
        start_claims = pricing._endpoint_claims(
            graph, cfg, origin=True, step=departure, timing_steps=0
        ) | pricing._visit_claims(
            cell, 0, departure + graph.takeoff_steps[0] + lane_steps, offsets
        )
        by_class.setdefault(int(variants.paid_class[index]), set()).add(
            view.active_claims(start_claims)
        )
    assert all(len(sets) == 1 for sets in by_class.values()), "a class mixed two paid sets"
    flattened = [next(iter(sets)) for sets in by_class.values()]
    assert len(flattened) == len(set(flattened)), "one paid set was split across classes"


def test_variant_prefilter_never_drops_the_reference_answer(monkeypatch):
    """The ground-delay pre-filter is an optimisation, not a change of answer."""

    cfg = _cfg(max_ground_delay_s=200.0)
    graph, params = _graph(cfg, dest=(5, -1), slack=4)
    rng = np.random.default_rng(23)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 30), cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    got = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert got[0] == pytest.approx(expected[0], abs=1e-8)
    assert got[1] == expected[1]
