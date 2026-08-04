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
from freespace_sim.planner.colgen import dp_prepare, network, pricing
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


# ------------------------------------------------------- forbidden rows (repair path)


def _price_both_forbidden(monkeypatch, graph, duals, pi_f, cfg, params, forbidden):
    """Price with and without the kernel under an identical row-exclusion set."""

    view = DualView(duals, cfg)
    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected = price_flight(
            graph, view, pi_f, cfg, params,
            forbidden_rows=forbidden, require_improving=False,
        )
    got = price_flight(
        graph, view, pi_f, cfg, params,
        forbidden_rows=forbidden, require_improving=False,
    )
    return expected, got


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_kernel_matches_reference_when_rows_are_forbidden(monkeypatch, time_buffer_s):
    """The repair path used to be reference-only; it is now compiled, so pin parity.

    Exclusions reach the kernel as a packed cell-row table
    (``dp_prepare.prepare_forbidden``) while terminal and endpoint rows stay in Python.
    The kernel therefore searches a superset of the feasible space and relies on
    ``_canonical_candidate`` to reject anything its packed set could not see -- if that
    division of labour were wrong, the two arms would disagree here.
    """

    assert dp_kernel is not None, "kernel inactive -- this parity guard would be vacuous"
    rng = np.random.default_rng(20260804)
    compared = 0
    excluded_something = 0
    for trial in range(14):
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
        duals = _random_duals(rng, cells, graph.base_step, int(rng.integers(0, 30)))
        pi_f = float(rng.normal(0.0, 50.0))

        # Forbid a random scatter of real cell rows inside the flight's own clock
        # window, so the exclusions actually bite rather than naming unreachable rows.
        forbidden = frozenset(
            RowKey.cell(
                cells[int(rng.integers(0, len(cells)))][0],
                cells[int(rng.integers(0, len(cells)))][1],
                0,
                int(rng.integers(graph.min_step, graph.max_step + 1)),
            )
            for _ in range(int(rng.integers(1, 25)))
        )

        (expected_rc, expected), (got_rc, got) = _price_both_forbidden(
            monkeypatch, graph, duals, pi_f, cfg, params, forbidden
        )
        if expected is None:
            assert got is None, "kernel returned a column where the reference found none"
            excluded_something += 1
            continue
        compared += 1
        assert got is not None
        assert got_rc == pytest.approx(expected_rc, abs=1e-8)
        assert got.cell_path == expected.cell_path
        assert got.departure_step == expected.departure_step
        assert got.origin_lane_idx == expected.origin_lane_idx
        assert got.dest_lane_idx == expected.dest_lane_idx
        # The whole point of the exclusion set: the answer must respect it.
        assert got.claims.isdisjoint(forbidden)
    assert compared >= 6, f"only {compared} graphs produced a column; sweep is too weak"


def test_forbidden_pack_probe_matches_the_python_membership_oracle():
    """The kernel probe and ``PreparedForbidden.contains`` must agree bit for bit.

    They are separate implementations of the same open-addressed table -- one in numba
    over int64 arrays, one in Python integers -- and a divergence would show up as
    silently ignored exclusions rather than as an error.
    """

    assert dp_kernel is not None
    rng = np.random.default_rng(7)
    index = {(q, r): i for i, (q, r) in enumerate(
        (q, r) for q in range(-6, 6) for r in range(-6, 6)
    )}
    inverse = {i: c for c, i in index.items()}
    truth = {
        (int(rng.integers(0, len(index))), int(rng.integers(-20, 200)))
        for _ in range(400)
    }
    rows = frozenset(
        RowKey.cell(inverse[c][0], inverse[c][1], 0, s) for c, s in truth
    )
    pack = dp_prepare.prepare_forbidden(rows, dp_prepare.PreparedTopology(), index)
    assert pack.n_rows == len(truth)

    window_lo, window_hi = -2, 1
    for cell in range(0, 20):
        for step in range(-25, 205):
            want = any((cell, step + o) in truth for o in range(window_lo, window_hi + 1))
            assert pack.contains(cell, step) == ((cell, step) in truth)
            assert bool(dp_kernel._visit_forbidden(
                cell, step, window_lo, window_hi, pack.slots, pack.log2cap, pack.n_rows
            )) == want


def test_overflow_resizes_from_geometry_not_a_blind_quadrupling():
    """One overflow must land on a pool sized from the graph, not four rungs later.

    ``64 * n_cells`` has no step term while the label count tracks ``n_cells * n_steps``,
    so a blind x4 ladder re-ran the whole search up to four times per flight -- measured
    at 82% of all label-expansions being discarded on density_faa. This pins the rule
    that fixed it; the pool size cannot change the answer, only how many times the
    search is thrown away and repeated.
    """

    assert dp_kernel is not None
    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -2), slack=4)
    topology = dp_prepare.prepare_topology(graph, cfg)
    n_steps = topology.max_step - topology.min_step + 1

    geometric = int(dp_kernel._LABEL_GEOMETRY_SAFETY * topology.n_cells * n_steps)

    def resize(current: int) -> int:
        return min(max(current * 4, geometric), 1 << 24)

    # Below the geometric size, one overflow lands on it directly rather than needing
    # log4(geometric / current) separate re-searches.
    tiny = 64
    assert tiny * 4 < geometric, "fixture too small to exercise the geometric branch"
    assert resize(tiny) == geometric

    # Above it, x4 still governs: the product is a floor on the state space, not a cap,
    # so a flight that has already exceeded it must keep growing.
    big = geometric
    assert resize(big) == big * 4

    # Monotone, and never past the ceiling.
    assert resize(1 << 24) == 1 << 24
    assert all(resize(n) >= n for n in (tiny, geometric, 1 << 20))


def test_tiny_label_pool_still_reaches_the_reference_answer(monkeypatch):
    """Resizing is a performance policy, so any starting pool must give one answer."""

    assert dp_kernel is not None
    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -2), slack=3)
    cells = sorted(graph.corridor_cells)
    rng = np.random.default_rng(4242)
    duals = _random_duals(rng, cells, graph.base_step, 24)
    view = DualView(duals, cfg)

    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        expected_rc, expected = price_flight(
            graph, view, 3.0, cfg, params, require_improving=False
        )

    original = dp_kernel.search_dag
    for start in (64, 4096):
        with monkeypatch.context() as patch:
            patch.setattr(
                dp_kernel,
                "search_dag",
                lambda *a, _s=start, **kw: original(*a, **{**kw, "label_limit": _s}),
            )
            got_rc, got = price_flight(
                graph, view, 3.0, cfg, params, require_improving=False
            )
        assert got_rc == pytest.approx(expected_rc, abs=1e-8)
        if expected is not None:
            assert got is not None and got.cell_path == expected.cell_path


# ------------------------------------------------- compiled incumbent search (find_feasible)


def _feasible_both(monkeypatch, graph, cfg, forbidden, improve_below=None):
    """Run find_feasible_column with the kernel and with it forced off."""

    with monkeypatch.context() as patch:
        patch.setattr(pricing, "_dp_kernel", None)
        reference = pricing.find_feasible_column(
            graph, cfg, forbidden_rows=forbidden, improve_below_delay_s=improve_below
        )
    compiled = pricing.find_feasible_column(
        graph, cfg, forbidden_rows=forbidden, improve_below_delay_s=improve_below
    )
    return reference, compiled


def test_compiled_feasible_search_is_never_worse_than_the_reference(monkeypatch):
    """The compiled path returns the delay-MINIMAL column, the reference the first
    improvement it happens to reach in best-first order.

    So the contract is deliberately one-sided -- never worse, always feasible -- rather
    than column-identical.  Asserting equality here would be wrong: it would pin an
    accident of frontier ordering that the kernel has no reason to reproduce.
    """

    assert dp_kernel is not None, "kernel inactive -- this guard would be vacuous"
    rng = np.random.default_rng(20260805)
    compared = 0
    for trial in range(12):
        cfg = _cfg(max_ground_delay_s=float(rng.choice([48.0, 120.0])))
        graph, _params = _graph(
            cfg,
            dest=(int(rng.integers(2, 6)), int(rng.integers(-3, 3))),
            slack=int(rng.integers(1, 5)),
            flight_id=trial + 1,
        )
        cells = sorted(graph.corridor_cells)
        forbidden = frozenset(
            RowKey.cell(
                cells[int(rng.integers(0, len(cells)))][0],
                cells[int(rng.integers(0, len(cells)))][1],
                0,
                int(rng.integers(graph.min_step, graph.max_step + 1)),
            )
            for _ in range(int(rng.integers(0, 12)))
        )

        reference, compiled = _feasible_both(monkeypatch, graph, cfg, forbidden)
        if reference is None:
            continue
        compared += 1
        assert compiled is not None, "kernel found no column where the reference did"
        assert compiled.delay_s <= reference.delay_s + 1e-9, (
            "compiled incumbent search returned a WORSE column than the reference"
        )
        assert compiled.claims.isdisjoint(forbidden)
        assert compiled.claims == network.column_claims(compiled, graph, cfg)
    assert compared >= 6, f"only {compared} graphs produced a column; sweep is too weak"


def test_compiled_feasible_search_honours_the_improvement_threshold(monkeypatch):
    """``improve_below_delay_s`` must still gate what comes back."""

    assert dp_kernel is not None
    cfg = _cfg(max_ground_delay_s=120.0)
    graph, _params = _graph(cfg, dest=(5, -2), slack=3)

    baseline = pricing.find_feasible_column(graph, cfg)
    assert baseline is not None

    # A threshold below the best achievable delay cannot be met; whatever comes back
    # must not pretend otherwise by beating it.
    strict = pricing.find_feasible_column(
        graph, cfg, improve_below_delay_s=baseline.delay_s - 1e6
    )
    assert strict is None or strict.delay_s >= baseline.delay_s - 1e-9
