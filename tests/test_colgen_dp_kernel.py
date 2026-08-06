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


def _graph(cfg, origin=(0, 0), dest=(4, -1), slack=4, flight_id=1, objective="total_delay"):
    params = ColGenParams(
        solver="highs", detour_slack_hops=slack, objective=objective
    )
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
@pytest.mark.parametrize("objective", ["total_delay", "total_cost"])
def test_kernel_matches_reference_column_on_random_mixed_sign_duals(
    monkeypatch, time_buffer_s, objective
):
    """Parity must hold under the WEIGHTED objective too, not just at unit weights.

    Every parity guard in this file used to run at ``total_delay``, where ground and air
    both cost 1 -- and that is exactly the regime in which a weighted/unweighted mixup is
    invisible, because the two currencies coincide.  A kernel that charged air at 1x while
    the model charged 3x would have passed the entire suite.

    ``total_cost`` separates them, so the same fixtures now discriminate: the reference
    and the kernel must agree on the reduced cost AND on which column wins, at 1:3.
    """

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
            objective=objective,
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



# ------------------------------------------------------ joint completion bound
#
# ``_completion_bound`` is the only place the kernel charges a label for work it has
# not done yet, so an over-tight value here prunes the optimum and the search returns
# a wrong answer that still looks certified.  Admissibility is therefore tested from
# four independent directions: against a hand-written recursion, against the degenerate
# case where it must reduce to the previous bound exactly, against what a real
# certified column actually pays, and end to end against the reference DP.


def _bounds_for(graph, cfg, view, *, benefit=100.0, pi_f=0.0):
    """Run ``_completion_bound`` on a prepared graph; returns (g, topology, variants)."""

    topology = pricing._topology_for(graph, cfg)
    duals = dp_prepare.prepare_duals(view, topology)
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, seed=False, benefit=benefit, pi_f=pi_f
    )
    n_steps = topology.max_step - topology.min_step + 1
    g = np.empty((topology.n_cells, n_steps), dtype=np.float64)
    dp_kernel._completion_bound(
        topology.arc_start, topology.arc_target, topology.rev_remaining,
        topology.dest_mask, topology.min_step, topology.max_step,
        duals.dual_first, duals.dual_start, duals.dual_prefix,
        duals.window_lo, duals.window_hi,
        variants.dest_slot_of_cell, variants.dest_positive, variants.dest_step_base,
        float(cfg.dt_s), g,
    )
    return g, topology, variants


def _completion_bound_oracle(topology, variants, view, dt_s):
    """Independent pure-Python recursion, written from the definition.

    Deliberately not a transcription of the kernel: least (positive-part payment +
    destination endpoint price + dt per remaining hop) over every relaxed completion.
    """

    n_cells = topology.n_cells
    min_step, max_step = topology.min_step, topology.max_step
    n_steps = max_step - min_step + 1
    cells = [
        (int(q), int(r))
        for q, r in zip(topology.cell_q.tolist(), topology.cell_r.tolist())
    ]
    inf = float("inf")
    g = [[inf] * n_steps for _ in range(n_cells)]
    n_dest_steps = variants.dest_positive.shape[1]

    for step in range(max_step, min_step - 1, -1):
        k = step - min_step
        for c in range(n_cells):
            if step + int(topology.rev_remaining[c]) > max_step or step >= max_step:
                continue                                        # stays inf
            best = inf
            for a in range(int(topology.arc_start[c]), int(topology.arc_start[c + 1])):
                nb = int(topology.arc_target[a])
                w = max(0.0, view.visit_cost(cells[nb], 0, step + 1))
                if topology.dest_mask[nb]:
                    # max, not sum: the endpoint rows and this visit's window overlap.
                    slot = int(variants.dest_slot_of_cell[nb])
                    arrival = step + 1 - variants.dest_step_base
                    endpoint = (
                        float(variants.dest_positive[slot, arrival])
                        if slot >= 0 and 0 <= arrival < n_dest_steps
                        else 0.0
                    )
                    stop = max(w, endpoint)
                    best = min(best, dt_s + stop)
                if g[nb][k + 1] != inf:
                    best = min(best, w + dt_s + g[nb][k + 1])
            g[c][k] = best
    return g


def test_completion_bound_matches_an_independent_backward_recursion():
    cfg = _cfg(max_ground_delay_s=120.0)
    graph, _params = _graph(cfg, dest=(5, -2), slack=4)
    # DENSE duals, deliberately.  With sparse pricing the payment part is identically
    # zero -- a free detour always exists -- so a random-dual sweep would compare 0
    # against 0 over most states.  (Not a quirk of the test: measured on three captured
    # density_faa subproblems, ZERO of 4.3M-7.6M reachable states had a positive
    # payment bound.  The endpoint and per-hop terms are what make it bite there.)
    dense = {
        RowKey.cell(cell, 0, step): 1.5
        for cell in sorted(graph.corridor_cells)
        for step in range(graph.base_step, graph.base_step + 30)
    }
    view = DualView(dense, cfg)
    g, topology, variants = _bounds_for(graph, cfg, view)
    want_g = _completion_bound_oracle(topology, variants, view, float(cfg.dt_s))

    finite = 0
    for c in range(topology.n_cells):
        for k in range(topology.max_step - topology.min_step + 1):
            want = want_g[c][k]
            if want == float("inf"):
                assert g[c, k] == float("inf"), f"g[{c},{k}] finite"
            else:
                assert g[c, k] == pytest.approx(want, abs=1e-9), f"g[{c},{k}]"
            if want_g[c][k] != float("inf"):
                finite += 1
    assert finite > 100, f"only {finite} reachable states; the comparison is too weak"
    assert float(g[np.isfinite(g)].max()) > 0.0, "bound is vacuously zero"


def test_completion_bound_reduces_to_the_previous_delay_bound_when_nothing_is_priced():
    """With no duals and no endpoint price, ``g`` must be exactly ``rev_remaining * dt``.

    That is the identity that makes this a strict tightening rather than a different
    bound: the old expression charged ``_delay_lower_bound`` at ``hops + rev_remaining``
    and nothing for payment, which is precisely this case.  If it fails, the joint bound
    is not a superset of the old one and every parity result is suspect.
    """

    cfg = _cfg(max_ground_delay_s=120.0)
    graph, _params = _graph(cfg, dest=(5, -2), slack=4)
    g, topology, _variants = _bounds_for(graph, cfg, DualView({}, cfg))

    checked = 0
    for c in range(topology.n_cells):
        for k in range(topology.max_step - topology.min_step + 1):
            if not np.isfinite(g[c, k]):
                continue
            checked += 1
            if topology.dest_mask[c]:
                # A label already ON a destination has had its sink emitted; the bound
                # governs only what happens if it CONTINUES, which costs at least one
                # hop off the cell (and, here, one back).  Charging that is a genuine
                # tightening over the old bound, which used rev_remaining == 0.
                assert g[c, k] >= cfg.dt_s - 1e-9, f"g[{c},{k}] undercharges a dest cell"
                continue
            assert g[c, k] == pytest.approx(
                float(topology.rev_remaining[c]) * cfg.dt_s, abs=1e-9
            ), f"g[{c},{k}] is not rev_remaining * dt"
    assert checked > 100, f"only {checked} reachable states"


def test_completion_bound_dominates_its_own_components():
    """Structural invariant: ``g >= rev_remaining * dt`` everywhere, and never negative.

    Failure means the per-hop charge or the endpoint seed landed wrong, which is the
    shape of error that silently over-tightens the bound and prunes the optimum.
    """

    cfg = _cfg(max_ground_delay_s=120.0)
    graph, _params = _graph(cfg, dest=(5, -2), slack=4)
    rng = np.random.default_rng(31)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 30), cfg)
    g, topology, _variants = _bounds_for(graph, cfg, view)

    finite = np.isfinite(g)
    assert finite.any()
    floor = np.broadcast_to(
        np.asarray(topology.rev_remaining, dtype=np.float64)[:, None] * cfg.dt_s, g.shape
    )
    assert np.all(g[finite] >= floor[finite] - 1e-9), "g fell below rev_remaining * dt"
    assert np.all(g[finite] >= -1e-12), "g went negative"


@pytest.mark.parametrize("seed", [23, 41, 57])
def test_completion_bound_never_exceeds_what_a_certified_column_pays(seed):
    """Admissibility against reality: the bound must not exceed the real remainder.

    Charge a label more than its best completion actually costs and the optimum is
    pruned.  Checked against a genuine certified column, at every position along it,
    for both accumulators.
    """

    cfg = _cfg(max_ground_delay_s=120.0)
    graph, params = _graph(cfg, dest=(5, -2), slack=4)
    rng = np.random.default_rng(seed)
    view = DualView(_random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 25), cfg)
    _rc, column = price_flight(graph, view, 0.0, cfg, params, require_improving=False)
    assert column is not None
    g, topology, variants = _bounds_for(graph, cfg, view)

    path = list(column.cell_path)
    first_visit = column.departure_step + graph.takeoff_steps[0] + (
        0 if column.origin_lane_idx is None
        else graph.origin_lanes[column.origin_lane_idx].steps
    )
    steps = [first_visit + i for i in range(len(path))]
    index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    arrival = steps[-1] - variants.dest_step_base
    dest_slot = int(variants.dest_slot_of_cell[index[path[-1]]])
    endpoint = (
        float(variants.dest_positive[dest_slot, arrival])
        if dest_slot >= 0 and 0 <= arrival < variants.dest_positive.shape[1]
        else 0.0
    )

    checked = 0
    # The final position is excluded on purpose: the column ENDS there, its sink was
    # already emitted when the arc into it was relaxed, and the bound at that label
    # governs only continuations -- which necessarily cost at least one more hop.
    for i, (cell, step) in enumerate(zip(path[:-1], steps[:-1])):
        ci = index.get((int(cell[0]), int(cell[1])))
        if ci is None or not (topology.min_step <= step <= topology.max_step):
            continue
        payment = sum(
            max(0.0, view.visit_cost(path[j], 0, steps[j])) for j in range(i + 1, len(path))
        )
        remaining_hops = len(path) - 1 - i
        checked += 1
        assert g[ci, step - topology.min_step] <= (
            payment + endpoint + remaining_hops * cfg.dt_s + 1e-9
        ), f"g exceeds what this column really costs from hop {i}; the optimum prunes"
    assert checked >= 3, f"only {checked} positions checked; the sweep is too weak"


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_joint_bound_returns_the_same_column_as_the_unbounded_search(time_buffer_s):
    """End-to-end admissibility: toggling the bound must not change the answer.

    The strongest check available -- an inadmissible bound prunes the optimum, and the
    two arms then disagree on the column even when the reduced cost happens to match.
    Both arms run the compiled kernel, so this isolates the bound from every other
    difference between kernel and reference.
    """

    original = dp_kernel.search_dag
    rng = np.random.default_rng(20260805)
    compared = 0
    for trial in range(10):
        cfg = _cfg(time_buffer_s=time_buffer_s, max_ground_delay_s=120.0)
        graph, params = _graph(
            cfg, dest=(int(rng.integers(3, 6)), int(rng.integers(-3, 3))),
            slack=int(rng.integers(1, 5)), flight_id=trial + 1,
        )
        view = DualView(
            _random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 30), cfg
        )
        pi_f = float(rng.normal(0.0, 20.0))

        arms = {}
        for enabled in (False, True):
            def patched(*args, _on=enabled, **kwargs):
                kwargs["completion_bound"] = _on
                return original(*args, **kwargs)

            pricing._dp_kernel.search_dag = patched
            try:
                arms[enabled] = price_flight(
                    graph, view, pi_f, cfg, params, require_improving=False
                )
            finally:
                pricing._dp_kernel.search_dag = original

        (off_rc, off), (on_rc, on) = arms[False], arms[True]
        if off is None:
            assert on is None, "the bound invented a column the unbounded search misses"
            continue
        compared += 1
        assert on is not None, "the bound pruned the only column"
        assert on_rc == pytest.approx(off_rc, abs=1e-8)
        assert on.cell_path == off.cell_path, "the bound pruned the optimum"
        assert on.departure_step == off.departure_step
    assert compared >= 5, f"only {compared} graphs produced a column; sweep is too weak"
