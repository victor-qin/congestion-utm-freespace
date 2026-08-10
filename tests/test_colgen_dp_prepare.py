"""The flat-array packing says exactly what the object API says.

``dp_prepare`` is answer-neutral by intent: every array restates something
:class:`FlightGraph` or :mod:`.pricing` already computes. That intent is worth nothing
unless it is checked arc-for-arc, so these tests compare against the oracle rather than
against a recorded snapshot -- a snapshot would keep passing after the oracle moved, which
is the failure mode the whole compiled-pricing effort has to avoid.
"""

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.dp_prepare import (
    ARC_FIRST,
    ARC_FIRST_LAST,
    ARC_INTERNAL,
    ARC_LAST,
    PricingWorkspace,
    endpoint_row_ids,
    prepare_forbidden,
    prepare_rows,
    prepare_topology,
)
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.objective import DELAY_MODEL, CostModel
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.types import FlightRequest, Terminal, vec


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


def _plain_graph(cfg, origin=(0, 0), dest=(4, -1), overrun=4):
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    request = FlightRequest(1, _point(origin, cfg), _point(dest, cfg), 0.0, 0.0)
    return build_flight_graph(request, cfg, (), params)


def _terminal_graph(cfg, origin=(0, 0), dest=(4, -1), overrun=4):
    o, d = _point(origin, cfg), _point(dest, cfg)
    o_term, d_term = Terminal("prep-A", 1, radius=90.0), Terminal("prep-B", 1, radius=90.0)
    request = FlightRequest(2, o, d, 0.0, 0.0, origin_terminal=o_term, dest_terminal=d_term)
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    return build_flight_graph(request, cfg, [(o, o_term), (d, d_term)], params)


GRAPHS = {"plain": _plain_graph, "terminal": _terminal_graph}


def _origin_lanes(fg):
    """The lane ids `_origin_options` offers, `None` for a laneless origin."""

    return [lane for lane, _cell, _steps in pricing._origin_options(fg)]


def _duals(fg, cfg, seed):
    import random

    rng = random.Random(seed)
    return {
        RowKey.cell(cell[0], cell[1], 0, step): rng.uniform(-2.0, 40.0)
        for cell in list(fg.corridor_cells)[:40]
        for step in range(fg.min_step, min(fg.min_step + 25, fg.max_step + 1))
        if rng.random() < 0.55
    }


# --------------------------------------------------------------------- topology


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_topology_arcs_and_roles_match_the_graph_oracle(shape):
    """Every CSR arc, in order, is what ``outgoing_neighbors`` returns -- roles included.

    Arc ORDER matters as much as arc membership: the reference iterates neighbours in
    ``AXIAL_NEIGHBORS`` order and its dominance keeps the first label to reach a state, so
    a reordered CSR would break ties the other way while looking equivalent.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    assert topo.ok, topo.unsupported_reason

    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    assert cells == sorted(cells), "interning must be a function of the cell set alone"

    for i, cell in enumerate(cells):
        expected = [t for t in fg.outgoing_neighbors(cell) if t in set(cells)]
        lo, hi = int(topo.arc_start[i]), int(topo.arc_start[i + 1])
        got = [cells[int(topo.arc_target[a])] for a in range(lo, hi)]
        assert got == expected, f"arc order/membership diverged at {cell}"
        for a, target in zip(range(lo, hi), expected):
            mask = int(topo.arc_roles[a])
            assert bool(mask & ARC_INTERNAL) == fg.hop_allowed_for_role(
                cell, target, first=False, last=False
            )
            assert bool(mask & ARC_FIRST) == fg.hop_allowed_for_role(
                cell, target, first=True, last=False
            )
            assert bool(mask & ARC_LAST) == fg.hop_allowed_for_role(
                cell, target, first=False, last=True
            )
            assert bool(mask & ARC_FIRST_LAST) == fg.hop_allowed_for_role(
                cell, target, first=True, last=True
            )


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_topology_reads_the_ceiling_rather_than_rebuilding_it(shape):
    """``air_hop_limit`` is ``fg.max_air_hops``, the one bound the search has.

    Pinned because the predecessor of this code carried a second copy spelled
    ``shortest_hops + detour_slack_hops``; issue #78 removed that knob precisely because
    the two agreed at the shipped default and diverged everywhere else.
    """

    cfg = _cfg()
    for overrun in (0, 1, 3, 9):
        fg = GRAPHS[shape](cfg, overrun=overrun)
        topo = prepare_topology(fg, cfg)
        if not topo.ok:
            continue
        assert topo.air_hop_limit == fg.max_air_hops
        assert topo.shortest_hops == fg.shortest_hops
        assert not hasattr(fg, "detour_slack_hops"), "the collapsed knob came back"


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_reverse_remaining_is_admissible_and_no_looser_than_hex_distance(shape):
    """The reverse BFS never over-estimates, and is a real distance where reachable.

    The reference bounds completions with plain hex distance
    (``pricing._distance_lower_bound``). A reverse BFS over real arcs is >= that -- it
    cannot cut corners the corridor does not contain -- and must still be reachable-exact,
    which is what makes it a valid substitute rather than merely a different number.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    assert topo.ok, topo.unsupported_reason
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    destinations = frozenset(pricing._destination_options(fg))

    for i, cell in enumerate(cells):
        remaining = int(topo.rev_remaining[i])
        hex_lb = min(hg.hex_distance(cell, d) for d in destinations)
        if remaining >= dp_prepare.UNREACHABLE:
            continue
        assert remaining >= hex_lb, (
            f"{cell}: reverse BFS {remaining} below the hex bound {hex_lb}, so it would "
            "prune a completion the reference keeps"
        )
        assert (remaining == 0) == (cell in destinations)


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_hex_remaining_is_the_references_own_distance_bound(shape):
    """``hex_remaining`` IS ``pricing._distance_lower_bound``, cell for cell.

    The priced search must read this and not ``rev_remaining``. Both are admissible, but
    the reverse BFS is *tighter*, and substituting it would prune labels the reference
    explores -- optimality survives, the explored set does not, and the search can then
    certify a different equally optimal column. The reference says as much in
    ``_arc_delay_lower_bound_s``: hex distance is chosen deliberately because it ignores
    walls and corridor shape.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    assert topo.ok
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    destinations = frozenset(pricing._destination_options(fg))

    for i, cell in enumerate(cells):
        assert int(topo.hex_remaining[i]) == pricing._distance_lower_bound(cell, destinations)
    # Looser everywhere, by construction: hex distance ignores the arcs the BFS follows.
    reachable = topo.rev_remaining < dp_prepare.UNREACHABLE
    assert np.all(topo.hex_remaining[reachable] <= topo.rev_remaining[reachable])

    # Recorded because it bounds what this file can prove: on these small corridors the two
    # arrays are IDENTICAL -- convex enough that following arcs costs nothing over straight
    # hex distance -- so no unit fixture here can detect the kernel reading the wrong one.
    # The guard against that is `analysis/ab_colgen_parity.py`'s density arms, where the
    # corridors are ~105 hops long and bend around real walls.
    assert np.array_equal(topo.hex_remaining, topo.rev_remaining), (
        "a fixture corridor became non-convex; that is fine, but the note above is now stale"
    )


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_claim_only_cells_are_interned_but_unenterable(shape):
    """Endpoint-disc cells outside the reachable set get row ids and nothing else.

    An endpoint's hover cylinder claims a disc around the point, and its rim regularly
    falls outside the forward-reachable set -- measured at 4 of the first 12
    ``density_faa_wing_zipline`` flights, one cell each. Those cells must be numbered or
    the endpoint dwell goes partly unpriced, which is a *cheaper wrong answer*. They must
    equally stay out of the search, which is what the two assertions below pin.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    assert topo.ok and rows.ok

    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    reachable = set(dp_prepare._reachable_cells(
        fg, [c for _l, c, _s in pricing._origin_options(fg)]
    ))
    for i, cell in enumerate(cells):
        if cell in reachable:
            continue
        assert int(topo.arc_start[i]) == int(topo.arc_start[i + 1]), (
            f"claim-only cell {cell} has arcs, so the search could enter it"
        )
        assert int(topo.rev_remaining[i]) >= dp_prepare.UNREACHABLE
        assert not topo.dest_mask[i]

    # And every disc cell really did get an id, which is the point of interning them.
    for index in list(rows.origin_disc) + list(rows.dest_disc):
        assert 0 <= int(index) < rows.n_cells


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_topology_arrays_are_read_only(shape):
    """Shared across threads without a lock, so nothing may write to them."""

    cfg = _cfg()
    topo = prepare_topology(GRAPHS[shape](cfg), cfg)
    assert topo.ok
    for name in ("cell_q", "arc_start", "arc_target", "arc_roles", "rev_remaining"):
        with pytest.raises(ValueError):
            getattr(topo, name)[0] = 0


def test_multi_level_is_refused_before_a_graph_can_even_exist():
    """The level guard lives in ``build_flight_graph``, so nothing downstream sees one.

    ``prepare_topology`` carries its own single-level check anyway, mirroring the equally
    unreachable one in ``pricing._best_column``. Both are defence against a future
    multi-level graph reaching code written for one level -- the failure would otherwise be
    a silently wrong answer, since every row here is built at ``level=0``. This test pins
    where the real refusal happens so the guards are not mistaken for live code paths.
    """

    cfg = _cfg(flight_levels_m=(30.0, 100.0), airspace_ceiling_m=200.0)
    request = FlightRequest(1, _point((0, 0), cfg), _point((4, -1), cfg), 0.0, 0.0)
    with pytest.raises(NotImplementedError, match="single flight level"):
        build_flight_graph(request, cfg, (), ColGenParams(solver="highs"))

    # And the mirror guard answers rather than raising, so a caller can fall back.
    fake = object.__new__(type(_plain_graph(_cfg())))
    object.__setattr__(fake, "levels", (30.0, 100.0))
    assert not prepare_topology(fake, cfg).ok


# ------------------------------------------------------------------------- rows


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_endpoint_row_ids_decode_to_the_reference_claim_set(shape):
    """The arithmetic numbering reproduces ``_endpoint_claims`` exactly, key for key.

    Swept over both endpoints and a range of ``timing_steps`` because the two endpoint
    shapes do NOT share a step rule -- a terminal ignores ``timing_steps`` and pads
    nothing, a bare point rounds outward by a drift that scales with it. A single span
    formula passes on one shape and silently mis-claims on the other.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    assert rows.ok, rows.unsupported_reason
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))

    def decode(row_id: int) -> RowKey:
        assert row_id >= 0, "endpoint row fell outside the numbered clock"
        resource, offset = divmod(row_id, rows.n_steps)
        step = offset + rows.step0
        if resource < rows.n_cells:
            q, r = cells[resource]
            return RowKey.cell(q, r, 0, step)
        slot = resource - rows.n_cells
        terminal = fg.origin_terminal if slot == rows.origin_term_slot else fg.dest_terminal
        return RowKey.term(terminal.id, step)

    checked = 0
    for origin in (True, False):
        for step in range(topo.min_step, min(topo.min_step + 25, topo.max_step + 1)):
            for timing_steps in (0, 1, 3, 11, 40):
                expected = pricing._endpoint_claims_uncached(
                    fg, cfg, origin=origin, step=step, timing_steps=timing_steps
                )
                got = {decode(r) for r in endpoint_row_ids(
                    rows, cfg, origin=origin, step=step, timing_steps=timing_steps
                )}
                assert got == expected, (shape, origin, step, timing_steps)
                checked += 1
    assert checked > 0


def test_row_numbering_is_injective_and_bounded():
    """Two resources never collide, and an out-of-clock step reports -1 rather than wrap."""

    cfg = _cfg()
    fg = _terminal_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    assert rows.ok

    seen = set()
    for cell_index in range(min(rows.n_cells, 40)):
        for step in range(rows.step0, rows.step0 + min(rows.n_steps, 40)):
            row_id = rows.row_of_cell(cell_index, step)
            assert 0 <= row_id < rows.n_rows
            assert row_id not in seen
            seen.add(row_id)
    for slot in range(rows.n_terminals):
        for step in range(rows.step0, rows.step0 + min(rows.n_steps, 40)):
            row_id = rows.row_of_term(slot, step)
            assert 0 <= row_id < rows.n_rows
            assert row_id not in seen
            seen.add(row_id)

    assert rows.row_of_cell(0, rows.step0 - 1) == -1
    assert rows.row_of_cell(0, rows.step0 + rows.n_steps) == -1
    assert rows.row_of_cell(rows.n_cells, rows.step0) == -1


def test_rows_hold_no_table_so_memory_stays_off_the_graph():
    """Nothing per-graph may be O(cells x steps) -- that product is ~3.6M on density.

    A dense table would be ~14 MB per flight and ~65 GB across a 4,636-flight scenario,
    which is the difference between this design scaling and not.
    """

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    assert rows.ok

    product = rows.n_cells * rows.n_steps
    assert product > 0
    stored = sum(
        int(getattr(rows, name).nbytes) for name in ("origin_disc", "dest_disc")
    )
    assert stored < product, (
        f"PreparedRows stores {stored} bytes against a cell-step product of {product}; "
        "it must number rows arithmetically, not tabulate them"
    )


# -------------------------------------------------------------------- forbidden


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_forbidden_bitset_answers_exactly_the_python_membership_test(shape):
    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    assert rows.ok
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))

    rng = np.random.default_rng(7)
    chosen = []
    for _ in range(60):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(topo.min_step, topo.max_step + 1))
        chosen.append(RowKey.cell(cell, 0, step))
    if rows.n_terminals:
        term = fg.origin_terminal
        chosen += [RowKey.term(term.id, int(rng.integers(topo.min_step, topo.max_step)))
                   for _ in range(10)]
    forbidden = frozenset(chosen)

    packed = prepare_forbidden(forbidden, fg, rows, topo)
    assert packed.n_unmapped == 0, "an in-universe row failed to map, so a claim goes unchecked"

    def bit(row_id: int) -> bool:
        return bool(int(packed.bits[row_id >> 6]) >> (row_id & 63) & 1)

    for index, cell in enumerate(cells):
        for step in range(topo.min_step, min(topo.min_step + 30, topo.max_step + 1)):
            key = RowKey.cell(cell, 0, step)
            assert bit(rows.row_of_cell(index, step)) == (key in forbidden), (cell, step)


def test_forbidden_ignores_rows_this_flight_can_never_claim():
    """Foreign cells and other levels are out of universe, not unmapped."""

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    far = RowKey.cell((999, -999), 0, topo.min_step)
    other_level = RowKey.cell(
        (int(topo.cell_q[0]), int(topo.cell_r[0])), 1, topo.min_step
    )
    packed = prepare_forbidden(frozenset({far, other_level}), fg, rows, topo)
    assert packed.n_set == 0 and packed.n_unmapped == 0 and not packed.any


def test_forbidden_counts_an_in_universe_row_it_cannot_number():
    """A claimable resource at an un-numbered step must be reported, never dropped.

    Dropping it would let the compiled search return a column touching a saturated row --
    the one failure mode of this structure that produces a wrong answer rather than a slow
    one, so the caller is told and refuses the compiled path.
    """

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    cell = (int(topo.cell_q[0]), int(topo.cell_r[0]))
    outside = RowKey.cell(cell, 0, rows.step0 + rows.n_steps + 5)
    packed = prepare_forbidden(frozenset({outside}), fg, rows, topo)
    assert packed.n_unmapped == 1 and packed.n_set == 0


# ------------------------------------------------------------------------ duals


def _duals_for(fg, topo, rng, count=120):
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    duals = {}
    for _ in range(count):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(topo.min_step - 2, topo.max_step + 2))
        key = RowKey.cell(cell, 0, step)
        duals[key] = duals.get(key, 0.0) + float(rng.gamma(2.0, 3.0))
    for terminal in (fg.origin_terminal, fg.dest_terminal):
        if terminal is None:
            continue
        for _ in range(20):
            step = int(rng.integers(topo.min_step, topo.max_step))
            key = RowKey.term(terminal.id, step)
            duals[key] = duals.get(key, 0.0) + float(rng.gamma(2.0, 3.0))
    return duals


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_prepared_visit_cost_is_bit_identical_to_dualview(shape):
    """Not approximately equal -- the same float.

    ``visit_cost`` sits in the innermost arc loop and its value flows straight into a
    label score, which dominance then compares for equality. An ulp of drift would let the
    compiled search break a tie the reference breaks the other way and return a different
    (equally optimal) column, which is exactly the divergence a parity test exists to
    catch.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    rng = np.random.default_rng(23)
    view = pricing.DualView(_duals_for(fg, topo, rng), cfg)
    prepared = dp_prepare.prepare_duals(view, fg, topo, rows)

    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    checked = 0
    for index, cell in enumerate(cells):
        for step in range(topo.min_step, min(topo.min_step + 40, topo.max_step + 1)):
            assert prepared.visit_cost(index, step) == view.visit_cost(cell, 0, step), (
                cell, step
            )
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_prepared_row_cost_is_bit_identical_to_dualview(shape):
    """Per-row prices must be the stored values, not prefix-sum differences.

    ``claim_cost`` sums exactly these numbers with ``math.fsum``. Recovering a row's price
    as ``prefix[k+1] - prefix[k]`` looks equivalent and is not: ``(a + v) - a != v`` in
    floating point once ``a`` is large relative to ``v``, which is the normal case partway
    along a busy cell's series.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    rng = np.random.default_rng(29)
    view = pricing.DualView(_duals_for(fg, topo, rng), cfg)
    prepared = dp_prepare.prepare_duals(view, fg, topo, rows)
    assert prepared.n_out_of_range == 0

    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    for index, cell in enumerate(cells):
        for step in range(topo.min_step, min(topo.min_step + 30, topo.max_step + 1)):
            row_id = rows.row_of_cell(index, step)
            assert prepared.row_cost(row_id) == view.row_cost(RowKey.cell(cell, 0, step))
    for slot, terminal in enumerate(
        [t for t in (fg.origin_terminal, fg.dest_terminal) if t is not None][: rows.n_terminals]
    ):
        for step in range(topo.min_step, min(topo.min_step + 30, topo.max_step + 1)):
            row_id = rows.row_of_term(slot, step)
            assert prepared.row_cost(row_id) == view.row_cost(RowKey.term(terminal.id, step))


def test_prepared_duals_carry_the_negative_credit_shortcut():
    """``max_negative_credit`` gates a whole-search shortcut, so it must come across."""

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    cell = (int(topo.cell_q[0]), int(topo.cell_r[0]))
    view = pricing.DualView({RowKey.cell(cell, 0, topo.min_step): -1e-9}, cfg)
    prepared = dp_prepare.prepare_duals(view, fg, topo, rows)
    assert prepared.max_negative_credit == view.max_negative_credit != 0.0


def test_unpriced_resources_report_zero_not_a_missing_series():
    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    prepared = dp_prepare.prepare_duals(pricing.DualView({}, cfg), fg, topo, rows)
    assert prepared.visit_cost(0, topo.min_step) == 0.0
    assert prepared.row_cost(rows.row_of_cell(0, topo.min_step)) == 0.0
    assert prepared.row_cost(-1) == 0.0


# --------------------------------------------------------------------- variants


def _reference_roots(fg, cfg, view, model, air_hop_limit):
    """Re-derive ``_best_column``'s root labels through pricing's own helpers.

    Built from the reference's components rather than from ``prepare_variants``' logic, so
    a packing bug shows up as a disagreement. It cannot catch a shared misunderstanding of
    those helpers -- that is what end-to-end column parity is for -- but it does catch the
    failures this structure can introduce on its own: a dropped guard, a mis-ordered union,
    a lane skipped, an unweighted score.
    """

    offsets = view.offsets
    destinations = frozenset(pricing._destination_options(fg))
    roots = {}
    for departure_step in range(fg.base_step, fg.latest_departure_step + 1):
        ground_delay_s = (departure_step - fg.base_step) * cfg.dt_s
        ground_score = -model.ground_weight * ground_delay_s
        origin_claims = pricing._endpoint_claims(
            fg, cfg, origin=True, step=departure_step, timing_steps=0
        )
        for lane_idx, cell, lane_steps in pricing._origin_options(fg):
            distance_to_go = min(hg.hex_distance(cell, d) for d in destinations)
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            if start_step >= fg.max_step:
                continue
            if start_step + distance_to_go > fg.max_step:
                continue
            if distance_to_go > air_hop_limit:
                continue
            start_claims = origin_claims | pricing._visit_claims(cell, 0, start_step, offsets)
            lane_dist = None if lane_idx is None else fg.origin_lanes[lane_idx].dist
            leg = pricing._fold_leg_s(fg.request.origin, fg.origin_terminal, lane_dist, cfg)
            roots[(departure_step, -1 if lane_idx is None else lane_idx)] = (
                cell,
                start_step,
                ground_score - model.air_weight * leg - view.claim_cost(start_claims),
                view.active_claims(start_claims),
            )
    return roots


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_variants_are_the_reference_root_labels(shape):
    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    rng = np.random.default_rng(41)
    view = pricing.DualView(_duals_for(fg, topo, rng), cfg)
    variants = dp_prepare.prepare_variants(fg, cfg, view, topo, rows)
    assert variants.ok

    expected = _reference_roots(fg, cfg, view, DELAY_MODEL, topo.air_hop_limit)
    cells = list(zip(topo.cell_q.tolist(), topo.cell_r.tolist()))
    got = {}
    for i in range(variants.n_variants):
        got[(int(variants.departure_step[i]), int(variants.lane_idx[i]))] = (
            cells[int(variants.cell[i])],
            int(variants.start_step[i]),
            float(variants.score[i]),
        )
    assert set(got) == set(expected), "variant set differs from the reference roots"
    for key, (cell, start_step, score) in got.items():
        want_cell, want_start, want_score, _paid = expected[key]
        assert (cell, start_step) == (want_cell, want_start), key
        assert score == want_score, key   # bit-identical: this IS the label score


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_paid_class_merges_exactly_the_roots_the_reference_merges(shape):
    """Equal paid-row sets share a class; unequal ones never do.

    The reference's dominance key holds the paid-row SET, so two roots at different
    departures that paid the same rows may merge downstream. Interning on variant id
    instead stays optimal but explores more and can break a tie the other way -- a
    divergence that shows up as a different (equally good) column, not as a failure.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    rng = np.random.default_rng(43)
    view = pricing.DualView(_duals_for(fg, topo, rng), cfg)
    variants = dp_prepare.prepare_variants(fg, cfg, view, topo, rows)
    expected = _reference_roots(fg, cfg, view, DELAY_MODEL, topo.air_hop_limit)

    by_class: dict[int, set] = {}
    for i in range(variants.n_variants):
        key = (int(variants.departure_step[i]), int(variants.lane_idx[i]))
        by_class.setdefault(int(variants.paid_class[i]), set()).add(expected[key][3])
    for paid_class, sets in by_class.items():
        assert len(sets) == 1, f"class {paid_class} merged distinct paid-row sets"
    assert len({next(iter(s)) for s in by_class.values()}) == len(by_class), (
        "two classes hold the same paid-row set, so roots the reference merges stay apart"
    )


def test_variant_scores_are_denominated_in_the_objective():
    """A ground-heavy model must move the score, or the kernel prices the wrong thing.

    ``[[colgen-label-score-currency]]``: the label score is the search's ranking currency
    and dominance prunes on it, so an unweighted score calls two labels tied where the
    objective strictly prefers one. Silent -- the search still returns *a* column.
    """

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    view = pricing.DualView({}, cfg)

    plain = dp_prepare.prepare_variants(fg, cfg, view, topo, rows, model=DELAY_MODEL)
    heavy = dp_prepare.prepare_variants(
        fg, cfg, view, topo, rows, model=CostModel(ground_weight=7.0, air_weight=1.0)
    )
    assert plain.n_variants == heavy.n_variants

    moved = 0
    for i in range(plain.n_variants):
        ground = (int(plain.departure_step[i]) - fg.base_step) * cfg.dt_s
        if ground == 0.0:
            continue
        # Ground is the only term the weighting changed, and it enters the score negated.
        assert float(heavy.score[i]) == pytest.approx(
            float(plain.score[i]) - 6.0 * ground, abs=1e-9
        )
        assert float(heavy.ground_delay_s[i]) == pytest.approx(7.0 * ground)
        moved += 1
    assert moved > 0, "fixture has no ground delay, so it cannot see the weighting"


def test_cutoff_prefilters_departures_without_dropping_a_winner():
    """The ground prefilter may only drop departures that provably cannot beat the cutoff."""

    cfg = _cfg()
    fg = _plain_graph(cfg)
    topo = prepare_topology(fg, cfg)
    rows = prepare_rows(fg, cfg, topo)
    view = pricing.DualView({}, cfg)

    unfiltered = dp_prepare.prepare_variants(fg, cfg, view, topo, rows, benefit=1000.0)
    assert unfiltered.n_departures_prefiltered == 0

    # A cutoff just under the best possible root score keeps only the earliest departures.
    best = float(max(unfiltered.score))
    filtered = dp_prepare.prepare_variants(
        fg, cfg, view, topo, rows, benefit=1000.0, cost_cutoff=1000.0 + best - 1e-9
    )
    assert filtered.n_departures_prefiltered > 0
    assert filtered.n_variants < unfiltered.n_variants
    kept = {int(filtered.departure_step[i]) for i in range(filtered.n_variants)}
    # Every surviving variant must be one the unfiltered pass also produced, unchanged.
    lookup = {
        (int(unfiltered.departure_step[i]), int(unfiltered.lane_idx[i])): float(unfiltered.score[i])
        for i in range(unfiltered.n_variants)
    }
    for i in range(filtered.n_variants):
        key = (int(filtered.departure_step[i]), int(filtered.lane_idx[i]))
        assert float(filtered.score[i]) == lookup[key]
    # And the best root survived -- a prefilter that drops the winner is the failure mode.
    assert max(kept) >= 0 and float(max(filtered.score)) == best


# -------------------------------------------------------------------- workspace


def test_workspace_grows_geometrically_and_reuses_its_arena():
    """Growth is a power-of-two ladder, and a big-enough buffer is never reallocated."""

    ws = PricingWorkspace()
    ws.ensure(1000, 10)
    first = ws.stamp
    assert ws.n_rows_capacity >= 1000
    ws.ensure(900, 5)
    assert ws.stamp is first, "a sufficient buffer must be kept, not re-allocated"

    ws.ensure(ws.n_rows_capacity + 1, 10)
    assert ws.stamp is not first
    assert ws.n_rows_capacity & (ws.n_rows_capacity - 1) == 0, "ladder must stay a power of two"


def test_workspace_generation_stamp_never_reads_stale():
    """A regrown buffer restarts stamping, or fresh zeros would read as 'already seen'."""

    ws = PricingWorkspace()
    ws.ensure(2048, 16)
    for _ in range(5):
        ws.next_generation()
    generation = ws.stamp_gen
    ws.stamp[3] = generation                      # pretend row 3 was visited
    ws.ensure(ws.n_rows_capacity + 1, 16)         # regrow
    assert ws.stamp_gen == 0
    assert ws.next_generation() == 1
    assert int(ws.stamp[3]) == 0, "regrown arena must not carry a live stamp"


# ------------------------------------------------------- the completion gate, split lazily


def _eager_envelope(envelopes, departure_step, lane_idx):
    """Rebuild `completion_envelope` the way it was before the lazy split.

    Deliberately NOT a call to `envelope()`: this is the independent re-derivation the split
    has to agree with, so it recomputes both halves in one pass exactly as the reference's
    closure does.
    """

    import math

    fg = envelopes._fg
    lane_steps = 0 if lane_idx is None else fg.origin_lanes[lane_idx].steps
    corridor_start = departure_step + fg.takeoff_steps[0] + lane_steps
    max_total_hops = min(fg.max_step - corridor_start, fg.max_air_hops)
    delay_lbs = [math.inf]
    dest_costs = [math.inf]
    incumbent = envelopes.incumbent
    for total_hops in range(1, max_total_hops + 1):
        delay_lb = envelopes.delay_lower_bound(departure_step, lane_idx, total_hops, 0)
        if incumbent is not None and (
            envelopes.benefit - envelopes._pi_f - delay_lb
            + envelopes._view.max_negative_credit
            < incumbent[0] - pricing._RECOMPUTE_EPS
        ):
            break
        cost = envelopes._destination_cost(corridor_start + total_hops, total_hops)
        if not math.isfinite(cost):
            delay_lbs.append(math.inf)
            dest_costs.append(math.inf)
            continue
        delay_lbs.append(delay_lb)
        dest_costs.append(cost)
    return tuple(delay_lbs), tuple(dest_costs)


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_completion_envelope_split_reproduces_the_eager_form(shape):
    """Deferring the destination half must not move the envelope, in value or in LENGTH.

    The length is the part that would fail silently. It is decided by a `break` that reads
    `delay_lb` alone -- which is why the halves can be split at all -- and it is itself a
    prune, since `can_compete` returns False outright on `first_hops >= len(delay_lbs)`.
    An envelope that came back one entry shorter would prune labels the reference keeps and
    return a different, equally optimal column.
    """

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    view = pricing.DualView(_duals(fg, cfg, 606), cfg)
    seed = pricing.seed_column(fg, cfg)
    incumbent = (
        100.0 - seed.delay_s - view.claim_cost(seed.claims),
        seed,
    )
    envelopes = dp_prepare.CompletionEnvelopes(
        fg, cfg, view, benefit=100.0, pi_f=0.0, incumbent=incumbent
    )

    checked = 0
    for departure_step in range(fg.base_step, min(fg.base_step + 6, fg.latest_departure_step + 1)):
        for lane_idx in _origin_lanes(fg):
            expected = _eager_envelope(envelopes, departure_step, lane_idx)
            assert envelopes.envelope(departure_step, lane_idx) == expected
            checked += 1
    assert checked >= 3, "the fixture swept too few keys to be a real check"


@pytest.mark.parametrize("shape", sorted(GRAPHS))
def test_can_compete_agrees_with_the_eager_envelope_everywhere(shape):
    """The gate's verdict, swept over the arguments the search actually supplies.

    `minimum_total_hops` is swept across the envelope's length because the 69.2% of real
    calls that are answered by `first_hops >= len(delay_lbs)` never touch the destination
    half at all -- so a split that got the length right and the values wrong would pass on
    the common case and fail only where it decides the answer.
    """

    import math

    cfg = _cfg()
    fg = GRAPHS[shape](cfg)
    view = pricing.DualView(_duals(fg, cfg, 606), cfg)
    seed = pricing.seed_column(fg, cfg)
    incumbent = (100.0 - seed.delay_s - view.claim_cost(seed.claims), seed)

    def expected(envelopes, departure_step, lane_idx, minimum_total_hops, paid, exact):
        delay_lbs, dest_costs = _eager_envelope(envelopes, departure_step, lane_idx)
        first_hops = max(1, minimum_total_hops)
        if first_hops >= len(delay_lbs):
            return False
        prefix = envelopes.incumbent_prefix
        origin_lane_tie = -1 if lane_idx is None else lane_idx
        paid_lb = max(0.0, paid - (0.0 if exact else pricing._RECOMPUTE_EPS))
        for total_hops in range(first_hops, len(delay_lbs)):
            union = max(paid_lb, dest_costs[total_hops])
            bound = (
                envelopes.benefit - envelopes._pi_f - delay_lbs[total_hops] - union
                + envelopes._view.max_negative_credit
            )
            if bound > incumbent[0] + pricing._SCORE_EPS:
                return True
            if not exact and bound >= incumbent[0] - pricing._RECOMPUTE_EPS:
                return True
            if abs(bound - incumbent[0]) <= pricing._SCORE_EPS:
                if (total_hops, departure_step, origin_lane_tie,
                        envelopes.destination_lane_tie) <= prefix:
                    return True
        return False

    checked = agreed_true = 0
    for departure_step in range(fg.base_step, min(fg.base_step + 4, fg.latest_departure_step + 1)):
        for lane_idx in _origin_lanes(fg):
            fresh = dp_prepare.CompletionEnvelopes(
                fg, cfg, view, benefit=100.0, pi_f=0.0, incumbent=incumbent
            )
            length = len(_eager_envelope(fresh, departure_step, lane_idx)[0])
            for hops in (0, 1, max(1, length // 2), length - 1, length, length + 5):
                for paid, exact in ((0.0, True), (5.0, False), (math.inf, False)):
                    lazy = dp_prepare.CompletionEnvelopes(
                        fg, cfg, view, benefit=100.0, pi_f=0.0, incumbent=incumbent
                    )
                    got = lazy.can_compete(
                        departure_step, lane_idx, hops, paid, paid_duals_exact=exact
                    )
                    want = expected(fresh, departure_step, lane_idx, hops, paid, exact)
                    assert got is want, (departure_step, lane_idx, hops, paid, exact)
                    checked += 1
                    agreed_true += int(got)
    assert checked >= 50, f"only {checked} verdicts compared"
    assert 0 < agreed_true < checked, "the sweep never exercised both verdicts"
