"""The compiled pricing kernel's primitives, against the Python they must reproduce.

Every assertion here is an *identity*, not a tolerance.  That is the point of testing these
separately: the search they support prunes on ``SCORE_EPS = 1e-12`` bands, so a primitive
that is merely accurate to a few ulps moves labels across a dominance boundary and returns a
different -- equally optimal, equally plausible -- column.  Nothing raises when that happens,
which is why it has to be caught one function at a time.
"""
from __future__ import annotations

import math
import pickle           # round-tripping this test's OWN enum members; nothing external is read
import random
import time
import types
from collections import Counter

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.objective import DELAY_MODEL, CostModel, cost_model
from freespace_sim.planner.colgen.pricing import DualView
from freespace_sim.types import FlightRequest, Terminal, vec

njit = pytest.importorskip("numba", reason="the compiled kernel needs numba").njit
dp_kernel = pytest.importorskip("freespace_sim.planner.colgen.dp_kernel")


# --------------------------------------------------------------------------- exact sums


@njit(cache=True)
def _fsum_array(values):
    """Drive the kernel's incremental partial expansion over a whole array."""

    partials = np.zeros(dp_kernel.FSUM_MAX_PARTIALS, np.float64)
    n = 0
    for i in range(values.shape[0]):
        n = dp_kernel._fsum_add(partials, n, values[i])
        if n < 0:
            return math.nan
    return dp_kernel._fsum_finalize(partials, n)


def _kernel_fsum(values):
    return _fsum_array(np.asarray(values, dtype=np.float64))


# CPython's own `math.fsum` test vectors (Lib/test/test_math.py).  They are chosen to hit the
# exactly-half-way rounding case, which is the branch a naive "sum the partials from the top"
# implementation gets wrong -- and gets wrong only on inputs where the answer is a tie.
_FSUM_CASES = [
    ([], 0.0),
    ([0.0], 0.0),
    ([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
    ([2.0**53, -0.5, -(2.0**-54)], 2.0**53 - 1.0),
    ([2.0**53, 1.0, 2.0**-100], 2.0**53 + 2.0),
    ([2.0**53 + 10.0, 1.0, 2.0**-100], 2.0**53 + 12.0),
    ([2.0**53 - 4.0, 0.5, 2.0**-54], 2.0**53 - 3.0),
    ([1e16, 1.0, 1e-16], 10000000000000002.0),
    ([1e16 - 2.0, 1.0 - 2.0**-53, -(1e16 - 2.0), -(1.0 - 2.0**-53)], 0.0),
    ([0.1] * 10, 1.0),
]


@pytest.mark.parametrize("values,expected", _FSUM_CASES)
def test_kernel_fsum_matches_math_fsum_on_adversarial_magnitudes(values, expected):
    assert _kernel_fsum(values) == expected == math.fsum(values)


def test_kernel_fsum_matches_math_fsum_on_random_wide_exponents():
    """Random draws spanning the exponent range, where a running `+=` visibly drifts."""

    rng = random.Random(20260809)
    naive_differed = 0
    for _ in range(400):
        values = [
            rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-60, 60) for _ in range(rng.randint(2, 40))
        ]
        exact = math.fsum(values)
        assert _kernel_fsum(values) == exact, values
        naive = 0.0
        for value in values:
            naive += value
        if naive != exact:
            naive_differed += 1
    # The test is only meaningful if these inputs actually separate the two summations.
    assert naive_differed > 40, f"only {naive_differed}/400 draws distinguished fsum from `+=`"


def test_kernel_fsum_reports_expansion_overflow_rather_than_truncating():
    """A full partial expansion returns -1; it never silently drops terms.

    Unreachable in the kernel's own use -- one visit window is a handful of rows -- which is
    exactly why it must fail loudly if the assumption ever stops holding.
    """

    non_overlapping = [2.0**exponent for exponent in range(-1000, 1000, 25)]
    assert len(non_overlapping) > dp_kernel.FSUM_MAX_PARTIALS
    assert math.isnan(_kernel_fsum(non_overlapping))


# ------------------------------------------------------------------------- dual queries


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


def _point(cell, cfg: SimConfig):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _graph(cfg: SimConfig, *, overrun: int = 4):
    request = FlightRequest(11, _point((0, 0), cfg), _point((4, -1), cfg), 0.0, 0.0)
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    return build_flight_graph(request, cfg, (), params), params


def _cell_index(topology) -> dict[tuple[int, int], int]:
    return {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }


def _random_duals(graph, cfg, seed):
    """Prices on real rows, at magnitudes wide enough that summation order matters."""

    rng = random.Random(seed)
    duals: dict[RowKey, float] = {}
    for cell in list(graph.corridor_cells)[:40]:
        for step in range(graph.min_step, min(graph.min_step + 25, graph.max_step + 1)):
            if rng.random() < 0.55:
                duals[RowKey.cell(cell[0], cell[1], 0, step)] = rng.uniform(-2.0, 40.0) * (
                    10.0 ** rng.randint(-6, 3)
                )
    return duals


def test_kernel_dual_queries_are_bit_identical_to_the_prepared_view():
    """`_range_sum`/`_visit_cost`/`_row_cost` reproduce `PreparedDuals`, which pins `DualView`.

    Both dual paths are swept, because they are different arithmetic and only one of them
    can be derived from prefix sums: a window is a subtraction of two stored partials, while
    a single row's price is a stored value that `prefix[k+1] - prefix[k]` would not recover.
    """

    cfg = _cfg()
    graph, _ = _graph(cfg)
    view = DualView(_random_duals(graph, cfg, 4242), cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    duals = dp_prepare.prepare_duals(view, graph, topology, rows)

    checked_nonzero = 0
    for cell_index in range(topology.n_cells):
        for visit_step in range(rows.step0, rows.step0 + min(rows.n_steps, 60)):
            expected = duals.visit_cost(cell_index, visit_step)
            actual = dp_kernel._visit_cost(
                duals.cell_series,
                duals.series_first,
                duals.series_start,
                duals.series_prefix,
                cell_index,
                visit_step,
                duals.offsets_lo,
                duals.offsets_hi,
            )
            assert actual == expected, (cell_index, visit_step)
            if expected != 0.0:
                checked_nonzero += 1
    assert checked_nonzero > 100, "the fixture priced too few windows to be a real check"

    for position in range(duals.row_id.shape[0]):
        row = int(duals.row_id[position])
        assert dp_kernel._row_cost(duals.row_id, duals.row_value, row) == duals.row_cost(row)
    # Unpriced and out-of-universe rows both answer zero, and must not read past the ends.
    for row in (-1, 0, rows.n_rows - 1, rows.n_rows + 1000):
        assert dp_kernel._row_cost(duals.row_id, duals.row_value, row) == duals.row_cost(row)


# --------------------------------------------------- prepare_duals scans its OWN resources


def _terminal_graph(cfg: SimConfig, *, overrun: int = 4):
    """A graph with both endpoints in terminal airspace, so terminal rows exist to price."""

    origin, dest = _point((0, 0), cfg), _point((4, -1), cfg)
    o_term, d_term = Terminal("dual-A", 1, radius=90.0), Terminal("dual-B", 1, radius=90.0)
    request = FlightRequest(
        12, origin, dest, 0.0, 0.0, origin_terminal=o_term, dest_terminal=d_term
    )
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    graph = build_flight_graph(request, cfg, [(origin, o_term), (dest, d_term)], params)
    return graph, params


def _mixed_duals(graph, cfg, seed):
    """This flight's rows, plus every category of row it does NOT own.

    The foreign rows are the whole point. The old loop walked the global mapping and
    filtered them out one at a time; a rewrite that walks this flight's resources instead
    has to reach the identical verdict *without ever looking at them*, so a fixture with
    only own-rows would pass while proving nothing.
    """

    rng = random.Random(seed)
    duals: dict[RowKey, float] = dict(_random_duals(graph, cfg, seed))
    own = list(graph.corridor_cells)[:8]
    # Cells no corridor of this flight reaches.
    for q in range(60, 90):
        for step in range(graph.min_step, graph.min_step + 6):
            duals[RowKey.cell(q, -q, 0, step)] = rng.uniform(0.5, 9.0)
    # A level this single-level flight cannot fly -- the `key.level != 0` skip.
    for cell in own:
        duals[RowKey.cell(cell[0], cell[1], 1, graph.min_step)] = rng.uniform(0.5, 9.0)
    # Own cells at steps outside the row numbering -- the `row < 0` / n_out_of_range path.
    for cell in own:
        duals[RowKey.cell(cell[0], cell[1], 0, graph.min_step - 500)] = rng.uniform(0.5, 9.0)
    # Its own two terminals, and one it never touches.
    for term_id in (graph.origin_terminal.id, graph.dest_terminal.id, "dual-foreign"):
        for step in range(graph.min_step, graph.min_step + 10):
            duals[RowKey.term(term_id, step)] = rng.uniform(0.5, 9.0)
    return duals


def _reference_pairs(view, cell_index, term_slot, rows):
    """`prepare_duals`' pre-change loop, kept verbatim so the rewrite has an oracle.

    Inlined rather than imported on purpose: this is the definition of the old behaviour,
    and it has to survive the source change that deletes it.
    """

    pairs: list[tuple[int, float]] = []
    n_out_of_range = 0
    for key, value in view._duals.items():
        if key.kind == "cell":
            if key.level != 0:
                continue
            index = cell_index.get(key.cell_coord)
            if index is None:
                continue
            row = rows.row_of_cell(index, key.step)
        else:
            slot = term_slot.get(key.terminal_id)
            if slot is None:
                continue
            row = rows.row_of_term(slot, key.step)
        if row < 0:
            n_out_of_range += 1
            continue
        pairs.append((row, float(value)))
    pairs.sort()
    return pairs, n_out_of_range


def _term_slot(graph, rows) -> dict:
    slot: dict = {}
    if rows.origin_is_terminal:
        slot[graph.origin_terminal.id] = rows.origin_term_slot
    if rows.dest_is_terminal:
        slot[graph.dest_terminal.id] = rows.dest_term_slot
    return slot


def test_prepare_duals_matches_a_full_scan_of_the_dual_mapping():
    """Walking this flight's resources must equal walking every global row and filtering.

    Compared with `==` on the values rather than `allclose`: the point of keeping
    `DualView`'s step buckets instead of differencing them back out of the prefix sums is
    that they are the SAME floats, and an approximate check would not notice if they
    stopped being.
    """

    cfg = _cfg()
    graph, _ = _terminal_graph(cfg)
    assert graph.origin_terminal is not None and graph.dest_terminal is not None
    view = DualView(_mixed_duals(graph, cfg, 90210), cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)

    expected, expected_out_of_range = _reference_pairs(
        view, _cell_index(topology), _term_slot(graph, rows), rows
    )
    assert expected, "the fixture priced no rows this flight owns"
    assert expected_out_of_range > 0, "the fixture never exercised the out-of-range path"

    prepared = dp_prepare.prepare_duals(view, graph, topology, rows)
    assert prepared.row_id.tolist() == [row for row, _ in expected]
    assert prepared.row_value.tolist() == [value for _, value in expected]
    assert prepared.n_out_of_range == expected_out_of_range
    # And the fixture has to reject a lot, or "equal" is trivially true: the rows this
    # flight does not own are the only ones whose handling the rewrite actually changes.
    rejected = len(view._duals) - len(expected)
    assert rejected > 150, f"the fixture barely filtered: {rejected} of {len(view._duals)}"


def test_prepare_duals_never_scans_the_global_dual_mapping():
    """The cost claim, pinned as a behaviour rather than a benchmark.

    A wall-clock assertion would be worthless on a shared machine; "it did not iterate the
    global mapping even once" is exact and machine-independent. The old loop's cost was
    O(all rows) *per flight*, so it grows with the master's materialized row count while
    the flight stays the same size -- which is why this is the property worth freezing.
    """

    cfg = _cfg()
    graph, _ = _terminal_graph(cfg)
    view = DualView(_mixed_duals(graph, cfg, 4711), cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    reference = dp_prepare.prepare_duals(view, graph, topology, rows)

    class _RefusesToBeScanned(dict):
        def items(self):
            raise AssertionError("prepare_duals scanned the global dual mapping")

        def __iter__(self):
            raise AssertionError("prepare_duals iterated the global dual mapping")

    view._duals = _RefusesToBeScanned(view._duals)
    prepared = dp_prepare.prepare_duals(view, graph, topology, rows)
    assert prepared.row_id.tolist() == reference.row_id.tolist()
    assert prepared.row_value.tolist() == reference.row_value.tolist()


def test_dual_view_step_buckets_hold_the_same_floats_as_the_flat_mapping():
    """The retained buckets must be the accumulated value, bit for bit -- not a recompute."""

    cfg = _cfg()
    graph, _ = _terminal_graph(cfg)
    view = DualView(_mixed_duals(graph, cfg, 1337), cfg)

    for key, value in view._duals.items():
        if key.kind == "cell":
            bucket = dict(view._cell_steps[(key.cell_coord, key.level)])
        else:
            bucket = dict(view._terminal_steps[key.terminal_id])
        assert bucket[key.step] == value, key


def test_cell_and_terminal_row_ids_never_collide():
    """Two pairs sharing a row id would make `pairs.sort()` depend on insertion order.

    `prepare_duals` sorts `(row, value)` tuples, so a duplicate row would be tie-broken by
    VALUE and the result would depend on the order rows were appended -- exactly what the
    rewrite changes. Injectivity is what makes the sort a total order and the change safe.
    """

    cfg = _cfg()
    graph, _ = _terminal_graph(cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)

    seen: dict[int, tuple] = {}
    steps = range(rows.step0, rows.step0 + rows.n_steps)
    for index in range(rows.n_cells):
        for step in steps:
            row = rows.row_of_cell(index, step)
            assert row not in seen, (row, seen.get(row), ("cell", index, step))
            seen[row] = ("cell", index, step)
    for slot in range(rows.n_terminals):
        for step in steps:
            row = rows.row_of_term(slot, step)
            assert row not in seen, (row, seen.get(row), ("term", slot, step))
            seen[row] = ("term", slot, step)


# ------------------------------------------------------------------- forbidden-row test


def test_kernel_forbidden_bitset_matches_the_prepared_set():
    """Every row in the flight's universe answers as `prepare_forbidden` recorded it."""

    cfg = _cfg()
    graph, _ = _graph(cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)

    rng = random.Random(99)
    forbidden = set()
    for cell in list(graph.corridor_cells)[:25]:
        for step in range(graph.min_step, min(graph.min_step + 30, graph.max_step + 1)):
            if rng.random() < 0.3:
                forbidden.add(RowKey.cell(cell[0], cell[1], 0, step))
    pack = dp_prepare.prepare_forbidden(forbidden, graph, rows, topology)
    assert pack.n_set > 50, "the fixture forbade too few rows to be a real check"

    index_of = _cell_index(topology)
    expected_ids = set()
    for row in forbidden:
        cell_index = index_of.get(row.cell_coord, -1)
        if cell_index >= 0:
            row_id = rows.row_of_cell(cell_index, row.step)
            if row_id >= 0:
                expected_ids.add(row_id)

    for row_id in range(rows.n_rows):
        assert dp_kernel._row_forbidden(pack.bits, row_id) == (row_id in expected_ids)
    # A row outside the numbering is not forbidden -- the flight cannot claim it at all.
    assert not dp_kernel._row_forbidden(pack.bits, -1)
    assert not dp_kernel._row_forbidden(pack.bits, rows.n_rows + 4096)


def test_kernel_forbidden_bitset_is_empty_when_nothing_is_forbidden():
    cfg = _cfg()
    graph, _ = _graph(cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    pack = dp_prepare.prepare_forbidden(frozenset(), graph, rows, topology)
    assert pack.n_set == 0
    for row_id in (0, 1, rows.n_rows - 1):
        assert not dp_kernel._row_forbidden(pack.bits, row_id)


# ------------------------------------------------------------------------ label compare


class _LabelPool:
    """Mirror one set of labels as both `pricing._Label` objects and kernel arrays."""

    def __init__(self):
        self.score: list[float] = []
        self.hops: list[int] = []
        self.departure: list[int] = []
        self.lane: list[int] = []
        self.parent: list[int] = []
        self.cell: list[int] = []
        self.labels: list[pricing._Label] = []

    def add(self, *, score, departure_step, lane_idx, path, parent):
        self.score.append(float(score))
        self.hops.append(len(path) - 1)
        self.departure.append(int(departure_step))
        self.lane.append(-1 if lane_idx is None else int(lane_idx))
        self.parent.append(parent)
        self.cell.append(path[-1])
        self.labels.append(
            pricing._Label(
                float(score),
                int(departure_step),
                lane_idx,
                tuple((cell, 0) for cell in path),
                frozenset(),
            )
        )
        return len(self.labels) - 1

    def arrays(self):
        return (
            np.asarray(self.score, np.float64),
            np.asarray(self.hops, np.int32),
            np.asarray(self.departure, np.int32),
            np.asarray(self.lane, np.int32),
            np.asarray(self.parent, np.int32),
            np.asarray(self.cell, np.int32),
        )


def _random_pool(rng, n_labels=140, depth=6):
    """Labels sharing prefixes, so path comparison is reached rather than short-circuited."""

    pool = _LabelPool()
    roots = []
    for _ in range(8):
        cell = rng.randint(0, 3)
        roots.append(pool.add(score=0.0, departure_step=rng.randint(0, 2), lane_idx=None,
                              path=[cell], parent=-1))
    live = list(roots)
    while len(pool.labels) < n_labels:
        parent = rng.choice(live)
        if pool.hops[parent] >= depth:
            continue
        path = [pool.cell[i] for i in _chain(pool.parent, parent)] + [rng.randint(0, 3)]
        # Scores drawn on the SCORE_EPS scale, so the epsilon band is genuinely exercised
        # rather than every comparison resolving on magnitude alone.
        score = pool.score[parent] - rng.choice([0.0, 0.5e-12, 1.0e-12, 2.0e-12, 1.0])
        live.append(
            pool.add(
                score=score,
                departure_step=pool.departure[parent],
                lane_idx=None if pool.lane[parent] < 0 else pool.lane[parent],
                path=path,
                parent=parent,
            )
        )
    return pool


def _chain(parent, node):
    out = []
    while node >= 0:
        out.append(node)
        node = parent[node]
    return list(reversed(out))


def test_kernel_path_compare_matches_python_tuple_ordering():
    """`_path_cmp` is Python's tuple comparison, prefix rule included."""

    rng = random.Random(7)
    pool = _random_pool(rng)
    score, hops, departure, lane, parent, cell = pool.arrays()
    scratch_a = np.zeros(64, np.int32)
    scratch_b = np.zeros(64, np.int32)

    compared_prefixes = 0
    for _ in range(3000):
        a = rng.randrange(len(pool.labels))
        b = rng.randrange(len(pool.labels))
        path_a = tuple(pool.cell[i] for i in _chain(pool.parent, a))
        path_b = tuple(pool.cell[i] for i in _chain(pool.parent, b))
        expected = int(path_a > path_b) - int(path_a < path_b)
        assert dp_kernel._path_cmp(a, b, parent, cell, scratch_a, scratch_b) == expected
        if path_a != path_b and (
            path_a[: len(path_b)] == path_b or path_b[: len(path_a)] == path_a
        ):
            compared_prefixes += 1
    assert compared_prefixes > 20, "no common-prefix pairs were compared"


def test_kernel_prefer_matches_the_reference_dominance_rule():
    """`_prefer` agrees with `pricing._prefer` on every pair, ties included."""

    rng = random.Random(11)
    pool = _random_pool(rng)
    score, hops, departure, lane, parent, cell = pool.arrays()
    scratch_a = np.zeros(64, np.int32)
    scratch_b = np.zeros(64, np.int32)

    within_band = 0
    for _ in range(4000):
        a = rng.randrange(len(pool.labels))
        b = rng.randrange(len(pool.labels))
        expected = pricing._prefer(pool.labels[a], pool.labels[b])
        actual = dp_kernel._prefer(
            a, b, score, hops, departure, lane, parent, cell, scratch_a, scratch_b
        )
        assert actual == expected, (a, b, pool.score[a], pool.score[b])
        if abs(pool.score[a] - pool.score[b]) <= dp_kernel.SCORE_EPS:
            within_band += 1
    assert within_band > 200, f"only {within_band} pairs landed in the tie band"

    # An empty slot always loses, which is how a first insertion is spelled.
    assert dp_kernel._prefer(
        0, -1, score, hops, departure, lane, parent, cell, scratch_a, scratch_b
    )


def test_kernel_prefer_reproduces_the_non_transitive_epsilon_band():
    """Three scores a hair apart: a ties b, b ties c, yet c beats a -- in both worlds.

    Pinned deliberately.  It is the reason the kernel has to reproduce the reference's
    insertion ORDER rather than treating dominance as a set operation: with a
    non-transitive rule, which label survives depends on which arrived first.
    """

    pool = _LabelPool()
    a = pool.add(score=0.0, departure_step=0, lane_idx=None, path=[0], parent=-1)
    b = pool.add(score=0.6e-12, departure_step=0, lane_idx=None, path=[1], parent=-1)
    c = pool.add(score=1.4e-12, departure_step=0, lane_idx=None, path=[2], parent=-1)
    score, hops, departure, lane, parent, cell = pool.arrays()
    scratch_a = np.zeros(8, np.int32)
    scratch_b = np.zeros(8, np.int32)

    def kernel(new, old):
        return dp_kernel._prefer(
            new, old, score, hops, departure, lane, parent, cell, scratch_a, scratch_b
        )

    for new, old in ((a, b), (b, a), (b, c), (c, b), (a, c), (c, a)):
        assert kernel(new, old) == pricing._prefer(pool.labels[new], pool.labels[old])

    # The cycle itself, stated outright: `a` and `b` are within the band so the smaller path
    # wins, likewise `b` and `c` -- but `a` and `c` are 1.4e-12 apart, which is outside it,
    # so the score decides and reverses the chain.
    assert kernel(a, b), "a displaces b on the path tie-break"
    assert kernel(b, c), "b displaces c on the path tie-break"
    assert kernel(c, a), "c displaces a on score, closing the cycle"
    assert not kernel(a, c)


# ------------------------------------------------------------------------- state hashing


def test_kernel_state_hash_separates_the_reference_key_fields():
    """Each component of `(cell, recent, paid_class, first_hop)` changes the hash.

    A hash collision is survivable -- the probe verifies the full key -- but a hash that
    ignored a field would make collisions systematic on exactly the states that must stay
    distinct, so this pins that every field is mixed in, `recent`'s LENGTH included.
    """

    recent = np.asarray([3, 5, 7, -1], np.int32)
    base = dp_kernel._state_hash(2, recent, 3, 4, 8, 9)
    assert dp_kernel._state_hash(3, recent, 3, 4, 8, 9) != base
    assert dp_kernel._state_hash(2, recent, 2, 4, 8, 9) != base
    assert dp_kernel._state_hash(2, recent, 3, 5, 8, 9) != base
    assert dp_kernel._state_hash(2, recent, 3, 4, -1, -1) != base
    assert dp_kernel._state_hash(2, recent, 3, 4, 9, 8) != base
    other = np.asarray([3, 5, 8, -1], np.int32)
    assert dp_kernel._state_hash(2, other, 3, 4, 8, 9) != base
    # Same inputs, same hash: the table depends on it being a function.
    assert dp_kernel._state_hash(2, recent, 3, 4, 8, 9) == base


def test_kernel_mix_stays_inside_the_table():
    for log2cap in (4, 8, 16):
        for value in (0, 1, 7, 1 << 20, (1 << 63) - 1):
            slot = dp_kernel._mix(np.uint64(value), log2cap)
            assert 0 <= slot < (1 << log2cap)


def test_warm_kernel_compiles_every_primitive():
    assert dp_kernel.warm_kernel() is True


# ------------------------------------------------------------------------ the whole search


def _reference_candidates(
    graph, cfg, view, params, monkeypatch, *, forbidden=frozenset(), incumbent=None,
    keep_roots=None, **kwargs
):
    """Every sink proposal `_best_column` registers, in the order it registers them.

    Spies on `_Candidate` rather than re-deriving the set: that constructor runs exactly
    once per accepted `(sink label, destination lane)` inside `consider_sink`, so this is
    the reference's candidate list by construction rather than by reimplementation.

    Call this AFTER `_kernel_candidates`, never before: `pricing._sink_certifier` builds a
    `_Candidate` too, so a spy left installed would record the kernel's certifications as
    if they were the reference's proposals.
    """

    recorded = []
    real = pricing._Candidate

    def spy(reduced_cost, delay_s, label, dest_lane_idx):
        candidate = real(reduced_cost, delay_s, label, dest_lane_idx)
        recorded.append(candidate)
        return candidate

    monkeypatch.setattr(pricing, "_Candidate", spy)
    pricing._best_column(
        graph, view, 0.0, cfg, kwargs.pop("benefit", 100.0), forbidden,
        seed=False, incumbent=incumbent, model=kwargs.pop("model", None) or DELAY_MODEL,
        keep_roots=keep_roots,
    )
    return recorded


def _kernel_candidates(
    graph, cfg, view, model, forbidden=frozenset(), *, incumbent=None, benefit=100.0,
    keep_roots=None, **kwargs,
):
    """The compiled search's sink proposals, driven through the full host protocol.

    Deliberately wires up `envelopes` and `certify` rather than running the kernel bare:
    the completion gate and the mid-sweep incumbent are the two things that make the
    kernel's explored set the reference's, and a helper that omitted them would be
    comparing a looser search against the oracle and calling the difference acceptable.
    """

    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    duals = dp_prepare.prepare_duals(view, graph, topology, rows)
    envelopes = dp_prepare.CompletionEnvelopes(
        graph, cfg, view, benefit=benefit, pi_f=0.0, model=model,
        forbidden_rows=forbidden, incumbent=incumbent,
    )
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, rows, benefit=benefit, pi_f=0.0,
        cost_cutoff=None if incumbent is None else incumbent[0],
        model=model, forbidden_rows=forbidden, envelopes=envelopes,
        keep_roots=keep_roots,
    )
    pack = dp_prepare.prepare_forbidden(forbidden, graph, rows, topology)
    result = dp_kernel.price_dag(
        topology, rows, duals, variants, pack,
        air_weight=model.air_weight, dt_s=cfg.dt_s,
        benefit=benefit, pi_f=0.0,
        envelopes=envelopes,
        certify=pricing._sink_certifier(
            graph, view, 0.0, cfg, benefit, forbidden, model
        ),
        **kwargs,
    )
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    out = []
    for departure, lane, dest_lane, _step, label in result.candidates:
        out.append((
            departure,
            None if lane < 0 else lane,
            None if dest_lane < 0 else dest_lane,
            tuple(cells[c] for c in result.paths[label]),
        ))
    return result, out


def _terminal_graph(cfg, *, overrun: int = 4):
    origin, dest = _point((0, 0), cfg), _point((4, -1), cfg)
    o_term, d_term = Terminal("kern-A", 1, radius=90.0), Terminal("kern-B", 1, radius=90.0)
    request = FlightRequest(
        12, origin, dest, 0.0, 0.0, origin_terminal=o_term, dest_terminal=d_term
    )
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    graph = build_flight_graph(request, cfg, [(origin, o_term), (dest, d_term)], params)
    return graph, params


GRAPH_SHAPES = {"plain": _graph, "terminal": _terminal_graph}


def _reference_sink_set(graph, cfg, view, params, monkeypatch, model, **kwargs):
    return {
        (
            candidate.label.departure_step,
            candidate.label.origin_lane_idx,
            candidate.dest_lane_idx,
            tuple((q, r) for q, r in candidate.label.path),
        )
        for candidate in _reference_candidates(
            graph, cfg, view, params, monkeypatch, model=model, **kwargs
        )
    }


def _certification_trace(monkeypatch):
    """Record what `_canonical_candidate` is asked to certify, in order.

    The reference calls it from two places -- ``consider_sink``, throughout the sweep, and
    Tier 2, once the sweep is over -- so a run's trace is its mid-sweep incumbent
    trajectory followed by its ranking. The kernel host calls it only from
    ``_sink_certifier``, so its trace IS the incumbent trajectory alone.
    """

    seen = []
    real = pricing._canonical_candidate

    def spy(candidate, *args, **kwargs):
        seen.append(
            (
                candidate.label.departure_step,
                candidate.label.origin_lane_idx,
                candidate.dest_lane_idx,
                tuple((q, r) for q, r in candidate.label.path),
            )
        )
        return real(candidate, *args, **kwargs)

    monkeypatch.setattr(pricing, "_canonical_candidate", spy)
    return seen


@pytest.mark.parametrize("shape", sorted(GRAPH_SHAPES))
def test_kernel_proposes_exactly_the_reference_sinks_by_shape(shape, monkeypatch):
    """The kernel's sink set EQUALS the reference's, on both endpoint shapes.

    Equality, not inclusion, and the difference is the whole of Phase 2d. Missing a sink
    was always a correctness failure. **Extra** sinks used to be the accepted cost of not
    applying `completion_can_compete` -- and they are not free: a looser search does not
    merely explore a superset, because its extra labels win dominance slots and evict the
    reference's survivors, whose sinks are then never generated at all. This fixture used
    to lose 12 that way, with every guard correct and every label score bit-identical to
    the reference's at every prefix. `[[pruning-not-neutral-under-dominance]]`.

    The terminal shape is the one that matters: a terminal origin turns on
    `track_first_hop`, so the dominance key grows a field that is inert on the plain
    fixture, and both endpoints claim *term* rows under the unpadded span rule instead of
    *cell* rows. Neither is exercised by `colgen_test`-shaped graphs, and both are the
    density shape -- which is to say, the only shape any registered scenario produces.
    """

    cfg = _cfg()
    graph, params = GRAPH_SHAPES[shape](cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model)
    assert result.ok, result.status
    if shape == "terminal":
        topology = dp_prepare.prepare_topology(graph, cfg)
        assert topology.track_first_hop, "fixture no longer exercises the first-hop field"

    reference_side = _reference_sink_set(graph, cfg, view, params, monkeypatch, model)
    assert reference_side, "the reference proposed nothing, so this proves nothing"
    missing = reference_side - set(kernel_side)
    extra = set(kernel_side) - reference_side
    assert not missing, f"{shape}: kernel missed {len(missing)}, e.g. {sorted(missing)[0]}"
    assert not extra, f"{shape}: kernel invented {len(extra)}, e.g. {sorted(extra)[0]}"


def test_kernel_proposes_exactly_the_reference_sinks(monkeypatch):
    """The same equality on a larger plain graph, with a different dual draw.

    Kept separate from the shape sweep because it is the size check: a fixture small
    enough to agree by luck proves nothing about a search whose whole difficulty is which
    of several thousand labels wins a dominance slot.
    """

    cfg = _cfg()
    graph, params = _graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 31337), cfg)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model)
    assert result.ok, result.status
    assert kernel_side, "the kernel proposed nothing, so this test proves nothing"

    reference_side = _reference_sink_set(graph, cfg, view, params, monkeypatch, model)
    assert len(reference_side) > 200, "the fixture is too small to be a real check"
    assert set(kernel_side) == reference_side


def _gated_variants(graph, cfg, view, model, incumbent=None, benefit=100.0, keep_roots=None):
    """`prepare_variants` exactly as `_bootstrap_incumbent` and `_best_column_compiled` call
    it -- WITH the completion gate, which is the part that has to match."""

    topology, rows = dp_prepare.prepared_for(graph, cfg)
    envelopes = dp_prepare.CompletionEnvelopes(
        graph, cfg, view, benefit=benefit, pi_f=0.0, model=model, incumbent=incumbent
    )
    return dp_prepare.prepare_variants(
        graph, cfg, view, topology, rows, benefit=benefit, pi_f=0.0,
        cost_cutoff=None if incumbent is None else incumbent[0],
        model=model, envelopes=envelopes, keep_roots=keep_roots,
    )


def _rank_roots(graph, cfg, view, model, k, incumbent=None, benefit=100.0):
    """The allowlist `_bootstrap_incumbent` builds: top-k GATED roots by variant score."""

    variants = _gated_variants(graph, cfg, view, model, incumbent=incumbent, benefit=benefit)
    order = np.argsort(-variants.score, kind="stable")[:k]
    keep = frozenset(
        (int(variants.departure_step[i]), int(variants.lane_idx[i])) for i in order
    )
    return variants, keep


def test_the_bootstrap_selects_only_roots_that_survive_the_full_gate():
    """The invariant behind the one failure on this path that does not raise.

    If the ranking is taken over a root set the restricted search will not see -- which is
    what happens if the ranking call omits `envelopes` -- the top-scoring root can be one
    `completion_can_compete` rejects. `prepare_variants` then returns EMPTY for the
    bootstrap, the search reports `(-inf, None)` successfully, `_bootstrap_incumbent` hands
    its incumbent straight back, and nothing anywhere raises: the bootstrap silently becomes
    a no-op that merely reads as "not worth much". Also covers the `-1`-for-a-bare-origin
    key encoding, whose failure mode is identical.

    HONEST LIMIT: neither fixture here gates hard enough to reproduce that on its own -- the
    top ungated root survives on both, checked. The real reproduction is `colgen_test`'s
    flight 0, where the gate takes 13,515 roots to 97 and the ungated winner is not among
    them. So this pins the property, and `prof_colgen_cutoff`'s `bt_lab` column (zero labels
    against a nonzero `boot_s`) is what catches it in the field.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)
    seed = pricing.seed_column(graph, cfg, model=model)
    incumbent = (
        model.reduced_cost(
            benefit=100.0, cost=seed.delay_s, dual_cost=view.claim_cost(seed.claims), pi_f=0.0
        ),
        seed,
    )

    _variants, keep = _rank_roots(graph, cfg, view, model, 3, incumbent=incumbent)
    assert len(keep) == 3, "the fixture has too few roots to select from"

    restricted = _gated_variants(
        graph, cfg, view, model, incumbent=incumbent, keep_roots=keep
    )
    survivors = {
        (int(d), int(lane))
        for d, lane in zip(restricted.departure_step.tolist(), restricted.lane_idx.tolist())
    }
    assert survivors == keep, "a selected root did not survive the gate the search applies"


def test_the_bootstrap_picks_the_highest_scoring_root():
    """Ranked, not truncated -- the difference between this and the prefix it replaces.

    A prefix takes the earliest departures and misses an optimum that departs late; `score`
    is the root's own upper bound and orders by promise instead. The tie-break must be
    stable, because the allowlist has to be a pure function of the graph, the duals and the
    incumbent -- if it were not, the two searches could be handed different sets.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    variants, keep = _rank_roots(graph, cfg, view, model, 1)
    best = int(np.argmax(variants.score))
    assert keep == {(int(variants.departure_step[best]), int(variants.lane_idx[best]))}

    _variants_again, keep_again = _rank_roots(graph, cfg, view, model, 1)
    assert keep_again == keep, "the ranking is not deterministic"


def test_kernel_and_reference_restrict_roots_identically(monkeypatch):
    """The bootstrap's restriction axis, held to the same equality as the full search.

    `_bootstrap_incumbent` runs whichever of the two searches is available over the same
    allowlist, so if they honoured it differently the bootstrap would hand the main search a
    cutoff that depends on which path ran -- and `Declined`'s contract, that the compiled
    search explored exactly what the reference would, would be false for the search that
    consumed it.

    Equality, not inclusion, for the reason the unrestricted case documents: a looser search
    does not merely explore a superset, because its extra labels win dominance slots and
    evict survivors whose sinks are then never generated at all.
    """

    cfg = _cfg()
    graph, params = _graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 31337), cfg)
    _variants, keep = _rank_roots(graph, cfg, view, model, 4)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model, keep_roots=keep)
    assert result.ok, result.status
    assert kernel_side, "the restricted kernel proposed nothing, so this proves nothing"

    reference_side = _reference_sink_set(
        graph, cfg, view, params, monkeypatch, model, keep_roots=keep
    )
    assert set(kernel_side) == reference_side

    # The restriction has to actually bite, or the equality above is the unrestricted test
    # wearing a disguise.
    unrestricted = _reference_sink_set(graph, cfg, view, params, monkeypatch, model)
    assert len(unrestricted) > len(reference_side)

    # Deliberately NOT asserting how the two sets nest, in either direction.
    #
    # A departure PREFIX produced sinks the unrestricted search never did: it keeps the
    # earliest roots, which are exactly the ones later better-scoring roots dominate away in
    # the full search, so their descendants survive only under the restriction.  Ranking by
    # `score` keeps the roots that win anyway, and on this fixture the restricted set comes
    # out a clean subset -- i.e. it perturbs the search less, which is a point in its favour.
    # But that is an observation about one fixture and not a property anyone has proved, so
    # `_bootstrap_incumbent` may still only ever return an INCUMBENT: a bootstrap candidate
    # flowing into `_certify_candidates` could otherwise put a sink into the real ranking
    # that the reference would never have generated
    # (`[[pruning-not-neutral-under-dominance]]`).


def test_kernel_certifies_the_same_sinks_in_the_same_order_as_the_reference(monkeypatch):
    """The pause protocol reproduces the reference's mid-sweep incumbent, step for step.

    This is the direct test of pause-and-resume, and it is sharper than comparing final
    answers. ``consider_sink`` canonicalizes a sink exactly when its provisional reduced
    cost beats the live incumbent, so the *sequence* of canonicalizations IS the cutoff
    trajectory -- and the cutoff is what every later layer prunes against. Two searches
    that agree on that sequence agree on every prune between them.

    The kernel's trace must be a strict PREFIX of the reference's, not merely a subset:
    the reference appends its Tier 2 ranking to the same trace once the sweep is over,
    and the kernel host's certifier stops at the end of the sweep.

    The label pool is sized generously on purpose. A budget restart re-runs the search
    from its first layer and re-certifies everything, so the trace would then be one run's
    certifications concatenated with another's -- a real property of the host, but not the
    one under test here, and `attempts` is asserted so a silent restart cannot hide.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    kernel_trace = _certification_trace(monkeypatch)
    result, _ = _kernel_candidates(graph, cfg, view, model, label_capacity=1 << 18)
    assert result.ok, result.status
    assert result.attempts == 1, "the search restarted, so the trace spans two runs"
    kernel_trace = list(kernel_trace)
    monkeypatch.undo()

    reference_trace = _certification_trace(monkeypatch)
    pricing._best_column(
        graph, view, 0.0, cfg, 100.0, frozenset(), seed=False, incumbent=None, model=model
    )

    assert kernel_trace, "no sink was ever certified, so the pause protocol never ran"
    assert reference_trace[: len(kernel_trace)] == kernel_trace
    # The certifications are a vanishing fraction of the sinks; that ratio is what makes
    # pausing per improvement a different proposition from pausing per sink.
    assert len(kernel_trace) < 0.01 * len(result.candidates)


def test_kernel_matches_the_reference_when_warm_started_with_an_incumbent(monkeypatch):
    """The production path: `price_flight` always hands `_best_column` a seed incumbent.

    A live incumbent from the very first root changes the search qualitatively rather than
    quantitatively. ``prepare_variants``' root gate now prunes departures outright, and
    every surviving variant's completion envelope is frozen against this one cutoff --
    which is what the reference's memo does too, since it builds them all inside its root
    loop before any arc is relaxed. Without an incumbent that whole code path is dead, so
    the un-warmed tests above do not cover it.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    seed = pricing.seed_column(graph, cfg, model=model)
    incumbent = (
        model.reduced_cost(
            benefit=100.0, cost=seed.delay_s, dual_cost=view.claim_cost(seed.claims), pi_f=0.0
        ),
        seed,
    )

    result, kernel_side = _kernel_candidates(graph, cfg, view, model, incumbent=incumbent)
    assert result.ok, result.status

    reference_side = _reference_sink_set(
        graph, cfg, view, params, monkeypatch, model, incumbent=incumbent
    )
    assert reference_side, "the warm start pruned everything, so this proves nothing"
    assert set(kernel_side) == reference_side


# ------------------------------------------------------------------------ the bootstrap


def _bootstrap_fixture(seed=606, overrun=4):
    cfg = _cfg()
    graph, params = _terminal_graph(cfg, overrun=overrun)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, seed), cfg)
    return cfg, graph, params, model, view


def test_a_restricted_search_leaves_the_graphs_budget_memo_alone():
    """The bootstrap shares `dag_budget` with the search it only means to inform.

    Both directions are wrong and both are silent. Reading it, a 4-departure bootstrap would
    allocate the FULL search's pool for a search that needs a sliver. Writing it, the memo
    would end up holding the bootstrap's tiny budget -- and since the write sits past the
    `status != OK` guard, a DECLINING flight (the one the bootstrap exists for) would keep
    that number and re-climb the ladder from it on every later iteration, which is the 20%
    cost that made raising the ceiling a regression in the first place.
    """

    cfg, graph, _params, model, view = _bootstrap_fixture()

    pricing._best_column_compiled(graph, view, 0.0, cfg, 100.0, frozenset(), model=model)
    with graph._search_cache.lock:
        full_budget = graph._search_cache.dag_budget
    assert full_budget is not None, "the unrestricted search recorded nothing to protect"

    pricing._best_column_compiled(
        graph, view, 0.0, cfg, 100.0, frozenset(), model=model,
        keep_roots=_rank_roots(graph, cfg, view, model, 1)[1], record_budget=False,
    )
    with graph._search_cache.lock:
        assert graph._search_cache.dag_budget == full_budget


def test_the_bootstrap_only_ever_returns_a_certified_improvement():
    """Uncertified is the one failure here that does not raise.

    A cutoff above the true optimum discards it, and the search then returns a worse column
    with no crash, no fallback and no signal. The defence is that `_bootstrap_incumbent`
    never scores anything itself -- it returns what one of the two real searches returned,
    and both have already been through `_canonical_candidate`. This checks the property that
    guarantees: the score it hands back is achievable by the column it hands back.
    """

    cfg, graph, params, model, view = _bootstrap_fixture()
    benefit = 100.0

    for roots in (1, 2, 4, 64):
        out = pricing._bootstrap_incumbent(
            graph, view, 0.0, cfg, benefit, frozenset(), model,
            incumbent=None, roots=roots,
        )
        if out is None:
            continue
        score, column = out
        achievable = model.reduced_cost(
            benefit=benefit, cost=column.delay_s,
            dual_cost=view.claim_cost(column.claims), pi_f=0.0,
        )
        assert score == achievable, f"{roots}: score is not the column's own"

    # Given an incumbent it cannot beat, it returns that incumbent UNCHANGED -- so a
    # bootstrap that finds nothing leaves the caller exactly where it was.
    unbeatable = (1e9, pricing.seed_column(graph, cfg, model=model))
    assert pricing._bootstrap_incumbent(
        graph, view, 0.0, cfg, benefit, frozenset(), model,
        incumbent=unbeatable, roots=4,
    ) is unbeatable


def test_the_bootstrap_reaches_the_real_search_only_as_an_incumbent(monkeypatch):
    """Its column may be a cutoff and must never be a candidate.

    A restricted search's labels win dominance slots, so it emits sinks the unrestricted
    search never does (see `test_kernel_and_reference_restrict_departures_identically`). One
    of those entering the real ranking would be a column the reference could not have
    returned -- optimal or not, that is a different answer. The separation is what keeps the
    bootstrap a pruning aid rather than a second source of columns.
    """

    cfg, graph, params, model, view = _bootstrap_fixture()
    seen = []
    real = pricing._certify_candidates

    def spy(candidates, *args, **kwargs):
        seen.append((len(candidates), kwargs.get("incumbent")))
        return real(candidates, *args, **kwargs)

    monkeypatch.setattr(pricing, "_certify_candidates", spy)
    boot = pricing._bootstrap_incumbent(
        graph, view, 0.0, cfg, 100.0, frozenset(), model,
        incumbent=None, roots=2,
    )
    assert seen, "the bootstrap never reached the ranking, so this proves nothing"
    assert boot is not None, "the bootstrap found nothing, so this proves nothing"

    seen.clear()
    pricing._best_column_compiled(
        graph, view, 0.0, cfg, 100.0, frozenset(), incumbent=boot, model=model
    )
    assert len(seen) == 1
    _n_candidates, ranked_incumbent = seen[0]
    # The real search's ranking starts from the kernel's own mid-sweep incumbent, which the
    # bootstrap's cutoff can only have improved -- never from a bootstrap CANDIDATE.
    assert ranked_incumbent is not None
    assert ranked_incumbent[0] >= boot[0]


def test_kernel_survives_a_budget_restart_with_the_same_answer(monkeypatch):
    """A pool too small to finish is re-run from scratch, and must re-run identically.

    The restart is where the envelope memo can betray parity: envelopes frozen against a
    mid-sweep incumbent are a strictly stronger prune, so a second attempt starting from
    the cutoff the first one *reached* would explore less than the first -- and the
    reference's column is defined by a search that never restarted. `rewind` is what puts
    the cutoff and the memo back, and this is the test that it does.

    The capacity is deliberately far too small, so several restarts happen rather than one.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model, label_capacity=1024)
    assert result.ok, result.status
    assert result.attempts > 1, "the fixture no longer forces a restart"

    reference_side = _reference_sink_set(graph, cfg, view, params, monkeypatch, model)
    assert set(kernel_side) == reference_side


def test_kernel_declines_at_the_capacity_ceiling_rather_than_allocating_past_it(monkeypatch):
    """Budget growth stops somewhere, and the stop is a decline rather than a `MemoryError`.

    `_best_column_compiled` documents "a budget the kernel could not grow into" as one of
    its `Declined` cases, and without a ceiling that case is unreachable: the retry
    loop grows geometrically and `np.zeros` raises long before `max_attempts` runs out.
    `MemoryError` is caught nowhere on the pricing path, so it would take the solve down --
    and under a worker pool an OOM-killed worker hangs the sweep forever
    (`pricing_pool`'s module docstring).

    The fixture is the one `test_kernel_survives_a_budget_restart_with_the_same_answer`
    uses, which needs several restarts at this capacity.  Pinning the ceiling AT the
    starting capacity is therefore the sharpest form of the contract: the same search that
    completes when it may grow now returns `STATUS_LABEL_LIMIT` without ever reallocating.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    monkeypatch.setattr(dp_kernel, "MAX_LABEL_CAPACITY", 1024)
    result, _ = _kernel_candidates(graph, cfg, view, model, label_capacity=1024)

    assert not result.ok
    assert result.status == dp_kernel.STATUS_LABEL_LIMIT
    assert result.attempts == 1, "the ceiling was reached, so nothing should have regrown"


def _quiet_kernel_telemetry(monkeypatch):
    """Reset the per-process warn flags and tally, so a test sees only its own run.

    All three are module globals by design -- warn ONCE per process, tally per process --
    which makes them order-dependent across tests unless each one starts clean.
    """

    monkeypatch.setattr(pricing, "_kernel_restart_warned", False)
    monkeypatch.setattr(pricing, "_kernel_budget_warned", False)
    monkeypatch.setattr(
        pricing, "_KERNEL_STATS",
        Counter({"priced": 0, "fell_back": 0, "label_restarts": 0, "budget_declined": 0}),
    )


def _compiled_or_fail(*args, **kwargs):
    """``_best_column_compiled``'s answer, failing the test if it declined instead.

    Every caller here is asserting that the compiled path SERVED this graph, so a
    ``Declined`` is a test failure and the reason is the useful part of the message.
    Collected in one place rather than repeated per call site, the way ``_feasible_both``
    already is for the other search.
    """

    outcome = pricing._best_column_compiled(*args, **kwargs)
    assert not isinstance(outcome, pricing.Declined), f"compiled search declined: {outcome}"
    return outcome


def test_a_restarted_label_pool_is_counted_and_announced_once(monkeypatch, capsys):
    """A restart is invisible except as a slow flight, so it says so on the way past.

    `DagResult.attempts` has always carried this -- "the number to read when a flight is
    unexpectedly expensive" -- and nothing read it.  The graph-cached `dag_budget` is the
    documented way in: seeding it small is exactly what a first iteration on a big flight
    does to itself.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)
    _quiet_kernel_telemetry(monkeypatch)
    graph._search_cache.dag_budget = (1024, 14, 4096)

    _rc, column = _compiled_or_fail(
        graph, view, 0.0, cfg, 100.0, frozenset(), model=model
    )

    assert column is not None, "a restart must not change the outcome"
    assert pricing.kernel_stats()["label_restarts"] >= 1
    assert pricing.kernel_stats()["budget_declined"] == 0
    warning = capsys.readouterr().err
    assert "restarted its label pool" in warning
    assert f"flight {graph.request.flight_id}" in warning


def test_a_search_that_hits_the_capacity_ceiling_says_so_before_falling_back(
    monkeypatch, capsys
):
    """The decline names the constant, because it is a knob and not a wall.

    This is the one cause of `fell_back` that gets WORSE as the instance grows, so it is
    also the one worth telling apart from "numba is missing" -- which has carried its own
    warn-once for the same reason since the compiled path shipped.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)
    _quiet_kernel_telemetry(monkeypatch)
    monkeypatch.setattr(dp_kernel, "MAX_LABEL_CAPACITY", 1024)
    graph._search_cache.dag_budget = (1024, 14, 4096)

    outcome = pricing._best_column_compiled(
        graph, view, 0.0, cfg, 100.0, frozenset(), model=model
    )

    # The REASON, not just "it declined": this is the one cause that gets worse as the
    # instance grows, and the one whose remedy is a constant rather than an install.
    assert outcome is pricing.Declined.LABEL_BUDGET
    assert pricing.kernel_stats()["budget_declined"] == 1
    warning = capsys.readouterr().err
    assert "MAX_LABEL_CAPACITY" in warning
    assert "pure-Python reference" in warning


def test_destination_lane_tie_comes_from_the_envelopes_not_the_packed_lanes(monkeypatch):
    """The kernel must tie-break on the REFERENCE's destination lane, not the packed one.

    Two different mins. `CompletionEnvelopes` takes it over every lane of every
    `_destination_options(fg)` entry, verbatim as `pricing.py` does.
    `topology.dest_lane_idx` holds only the lanes of destination cells that survived
    interning (`cells = set(reachable) | claim_only`), a filter `prepare_topology` spells
    out itself as `[index[cell] for cell in destination_options if cell in index]`.

    They coincide until a destination cell is not forward-reachable from any origin. If
    that cell held the lowest lane, the packed min is HIGHER, and the value reaches
    `_prefix_le`'s four-field comparison -- so the compiled search could certify a
    different, equally optimal column while still reporting `proved=True`.

    Measured: the filter drops nothing across 260 flights on `colgen_test` and the four
    density arms, so no natural fixture exhibits it. Rather than wait for one, this pins
    the SOURCE: the envelope's value is what reaches the kernel even when the packed array
    disagrees, which is what makes the divergence unreachable instead of merely unobserved.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    duals = dp_prepare.prepare_duals(view, graph, topology, rows)
    envelopes = dp_prepare.CompletionEnvelopes(
        graph, cfg, view, benefit=100.0, pi_f=0.0, model=model, forbidden_rows=frozenset(),
    )
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, rows, benefit=100.0, pi_f=0.0, cost_cutoff=None,
        model=model, forbidden_rows=frozenset(), envelopes=envelopes,
    )
    pack = dp_prepare.prepare_forbidden(frozenset(), graph, rows, topology)

    packed_min = int(topology.dest_lane_idx.min())
    sentinel = packed_min - 7  # a value the packed array cannot produce
    envelopes.destination_lane_tie = sentinel

    seen = []
    real = dp_kernel._price_dag

    def spy(*args):
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(dp_kernel, "_price_dag", spy)
    dp_kernel.price_dag(
        topology, rows, duals, variants, pack,
        air_weight=model.air_weight, dt_s=cfg.dt_s, benefit=100.0, pi_f=0.0,
        envelopes=envelopes,
    )

    assert seen, "the kernel never ran, so this asserts nothing"
    # Scalars only: most of the argument list is numpy arrays, and `in` on those compares
    # elementwise and raises rather than answering.
    scalars = [
        int(arg) for arg in seen[0]
        if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool)
    ]
    assert sentinel in scalars, "the kernel took the packed min, not the envelope's tie"


def test_a_correctness_stop_is_not_offered_the_budget_remedy(monkeypatch, capsys):
    """`FSUM_OVERFLOW` reaches the same branch as a full pool and means the opposite.

    Both are "the compiled search declined", so both land in `_warn_budget_growth` -- but
    one is a knob and the other is the kernel refusing to report a score it cannot stand
    behind.  Telling someone to raise a capacity ceiling for the second would be worse than
    printing nothing, which is why the remedy is chosen from the status rather than shared.

    Driven through `_warn_budget_growth` directly: the condition is unreachable on a real
    graph (this kernel's arc sums are a handful of terms), and a test that could only reach
    it by faking the search would be asserting the fake.
    """

    _quiet_kernel_telemetry(monkeypatch)

    class _Declined:
        status = dp_kernel.STATUS_FSUM_OVERFLOW
        attempts = 1
        budget = (65536, 14, 4096)

    graph, _params = _terminal_graph(_cfg())
    pricing._warn_budget_growth(dp_kernel, graph, _Declined())

    assert pricing.kernel_stats()["budget_declined"] == 1
    warning = capsys.readouterr().err
    assert "FSUM_OVERFLOW" in warning
    assert "CORRECTNESS stop" in warning
    assert "MAX_LABEL_CAPACITY" not in warning, "budget advice on a correctness stop"


def test_declined_survives_a_process_boundary():
    """The bug the `object()` sentinel it replaces actually had.

    A module-level `object()` pickles happily and arrives in a worker as a DIFFERENT
    instance, so `result is _UNPROVED` was always False across a process boundary. Nothing
    sequential could catch that -- nothing is pickled there -- so it would have surfaced
    first on a production timeout under a pool. An `Enum` pickles by name and round-trips
    to the same object.

    Narrow to the round trip on purpose: that a decline is distinguishable from `None` is
    already tested THROUGH `find_feasible_column`, which is the better test of it.
    """

    for member in pricing.Declined:
        assert pickle.loads(pickle.dumps(member)) is member


def test_every_kernel_stop_status_maps_to_its_own_reason():
    """Distinct reasons, and an unmapped status degrades to a fallback rather than a crash.

    "numba isn't installed" and "a Shewchuk partial expansion saturated on real data" used
    to be the same value, and they warrant opposite responses. The catch-all matters for a
    different reason: a closed `[...]` lookup would turn a future kernel status into a
    `KeyError` raised out of the function whose entire job is to decline gracefully.
    """

    mapped = {
        dp_kernel.STATUS_LABEL_LIMIT: pricing.Declined.LABEL_BUDGET,
        dp_kernel.STATUS_STATE_LIMIT: pricing.Declined.STATE_BUDGET,
        dp_kernel.STATUS_CANDIDATE_LIMIT: pricing.Declined.HEAP_BUDGET,
        dp_kernel.STATUS_FSUM_OVERFLOW: pricing.Declined.FSUM_OVERFLOW,
    }
    for status, expected in mapped.items():
        assert pricing._status_reason(dp_kernel, status) is expected
    assert len(set(mapped.values())) == len(mapped), "two statuses share a reason"

    unmapped = max(dp_kernel.STATUS_NAMES) + 100
    assert pricing._status_reason(dp_kernel, unmapped) is pricing.Declined.KERNEL_STATUS


def test_each_structural_decline_names_its_own_cause(monkeypatch):
    """The four guards that fire before the kernel runs, each with its own reason.

    These are the ones a `proved=False` boolean flattened into a single value: "fix your
    install", "this scope is unimplemented", and "the packer refused this graph" want
    different responses from whoever reads the run.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)
    call = (graph, view, 0.0, cfg, 100.0, frozenset())

    monkeypatch.setattr(pricing, "_dp_kernel", lambda: None)
    assert pricing._best_column_compiled(*call, model=model) is pricing.Declined.NO_NUMBA
    monkeypatch.undo()

    # Only `len(fg.levels)` is read before this guard returns, so a stand-in is honest here
    # and building a genuinely multi-level graph would be testing the builder instead.
    two_level = types.SimpleNamespace(levels=(0, 1))
    assert pricing._best_column_compiled(
        two_level, view, 0.0, cfg, 100.0, frozenset(), model=model
    ) is pricing.Declined.MULTI_LEVEL

    for refused, expected in (
        ("topology", pricing.Declined.TOPOLOGY),
        ("rows", pricing.Declined.ROWS),
    ):
        topology, rows = dp_prepare.prepared_for(graph, cfg)
        stubs = {"topology": topology, "rows": rows}
        stubs[refused] = types.SimpleNamespace(ok=False, unsupported_reason="stubbed")
        monkeypatch.setattr(
            dp_prepare, "prepared_for",
            lambda *a, **k: (stubs["topology"], stubs["rows"]),
        )
        assert pricing._best_column_compiled(*call, model=model) is expected
        monkeypatch.undo()


@pytest.mark.parametrize(
    "capacity, step_reached, expected",
    [
        # No layer was relaxed at all, so there is nothing to extrapolate from: double.
        (1000, -1, 2000),
        # Filled a tenth of the way in; 1.25 * 10x wants 12.5x, and the ceiling caps it.
        (1000, 9, 8000),
        # Filled a third of the way; 1.25 * 3x = 3.75x, inside both bounds.
        (1000, 32, 3787),
        # Nearly finished; 1.25 * 1.02x is below the doubling floor, which wins.
        (1000, 97, 2000),
    ],
)
def test_kernel_retry_capacity_is_bounded_in_both_directions(
    capacity, step_reached, expected
):
    """Extrapolating beats doubling, but only between a floor and a ceiling.

    The floor matters because labels per step are not uniform -- the frontier widens before
    it plateaus, so an estimate taken early reads low and would ask for less than doubling.
    The ceiling matters because a pathological early fill would otherwise ask for gigabytes:
    at 40 bytes a label, 8x of a 13.3M pool is already 4.3 GB.
    """

    assert dp_kernel._next_label_capacity(capacity, step_reached, 0, 99) == expected


def test_kernel_honours_forbidden_rows(monkeypatch):
    """Repair's exclusion set, applied inside the kernel rather than by a Python fallback.

    Required coverage rather than a nice-to-have: the sweep always passes ``_EMPTY_ROWS``
    (solver.py), so without a test that constructs one explicitly the whole bitset path
    ships unexercised.

    The inclusion is one-directional here, deliberately. ``consider_sink`` drops a sink
    whose destination-endpoint or path claims touch an excluded row, and doing that in the
    kernel would mean reproducing both endpoint span rules in numba -- a second place for
    them to drift, to save work that Tier 2 redoes anyway. So the kernel proposes those
    sinks and the assertion below pins exactly that: every extra it proposes is one the
    reference rejected on a forbidden gate, and none of them can survive certification.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    rng = random.Random(2718)
    forbidden = frozenset(
        RowKey.cell(cell[0], cell[1], 0, step)
        for cell in sorted(graph.corridor_cells)[:12]
        for step in range(graph.min_step + 4, graph.min_step + 16)
        if rng.random() < 0.25
    )
    assert len(forbidden) > 10, "the fixture forbade too few rows to be a real check"

    result, kernel_side = _kernel_candidates(graph, cfg, view, model, forbidden)
    assert result.ok, result.status

    reference_side = _reference_sink_set(
        graph, cfg, view, params, monkeypatch, model, forbidden=forbidden
    )
    assert reference_side, "the exclusion set killed every sink, so this proves nothing"
    missing = reference_side - set(kernel_side)
    assert not missing, f"kernel missed {len(missing)}, e.g. {sorted(missing)[0]}"

    extra = set(kernel_side) - reference_side
    for departure_step, origin_lane, dest_lane, path in extra:
        label = pricing._Label(0.0, departure_step, origin_lane, path, frozenset())
        claims = pricing._path_claims(graph, cfg, label, dest_lane)
        assert not claims.isdisjoint(forbidden), (
            "the kernel proposed a sink the reference did not, and it is not one the "
            "forbidden-row gates explain"
        )


# ------------------------------------------------------------------- label state helpers


def test_kernel_arc_role_bits_match_dp_prepare():
    """Restated for numba as compile-time constants; they must not drift from the packer."""

    assert dp_kernel._ARC_INTERNAL == dp_prepare.ARC_INTERNAL
    assert dp_kernel._ARC_FIRST == dp_prepare.ARC_FIRST
    assert dp_kernel._ARC_LAST == dp_prepare.ARC_LAST
    assert dp_kernel._ARC_FIRST_LAST == dp_prepare.ARC_FIRST_LAST


def test_kernel_role_gate_matches_the_graphs_own_verdict():
    """`_role_allows` reads a packed mask the way `fg.hop_allowed_for_role` answers."""

    cfg = _cfg()
    graph, _ = _graph(cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))

    checked = 0
    for source_index, source in enumerate(cells):
        for arc in range(int(topology.arc_start[source_index]),
                         int(topology.arc_start[source_index + 1])):
            target = cells[int(topology.arc_target[arc])]
            roles = int(topology.arc_roles[arc])
            for first in (False, True):
                for last in (False, True):
                    assert dp_kernel._role_allows(roles, first, last) == (
                        graph.hop_allowed_for_role(source, target, first=first, last=last)
                    )
                    checked += 1
    assert checked > 200, "the fixture had too few arcs to be a real check"


def test_kernel_recent_matches_the_references_tuple():
    """`_fill_recent` reproduces `(neighbour, *recent[:depth - 1])`, length included."""

    rng = random.Random(3)
    pool = _random_pool(rng, n_labels=90, depth=7)
    _score, _hops, _dep, _lane, parent, cell = pool.arrays()

    for depth in (2, 3, 4):
        out = np.zeros(depth, np.int32)
        for label in range(len(pool.labels)):
            path = [pool.cell[i] for i in _chain(pool.parent, label)]
            expected = tuple(reversed(path))[:depth]
            n = dp_kernel._fill_recent(label, depth, parent, cell, out)
            assert tuple(out[:n].tolist()) == expected
            assert n == min(len(path), depth)


def test_kernel_recent_compare_matches_python_tuple_ordering():
    """Including the prefix rule -- a shorter history sorts before a longer one."""

    rng = random.Random(5)
    for _ in range(2000):
        a = [rng.randint(0, 3) for _ in range(rng.randint(1, 4))]
        b = [rng.randint(0, 3) for _ in range(rng.randint(1, 4))]
        buf_a = np.asarray(a + [0] * 4, np.int32)
        buf_b = np.asarray(b + [0] * 4, np.int32)
        expected = int(tuple(a) > tuple(b)) - int(tuple(a) < tuple(b))
        assert dp_kernel._recent_cmp(buf_a, len(a), buf_b, len(b)) == expected


def test_kernel_layer_sort_matches_the_references_sorted_call():
    """`_sort_layer` orders a layer exactly as `sorted(..., key=(cell, recent, tie_key))`.

    Load-bearing rather than cosmetic: relaxation order decides which label reaches the
    next layer's slot first, and `_prefer` is non-transitive inside its epsilon band.
    """

    rng = random.Random(13)
    pool = _random_pool(rng, n_labels=300, depth=6)
    score, hops, departure, lane, parent, cell = pool.arrays()
    depth = 3

    for n in (0, 1, 2, 17, 300):
        items = np.asarray(rng.sample(range(len(pool.labels)), n), np.int32)
        buffer = np.zeros(max(n, 1), np.int32)
        recent_a, recent_b = np.zeros(depth, np.int32), np.zeros(depth, np.int32)
        scratch_a, scratch_b = np.zeros(64, np.int32), np.zeros(64, np.int32)
        expected = sorted(
            items.tolist(),
            key=lambda label: (
                pool.cell[label],
                tuple(reversed([pool.cell[i] for i in _chain(pool.parent, label)]))[:depth],
                (
                    pool.hops[label],
                    pool.departure[label],
                    pool.lane[label],
                    tuple(pool.cell[i] for i in _chain(pool.parent, label)),
                ),
            ),
        )
        dp_kernel._sort_layer(
            items, buffer, n, depth, score, cell, parent, hops, departure, lane,
            recent_a, recent_b, scratch_a, scratch_b,
        )
        assert items.tolist() == expected, n


def test_kernel_state_table_finds_what_it_inserted_and_separates_keys():
    """Insert/lookup round trip, and two labels differing in one key field never merge."""

    rng = random.Random(17)
    pool = _random_pool(rng, n_labels=200, depth=5)
    _score, _hops, _dep, _lane, parent, cell = pool.arrays()
    depth = 3
    log2cap = 10
    slot_label = np.full(1 << log2cap, -1, np.int32)
    # uint64, matching `_state_hash`'s return: an int64 slot would overflow on any key whose
    # Fibonacci mix lands above 2**63, which is half of them.
    slot_hash = np.zeros(1 << log2cap, np.uint64)
    # One paid class per label id parity, so the field genuinely varies.
    variant = np.arange(len(pool.labels), dtype=np.int32)
    var_paid_class = np.asarray([i % 3 for i in range(len(pool.labels))], np.int32)
    probe = np.zeros(depth, np.int32)
    recent = np.zeros(depth, np.int32)
    no_first = np.full(len(pool.labels), -1, np.int32)

    placed = {}
    for label in range(len(pool.labels)):
        n = dp_kernel._fill_recent(label, depth, parent, cell, recent)
        paid_class = int(var_paid_class[label])
        key_hash = np.uint64(dp_kernel._state_hash(int(cell[label]), recent, n, paid_class, -1, -1))
        slot, found = dp_kernel._state_find(
            slot_label, slot_hash, log2cap, key_hash, depth,
            int(cell[label]), recent, n, paid_class, -1, -1,
            cell, parent, variant, var_paid_class, no_first, no_first, probe,
        )
        assert slot >= 0
        key = (int(cell[label]), tuple(recent[:n].tolist()), paid_class)
        if key in placed:
            assert found and slot_label[slot] == placed[key]
        else:
            assert not found
            slot_label[slot] = label
            slot_hash[slot] = key_hash
            placed[key] = label

    # Every distinct key got its own slot -- no two merged.
    assert len(placed) == len({int(slot_label[s]) for s in range(1 << log2cap)
                               if slot_label[s] >= 0})
    assert len(placed) > 30, "the fixture produced too few distinct states"


def test_kernel_visit_forbidden_matches_the_reference_window_test():
    """`_visit_hits_forbidden` over row ids answers as `pricing._visit_hits_forbidden` does."""

    cfg = _cfg()
    graph, _ = _graph(cfg)
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    offsets = DualView({}, cfg).offsets

    rng = random.Random(23)
    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    forbidden = set()
    for cell in cells[:20]:
        for step in range(graph.min_step, min(graph.min_step + 30, graph.max_step + 1)):
            if rng.random() < 0.25:
                forbidden.add(RowKey.cell(cell[0], cell[1], 0, step))
    pack = dp_prepare.prepare_forbidden(forbidden, graph, rows, topology)

    hits = 0
    for cell_index, cell in enumerate(cells):
        for visit_step in range(rows.step0, rows.step0 + min(rows.n_steps, 50)):
            expected = pricing._visit_hits_forbidden(cell, 0, visit_step, offsets, forbidden)
            actual = dp_kernel._visit_hits_forbidden(
                pack.bits, rows.n_steps, rows.step0, cell_index, visit_step,
                offsets[0], offsets[1],
            )
            assert actual == expected, (cell, visit_step)
            hits += int(expected)
    assert hits > 50, "the fixture forbade too little to exercise the positive branch"


def test_kernel_paid_correction_matches_the_references_fsum_expression():
    """The double-charge correction, term for term and in the same summation order."""

    partials = np.zeros(dp_kernel.FSUM_MAX_PARTIALS, np.float64)
    # Two classes: class 0 has rows on cells 5 and 9, class 1 is empty.
    paid_start = np.asarray([0, 4, 4], np.int32)
    paid_cell = np.asarray([5, 5, 5, 9], np.int32)
    paid_step = np.asarray([10, 11, 12, 11], np.int32)
    paid_value = np.asarray([1e16, 1.0, -1e16, 7.5], np.float64)

    # Window [11, 12] on cell 5 picks rows 1 and 2 -- magnitudes chosen so `+=` would lose
    # the 1.0 entirely and only an exact sum recovers it.
    value, ok = dp_kernel._paid_visit_correction(
        paid_start, paid_cell, paid_step, paid_value, 0, 5, 11, 0, 1, partials
    )
    assert ok and value == math.fsum([1.0, -1e16])

    # A cell with no paid rows in range answers zero, which is the reference's `in` guard.
    value, ok = dp_kernel._paid_visit_correction(
        paid_start, paid_cell, paid_step, paid_value, 0, 7, 11, 0, 1, partials
    )
    assert ok and value == 0.0
    value, ok = dp_kernel._paid_visit_correction(
        paid_start, paid_cell, paid_step, paid_value, 1, 5, 11, 0, 1, partials
    )
    assert ok and value == 0.0


# -------------------------------------------------------------- the production entry point


def test_price_flight_takes_the_compiled_path_and_returns_the_reference_column():
    """`price_flight` proves the compiled search and returns exactly what the oracle does.

    The end-to-end shape of Phase 2c, and the one that would otherwise ship silently wrong:
    every earlier test drives `price_dag` directly, so none of them notices if
    `_best_column_compiled` falls back on every flight -- which reads as a correct, slow
    solve rather than as a failure. Hence `_compiled_or_fail`, which fails on a `Declined`,
    as well as the assertion on the column.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)
    seed = pricing.seed_column(graph, cfg, model=model)
    incumbent = (
        model.reduced_cost(
            benefit=100.0, cost=seed.delay_s, dual_cost=view.claim_cost(seed.claims), pi_f=0.0
        ),
        seed,
    )

    rc, column = _compiled_or_fail(
        graph, view, 0.0, cfg, 100.0, frozenset(), incumbent=incumbent, model=model
    )
    assert column is not None

    ref_rc, ref_column = pricing._best_column(
        graph, view, 0.0, cfg, 100.0, frozenset(), seed=False,
        incumbent=incumbent, model=model,
    )
    assert column == ref_column
    assert rc == ref_rc


def test_compiled_path_honours_forbidden_rows_without_falling_back():
    """Repair runs INSIDE the kernel; a non-empty exclusion set is not a fallback trigger.

    Required rather than optional: the sweep always passes `_EMPTY_ROWS` (solver.py), so
    without this the whole repair path ships untested, and routing repair to Python was
    rejected because it is O(flights) inside the greedy.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    rng = random.Random(4242)
    forbidden = frozenset(
        RowKey.cell(cell[0], cell[1], 0, step)
        for cell in sorted(graph.corridor_cells)[:12]
        for step in range(graph.min_step + 4, graph.min_step + 16)
        if rng.random() < 0.25
    )
    assert len(forbidden) > 10

    rc, column = _compiled_or_fail(
        graph, view, 0.0, cfg, 100.0, forbidden, incumbent=None, model=model
    )
    ref_rc, ref_column = pricing._best_column(
        graph, view, 0.0, cfg, 100.0, forbidden, seed=False, incumbent=None, model=model
    )
    assert column == ref_column
    assert rc == ref_rc
    if column is not None:
        assert column.claims.isdisjoint(forbidden)


def test_compiled_path_respects_the_pricing_deadline():
    """An expired deadline raises `PricingTimeout` rather than running to completion.

    An `@njit(nogil=True)` function cannot read a clock, and with geometric budget growth
    one call can run for minutes -- so the deadline is carried by a watchdog that sets the
    kernel's `cancel` flag, polled per time layer. `solver.py` turns the exception into
    `termination_reason = "time_limit"`, so swallowing it here would report a converged
    solve that was actually cut off.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    with pytest.raises(pricing.PricingTimeout):
        pricing._best_column_compiled(
            graph, view, 0.0, cfg, 100.0, frozenset(),
            incumbent=None, deadline=time.monotonic() - 1.0, model=model,
        )


def test_compiled_path_weights_the_label_score_in_the_objective_currency():
    """A ground-heavy `CostModel` must still return the reference's column.

    `[[colgen-label-score-currency]]`: at unit weights `ground + flown` is invariant within
    a time layer, so the objective and raw seconds coincide and a mis-weighted score is
    dormant. Under `total_cost` they diverge -- trading one step of ground for one hop of
    air is free in seconds and worth `2*dt` in cost -- so only an asymmetric model can tell
    the two apart.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = CostModel(ground_weight=9.0, air_weight=1.0)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    rc, column = _compiled_or_fail(
        graph, view, 0.0, cfg, 100.0, frozenset(), incumbent=None, model=model
    )
    ref_rc, ref_column = pricing._best_column(
        graph, view, 0.0, cfg, 100.0, frozenset(), seed=False, incumbent=None, model=model
    )
    assert column == ref_column
    assert rc == ref_rc


@pytest.mark.parametrize("overrun", [0, 1, 3, 9])
def test_compiled_path_matches_the_reference_across_hop_ceilings(overrun):
    """The route-length bound is the only knob shaping the corridor, so sweep it.

    `max_air_overrun_hops` sizes `max_air_hops`, which is simultaneously the ceiling, the
    per-arc lookahead that truncates weaving, and (post-#78) the corridor. A kernel that
    read the ceiling correctly but reconstructed the lookahead would pass at the shipped
    value and fail everywhere else, which is what `[[colgen-ceiling-pairs-with-slack]]`
    describes from the other direction.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg, overrun=overrun)
    model = cost_model(cfg, params)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    rc, column = _compiled_or_fail(
        graph, view, 0.0, cfg, 100.0, frozenset(), incumbent=None, model=model
    )
    ref_rc, ref_column = pricing._best_column(
        graph, view, 0.0, cfg, 100.0, frozenset(), seed=False, incumbent=None, model=model
    )
    assert column == ref_column
    assert rc == ref_rc


def test_prepared_topology_is_built_once_per_graph():
    """The packing is cached on the graph, because it is rebuilt every colgen iteration.

    Measured at up to 80% of the compiled search's own time on a cheap density flight --
    enough to make the compiled path a regression on the majority of a real sweep. The
    assertion is on the CALL COUNT rather than on wall time so it cannot pass by accident
    on a fast machine.
    """

    cfg = _cfg()
    graph, _ = _terminal_graph(cfg)
    calls = []
    real = dp_prepare.prepare_topology

    def counted(fg, config):
        calls.append(fg.request.flight_id)
        return real(fg, config)

    dp_prepare.prepare_topology = counted
    try:
        first = dp_prepare.prepared_for(graph, cfg)
        second = dp_prepare.prepared_for(graph, cfg)
    finally:
        dp_prepare.prepare_topology = real

    assert len(calls) == 1, f"prepare_topology ran {len(calls)} times for one graph"
    assert first[0] is second[0] and first[1] is second[1]
    # A transported graph starts cold: the lock cannot be pickled and the payload is a
    # memo, so `__reduce__` deliberately drops it.
    assert type(graph._search_cache)().prepared is None


def _random_case(rng, index):
    """One randomized pricing subproblem: geometry, hop ceiling, endpoints and duals."""

    cfg = _cfg(max_ground_delay_s=rng.choice((16.0, 48.0, 96.0)))
    origin = (0, 0)
    dest = (rng.randint(2, 5), rng.randint(-3, 1))
    if dest == origin:
        dest = (3, -1)
    overrun = rng.choice((0, 1, 3, 9))
    terminal = bool(index % 2)
    o, d = _point(origin, cfg), _point(dest, cfg)
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    if terminal:
        o_term = Terminal(f"rnd-A{index}", 1, radius=90.0)
        d_term = Terminal(f"rnd-B{index}", 1, radius=90.0)
        request = FlightRequest(
            index, o, d, 0.0, 0.0, origin_terminal=o_term, dest_terminal=d_term
        )
        statics = [(o, o_term), (d, d_term)]
    else:
        request = FlightRequest(index, o, d, 0.0, 0.0)
        statics = ()
    graph = build_flight_graph(request, cfg, statics, params)
    model = rng.choice(
        (cost_model(cfg, params), CostModel(ground_weight=9.0, air_weight=1.0))
    )
    return cfg, graph, model, DualView(_random_duals(graph, cfg, rng.randrange(1 << 30)), cfg)


@pytest.mark.parametrize("bootstrap", [0, 1])
def test_compiled_path_returns_the_same_column_as_the_reference_over_random_graphs(bootstrap):
    """The load-bearing sweep: the same COLUMN, not merely the same reduced cost.

    Reduced-cost equality is the weak claim and the one that hides the real failure. Two
    equally optimal columns score identically, so a search that broke a dominance tie the
    other way passes an RC check and still returns a different trajectory -- which changes
    the next iteration's duals and compounds across the solve. Hence `column == ref_column`.

    Randomized over the axes that reshape the search rather than merely rescale it:
    geometry (so `shortest_hops` and the corridor move), the hop ceiling (the only
    route-length bound post-#78, which is simultaneously ceiling, per-arc lookahead and
    corridor), the endpoint shape (terminal turns on `track_first_hop` and the unpadded
    `term`-row span rule), the objective weights (`[[colgen-label-score-currency]]`), and
    the duals.

    Run with the bootstrap off and on. On, BOTH searches receive its cutoff -- the same
    object, from `price_flight`'s single call -- so the equality claim is unchanged and the
    space each explores is merely smaller. That is the whole reason the bootstrap lives in
    the shared caller: applied to one search only, this sweep is what would fail.
    """

    rng = random.Random(20260810)
    compared = nontrivial = 0
    for index in range(40):
        cfg, graph, model, view = _random_case(rng, index)
        incumbent = None
        if index % 3 == 0:
            try:
                seed = pricing.seed_column(graph, cfg, model=model)
            except ValueError:
                seed = None
            if seed is not None:
                incumbent = (
                    model.reduced_cost(
                        benefit=100.0,
                        cost=seed.delay_s,
                        dual_cost=view.claim_cost(seed.claims),
                        pi_f=0.0,
                    ),
                    seed,
                )

        if bootstrap:
            incumbent = pricing._bootstrap_incumbent(
                graph, view, 0.0, cfg, 100.0, frozenset(), model,
                incumbent=incumbent, roots=bootstrap,
            )

        outcome = pricing._best_column_compiled(
            graph, view, 0.0, cfg, 100.0, frozenset(), incumbent=incumbent, model=model
        )
        if isinstance(outcome, pricing.Declined):
            continue
        rc, column = outcome
        ref_rc, ref_column = pricing._best_column(
            graph, view, 0.0, cfg, 100.0, frozenset(), seed=False,
            incumbent=incumbent, model=model,
        )
        compared += 1
        nontrivial += column is not None
        assert column == ref_column, f"case {index}: different column"
        assert rc == ref_rc, f"case {index}: {rc!r} != {ref_rc!r}"

    assert compared >= 30, f"only {compared} cases ran compiled; the sweep proves little"
    assert nontrivial >= 20, f"only {nontrivial} cases found a column at all"


# ------------------------------------------------------ the compiled feasible search


def _feasible_both(graph, cfg, model, **kwargs):
    """Run `find_feasible_column` compiled and again with the compiled path refused.

    The graph is prepared and seeded first, deliberately. In production pricing has already
    done both, and timing or comparing a COLD `prepare_topology` against a reference that
    never pays it measures the cache rather than the search -- which is exactly the mistake
    that made this search first read as a 1.03x regression.
    """

    dp_prepare.prepared_for(graph, cfg)
    pricing.seed_column(graph, cfg, model=model)
    compiled = pricing.find_feasible_column(graph, cfg, model=model, **kwargs)
    real = pricing._feasible_compiled
    pricing._feasible_compiled = lambda *a, **k: pricing.Declined.NO_NUMBA
    try:
        reference = pricing.find_feasible_column(graph, cfg, model=model, **kwargs)
    finally:
        pricing._feasible_compiled = real
    return compiled, reference


@pytest.mark.parametrize("shape", sorted(GRAPH_SHAPES))
def test_compiled_feasible_search_returns_the_reference_column(shape):
    """The greedy's incumbent search, compiled, returns the same column.

    A best-first search rather than the priced DP's layered one, so almost none of
    `_price_dag` applies: the frontier is a heap ordered by an admissible delay bound, the
    dominance table is global with `step` inside the key, and what it keeps per state is the
    lexicographically smallest PATH rather than the best score. Getting that last rule wrong
    yields a search that is still optimal and still returns a different column.
    """

    cfg = _cfg()
    graph, params = GRAPH_SHAPES[shape](cfg)
    model = cost_model(cfg, params)
    compiled, reference = _feasible_both(graph, cfg, model)
    assert compiled == reference
    assert compiled is not None, "the fixture has no feasible column, so this proves nothing"


def test_compiled_feasible_search_honours_the_early_improvement_exit():
    """`improve_below_delay_s` returns the FIRST certified strict improvement.

    That early exit is what makes this an incumbent heuristic rather than an oracle, and it
    is order-sensitive: it returns whichever improving column the frontier reaches first, so
    a kernel that expanded in a different order would return a different -- also valid --
    column and quietly change the greedy's selection.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    dp_prepare.prepared_for(graph, cfg)
    baseline = pricing.find_feasible_column(graph, cfg, model=model)
    assert baseline is not None

    threshold = baseline.delay_s + 10.0 * cfg.dt_s
    compiled, reference = _feasible_both(
        graph, cfg, model, improve_below_delay_s=threshold
    )
    assert compiled == reference
    assert compiled is not None and compiled.delay_s < threshold


def test_compiled_feasible_search_honours_forbidden_rows():
    """Repair's exclusion set, in the search that the greedy actually calls for repair."""

    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    rng = random.Random(97531)
    forbidden = frozenset(
        RowKey.cell(cell[0], cell[1], 0, step)
        for cell in sorted(graph.corridor_cells)[:10]
        for step in range(graph.min_step + 3, graph.min_step + 14)
        if rng.random() < 0.3
    )
    assert len(forbidden) > 8

    compiled, reference = _feasible_both(graph, cfg, model, forbidden_rows=forbidden)
    assert compiled == reference
    if compiled is not None:
        assert compiled.claims.isdisjoint(forbidden)


def test_compiled_feasible_search_declines_rather_than_reporting_infeasible():
    """A `Declined` is not `None`, and the difference is a flight that cannot fly.

    `find_feasible_column` returns `None` for a genuinely infeasible flight. If the compiled
    path signalled "I declined" with the same value, a missing numba would read as an
    infeasible flight and the greedy would drop it instead of falling back.
    """

    assert pricing.Declined.NO_NUMBA is not None
    cfg = _cfg()
    graph, params = _terminal_graph(cfg)
    model = cost_model(cfg, params)
    real = pricing._dp_kernel
    pricing._dp_kernel = lambda: None
    try:
        column = pricing.find_feasible_column(graph, cfg, model=model)
    finally:
        pricing._dp_kernel = real
    assert column is not None, "the reference fallback did not run"
