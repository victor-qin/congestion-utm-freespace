"""The compiled pricing kernel's primitives, against the Python they must reproduce.

Every assertion here is an *identity*, not a tolerance.  That is the point of testing these
separately: the search they support prunes on ``SCORE_EPS = 1e-12`` bands, so a primitive
that is merely accurate to a few ulps moves labels across a dominance boundary and returns a
different -- equally optimal, equally plausible -- column.  Nothing raises when that happens,
which is why it has to be caught one function at a time.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView
from freespace_sim.types import FlightRequest, vec

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
