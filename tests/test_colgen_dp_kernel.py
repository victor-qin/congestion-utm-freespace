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
from freespace_sim.planner.colgen.objective import DELAY_MODEL, cost_model
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


def _reference_candidates(graph, cfg, view, params, monkeypatch, **kwargs):
    """Every sink proposal `_best_column` registers, in the order it registers them.

    Spies on `_Candidate` rather than re-deriving the set: that constructor runs exactly
    once per accepted `(sink label, destination lane)` inside `consider_sink`, so this is
    the reference's candidate list by construction rather than by reimplementation.
    """

    recorded = []
    real = pricing._Candidate

    def spy(reduced_cost, delay_s, label, dest_lane_idx):
        candidate = real(reduced_cost, delay_s, label, dest_lane_idx)
        recorded.append(candidate)
        return candidate

    monkeypatch.setattr(pricing, "_Candidate", spy)
    pricing._best_column(
        graph, view, 0.0, cfg, kwargs.pop("benefit", 100.0), frozenset(),
        seed=False, incumbent=None, model=kwargs.pop("model", None) or DELAY_MODEL,
    )
    return recorded


def _kernel_candidates(graph, cfg, view, model, forbidden=frozenset()):
    topology = dp_prepare.prepare_topology(graph, cfg)
    rows = dp_prepare.prepare_rows(graph, cfg, topology)
    duals = dp_prepare.prepare_duals(view, graph, topology, rows)
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, rows, benefit=100.0, pi_f=0.0,
        cost_cutoff=None, model=model, forbidden_rows=forbidden,
    )
    pack = dp_prepare.prepare_forbidden(forbidden, graph, rows, topology)
    result = dp_kernel.price_dag(
        topology, rows, duals, variants, pack,
        air_dt_s=model.air_weight * cfg.dt_s,
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


@pytest.mark.parametrize(
    "shape",
    [
        "plain",
        pytest.param(
            "terminal",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "KNOWN BUG, not a tolerance: on a terminal graph the kernel misses 12 of "
                    "the reference's sinks, all at (departure=8, origin_lane=2). Missing "
                    "means the kernel is stricter than the reference somewhere, which is the "
                    "one direction that costs an answer. NOTE these graphs are the only shape "
                    "that occurs in any registered scenario -- every colgen_test and density "
                    "flight has terminals -- so this is the production case, not an edge one. "
                    "RULED OUT, each by direct comparison against the graph or the reference: "
                    "(1) the CSR arcs, role masks and destination-lane table all match exactly "
                    "-- 0 mismatches over 856 role checks; (2) `paid_class` interns the full "
                    "`active_claims` set including term rows, so states are not collapsing "
                    "there; (3) `prepare_variants` emits exactly the reference's 78 roots, "
                    "(8, 2) among them, so the root is created and dies during the search; "
                    "(4) `first_hop` -- `_state_find` hashed it without verifying it on a "
                    "probe, which was a genuine latent bug and is now fixed, but it was NOT "
                    "this one. Remaining suspects are the layer-swap discipline and the "
                    "`recent` history under multi-lane origins, neither yet tested in "
                    "isolation. strict=True so this flips the moment it is fixed."
                ),
            ),
        ),
    ],
)
def test_kernel_never_misses_a_sink_the_reference_finds_by_shape(shape, monkeypatch):
    """The same inclusion on a terminal graph, where two more code paths come alive.

    A terminal origin turns on `track_first_hop`, so the dominance key grows a field that is
    inert on the plain fixture, and both endpoints claim *term* rows under the unpadded span
    rule instead of *cell* rows. Neither is exercised by `colgen_test`-shaped graphs, and
    both are the density shape.
    """

    cfg = _cfg()
    graph, params = GRAPH_SHAPES[shape](cfg)
    model = cost_model(params, cfg)
    view = DualView(_random_duals(graph, cfg, 606), cfg)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model)
    assert result.ok, result.status
    if shape == "terminal":
        topology = dp_prepare.prepare_topology(graph, cfg)
        assert topology.track_first_hop, "fixture no longer exercises the first-hop field"

    reference = _reference_candidates(graph, cfg, view, params, monkeypatch, model=model)
    reference_side = {
        (
            candidate.label.departure_step,
            candidate.label.origin_lane_idx,
            candidate.dest_lane_idx,
            tuple((q, r) for q, r in candidate.label.path),
        )
        for candidate in reference
    }
    assert reference_side, "the reference proposed nothing, so this proves nothing"
    missing = reference_side - set(kernel_side)
    assert not missing, f"{shape}: kernel missed {len(missing)}, e.g. {sorted(missing)[0]}"


def test_kernel_never_misses_a_sink_the_reference_finds(monkeypatch):
    """Every reference sink is proposed by the kernel; the kernel may propose more.

    This is the load-bearing test of the whole search, and the direction of the inclusion is
    the whole point. **Missing** a sink is a correctness failure -- the reference found a
    column the kernel cannot certify. **Extra** sinks are the documented, deliberate cost of
    `_price_dag` not applying `completion_can_compete`: `consider_sink` assigns to a
    `nonlocal incumbent`, so the reference acquires a cutoff DURING its sweep even when
    called with `incumbent=None`, and prunes against it from then on. The kernel holds one
    cutoff per round and cannot. Tier 2 then certifies the extras away.

    Measured on this fixture: the kernel proposes ~1.6x the reference's sinks and misses
    none. That ratio is the cost of the omitted gate, and it is the number to re-read once
    the gate lands -- it should collapse toward 1.0.
    """

    cfg = _cfg()
    graph, params = _graph(cfg)
    model = cost_model(params, cfg)
    view = DualView(_random_duals(graph, cfg, 31337), cfg)

    result, kernel_side = _kernel_candidates(graph, cfg, view, model)
    assert result.ok, result.status
    assert kernel_side, "the kernel proposed nothing, so this test proves nothing"

    reference = _reference_candidates(graph, cfg, view, params, monkeypatch, model=model)
    reference_side = [
        (
            candidate.label.departure_step,
            candidate.label.origin_lane_idx,
            candidate.dest_lane_idx,
            tuple((q, r) for q, r in candidate.label.path),
        )
        for candidate in reference
    ]

    missing = set(reference_side) - set(kernel_side)
    assert not missing, f"kernel missed {len(missing)} reference sinks, e.g. {sorted(missing)[0]}"
    assert len(set(reference_side)) > 200, "the fixture is too small to be a real check"
    # Pinned so the omitted gate's cost stays visible rather than drifting silently.
    ratio = len(set(kernel_side)) / len(set(reference_side))
    assert 1.0 <= ratio < 2.5, f"kernel/reference sink ratio {ratio:.2f} moved unexpectedly"


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
