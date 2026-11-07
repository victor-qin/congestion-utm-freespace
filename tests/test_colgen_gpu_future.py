"""Future-cost relaxations preserve exact positive-price lower bounds and deadlines."""

from decimal import Decimal, localcontext
from dataclasses import replace
from functools import cache
from types import SimpleNamespace
import time

import numpy as np
import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.dp_prepare import UNREACHABLE, PreparedTopology, PreparedVariants
from freespace_sim.planner.colgen.gpu_future import FutureBounds
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.pricing_platform import PricingProblem
from tests.test_colgen_solver import _cfg, _params, _request


class _ReferenceRuntime:
    """Exercise the dispatch contract without initializing or claiming to test a GPU."""

    def __init__(self) -> None:
        """Count relaxed-layer invocations independently of the implementation's stats."""
        self.dispatches = 0

    def run(
        self,
        name: str,
        count: int,
        arrays: tuple[np.ndarray, ...],
        outputs: tuple[int, ...],
        static: dict[int, str] | None = None,
        write_only: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, ...]:
        """Evaluate the requested FP32 relaxation, including downward device additions.

        Parameters
        ------------
        - name (str): requested kernel.
        - count (int): number of local cells to relax.
        - arrays (tuple[np.ndarray, ...]): contiguous buffers in the GPU ABI's order.
        - outputs (tuple[int, ...]): output-buffer positions.
        - static (dict[int, str] | None): immutable cached inputs.
        - write_only (tuple[int, ...]): outputs that do not need their host contents uploaded.

        Return
        --------
        - output (tuple[np.ndarray, ...]): simulated device results for the caller.
        """
        assert name == "future_bounds"
        assert outputs == (5,)
        control, graph, following, visit, air, output = arrays
        assert int(control[0]) == count == len(graph)
        assert all(array.flags.c_contiguous for array in arrays)
        assert static is not None

        def add(first: np.float32, second: np.float32) -> np.float32:
            """Model the device's nonnegative downward addition and unreachable infinity."""
            if np.isposinf(first) or np.isposinf(second):
                return np.float32(np.inf)
            with np.errstate(over="ignore", under="ignore"):
                return np.nextafter(first + second, np.float32(0.0))

        for cell in range(count):
            output[cell] = air[1] if graph[cell, 6] else np.float32(np.inf)
            for target in graph[cell, :6]:
                if target >= 0:
                    output[cell] = min(
                        output[cell], add(add(air[0], visit[target]), following[target])
                    )
        self.dispatches += 1
        return (output.copy(),)


def _case() -> tuple[PricingProblem, PreparedTopology, PreparedVariants, pricing.DualView]:
    """Create competing short/priced and long/cheap paths plus an unreachable cell.

    Parameters
    ------------
    - none: fixed synthetic topology isolates the backward relaxation's requirements.

    Return
    --------
    - output (tuple): original problem, prepared graph, two departure roots and dual view.
    """
    cfg, params = _cfg(), _params()
    graph = build_flight_graph(_request(1, (0, 0), (3, 0), cfg), cfg, (), params)
    prices = {
        RowKey.cell((1, 0), 0, 11): 100.0,
        RowKey.cell((1, 0), 0, 12): 90.0,
        RowKey.cell((2, 0), 0, 10): 0.1,
        RowKey.cell((2, 0), 0, 11): 0.2,
        RowKey.cell((2, 0), 0, 15): 2.7,
        RowKey.cell((3, 0), 0, 12): -800.0,
        RowKey.cell((2, 0), 1, 11): 400.0,
        RowKey.cell((99, 99), 0, 11): 700.0,
        RowKey.terminal("unrelated", 11): 1000.0,
    }
    problem = PricingProblem(graph, cfg, params, prices, 0.0)
    topology = PreparedTopology(
        cell_q=np.arange(7, dtype=np.int32),
        cell_r=np.zeros(7, np.int32),
        arc_start=np.array([0, 2, 3, 4, 6, 7, 7, 7], np.int32),
        arc_target=np.array([1, 2, 5, 3, 5, 4, 5], np.int32),
        arc_roles=np.array([1, 2, 12, 1, 2, 1, 4], np.uint8),
        rev_remaining=np.array([2, 1, 2, 1, 1, 0, UNREACHABLE], np.int32),
        dest_mask=np.array([0, 0, 0, 0, 0, 1, 0], np.uint8),
        min_step=0,
        max_step=16,
    )
    variants = PreparedVariants(
        departure_step=np.array([10, 12, 12, 16, 16], np.int32),
        start_step=np.array([10, 12, 12, 16, 16], np.int32),
        cell=np.array([0, 0, 1, 5, 0], np.int32),
    )
    view = pricing.DualView(prices, cfg)
    view._offsets = (-1, 1)
    return problem, topology, variants, view


@pytest.mark.parametrize("cpu_recompute", [True, False])
@pytest.mark.parametrize("air", [0.3, 2e38, 1e-50])
def test_future_table_covers_all_relaxed_paths_and_visit_windows(
    monkeypatch: pytest.MonkeyPatch, cpu_recompute: bool, air: float
) -> None:
    """Visit-only bounds cover exhaustive arithmetic, including longer cheap routes.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): isolates visit arithmetic from rooted endpoint bounds.
    - cpu_recompute (bool): exercises independent FP64 refinement and unchecked FP32 storage.
    - air (float): tests ordinary values, FP32 overflow and FP32 underflow.

    Return
    --------
    - output (None): tests bounds, inclusive windows, ignored nonvisit rows and horizon.
    """
    problem, topology, variants, view = _case()
    monkeypatch.setattr(pricing, "_endpoint_claims", lambda *args, **kwargs: frozenset())
    runtime = _ReferenceRuntime()
    if cpu_recompute:
        def no_device(*args, **kwargs):
            """Certified preparation must not depend on any device recurrence result."""
            pytest.fail("certified future preparation dispatched an unused GPU recurrence")
        monkeypatch.setattr(runtime, "run", no_device)
    result = FutureBounds(
        problem,
        topology,
        variants,
        SimpleNamespace(air_per_hop=air),
        view,
        runtime,
        None,
        cpu_recompute=cpu_recompute,
    )

    @cache
    def exact(cell: int, step: int, first: bool = False) -> Decimal:
        """Enumerate all relaxed paths using decimal sums of the original binary64 inputs."""
        candidates = []
        with localcontext() as context:
            context.prec = 400
            if topology.dest_mask[cell]:
                candidates.append(Decimal(0))
            if step == topology.max_step:
                return min(candidates, default=Decimal("Infinity"))
            for arc in range(topology.arc_start[cell], topology.arc_start[cell + 1]):
                if not topology.arc_roles[arc] & (10 if first else 5):
                    continue
                target = int(topology.arc_target[arc])
                visit = sum(
                    (
                        Decimal.from_float(price)
                        for row, price in view._duals.items()
                        if row.kind == "cell"
                        and row.level == 0
                        and row.cell_coord == (target, 0)
                        and step + 1 + view.offsets[0] <= row.step <= step + 1 + view.offsets[1]
                        and price > 0.0
                    ),
                    Decimal(0),
                )
                candidates.append(Decimal.from_float(air) + visit + exact(target, step + 1))
        return min(candidates, default=Decimal("Infinity"))

    assert result.first_step == 10
    assert result.last_step == 16
    assert result.stats["future_dispatches"] == runtime.dispatches
    if not cpu_recompute:
        assert runtime.dispatches > 0
    assert result.stats["future_layers"] == 6
    assert result.stats["future_root_layers"] == 2
    assert result.stats["future_roots"] == 5
    assert result.stats["future_s"] >= result.stats["future_prepare_s"] >= 0.0
    assert result.stats["future_cpu_recompute"] is cpu_recompute
    assert result._costs.dtype == (np.float64 if cpu_recompute else np.float32)
    for step in range(result.first_step, result.last_step + 1):
        row = result.row(step)
        assert row.dtype == np.float64
        assert not row.flags.writeable
        for cell in range(topology.n_cells):
            actual = result.lookup(cell, step)
            expected = exact(cell, step)
            assert Decimal.from_float(actual) <= expected
            assert actual == row[cell]
            if cpu_recompute or float(expected) <= float(np.finfo(np.float32).max):
                assert actual == pytest.approx(
                    float(expected), rel=1e-14 if cpu_recompute else 1e-5
                )
    for cell, step in zip(variants.cell, variants.start_step, strict=True):
        actual = result.root_lookup(int(cell), int(step))
        expected = exact(int(cell), int(step), first=True)
        assert Decimal.from_float(actual) <= expected
        if cpu_recompute or float(expected) <= float(np.finfo(np.float32).max):
            assert actual == pytest.approx(float(expected), rel=1e-14 if cpu_recompute else 1e-5)
    if air == 0.3:
        assert result.root_lookup(0, 10) == pytest.approx(1.5)
        assert result.lookup(0, 10) == pytest.approx(190.6)
        assert result.root_lookup(1, 12) == pytest.approx(0.3)
        assert result.lookup(3, 12) == pytest.approx(0.6)
        assert result.lookup(0, 14) == pytest.approx(0.6)
        if cpu_recompute:
            # Frozen original FP64 results: deleting the unused device recurrence
            # must not change even the final bits of these independently stored bounds.
            assert result.lookup(0, 10).hex() == "0x1.7d33333333331p+7"
            assert result.root_lookup(0, 10).hex() == "0x1.7fffffffffffcp+0"
            assert result.root_lookup(0, 12).hex() == "0x1.3333333333330p+0"
    assert result.lookup(0, 15) == np.inf
    assert result.lookup(6, 10) == np.inf
    with pytest.raises(IndexError):
        result.row(9)
    with pytest.raises(IndexError):
        result.lookup(-1, 10)
    with pytest.raises(KeyError):
        result.root_lookup(3, 12)


@pytest.mark.parametrize("cpu_recompute", [True, False])
def test_future_arrival_tail_excludes_overlap_and_can_delay_stopping(
    monkeypatch: pytest.MonkeyPatch, cpu_recompute: bool
) -> None:
    """Tail prices are disjoint, time-specific and minimized against later arrival.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies precisely priced endpoint-claim boundaries.
    - cpu_recompute (bool): exercises verified and unchecked arithmetic and final-row storage.

    Return
    --------
    - output (None): boundary/earlier cell rows and negative prices cannot inflate the tail;
      terminal rows remain included, and destination continuation can lower the relaxed cost.
    """
    problem, _, _, _ = _case()

    def endpoints(graph, cfg, *, origin: bool, step: int, timing_steps: int):
        """Provide terminal rows and cell rows on both sides of the last visit's upper edge."""
        assert graph is problem.graph and cfg is problem.cfg
        assert origin is False and timing_steps == 0
        return frozenset(
            (
                RowKey.cell((1, 0), 0, step + 1),
                RowKey.cell((91, 0), 0, step + 2),
                RowKey.terminal("arrival", step - 3),
                RowKey.cell((1, 0), 0, step),
                RowKey.cell((93, 0), 0, step + 2),
            )
        )

    prices = {}
    for step in range(10, 17):
        prices.update(
            {
                RowKey.cell((1, 0), 0, step + 1): 0.01,
                RowKey.cell((91, 0), 0, step + 2): 99.7 if step == 10 else 0.1,
                RowKey.terminal("arrival", step - 3): 0.3,
                RowKey.cell((1, 0), 0, step): 0.01,
                RowKey.cell((93, 0), 0, step + 2): -50.0,
            }
        )
    view = pricing.DualView(prices, problem.cfg)
    view._offsets = (-1, 1)
    topology = PreparedTopology(
        cell_q=np.array([0, 1], np.int32),
        cell_r=np.array([0, 0], np.int32),
        arc_start=np.array([0, 1, 2], np.int32),
        arc_target=np.array([1, 0], np.int32),
        arc_roles=np.array([15, 1], np.uint8),
        rev_remaining=np.array([1, 0], np.int32),
        dest_mask=np.array([0, 1], np.uint8),
        max_step=16,
    )
    variants = PreparedVariants(
        departure_step=np.array([10, 10, 16], np.int32),
        start_step=np.array([10, 10, 16], np.int32),
        cell=np.array([0, 1, 1], np.int32),
    )
    monkeypatch.setattr(pricing, "_endpoint_claims", endpoints)
    result = FutureBounds(
        problem,
        topology,
        variants,
        SimpleNamespace(air_per_hop=0.3),
        view,
        _ReferenceRuntime(),
        None,
        cpu_recompute=cpu_recompute,
    )
    for actual, expected in (
        (result.lookup(1, 10), 1.03),
        (result.root_lookup(1, 10), 100.0),
        (result.root_lookup(0, 10), 0.73),
        (result.lookup(1, 16), 0.4),
        (result.root_lookup(1, 16), 0.4),
    ):
        assert actual <= expected
        assert actual == pytest.approx(expected, rel=1e-14 if cpu_recompute else 1e-5)


@pytest.mark.parametrize("cpu_recompute", [True, False])
@pytest.mark.parametrize("start", [-100, 100])
@pytest.mark.parametrize("extra_root", [None, 1, 3])
def test_endpoint_reachability_bound_covers_every_rooted_claim_union(
    monkeypatch: pytest.MonkeyPatch,
    cpu_recompute: bool,
    start: int,
    extra_root: int | None,
) -> None:
    """Count unreachable overlap once, retaining real overlap from every legal root.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies guaranteed endpoint rows on a tiny graph.
    - cpu_recompute (bool): checks both numerical modes against exact original-price sums.
    - start (int): shifts the same domain across positive and negative absolute clocks.
    - extra_root (int | None): adds an early overlapping or too-late alternate root after
      thirty-two ordinary roots, ensuring endpoint disjointness uses the full root domain.

    Return
    --------
    - output (None): verifies useful nonoverlap charges and exhaustive rooted route unions,
      including root visits, signed prices, absent cells, other levels and multiple arrivals.
    """
    problem, _, _, _ = _case()
    topology = PreparedTopology(
        cell_q=np.arange(6, dtype=np.int32),
        cell_r=np.zeros(6, np.int32),
        arc_start=np.array([0, 2, 3, 4, 4, 5, 5], np.int32),
        arc_target=np.array([1, 2, 3, 4, 3], np.int32),
        arc_roles=np.full(5, 15, np.uint8),
        rev_remaining=np.array([2, 1, 2, 0, 1, UNREACHABLE], np.int32),
        dest_mask=np.array([0, 0, 0, 1, 0, 0], np.uint8),
        max_step=start + 3,
    )
    root_cells, root_steps = [0] * 32, [start] * 32
    if extra_root is not None:
        root_cells.append(4)
        root_steps.append(start + extra_root)
    variants = PreparedVariants(
        departure_step=np.asarray(root_steps, np.int32),
        start_step=np.asarray(root_steps, np.int32),
        cell=np.asarray(root_cells, np.int32),
    )
    prices = {
        RowKey.cell((4, 0), 0, start + 2): 7.0,
        RowKey.cell((0, 0), 0, start): 5.0,
        RowKey.cell((1, 0), 0, start + 1): -11.0,
        RowKey.cell((99, 0), 0, start + 1): 0.25,
        RowKey.cell((1, 0), 1, start + 1): 2.0,
        RowKey.cell((5, 0), 0, start + 1): 3.0,
        RowKey.terminal("arrival", start): 1.5,
    }
    claims = frozenset(prices)
    monkeypatch.setattr(pricing, "_endpoint_claims", lambda *args, **kwargs: claims)
    view = pricing.DualView(prices, problem.cfg)
    view._offsets = (-2, 1)
    air = 0.3
    result = FutureBounds(
        problem, topology, variants, SimpleNamespace(air_per_hop=air), view,
        _ReferenceRuntime(), None, cpu_recompute=cpu_recompute,
    )
    tolerance = 1e-14 if cpu_recompute else 1e-5
    expected_tail = 6.75 if extra_root == 1 else 13.75
    assert result.lookup(3, start + 2) == pytest.approx(expected_tail, rel=tolerance)
    assert result.lookup(3, start + 3) == pytest.approx(6.75, rel=tolerance)
    assert result.root_lookup(0, start) == pytest.approx(expected_tail + 0.6, rel=tolerance)

    def completed_paths(path: tuple[int, ...], step: int) -> list[tuple[int, ...]]:
        """Enumerate finite rooted paths without consulting the implementation's bounds."""
        cell = path[-1]
        found = [path] if topology.dest_mask[cell] else []
        if step < topology.max_step:
            for arc in range(topology.arc_start[cell], topology.arc_start[cell + 1]):
                target = int(topology.arc_target[arc])
                found.extend(completed_paths((*path, target), step + 1))
        return found

    for root_cell, root_step in set(zip(root_cells, root_steps, strict=True)):
        for path in completed_paths((root_cell,), root_step):
            visits = [
                pricing._visit_claims((cell, 0), 0, root_step + hop, view.offsets)
                for hop, cell in enumerate(path)
            ]
            all_claims = claims.union(*visits)
            with localcontext() as context:
                context.prec = 400
                air_cost = Decimal.from_float(air) * (len(path) - 1)
                exact_positive = air_cost + sum(
                    (Decimal.from_float(max(0.0, prices.get(row, 0.0))) for row in all_claims),
                    Decimal(0),
                )
                exact_signed = air_cost + sum(
                    (Decimal.from_float(prices.get(row, 0.0)) for row in all_claims), Decimal(0)
                )
                prefix = frozenset()
                for hop, cell in enumerate(path):
                    prefix |= visits[hop]
                    suffix = (
                        result.root_lookup(cell, root_step)
                        if hop == 0 else result.lookup(cell, root_step + hop)
                    )
                    lower = Decimal.from_float(air) * hop + Decimal.from_float(suffix) + sum(
                        (Decimal.from_float(max(0.0, prices.get(row, 0.0))) for row in prefix),
                        Decimal(0),
                    )
                    assert lower <= exact_positive
                    assert lower + Decimal(-11) <= exact_signed


@pytest.mark.parametrize("cpu_recompute", [True, False])
def test_future_table_empty_roots_and_deadline_are_explicit(
    monkeypatch: pytest.MonkeyPatch, cpu_recompute: bool,
) -> None:
    """No roots need no dispatch, and an expired layer cannot publish a partial table.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): expires the deadline after a computed layer.
    - cpu_recompute (bool): exercises CPU recurrence and unchecked GPU recurrence deadlines.

    Return
    --------
    - output (None): checks terminal-only empty-root data and deadline propagation.
    """
    problem, topology, variants, view = _case()
    runtime = _ReferenceRuntime()
    geometry = SimpleNamespace(air_per_hop=0.0)
    empty = FutureBounds(problem, topology, PreparedVariants(), geometry, view, runtime, None)
    assert empty.first_step == empty.last_step == 16
    assert runtime.dispatches == empty.stats["future_dispatches"] == 0
    assert empty.lookup(0, 16) == np.inf
    assert empty.lookup(5, 16) == pytest.approx(2.7)
    deadline = time.monotonic() + 100.0
    original = FutureBounds._relax if cpu_recompute else runtime.run

    def expire(*args, **kwargs):
        """Move time past the deadline once the first completed layer is available."""
        output = original(*args, **kwargs)
        monkeypatch.setattr(time, "monotonic", lambda: deadline + 1.0)
        return output

    if cpu_recompute:
        monkeypatch.setattr(FutureBounds, "_relax", staticmethod(expire))
    else:
        monkeypatch.setattr(runtime, "run", expire)
    with pytest.raises(pricing.PricingTimeout, match="future-bound"):
        FutureBounds(
            problem, topology, variants, geometry, view, runtime, deadline,
            cpu_recompute=cpu_recompute,
        )


@pytest.mark.parametrize("limit", [3, 4, 11])
@pytest.mark.parametrize("budget", [0, 1200, 1_000_000])
@pytest.mark.parametrize("early_prices", [False, True])
def test_hop_tables_cover_all_rooted_prefixes_and_fall_back_on_missing_bins(
    monkeypatch: pytest.MonkeyPatch, limit: int, budget: int, early_prices: bool,
) -> None:
    """Hop refinement covers original arithmetic, roles, exclusions and incomplete budgets.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): isolates visit and hop costs from endpoint geometry.
    - limit (int): short, just-feasible and greater-than-six extra allowances.
    - budget (int): disables tables, stores only some bins or stores the full reachable domain.
    - early_prices (bool): truncates weighted support before the horizon; later suffixes
      must inherit the old lower bound instead of being falsely declared unreachable.

    Return
    --------
    - output (None): every rooted-prefix cost is covered; missing bins remain weaker bounds.
    """
    problem, topology, variants, view = _case()
    topology = replace(topology, air_hop_limit=limit)
    if early_prices:
        view = pricing.DualView({RowKey.cell((1, 0), 0, 11): 19.123456789}, problem.cfg)
        view._offsets = (-1, 1)
    monkeypatch.setattr(pricing, "_endpoint_claims", lambda *args, **kwargs: frozenset())
    for forbidden in (frozenset(), frozenset((RowKey.cell((4, 0), 0, 14),))):
        current = replace(problem, forbidden_rows=forbidden)
        table = FutureBounds(
            current, topology, variants, SimpleNamespace(air_per_hop=0.3), view,
            _ReferenceRuntime(), None, hop_max_bytes=budget,
        )
        old = FutureBounds(
            current, topology, variants, SimpleNamespace(air_per_hop=0.3), view,
            _ReferenceRuntime(), None, hop_max_bytes=0,
        )

        @cache
        def exact(cell: int, step: int, available: int, first: bool) -> Decimal:
            """Enumerate the role relaxation with the actual remaining budget in Decimal."""
            with localcontext() as context:
                context.prec = 100
                best = Decimal(0) if topology.dest_mask[cell] else Decimal("Infinity")
                if not available or step >= topology.max_step:
                    return best
                for arc in range(topology.arc_start[cell], topology.arc_start[cell + 1]):
                    if not topology.arc_roles[arc] & (10 if first else 5):
                        continue
                    target = int(topology.arc_target[arc])
                    claims = pricing._visit_claims((target, 0), 0, step + 1, view.offsets)
                    if not claims.isdisjoint(forbidden):
                        continue
                    cost = sum((Decimal.from_float(max(0., view.row_cost(row))) for row in claims),
                               Decimal.from_float(0.3))
                    best = min(best, cost + exact(target, step + 1, available - 1, False))
                return best

        for cell, start in zip(variants.cell, variants.start_step, strict=True):
            cell, start = int(cell), int(start)
            assert Decimal.from_float(table.root_lookup(cell, start)) <= exact(cell, start, limit, True)
            prefixes = [(cell, start, limit, True)]
            while prefixes:
                cell, step, available, first = prefixes.pop()
                actual = table.root_lookup(cell, step) if first else table.lookup(cell, step, available)
                assert Decimal.from_float(actual) <= exact(cell, step, available, first)
                if not first:
                    assert actual >= old.lookup(cell, step)
                if available and step < topology.max_step:
                    for arc in range(topology.arc_start[cell], topology.arc_start[cell + 1]):
                        if topology.arc_roles[arc] & (10 if first else 5):
                            prefixes.append((int(topology.arc_target[arc]), step + 1, available - 1, False))
        if budget == 0:
            assert table.hops.costs is None
            assert table.root_lookup(0, 10) == old.root_lookup(0, 10)
        elif budget == 1_000_000 and limit == 3 and not early_prices:
            assert table.root_lookup(0, 10) == np.inf
            if not forbidden:
                assert old.root_lookup(0, 10) < np.inf
        assert table.hops.stats['hop_table_bytes'] <= budget


def test_hop_snapshots_refresh_prices_roots_and_do_not_publish_interrupted_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only unpriced structures survive snapshots; interrupted weighted tables never escape.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): interrupts arithmetic inside the extra recurrence.

    Return
    --------
    - output (None): new/sign-changing rows and changed root domains match a fresh build.
    """
    from freespace_sim.planner.colgen import gpu_hops

    problem, topology, variants, view = _case()
    topology = replace(topology, air_hop_limit=4)
    monkeypatch.setattr(pricing, "_endpoint_claims", lambda *args, **kwargs: frozenset())
    geometry = SimpleNamespace(air_per_hop=0.3)
    first = FutureBounds(problem, topology, variants, geometry, view, _ReferenceRuntime(), None)
    frozen = first.hops.costs.copy()
    prices = dict(view._duals)
    prices[RowKey.cell((1, 0), 0, 11)] = -900.0
    prices[RowKey.cell((4, 0), 0, 13)] = 10.123456789
    changed = pricing.DualView(prices, problem.cfg)
    changed._offsets = view.offsets
    second = FutureBounds(problem, topology, variants, geometry, changed, _ReferenceRuntime(), None)
    assert first.hops.structure is second.hops.structure
    np.testing.assert_array_equal(first.hops.costs, frozen)
    assert not np.array_equal(first.hops.costs, second.hops.costs)
    fresh_problem, _, _, _ = _case()
    fresh = FutureBounds(fresh_problem, topology, variants, geometry, changed, _ReferenceRuntime(), None)
    np.testing.assert_array_equal(second.hops.costs, fresh.hops.costs)
    reduced = replace(variants, cell=variants.cell[1:], start_step=variants.start_step[1:])
    moved = FutureBounds(problem, topology, reduced, geometry, changed, _ReferenceRuntime(), None)
    assert moved.hops.structure is not second.hops.structure
    original = gpu_hops._add_lower
    calls = 0

    def expire(*args, **kwargs):
        """Interrupt after weighted recurrence work has begun, leaving geometry reusable."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise pricing.PricingTimeout("interrupted hop table")
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(gpu_hops, "_add_lower", expire)
        with pytest.raises(pricing.PricingTimeout, match="interrupted hop table"):
            FutureBounds(problem, topology, variants, geometry, changed, _ReferenceRuntime(), None)
    retry = FutureBounds(problem, topology, variants, geometry, changed, _ReferenceRuntime(), None)
    np.testing.assert_array_equal(retry.hops.costs, fresh.hops.costs)
