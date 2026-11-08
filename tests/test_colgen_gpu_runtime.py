"""Metal buffer reuse preserves uploads, independent results and batch lifetimes."""

from collections.abc import Callable
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from freespace_sim.planner.colgen.gpu_runtime import ComputeRuntime


class _Buffer:
    """Expose shared bytes through the small Metal buffer interface used by the runtime."""

    def __init__(self, length: int) -> None:
        """Initialize deterministic storage so skipped uploads can be observed."""
        self.data = bytearray(length)

    def contents(self) -> "_Buffer":
        """Stand in for the PyObjC shared-memory pointer."""
        return self

    def as_buffer(self, length: int) -> memoryview:
        """Expose only the requested logical portion of the allocation."""
        return memoryview(self.data)[:length]


def _runtime(
    monkeypatch: pytest.MonkeyPatch,
    execute: Callable[[list[_Buffer], int], None],
    backend: str = "metal",
) -> tuple[ComputeRuntime, list[weakref.ReferenceType[_Buffer]]]:
    """Run the production adapter against synchronous CPU-backed Metal objects.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): installs the fake backend without using a GPU.
    - execute (Callable): simulated kernel receives buffers and the dispatch count.
    - backend (str): adapter to exercise with CPU-backed allocations.

    Return
    --------
    - output (tuple): real runtime and weak references to all allocated buffers.
    """
    allocations = []

    if backend == "cuda":
        class Array(np.ndarray):
            """CPU storage exposing the CuPy upload interface without claiming CUDA execution."""

            def set(self, source):
                """Refresh the explicit allocation without replacing it."""
                self[:] = source

        def empty(shape, dtype):
            """Track CUDA-adapter allocations using the same lifetime assertions."""
            array = np.empty(shape, dtype).view(Array)
            allocations.append(weakref.ref(array))
            return array

        api = SimpleNamespace(
            empty=empty,
            asarray=lambda source: np.array(source).view(Array),
            asnumpy=lambda source: np.array(source),
            cuda=SimpleNamespace(
                Device=lambda: SimpleNamespace(id=0),
                runtime=SimpleNamespace(getDeviceProperties=lambda _: {"name": "CPU CUDA double"}),
                get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
            ),
            RawModule=lambda **kwargs: SimpleNamespace(get_function=lambda name:
                lambda grid, block, buffers: execute(buffers, buffers[0].size)),
        )
        monkeypatch.setitem(sys.modules, "cupy", api)
        return ComputeRuntime("cuda"), allocations

    def allocate(length: int, options: int) -> _Buffer:
        """Track allocations without retaining buffers after the runtime releases them."""
        assert options == 1
        buffer = _Buffer(length)
        allocations.append(weakref.ref(buffer))
        return buffer

    def command_buffer() -> SimpleNamespace:
        """Execute the simulated kernel only when the runtime waits for completion."""
        commands = []

        def encoder():
            """Keep each dependent encoder's bindings until the single block wait."""
            buffers, counts = [], []
            commands.append((buffers, counts))

            def bind(buffer: _Buffer, offset: int, index: int) -> None:
                """Keep each argument binding in its declared order."""
                assert offset == 0 and index == len(buffers)
                buffers.append(buffer)

            return SimpleNamespace(
                setComputePipelineState_=lambda pipeline: None,
                setBuffer_offset_atIndex_=bind,
                dispatchThreads_threadsPerThreadgroup_=lambda size, group: counts.append(size[0]),
                endEncoding=lambda: None,
            )

        return SimpleNamespace(
            computeCommandEncoder=encoder,
            commit=lambda: None,
            waitUntilCompleted=lambda: [execute(buffers, counts[0]) for buffers, counts in commands],
            status=lambda: 4,
        )

    pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 128)
    library = SimpleNamespace(newFunctionWithName_=lambda name: name)
    device = SimpleNamespace(
        name=lambda: "CPU test double",
        newLibraryWithSource_options_error_=lambda *args: (library, None),
        newComputePipelineStateWithFunction_error_=lambda function, error: (pipeline, None),
        newCommandQueue=lambda: SimpleNamespace(commandBuffer=command_buffer),
        newBufferWithLength_options_=allocate,
    )
    options = SimpleNamespace(setFastMathEnabled_=lambda enabled: None)
    api = SimpleNamespace(
        MTLCreateSystemDefaultDevice=lambda: device,
        MTLCompileOptions=SimpleNamespace(
            alloc=lambda: SimpleNamespace(init=lambda: options)
        ),
        MTLResourceStorageModeShared=1,
        MTLCommandBufferStatusCompleted=4,
        MTLSizeMake=lambda *sizes: sizes,
    )
    monkeypatch.setitem(sys.modules, "Metal", api)
    return ComputeRuntime("metal"), allocations


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_metal_reuse_refreshes_inputs_and_resets_read_write_outputs(monkeypatch, dtype):
    """Growing and shrinking dispatches refresh every input and retain independent results."""
    def execute(buffers: list[_Buffer], count: int) -> None:
        """Use the initial output values, as an atomic grouping table would."""
        source = np.frombuffer(buffers[0].data, dtype=dtype, count=count * 2)
        output = np.frombuffer(buffers[1].data, dtype=dtype, count=count * 2)
        np.testing.assert_array_equal(output, -1)
        output += source

    runtime, allocations = _runtime(monkeypatch, execute)
    saved = []
    for iteration, count in enumerate((6, 4, 9, 3, 9)):
        source = np.arange(count * 2, dtype=dtype).reshape(count, 2) + iteration * 100
        source.flags.writeable = False
        initialized = np.full_like(source, -1)
        (actual,) = runtime.run("group_best", count, (source, initialized), (1,))
        expected = source - 1
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(initialized, -1)
        saved.append((actual, expected))
        assert len(allocations) == (2 if iteration < 2 else 4)
        assert sum(reference() is not None for reference in allocations) == 2

    runtime.clear_cache()
    assert all(reference() is None for reference in allocations)
    assert runtime.dispatches == 5
    for actual, expected in saved:
        np.testing.assert_array_equal(actual, expected)


def test_metal_static_identity_and_batch_release(monkeypatch):
    """Static data is shared until replaced, while dynamic arguments remain kernel-local."""
    def execute(buffers: list[_Buffer], count: int) -> None:
        """Compute with a cached graph and refreshed dynamic values."""
        graph, values, output = (
            np.frombuffer(buffer.data, dtype=np.int32, count=count) for buffer in buffers
        )
        output[:] = graph + values

    runtime, allocations = _runtime(monkeypatch, execute)
    graph = np.array([3, 7], np.int32)
    for name, expected_allocations in (
        ("bound_roots", 3), ("bound_roots", 3), ("future_bounds", 5)
    ):
        values = np.array([expected_allocations, 4], np.int32)
        (actual,) = runtime.run(
            name, 2, (graph, values, np.empty_like(graph)), (2,), static={0: "graph"}
        )
        np.testing.assert_array_equal(actual, graph + values)
        assert len(allocations) == expected_allocations

    graph = np.array([30, 70], np.int32)
    (actual,) = runtime.run(
        "bound_roots", 2, (graph, values, np.empty_like(graph)), (2,), static={0: "graph"}
    )
    np.testing.assert_array_equal(actual, graph + values)
    assert len(allocations) == 6
    runtime.clear_cache()
    assert all(reference() is None for reference in allocations)
    runtime.run("bound_roots", 2, (graph, values, np.empty_like(graph)), (2,))
    assert len(allocations) == 9


def test_metal_write_only_skips_initial_values_and_zero_count_is_independent(monkeypatch):
    """Only explicitly declared output-only arguments may omit their initial upload."""
    def execute(buffers: list[_Buffer], count: int) -> None:
        """Fully overwrite output without depending on its preceding contents."""
        source, output = (
            np.frombuffer(buffer.data, dtype=np.float32, count=count) for buffer in buffers
        )
        assert np.all(output != -999)
        output[:] = source * 2

    runtime, allocations = _runtime(monkeypatch, execute)
    source = np.array([3.0, 4.0], np.float32)
    output = np.full(2, -999, np.float32)
    for _ in range(2):
        (actual,) = runtime.run("bound_roots", 2, (source, output), (1,), write_only=(1,))
        np.testing.assert_array_equal(actual, source * 2)
        np.testing.assert_array_equal(output, -999)
    assert len(allocations) == 2
    (empty_dispatch,) = runtime.run("bound_roots", 0, (source, output), (1,))
    empty_dispatch[0] = 0
    np.testing.assert_array_equal(output, -999)
    assert runtime.dispatches == 2
    for static, write_only in (({}, (0,)), ({1: "output"}, (1,))):
        with pytest.raises(ValueError, match="nonstatic output"):
            runtime.run(
                "bound_roots", 2, (source, output), (1,), static=static, write_only=write_only
            )


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_owned_buffers_survive_batch_cleanup_without_reupload(monkeypatch, backend):
    """Device data remains live across batches and cannot be bound to another device owner."""
    def execute(buffers, count):
        """Use an explicitly owned input and write an explicitly owned output."""
        source, output = (np.frombuffer(b.data, np.int32, count=count) for b in buffers)
        output[:] = source + 1

    runtime, allocations = _runtime(monkeypatch, execute, backend)
    source = np.array([3, 7], np.int32)
    owned = runtime.upload(source)
    output = runtime.upload(np.zeros_like(source))
    source[:] = -1
    uploaded = runtime.counters["upload_bytes"]
    for _ in range(2):
        runtime.clear_batch()
        (actual,) = runtime.run("bound_roots", 2, (owned, output), (1,))
        np.testing.assert_array_equal(actual, [4, 8])
    assert runtime.counters["upload_bytes"] == uploaded
    assert len(allocations) == 2
    other, _ = _runtime(monkeypatch, execute, backend)
    with pytest.raises(ValueError, match="another runtime"):
        other.run("bound_roots", 2, (owned, output), (1,))
    runtime.clear_cache()
    np.testing.assert_array_equal(runtime.read(owned), [3, 7])
    del owned, output
    assert all(ref() is None for ref in allocations)


def test_price_store_bounds_lifetime_and_preserves_complete_support(monkeypatch):
    """Complete snapshot tables refresh signs/new rows, survive cleanup, and obey LRU limits."""
    from freespace_sim.planner.colgen.gpu_prices import DevicePriceCache
    from freespace_sim.planner.colgen.pricing import DualView
    from freespace_sim.planner.colgen.network import RowKey
    from tests.test_colgen_pricing_platform import _cfg

    runtime, _ = _runtime(monkeypatch, lambda buffers, count: None)
    runtime.prices = DevicePriceCache(max_snapshots=1, max_bytes=1024 * 1024)
    raw = {RowKey.cell((-999, 99), 0, -4): 7.0,
           RowKey.cell((-999, 99), 0, -3): -2.0,
           RowKey.cell((0, 0), 1, 2): 9.0, RowKey.term("A", 3): -6.0}
    view = DualView(raw, _cfg())
    first = runtime.prices.get(view, runtime, lambda: None)
    before = dict(runtime.counters)
    runtime.clear_batch()
    assert runtime.prices.get(view, runtime, lambda: None) is first
    assert runtime.counters["upload_bytes"] == before["upload_bytes"]
    assert runtime.counters["allocations"] == before["allocations"]
    ranges, steps, values = map(runtime.read, (first.ranges, first.steps, first.values))
    for resource, windows in view._gpu_visit_prices.cells.items():
        start, stop = ranges[first.slots[resource]]
        assert dict(zip(map(int, steps[start:stop]), map(tuple, values[start:stop]))) == windows
    assert first.slots.get(((999, -999), 0), 0) == 0
    assert tuple(ranges[0]) == (0, 0)
    raw[RowKey.cell((-999, 99), 0, -4)] = -13.0
    raw[RowKey.cell((999, -999), 0, 777)] = 11.0
    changed = DualView(raw, _cfg())
    second = runtime.prices.get(changed, runtime, lambda: None)
    assert second is not first
    assert ((999, -999), 0) in second.slots
    assert list(runtime.prices.stores) == [changed]
    assert runtime.prices.nbytes == second.nbytes
    # Eviction releases cache ownership, not a table still used by an active batch.
    np.testing.assert_array_equal(runtime.read(first.values), values)
    calls = 0

    def expire():
        """Interrupt after sparse sums were built but before all device arrays publish."""
        nonlocal calls
        calls += 1
        if calls == len(view._gpu_visit_prices.cells) + 3:
            raise TimeoutError()

    with pytest.raises(TimeoutError):
        runtime.prices.get(view, runtime, expire)
    assert list(runtime.prices.stores) == [changed]
    runtime.prices.max_bytes = 0
    runtime.prices.clear()
    uncached = runtime.prices.get(view, runtime, lambda: None)
    assert not runtime.prices.stores and runtime.prices.nbytes == 0
    np.testing.assert_array_equal(runtime.read(uncached.values), values)


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_ordered_device_block_waits_once_and_reads_only_selected_rows(monkeypatch, backend):
    """A later kernel consumes the preceding result without a host copy or intermediate wait."""
    from freespace_sim.planner.colgen.gpu_buffers import Dispatch

    def execute(buffers, count):
        """Make the second invocation depend on bytes produced by the first."""
        source, output = (np.frombuffer(b.data, np.int32, count=count) for b in buffers)
        output[:] = source + 1

    runtime, _ = _runtime(monkeypatch, execute, backend)
    source = runtime.upload(np.array([[3, 7], [9, 11]], np.int32))
    middle = runtime.empty(source.shape, source.dtype)
    output = runtime.empty(source.shape, source.dtype)
    runtime.run_many((Dispatch('bound_roots', 4, (source, middle)),
                      Dispatch('group_best', 4, (middle, output))))
    assert runtime.dispatches == 2 and runtime.counters['waits'] == 1
    assert runtime.counters['upload_bytes'] == source.nbytes
    np.testing.assert_array_equal(runtime.read_rows(output, 1, 1), [[11, 13]])
    assert runtime.counters['download_bytes'] == 8
    with pytest.raises(ValueError, match='row range'):
        runtime.read_rows(output, 1, 2)
    other, _ = _runtime(monkeypatch, execute, backend)
    with pytest.raises(ValueError, match='foreign device'):
        other.run_many((Dispatch('bound_roots', 4, (source, middle)),))


@pytest.mark.parametrize('budget', [1, 8 * 9 * 4 + 1])
def test_partial_resident_allocation_releases_storage(monkeypatch, budget):
    """A memory limit during construction cannot retain a partially built search arena."""
    import gc
    from freespace_sim.planner.colgen import gpu_resident
    from freespace_sim.planner.colgen.gpu_pricing import _Stopped

    runtime, references = _runtime(monkeypatch, lambda buffers, count: None)
    batch = SimpleNamespace(runtime=runtime, options=SimpleNamespace(max_labels=8, chunk_size=2),
                            search_label_limit=8, stride=9, stats={})
    monkeypatch.setattr(gpu_resident, '_MAX_BYTES', budget)
    with pytest.raises(_Stopped, match='buffer_limit'):
        gpu_resident.ResidentSearch(batch)
    gc.collect()
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_integer_operands_bound_retention_and_keep_evicted_borrowers_alive(
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    """Hits skip builds/uploads; eviction, clear and failed builds preserve owner isolation.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies deterministic CPU-backed adapter storage.
    - backend (str): Metal or CUDA interface; this test makes no hardware claim.

    Return
    --------
    - output (None): requires byte-safe borrowers, no partial publication and complete release.
    """
    from freespace_sim.planner.colgen.gpu_operand_cache import OperandCache

    runtime, references = _runtime(monkeypatch, lambda buffers, count: None, backend)
    runtime.operands = cache = OperandCache(runtime._owner, max_entries=1)

    def build():
        """Supply a complete immutable generation, without search metadata."""
        return (np.arange(16, dtype=np.int32),)

    def forbidden():
        """A hit must not encode or upload the same operands again."""
        pytest.fail("Cached operand was rebuilt")

    first = cache.get(("first",), build, runtime, lambda: None, upload=(0,))
    runtime.clear_batch()
    uploads = runtime.counters["upload_bytes"]
    assert cache.get(("first",), forbidden, runtime, lambda: None, upload=(0,)) is first
    assert runtime.counters["upload_bytes"] == uploads
    assert runtime.counters["integer_upload_avoided_bytes"] == first.buffers[0].nbytes
    assert not first.arrays[0].flags.writeable
    second = cache.get(("second",), build, runtime, lambda: None, upload=(0,))
    assert runtime.counters["integer_cache_evictions"] == 1
    assert len(cache.stores) == 1
    np.testing.assert_array_equal(runtime.read(first.buffers[0]), build()[0])
    foreign, _ = _runtime(monkeypatch, lambda buffers, count: None, backend)
    with pytest.raises(ValueError, match="another runtime"):
        cache.get(("second",), forbidden, foreign, lambda: None, upload=(0,))
    other = foreign.operands.get(("second",), build, foreign, lambda: None, upload=(0,))
    assert other.buffers[0].owner is not second.buffers[0].owner
    foreign.clear_cache()
    del other

    for limit in ("max_host_bytes", "max_device_bytes"):
        saved = getattr(cache, limit)
        setattr(cache, limit, 1)
        skipped = cache.get((limit,), build, runtime, lambda: None, upload=(0,))
        assert (limit,) not in cache.stores
        np.testing.assert_array_equal(runtime.read(skipped.buffers[0]), build()[0])
        setattr(cache, limit, saved)
    del skipped

    cache.max_entries = 16
    for limit, size in (("max_host_bytes", first.host_bytes),
                        ("max_device_bytes", first.device_bytes)):
        cache.clear()
        saved = getattr(cache, limit)
        setattr(cache, limit, size + 16)
        held = cache.get(("a",), build, runtime, lambda: None, upload=(0,))
        cache.get(("b",), build, runtime, lambda: None, upload=(0,))
        assert len(cache.stores) == 1
        assert cache.host_bytes <= cache.max_host_bytes
        assert cache.device_bytes <= cache.max_device_bytes
        np.testing.assert_array_equal(runtime.read(held.buffers[0]), build()[0])
        setattr(cache, limit, saved)
    del held

    cache.clear()
    saved = cache.max_host_bytes
    cache.max_host_bytes = 2048
    view = cache.get(("view",), lambda: (np.arange(1024, dtype=np.int32)[:1],),
                     runtime, lambda: None, upload=(0,))
    assert not cache.stores and cache.host_bytes == 0
    assert view.host_bytes >= 4096
    np.testing.assert_array_equal(runtime.read(view.buffers[0]), [0])
    del view
    cache.max_host_bytes = saved

    for failure in (TimeoutError, OverflowError):
        calls = 0

        def fail_after_upload():
            """Interrupt after the upload but before a generation could become a hit."""
            nonlocal calls
            calls += 1
            if calls == 4:
                raise failure()

        with pytest.raises(failure):
            cache.get((failure.__name__,), build, runtime, fail_after_upload, upload=(0,))
        assert (failure.__name__,) not in cache.stores
    runtime.clear_cache()
    assert cache.host_bytes == cache.device_bytes == 0 and not cache.stores
    # Cache clearing cannot revoke handles already held by a command or invocation.
    np.testing.assert_array_equal(runtime.read(first.buffers[0]), build()[0])
    del first, second
    assert all(ref() is None for ref in references)
