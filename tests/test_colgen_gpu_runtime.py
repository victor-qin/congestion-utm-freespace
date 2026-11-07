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
) -> tuple[ComputeRuntime, list[weakref.ReferenceType[_Buffer]]]:
    """Run the production adapter against synchronous CPU-backed Metal objects.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): installs the fake backend without using a GPU.
    - execute (Callable): simulated kernel receives buffers and the dispatch count.

    Return
    --------
    - output (tuple): real runtime and weak references to all allocated buffers.
    """
    allocations = []

    def allocate(length: int, options: int) -> _Buffer:
        """Track allocations without retaining buffers after the runtime releases them."""
        assert options == 1
        buffer = _Buffer(length)
        allocations.append(weakref.ref(buffer))
        return buffer

    def command_buffer() -> SimpleNamespace:
        """Execute the simulated kernel only when the runtime waits for completion."""
        buffers = []
        counts = []

        def bind(buffer: _Buffer, offset: int, index: int) -> None:
            """Keep each argument binding in its declared order."""
            assert offset == 0 and index == len(buffers)
            buffers.append(buffer)

        encoder = SimpleNamespace(
            setComputePipelineState_=lambda pipeline: None,
            setBuffer_offset_atIndex_=bind,
            dispatchThreads_threadsPerThreadgroup_=lambda size, group: counts.append(size[0]),
            endEncoding=lambda: None,
        )
        return SimpleNamespace(
            computeCommandEncoder=lambda: encoder,
            commit=lambda: None,
            waitUntilCompleted=lambda: execute(buffers, counts[0]),
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
