"""Small runtime-compiled Metal/CUDA interface for the optional pricing prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class GPUUnavailable(RuntimeError):
    """The explicitly requested GPU backend cannot be initialized or executed."""


_ARGUMENTS = (
    "GLOBAL const int* control",
    "GLOBAL const int* labels",
    "GLOBAL const float* values",
    "GLOBAL const int* graph",
    "GLOBAL const int* roots",
    "GLOBAL const float* root_values",
    "GLOBAL const float* visits",
    "GLOBAL const int* blocked",
    "GLOBAL const int* paid_rows",
    "GLOBAL const float* paid_values",
    "GLOBAL int* output",
    "GLOBAL float* scores",
)


def _source(backend: str) -> str:
    """Wrap shared integer/FP32 operations in device-specific entry points and atomics.

    Parameters
    ------------
    - backend (str): Metal or CUDA compilation target.

    Return
    --------
    - output (str): complete runtime-compilable source with identical algorithm bodies.
    """
    body = Path(__file__).with_name("pricing_gpu.h").read_text()
    if backend == "metal":
        prefix = """#include <metal_stdlib>
using namespace metal;
#define GLOBAL device
#define INLINE inline
#define NEXT_DOWN(x) nextafter((x), 0.0f)
INLINE int compare_exchange(GLOBAL int* p, int expected, int desired) {
    device atomic_int* a = reinterpret_cast<device atomic_int*>(p);
    int observed = expected;
    while (!atomic_compare_exchange_weak_explicit(a, &observed, desired,
             memory_order_relaxed, memory_order_relaxed)) {
        if (observed != expected) return observed;
    }
    return expected;
}
"""
        expand = ", ".join(f"{arg} [[buffer({i})]]" for i, arg in enumerate(_ARGUMENTS))
        return (
            prefix
            + body
            + f"""
kernel void expand({expand}, uint tid [[thread_position_in_grid]]) {{
    expand_one(tid, control, labels, values, graph, roots, root_values,
               visits, blocked, paid_rows, paid_values, output, scores);
}}
kernel void group_best(device const int* control [[buffer(0)]],
                       device const int* labels [[buffer(1)]],
                       device const float* scores [[buffer(2)]],
                       device int* table [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {{
    group_one(tid, control, labels, scores, table);
}}
kernel void bound_roots(device const int* control [[buffer(0)]],
                        device const float* roots [[buffer(1)]],
                        device float* bounds [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {{
    root_bound_one(tid, control, roots, bounds);
}}
kernel void future_bounds(device const int* control [[buffer(0)]],
                          device const int* graph [[buffer(1)]],
                          device const float* next [[buffer(2)]],
                          device const float* visits [[buffer(3)]],
                          device const float* air [[buffer(4)]],
                          device float* output [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {{
    future_one(tid, control, graph, next, visits, air, output);
}}
"""
        )
    prefix = """#include <math.h>
#define GLOBAL
#define INLINE __device__ __forceinline__
#define NEXT_DOWN(x) nextafterf((x), 0.0f)
INLINE int compare_exchange(int* p, int expected, int desired) {
    return atomicCAS(p, expected, desired);
}
"""
    return (
        prefix
        + body
        + f"""
extern "C" __global__ void expand({", ".join(_ARGUMENTS)}) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    expand_one(tid, control, labels, values, graph, roots, root_values,
               visits, blocked, paid_rows, paid_values, output, scores);
}}
extern "C" __global__ void group_best(const int* control, const int* labels,
                                     const float* scores, int* table) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    group_one(tid, control, labels, scores, table);
}}
extern "C" __global__ void bound_roots(const int* control, const float* roots,
                                      float* bounds) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    root_bound_one(tid, control, roots, bounds);
}}
extern "C" __global__ void future_bounds(const int* control, const int* graph,
                                        const float* next, const float* visits,
                                        const float* air, float* output) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    future_one(tid, control, graph, next, visits, air, output);
}}
"""
    )


class ComputeRuntime:
    """Execute shared pricing kernels on an explicitly selected physical GPU."""

    def __init__(self, backend: str) -> None:
        """Compile the kernels; failure never silently selects a CPU device.

        Parameters
        ------------
        - backend (str): ``metal`` or ``cuda``.

        Return
        --------
        - output (None): initializes the selected device and compiled kernels.
        """
        self.backend = backend
        self.dispatches = 0
        self._static: dict[str, tuple[np.ndarray, object]] = {}
        self._metal_buffers: dict[tuple[str, int], tuple[int, object]] = {}
        try:
            if backend == "metal":
                import Metal

                self.api = Metal
                self.device = Metal.MTLCreateSystemDefaultDevice()
                if self.device is None:
                    raise GPUUnavailable("No Metal GPU is available")
                self.device_name = str(self.device.name())
                options = Metal.MTLCompileOptions.alloc().init()
                options.setFastMathEnabled_(False)
                library, error = self.device.newLibraryWithSource_options_error_(
                    _source(backend), options, None
                )
                if library is None:
                    raise GPUUnavailable(f"Metal pricing compilation failed: {error}")
                self.kernels = {}
                for name in ("expand", "group_best", "bound_roots", "future_bounds"):
                    pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
                        library.newFunctionWithName_(name), None
                    )
                    if pipeline is None:
                        raise GPUUnavailable(f"Metal {name} pipeline failed: {error}")
                    self.kernels[name] = pipeline
                self.queue = self.device.newCommandQueue()
            elif backend == "cuda":
                import cupy

                self.api = cupy
                self.device = cupy.cuda.Device()
                properties = cupy.cuda.runtime.getDeviceProperties(self.device.id)
                name = properties["name"]
                self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                module = cupy.RawModule(code=_source(backend), options=("--fmad=false",))
                self.kernels = {
                    name: module.get_function(name)
                    for name in ("expand", "group_best", "bound_roots", "future_bounds")
                }
            else:
                raise ValueError("gpu_backend must be 'metal' or 'cuda'")
        except (ImportError, RuntimeError) as exc:
            if isinstance(exc, GPUUnavailable):
                raise
            raise GPUUnavailable(f"Cannot initialize {backend} pricing: {exc}") from exc

    def clear_cache(self) -> None:
        """Release static and reusable Metal buffers at the end of their batch."""
        self._static.clear()
        self._metal_buffers.clear()

    def run(
        self,
        name: str,
        count: int,
        arrays: Sequence[np.ndarray],
        outputs: Sequence[int],
        static: dict[int, str] | None = None,
        write_only: Sequence[int] = (),
    ) -> tuple[np.ndarray, ...]:
        """Dispatch one kernel and return completed output buffers.

        Parameters
        ------------
        - name (str): compiled kernel name.
        - count (int): number of logical GPU invocations.
        - arrays (Sequence[np.ndarray]): contiguous int32/float32 buffer arguments.
        - outputs (Sequence[int]): argument indices to read after GPU completion.
        - static (dict[int, str] | None): immutable arguments cached for this batch.
        - write_only (Sequence[int]): nonstatic outputs whose initial content is unused;
          Metal skips their upload. The caller may read only fields the kernel writes.

        Return
        --------
        - output (tuple[np.ndarray, ...]): copies of the requested completed arrays.
        """
        if count == 0:
            return tuple(arrays[i].copy() for i in outputs)
        buffers = []
        static = static or {}
        if any(i not in outputs or i in static for i in write_only):
            raise ValueError("Write-only buffers must be nonstatic output arguments")
        for i, array in enumerate(arrays):
            if array.dtype not in (np.dtype("int32"), np.dtype("float32")):
                raise TypeError("GPU buffers must use int32 or float32")
            if not array.flags.c_contiguous:
                raise ValueError("GPU buffers must be contiguous")
            key = static.get(i)
            cached = self._static.get(key) if key is not None else None
            if cached is not None and cached[0] is array:
                buffers.append(cached[1])
                continue
            if self.backend == "metal":
                required = max(4, array.nbytes)
                slot = (name, i)
                reusable = self._metal_buffers.get(slot) if key is None else None
                if reusable is not None and reusable[0] >= required:
                    buffer = reusable[1]
                else:
                    buffer = self.device.newBufferWithLength_options_(
                        required, self.api.MTLResourceStorageModeShared
                    )
                    if buffer is None:
                        raise GPUUnavailable("Metal buffer allocation failed")
                    if key is None:
                        # Retain only the largest current allocation, not every past shape.
                        self._metal_buffers[slot] = (required, buffer)
                if array.nbytes and i not in write_only:
                    memoryview(buffer.contents().as_buffer(array.nbytes))[:] = memoryview(
                        array
                    ).cast("B")
            else:
                buffer = self.api.asarray(array if array.size else np.zeros(1, array.dtype))
            buffers.append(buffer)
            if key is not None:
                self._static[key] = (array, buffer)
        if self.backend == "metal":
            pipeline = self.kernels[name]
            command = self.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            for i, buffer in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buffer, 0, i)
            width = min(128, int(pipeline.maxTotalThreadsPerThreadgroup()))
            encoder.dispatchThreads_threadsPerThreadgroup_(
                self.api.MTLSizeMake(count, 1, 1), self.api.MTLSizeMake(width, 1, 1)
            )
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if command.status() != self.api.MTLCommandBufferStatusCompleted:
                raise GPUUnavailable(f"Metal pricing dispatch failed: {command.error()}")
            result = tuple(
                np.frombuffer(buffers[i].contents().as_buffer(arrays[i].nbytes), arrays[i].dtype)
                .copy()
                .reshape(arrays[i].shape)
                for i in outputs
            )
        else:
            self.kernels[name](((count + 127) // 128,), (128,), tuple(buffers))
            self.api.cuda.get_current_stream().synchronize()
            result = tuple(self.api.asnumpy(buffers[i]).reshape(arrays[i].shape) for i in outputs)
        self.dispatches += 1
        return result
