"""Runtime-owned immutable numeric operands, independent of search invocation state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from .gpu_buffers import DeviceBuffer

if TYPE_CHECKING:
    from .gpu_runtime import ComputeRuntime


def _key_bytes(value: object) -> int:
    """Conservatively charge shared key components again for each retained entry."""
    return sys.getsizeof(value) + (sum(map(_key_bytes, value))
                                 if isinstance(value, (tuple, frozenset)) else 0)


def _array_bytes(array: np.ndarray) -> int:
    """Charge the complete retained backing allocation, including headers of reshape views."""
    size = 0
    value: object = array
    while isinstance(value, (np.ndarray, memoryview)):
        size += sys.getsizeof(value)
        value = value.base if isinstance(value, np.ndarray) else value.obj
    return size + (sys.getsizeof(value) if value is not None else 0)


@dataclass(frozen=True, slots=True)
class OperandBlock:
    """A complete immutable build whose borrowers outlive cache eviction.

    Attributes
    ------------
    - arrays (tuple[np.ndarray, ...]): read-only host operands, never search state.
    - buffers (tuple[DeviceBuffer, ...]): matching selected uploads owned by one runtime.
    - host_bytes (int): retained arrays and conservatively charged key/container storage.
    - device_bytes (int): actual device allocation sizes, including empty-buffer minima.
    - build_s (float): measured host build time avoided on a hit; excludes upload time.
    - encoded_bytes (int): interval arrays encoded by this build, excluding borrowed inputs.
    """

    arrays: tuple[np.ndarray, ...]
    buffers: tuple[DeviceBuffer, ...]
    host_bytes: int
    device_bytes: int
    build_s: float
    encoded_bytes: int


class OperandCache:
    """Bound immutable operand retention separately from live invocation budgets.

    Attributes
    ------------
    - owner (object): runtime token required even for host-only cache hits.
    - stores (OrderedDict): least-recently-used complete operand blocks.
    - host_bytes (int): conservatively charged retained host storage.
    - device_bytes (int): retained device allocation bytes, not process RSS.
    - max_host_bytes (int): host retention limit; zero disables admission.
    - max_device_bytes (int): device retention limit; zero disables admission.
    - max_entries (int): metadata bound even for empty blocks.
    """

    def __init__(
        self, owner: object, *, max_host_bytes: int = 128 * 1024 * 1024,
        max_device_bytes: int = 128 * 1024 * 1024, max_entries: int = 8192,
    ) -> None:
        """Initialize an empty cache with independent host, device and entry limits.

        Parameters
        ------------
        - owner (object): unique token of the runtime permitted to borrow these buffers.
        - max_host_bytes (int): retained array/key storage limit; zero disables admission.
        - max_device_bytes (int): retained device allocation limit; zero disables admission.
        - max_entries (int): retained metadata limit; zero disables admission.

        Return
        --------
        - output (None): initializes empty storage without allocating device buffers.
        """
        self.owner = owner
        self.max_host_bytes, self.max_device_bytes = max_host_bytes, max_device_bytes
        self.max_entries = max_entries
        self.stores: OrderedDict[tuple, OperandBlock] = OrderedDict()
        self.host_bytes = self.device_bytes = 0

    def clear(self) -> None:
        """Release every cache reference; live borrowers still own their immutable buffers."""
        self.stores.clear()
        self.host_bytes = self.device_bytes = 0

    def get(
        self, key: tuple | None, build: Callable[[], tuple[np.ndarray, ...]],
        runtime: ComputeRuntime, check: Callable[[], None], *, upload: tuple[int, ...],
        encoded: tuple[int, ...] = (),
    ) -> OperandBlock:
        """Borrow exact operands or publish only after the complete build and all uploads.

        Parameters
        ------------
        - key (tuple | None): exact immutable dependencies; None forces invocation-local work.
        - build (Callable): returns fresh, complete arrays without retaining the invocation.
        - runtime (ComputeRuntime): sole device owner of this cache.
        - check (Callable): cooperative deadline check, including before publication and on hits.
        - upload (tuple[int, ...]): array indices to upload in binding order.
        - encoded (tuple[int, ...]): newly encoded interval arrays, for avoided-work accounting.

        Return
        --------
        - output (OperandBlock): complete read-only operands, independently owned by the caller.
          Failed/interrupted builds never become hits; oversize builds remain invocation-local.
        """
        if runtime._owner is not self.owner:
            raise ValueError("Operand cache belongs to another runtime")
        check()
        counters = runtime.counters
        if key is not None and key in self.stores:
            self.stores.move_to_end(key)
            block = self.stores[key]
            counters["integer_cache_hits"] += 1
            counters["integer_encoding_avoided_s"] += block.build_s
            counters["integer_encoding_avoided_bytes"] += block.encoded_bytes
            counters["integer_upload_avoided_bytes"] += sum(b.nbytes for b in block.buffers)
            return block
        counters["integer_cache_misses"] += 1
        started = time.monotonic()
        arrays = build()
        check()
        build_s = time.monotonic() - started
        counters["integer_encoding_s"] += build_s
        for array in arrays:
            array.flags.writeable = False
        buffers = []
        for index in upload:
            check()
            buffers.append(runtime.upload(arrays[index]))
        check()
        host_bytes = (_key_bytes(key) + sys.getsizeof(arrays) + sum(map(_array_bytes, arrays))
                      + 512 + len(buffers) * 128)
        block = OperandBlock(arrays, tuple(buffers), host_bytes,
                             sum(max(4, b.nbytes) for b in buffers), build_s,
                             sum(arrays[i].nbytes for i in encoded))
        if (key is not None and self.max_entries > 0 and self.max_host_bytes > 0
                and self.max_device_bytes > 0 and host_bytes <= self.max_host_bytes
                and block.device_bytes <= self.max_device_bytes):
            while self.stores and (len(self.stores) >= self.max_entries
                                  or self.host_bytes + host_bytes > self.max_host_bytes
                                  or self.device_bytes + block.device_bytes > self.max_device_bytes):
                _, old = self.stores.popitem(last=False)
                self.host_bytes -= old.host_bytes
                self.device_bytes -= old.device_bytes
                counters["integer_cache_evictions"] += 1
            self.stores[key] = block
            self.host_bytes += host_bytes
            self.device_bytes += block.device_bytes
        else:
            counters["integer_cache_skips"] += 1
        return block
