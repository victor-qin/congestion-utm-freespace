"""Exact numeric continuation keys shared by every compact Zeta search domain."""

from __future__ import annotations

import math
import struct

import cython
import numpy as np


if not cython.compiled:
    def _zeta_state_bits(value: float) -> int:
        """Match the native hash while identifying both representations of zero."""
        return 0 if value == 0.0 else struct.unpack("=Q", struct.pack("=d", value))[0]

    def _zeta_state_mix(first: int, second: int) -> int:
        """Match the native unsigned mixing; equality is always checked separately."""
        mask = (1 << 64) - 1
        second = (second + 0x9E3779B97F4A7C15) & mask
        second = ((second ^ (second >> 30)) * 0xBF58476D1CE4E5B9) & mask
        second = ((second ^ (second >> 27)) * 0x94D049BB133111EB) & mask
        return ((first ^ second ^ (second >> 31)) * 0x9E3779B97F4A7C15) & mask


class _StateIndex:
    """Intern exact, broad and length-independent continuations without tuple keys.

    Attributes
    ------------
    - size (int): stable IDs issued; entries are never retired or recycled.
    - slots (memoryview): open-addressed hash slots, each holding an entry ID or -1.
    - integers, clocks (memoryview): exact key fields, with no time or length rounding.
    - hashes (memoryview): cached hashes used only for lookup and resizing.
    """

    def __init__(self) -> None:
        """Allocate small geometric-growth tables owned by one pricing call."""
        self.size = 0
        self.slots = memoryview(np.full(1024, -1, dtype=np.int64))
        self.integers = memoryview(np.empty((5, 512), dtype=np.int64))
        self.clocks = memoryview(np.empty((3, 512), dtype=np.float64))
        self.hashes = memoryview(np.empty(512, dtype=np.uint64))

    def _grow(self) -> None:
        """Grow storage and rehash existing IDs without changing their identity or order."""
        capacity: cython.Py_ssize_t = self.integers.shape[1] * 2
        integers = np.empty((5, capacity), dtype=np.int64)
        clocks = np.empty((3, capacity), dtype=np.float64)
        hashes = np.empty(capacity, dtype=np.uint64)
        integers[:, :self.size] = self.integers
        clocks[:, :self.size] = self.clocks
        hashes[:self.size] = self.hashes
        self.integers, self.clocks, self.hashes = memoryview(integers), memoryview(clocks), memoryview(hashes)
        self.slots = memoryview(np.full(capacity * 2, -1, dtype=np.int64))
        mask: cython.ulonglong = self.slots.shape[0] - 1
        slot: cython.ulonglong
        index: cython.Py_ssize_t
        for index in range(self.size):
            slot = self.hashes[index] & mask
            while self.slots[slot] >= 0:
                slot = (slot + 1) & mask
            self.slots[slot] = index

    def _intern(
        self, kind: cython.longlong, node: cython.longlong, context: cython.longlong,
        stamp: cython.double, length: cython.double, canonical: cython.double,
        roles: cython.longlong, sequence: cython.longlong, hashed: cython.ulonglong,
    ) -> cython.Py_ssize_t:
        """Resolve collisions by all original fields before returning a stable group ID.

        Parameters
        ------------
        - kind (int): exact, broad or length-independent projection discriminator.
        - node, context, roles, sequence (int): original discrete fields of the projection.
        - stamp, length, canonical (float): original unrounded continuous fields.
        - hashed (int): unsigned lookup hash; equality still checks every field above.

        Return
        --------
        - output (int): existing or newly allocated stable group ID.
        """
        if self.size == self.integers.shape[1]:
            self._grow()
        mask: cython.ulonglong = self.slots.shape[0] - 1
        slot: cython.ulonglong = hashed & mask
        index: cython.Py_ssize_t = self.slots[slot]
        while index >= 0:
            if (self.hashes[index] == hashed and self.integers[0, index] == kind
                    and self.integers[1, index] == node and self.integers[2, index] == context
                    and self.integers[3, index] == roles and self.integers[4, index] == sequence
                    and self.clocks[0, index] == stamp and self.clocks[1, index] == length
                    and self.clocks[2, index] == canonical):
                return index
            slot = (slot + 1) & mask
            index = self.slots[slot]
        index = self.size
        self.integers[0, index] = kind
        self.integers[1, index] = node
        self.integers[2, index] = context
        self.integers[3, index] = roles
        self.integers[4, index] = sequence
        self.clocks[0, index] = stamp
        self.clocks[1, index] = length
        self.clocks[2, index] = canonical
        self.hashes[index] = hashed
        self.slots[slot] = index
        self.size += 1
        return index

    def keys(
        self, node: cython.longlong, context: cython.longlong, stamp: cython.double,
        length: cython.double, canonical: cython.double, terminal: cython.bint,
        first: cython.bint, sequence: cython.longlong, broad: cython.bint, shorter: cython.bint,
    ) -> tuple:
        """Return IDs for precisely the key projections allowed by the caller's guards.

        Parameters
        ------------
        - node, context, sequence (int): unchanged interned waypoint, context and sequence IDs.
        - stamp, length, canonical (float): exact arrival and both accumulated lengths.
        - terminal, first (bool): endpoint and first-segment roles.
        - broad, shorter (bool): whether the caller has certified these comparisons.

        Return
        --------
        - output (tuple): exact ID, optional sequence-independent ID, optional length-independent ID.
          Hash collisions never establish equivalence; all retained fields must compare equal.
        """
        if not (math.isfinite(stamp) and math.isfinite(length) and math.isfinite(canonical)):
            raise ValueError("native state keys require finite clocks and lengths")
        if node < 0 or context < 0 or sequence < 0:
            raise ValueError("native state keys require nonnegative interned IDs")
        roles: cython.longlong = 2 * terminal + first
        common: cython.ulonglong = _zeta_state_mix(node, context)
        common = _zeta_state_mix(_zeta_state_mix(common, roles), _zeta_state_bits(stamp))
        distances: cython.ulonglong = _zeta_state_mix(
            _zeta_state_mix(common, _zeta_state_bits(length)), _zeta_state_bits(canonical))
        exact = self._intern(0, node, context, stamp, length, canonical, roles, sequence,
                             _zeta_state_mix(distances, sequence))
        broad_id = self._intern(1, node, context, stamp, length, canonical, roles, 0,
                                _zeta_state_mix(distances, 1)) if broad else None
        length_id = self._intern(2, node, context, stamp, 0.0, 0.0, roles, 0,
                                 _zeta_state_mix(common, 2)) if broad and shorter else None
        return exact, broad_id, length_id
