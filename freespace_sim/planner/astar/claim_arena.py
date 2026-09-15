"""Flat, numba-visible storage for the occupancy claim journal.

``CompiledHexOccupancy`` records every committed (cell, step-span) as a packed int64 "claim".
Storing those as free-interval query pools made removing a flight rebuild every cell it touched
from its SURVIVORS — super-linear in congestion, far more than the released flight's own footprint.
Answering occupancy from the claims directly removes that cost, but numba cannot iterate a dict of
lists, so the storage has to be FLAT for a window build to read it without host-side flattening.

One ``int64`` arena holds every claim; each cell's claims occupy ONE contiguous slab within it,
described by ``start``/``length``/``cap`` arrays keyed by ``key = (cell << 1) | pool_idx`` (the same
key ``_claims`` uses). A window build then reads ``arena[start[key] : start[key] + length[key]]``
with no host-side work at all.

**Removal is a swap-remove**, which is what makes the whole thing worth doing: find the claim in its
slab and move the slab's last entry over it. Order within a slab is irrelevant — the window paint
ORs spans together and ``blocked_at`` is a membership test — so nothing depends on it, and the cost
is the flight's OWN footprint rather than everyone else's.

**Growth is the only source of garbage.** A full slab is re-homed at the arena tail at twice the
capacity and its old extent abandoned; ``compact`` reclaims those. Removals never fragment, so
compaction is unrelated to the destroy path.

``add_many`` is deliberately atomic: it computes the tail it needs BEFORE writing anything and
returns that number if the arena is too small, so the host can grow and call again with no risk of
double-applying a partially-written batch. It requires its input sorted by key, which is what lets
the capacity pass see each cell's whole batch at once.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                     # numba absent — same guard as `window`: this module is
    def njit(*_args, **_kwargs):        # imported at module level by `compiled_hex_occupancy`, whose
        def deco(fn):                   # own numba fallback is an ImportError guard around `.kernel`.
            def _needs_numba(*_a, **_kw):
                raise RuntimeError(
                    f"{fn.__name__} is a numba kernel and numba is not installed; the claim arena is "
                    f"only reachable from the compiled A* path")
            return _needs_numba
        return deco

_MIN_SLAB = 4                   # first allocation for a cell that gains its first claim
_GROWTH = 2                     # slab capacity multiplier; garbage per growth == the old capacity
_HEADROOM_NUM, _HEADROOM_DEN = 1, 4     # slack a compacted slab keeps, so the next add does not
#                                         immediately re-home it (see `compact_into`)


@njit(cache=True, nogil=True)
def add_many(keys, vals, n, arena, start, length, cap, tail, garbage):
    """Append ``n`` claims to their cells' slabs (``keys`` SORTED ascending, ``vals`` matching).

    A capacity pass runs first and is read-only, so on a shortfall NOTHING is written and the caller
    can grow and retry without tracking what a partial batch already applied. Requires keys sorted
    so each cell's whole batch is seen at once.

    Parameters
    ------------
    - keys (int64[:]): per-claim cell key ``(cell << 1) | pool_idx``, sorted ascending.
    - vals (int64[:]): packed claim for each key, permuted to match ``keys``.
    - n (int): number of claims to append (a prefix of ``keys``/``vals``).
    - arena (int64[:]): the shared claim buffer, mutated in place.
    - start (int64[:]): per-key slab start offset, rewritten when a slab is re-homed.
    - length (int64[:]): per-key live claim count, incremented here.
    - cap (int64[:]): per-key slab capacity, grown when a slab is re-homed.
    - tail (int64[1]): one-past-last used arena slot, advanced on re-home.
    - garbage (int64[1]): abandoned-slot counter, incremented on re-home.

    Return
    --------
    - output (int): 0 on success, else the extra arena slots required (in which case nothing is
      written).
    """
    need = 0
    i = 0
    while i < n:                                    # capacity pass — read-only
        k = keys[i]
        j = i
        while j < n and keys[j] == k:
            j += 1
        c = cap[k]
        target = length[k] + (j - i)
        while c < target:
            c = _MIN_SLAB if c == 0 else c * _GROWTH
        if c != cap[k]:
            need += c
        i = j
    free = arena.shape[0] - tail[0]
    if need > free:
        return need - free

    i = 0
    while i < n:                                    # apply pass
        k = keys[i]
        j = i
        while j < n and keys[j] == k:
            j += 1
        c = cap[k]
        target = length[k] + (j - i)
        if c < target:
            while c < target:
                c = _MIN_SLAB if c == 0 else c * _GROWTH
            ns = tail[0]
            tail[0] += c
            old = start[k]
            for m in range(length[k]):              # re-home the live entries, abandon the extent
                arena[ns + m] = arena[old + m]
            garbage[0] += cap[k]
            start[k] = ns
            cap[k] = c
        base = start[k] + length[k]
        for m in range(j - i):
            arena[base + m] = vals[i + m]
        length[k] += j - i
        i = j
    return 0


@njit(cache=True, nogil=True)
def remove_many(keys, vals, n, arena, start, length):
    """Swap-remove ``n`` claims from their cells' slabs.

    A miss (claim not present) is the drift signal ``_claims[key].remove`` raises ``ValueError``
    for; it is COUNTED and returned rather than thrown (numba cannot raise a useful exception here),
    so the caller decides. The count must be 0 in a consistent journal.

    Parameters
    ------------
    - keys (int64[:]): per-claim cell key ``(cell << 1) | pool_idx``.
    - vals (int64[:]): the packed claim to remove for each key.
    - n (int): number of claims to remove.
    - arena (int64[:]): the shared claim buffer, mutated in place.
    - start (int64[:]): per-key slab start offset (read only).
    - length (int64[:]): per-key live claim count, decremented per hit.

    Return
    --------
    - output (int): the number of claims NOT found (0 in a consistent journal).
    """
    missing = 0
    for i in range(n):
        k = keys[i]
        v = vals[i]
        s = start[k]
        ln = length[k]
        hit = -1
        for m in range(ln):
            if arena[s + m] == v:
                hit = m
                break
        if hit < 0:
            missing += 1
            continue
        arena[s + hit] = arena[s + ln - 1]          # order within a slab carries no meaning
        length[k] = ln - 1
    return missing


@njit(cache=True, nogil=True)
def blocked_at(key, s, arena, start, length, s0_shift, span_bits, field_mask):
    """Is ``key`` blocked at step ``s``? Diagnostic membership scan over the cell's claim slab.

    Parameters
    ------------
    - key (int): the cell key ``(cell << 1) | pool_idx`` whose slab to scan.
    - s (int): the step to test for membership in any claimed span.
    - arena (int64[:]): the shared claim buffer.
    - start (int64[:]): per-key slab start offset.
    - length (int64[:]): per-key live claim count.
    - s0_shift (int): right-shift recovering a claim's span start.
    - span_bits (int): right-shift recovering a claim's span end.
    - field_mask (int): mask isolating the span-end field after the shift.

    Return
    --------
    - output (bool): True if any claim in the slab covers step ``s``.
    """
    base = start[key]
    for m in range(length[key]):
        packed = arena[base + m]
        if (packed >> s0_shift) <= s <= ((packed >> span_bits) & field_mask):
            return True
    return False


@njit(cache=True, nogil=True)
def compact_into(arena, start, length, cap, dst, headroom_num, headroom_den):
    """Copy every live slab into ``dst`` back to back, rewriting ``start``/``cap``; return new tail.

    Only growth produces the garbage this reclaims, so this is off the destroy path. Each slab
    keeps a little headroom rather than being packed to exactly its length: with ``cap == length``
    the very next claim added to a cell would re-home the whole slab, so a tight pack hands back its
    own savings as fresh garbage on the following commit.

    Parameters
    ------------
    - arena (int64[:]): the source claim buffer (live slabs read from it).
    - start (int64[:]): per-key slab start offset, rewritten to the ``dst`` layout.
    - length (int64[:]): per-key live claim count (unchanged; read to size each slab).
    - cap (int64[:]): per-key slab capacity, rewritten to length + headroom.
    - dst (int64[:]): destination buffer the live slabs are packed into.
    - headroom_num (int): numerator of the per-slab headroom fraction.
    - headroom_den (int): denominator of the per-slab headroom fraction.

    Return
    --------
    - output (int): the new arena tail (one past the last packed slot).
    """
    tail = 0
    for k in range(start.shape[0]):
        ln = length[k]
        if ln == 0:
            start[k] = 0
            cap[k] = 0
            continue
        src = start[k]
        for m in range(ln):
            dst[tail + m] = arena[src + m]
        start[k] = tail
        c = ln + (ln * headroom_num) // headroom_den + 2
        cap[k] = c
        tail += c
    return tail


class ClaimArena:
    """Host-side owner of the arrays above: growth, compaction, and the packed-field constants."""

    def __init__(self, n_keys: int, s0_shift: int, span_bits: int, field_mask: int,
                 capacity: int = 1 << 16):
        """Allocate the arena and per-key slab arrays for ``n_keys`` cell keys.

        Parameters
        ------------
        - n_keys (int): number of distinct cell keys ``(cell << 1) | pool_idx``; sizes the
          ``start``/``length``/``cap`` arrays.
        - s0_shift (int): right-shift recovering a claim's span start (stored for :meth:`blocked`).
        - span_bits (int): right-shift recovering a claim's span end.
        - field_mask (int): mask isolating the span-end field.
        - capacity (int): initial arena size in claims (default ``1 << 16``).

        Return
        --------
        - output (None): initializes the arena, slab arrays, and packed-field constants.
        """
        self.n_keys = n_keys
        self._s0_shift, self._span_bits, self._field_mask = s0_shift, span_bits, field_mask
        self.arena = np.zeros(max(capacity, _MIN_SLAB), np.int64)
        self.start = np.zeros(n_keys, np.int64)
        self.length = np.zeros(n_keys, np.int64)
        self.cap = np.zeros(n_keys, np.int64)
        self.tail = np.zeros(1, np.int64)
        self.garbage = np.zeros(1, np.int64)

    # ---- maintenance ----
    def add(self, keys: np.ndarray, vals: np.ndarray) -> None:
        """Add a batch of claims, sorting, growing, and compacting as needed.

        Sorts by key because ``add_many``'s capacity pass needs each cell's batch contiguous, then
        retries up to three times: reclaim garbage first when it dominates (growth is the only thing
        that made it), otherwise grow the backing buffer.

        Parameters
        ------------
        - keys (np.ndarray): per-claim cell keys (any order; sorted here).
        - vals (np.ndarray): packed claim for each key, permuted to match the sort.

        Return
        --------
        - output (None): mutates the arena in place; raises ``RuntimeError`` if the batch cannot be
          satisfied after growing and compacting.
        """
        n = keys.shape[0]
        if n == 0:
            return
        order = np.argsort(keys, kind="stable")
        ks, vs = np.ascontiguousarray(keys[order]), np.ascontiguousarray(vals[order])
        for _ in range(3):
            short = add_many(ks, vs, n, self.arena, self.start, self.length, self.cap,
                             self.tail, self.garbage)
            if short == 0:
                self._maybe_compact()
                return
            if self.garbage[0] > self.tail[0] // 3:
                self.compact()                       # reclaim before growing: growth is the only
                continue                             # thing that made this garbage
            self._grow(short)
        raise RuntimeError("ClaimArena: could not satisfy a batch after growing and compacting")

    def _maybe_compact(self) -> None:
        """Reclaim when the arena has drifted well past the data it holds.

        Three triggers catch different waste: GARBAGE from re-homed slabs, TAIL capacity far beyond
        what compaction can attain, and an overgrown backing buffer. The attainable size matters:
        compaction deliberately keeps 25% per-slab headroom plus two slots per live key. Comparing
        tail directly with live claims made a one-claim slab (post-compact cap == 3) permanently
        satisfy ``tail > 2*live`` and rewrite the entire arena after every sparse commit.

        Checked after a SUCCESSFUL add, not only when one runs short, or the arena only ever reclaims
        under allocation pressure. Removals never fragment, so this is off the destroy path."""
        live = int(self.length.sum())
        n_live_keys = int(np.count_nonzero(self.length))
        attainable_tail = live + (live * _HEADROOM_NUM) // _HEADROOM_DEN + 2 * n_live_keys
        attainable_buffer = max(attainable_tail + attainable_tail // 8, 1 << 16)
        if (self.garbage[0] > self.tail[0] // 3
                or self.tail[0] > 2 * max(attainable_tail, 1)
                or self.arena.shape[0] > 2 * attainable_buffer):
            self.compact()

    def remove(self, keys: np.ndarray, vals: np.ndarray) -> None:
        """Swap-remove a batch of claims, raising if any are absent.

        Parameters
        ------------
        - keys (np.ndarray): per-claim cell keys.
        - vals (np.ndarray): the packed claim to remove for each key.

        Return
        --------
        - output (None): mutates the arena in place; raises ``ValueError`` if any claim to remove is
          not present (the journal and arena have drifted).
        """
        n = keys.shape[0]
        if n == 0:
            return
        missing = remove_many(keys, vals, n, self.arena, self.start, self.length)
        if missing:
            raise ValueError(
                f"ClaimArena: {missing} of {n} claims to remove were not present — the journal and "
                f"the arena have drifted")

    def blocked(self, key: int, s: int) -> bool:
        """True if any claim in ``key``'s slab covers step ``s`` (wraps ``blocked_at``)."""
        return bool(blocked_at(key, s, self.arena, self.start, self.length,
                               self._s0_shift, self._span_bits, self._field_mask))

    def slab(self, key: int) -> np.ndarray:
        """The cell's claims as one contiguous view — what a window build reads.

        Parameters
        ------------
        - key (int): the cell key ``(cell << 1) | pool_idx`` whose slab to view.

        Return
        --------
        - output (np.ndarray): a view of ``arena`` spanning the key's live claims (may be empty).
        """
        s = int(self.start[key])
        return self.arena[s:s + int(self.length[key])]

    def compact(self) -> None:
        """Rewrite every live slab back to back into a RIGHT-SIZED buffer.

        Sizing the destination to the live claims (plus headroom for the next round of growth) rather
        than to the current buffer is what actually returns the memory: compacting in place leaves
        the allocation at its high-water mark instead of shrinking it.
        """
        live = int(self.length.sum())
        n_live_keys = int(np.count_nonzero(self.length))
        want = live + (live * _HEADROOM_NUM) // _HEADROOM_DEN + 2 * n_live_keys
        dst = np.zeros(max(want + want // 8, 1 << 16), np.int64)
        self.tail[0] = compact_into(self.arena, self.start, self.length, self.cap, dst,
                                    _HEADROOM_NUM, _HEADROOM_DEN)
        self.arena = dst
        self.garbage[0] = 0

    def reset(self) -> None:
        """Zero the slab arrays and tail/garbage counters, dropping every claim (buffer kept)."""
        self.start[:] = 0
        self.length[:] = 0
        self.cap[:] = 0
        self.tail[0] = 0
        self.garbage[0] = 0

    # ---- diagnostics ----
    def _grow(self, shortfall: int) -> None:
        """Reallocate the arena larger (≥ double, ≥ ``2 * shortfall``), copying the live prefix."""
        size = max(self.arena.shape[0] * 2, self.arena.shape[0] + shortfall * 2, 1 << 16)
        grown = np.zeros(size, np.int64)
        grown[:self.tail[0]] = self.arena[:self.tail[0]]
        self.arena = grown

    @property
    def n_claims(self) -> int:
        """Total live claims across all slabs."""
        return int(self.length.sum())

    def nbytes(self) -> int:
        """Backing-store size in bytes (arena plus the ``start``/``length``/``cap`` arrays)."""
        return int(self.arena.nbytes + self.start.nbytes + self.length.nbytes + self.cap.nbytes)

    def as_dict(self) -> dict:
        """``{key: sorted(claims)}`` for every non-empty slab — the shape ``_claims`` holds, so the
        two can be compared directly. Diagnostics only: O(live claims) and allocation-heavy."""
        out = {}
        for k in np.nonzero(self.length)[0]:
            out[int(k)] = sorted(int(x) for x in self.slab(int(k)))
        return out
