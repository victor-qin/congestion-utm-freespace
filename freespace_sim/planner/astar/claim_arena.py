"""Flat, numba-visible storage for the occupancy claim journal.

``CompiledHexOccupancy`` records every committed (cell, step-span) as a packed int64 "claim", today
in a ``dict[key, list[int]]``. That dict is the reason the interval pools still exist: the pools are
a *query accelerator* derived from the claims, and removing a flight from them costs a rebuild of
each touched cell from its SURVIVORS — measured at 12.2x the released flight's own footprint at
``density_faa`` scale, and growing with congestion.

The obvious fix is to answer occupancy from the claims directly and delete the pools. Phase 0
measured what that needs (``context/lns_plan.md``):

  * a claim IS a blocked span, so painting the per-plan window bitmap from claims replaces the
    pools' invert-and-merge and is **2.81x faster** (0.514 ms against 1.443 ms per window);
  * but numba cannot iterate a dict of lists, so the host would have to flatten the window's claims
    first — **37.6 ms per window, 26x worse than the whole build it replaces**.

So the storage has to be flat, and this is it. One ``int64`` arena holds every claim; each cell's
claims occupy ONE contiguous slab within it, described by ``start``/``length``/``cap`` arrays keyed
by ``key = (cell << 1) | pool_idx`` (the same key ``_claims`` uses). A window build then reads
``arena[start[key] : start[key] + length[key]]`` with no host-side work at all.

**Removal is a swap-remove**, which is what makes the whole thing worth doing: find the claim in its
slab (~30 comparisons at full scale) and move the slab's last entry over it. Order within a slab is
irrelevant — the window paint ORs spans together and ``blocked_at`` is a membership test — so
nothing depends on it, and the cost is the flight's OWN footprint rather than everyone else's.

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
    """Append ``n`` claims (``keys`` SORTED ascending, ``vals`` permuted to match).

    Returns 0 on success, or the number of extra arena slots required — in which case **nothing has
    been written**. The capacity pass runs first precisely so a caller can grow and retry without
    tracking what a partial batch already applied."""
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
    """Swap-remove ``n`` claims. Returns the number NOT found, which must be 0 — a miss is the same
    drift signal ``_claims[key].remove`` raises ``ValueError`` for, reported rather than thrown so
    the caller decides (numba cannot raise a useful exception here)."""
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
    """Is ``key`` blocked at step ``s``? A membership scan over the cell's slab — the operation the
    interval pools exist to accelerate, and which the per-plan window makes unnecessary."""
    base = start[key]
    for m in range(length[key]):
        packed = arena[base + m]
        if (packed >> s0_shift) <= s <= ((packed >> span_bits) & field_mask):
            return True
    return False


@njit(cache=True, nogil=True)
def compact_into(arena, start, length, cap, dst, headroom_num, headroom_den):
    """Copy every live slab into ``dst`` back to back, rewriting ``start``/``cap``. Returns the new
    tail. Only growth produces the garbage this reclaims, so this is off the destroy path.

    Each slab keeps a little headroom rather than being packed to exactly its length. Packing tight
    is smaller for an instant and worse immediately after: with ``cap == length`` the very next claim
    added to a cell re-homes the whole slab, so a compaction would hand back its own savings as fresh
    garbage on the following commit."""
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
        """Add a batch. Sorts by key (``add_many``'s capacity pass needs each cell's batch contiguous),
        then grows and retries at most twice: once for the reported shortfall, once if compaction is
        the cheaper way to find it."""
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

        Two triggers, because they catch different things. GARBAGE past a third of the tail is
        fragmentation from re-homed slabs. TAIL past twice the live claims is the subtler one: slab
        capacities are powers of two, so a bulk load leaves every slab up to 2x its length even with
        no garbage at all, and the buffer doubles on top of that — measured at 146 MB of allocation
        holding 34 MB of claims, sitting just under a garbage-only threshold and never compacting.

        Checked after a SUCCESSFUL add, not only when one runs short, or the arena only ever reclaims
        under allocation pressure. Removals never fragment, so this is off the destroy path."""
        live = int(self.length.sum())
        if self.garbage[0] > self.tail[0] // 3 or self.tail[0] > 2 * max(live, 1):
            self.compact()

    def remove(self, keys: np.ndarray, vals: np.ndarray) -> None:
        n = keys.shape[0]
        if n == 0:
            return
        missing = remove_many(keys, vals, n, self.arena, self.start, self.length)
        if missing:
            raise ValueError(
                f"ClaimArena: {missing} of {n} claims to remove were not present — the journal and "
                f"the arena have drifted")

    def blocked(self, key: int, s: int) -> bool:
        return bool(blocked_at(key, s, self.arena, self.start, self.length,
                               self._s0_shift, self._span_bits, self._field_mask))

    def slab(self, key: int) -> np.ndarray:
        """The cell's claims as one contiguous view — what a window build reads."""
        s = int(self.start[key])
        return self.arena[s:s + int(self.length[key])]

    def compact(self) -> None:
        """Rewrite every live slab back to back into a RIGHT-SIZED buffer.

        Sizing the destination to the live claims (plus headroom for the next round of growth) rather
        than to the current buffer is what actually returns the memory — compacting in place leaves
        the allocation at its high-water mark, which is how the arena sat at 150 MB while holding
        34 MB of claims."""
        live = int(self.length.sum())
        n_live_keys = int(np.count_nonzero(self.length))
        want = live + (live * _HEADROOM_NUM) // _HEADROOM_DEN + 2 * n_live_keys
        dst = np.zeros(max(want + want // 8, 1 << 16), np.int64)
        self.tail[0] = compact_into(self.arena, self.start, self.length, self.cap, dst,
                                    _HEADROOM_NUM, _HEADROOM_DEN)
        self.arena = dst
        self.garbage[0] = 0

    def reset(self) -> None:
        self.start[:] = 0
        self.length[:] = 0
        self.cap[:] = 0
        self.tail[0] = 0
        self.garbage[0] = 0

    # ---- diagnostics ----
    def _grow(self, shortfall: int) -> None:
        size = max(self.arena.shape[0] * 2, self.arena.shape[0] + shortfall * 2, 1 << 16)
        grown = np.zeros(size, np.int64)
        grown[:self.tail[0]] = self.arena[:self.tail[0]]
        self.arena = grown

    @property
    def n_claims(self) -> int:
        return int(self.length.sum())

    def nbytes(self) -> int:
        return int(self.arena.nbytes + self.start.nbytes + self.length.nbytes + self.cap.nbytes)

    def as_dict(self) -> dict:
        """``{key: sorted(claims)}`` for every non-empty slab — the shape ``_claims`` holds, so the
        two can be compared directly. Diagnostics only: O(live claims) and allocation-heavy."""
        out = {}
        for k in np.nonzero(self.length)[0]:
            out[int(k)] = sorted(int(x) for x in self.slab(int(k)))
        return out
