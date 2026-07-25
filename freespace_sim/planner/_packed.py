"""Cache-line-aligned packed record arrays for the A* kernel's random-access structures (issue #8).

**Why.** Struct-of-arrays is the right layout for sequential sweeps and the wrong one for random
access. The kernel's g-hash was five separate allocations (``g_key``/``g_gen``/``g_val``/``g_came``/
``g_flag``), so a single node relaxation touched **five cache lines — 640 B on this machine's 128 B
lines — to move 33 B of payload** (5% line utilization). That is invisible in a solo profile (one
process has a whole 12 MB cluster L2 to itself) but dominates under concurrency: this machine's
P-cores share **12 MB of L2 per 4-core cluster**, so W workers each get ~12/W MB, and a 60k-expansion
search touching ~37 MB of cache lines misses essentially every probe.

Packing the same fields into one 32 B record — 4 records per 128 B line — was measured
(``analysis/prof_memory.py``) at **2.5x faster solo and 3.1x at 8 concurrent processes** on the
kernel's exact probe+relax pattern, and it cuts that structure's own concurrency tax from 2.28x to
1.85x. It is a pure layout change: the probe sequence, the comparisons and the stored values are
identical, so plans stay byte-identical.

**How the 32 B record holds 33 B of fields.** ``g_val`` is a float and the rest are integers, so the
buffer is allocated once as ``(rows, 4) int64`` and a ``float64`` *view of the same memory* addresses
the value column — no per-access bit-casting, and numba compiles both views to plain loads/stores.
The 1-byte closed flag is folded into bit 0 of the generation stamp (generations advance by 2, so
they are always even and the flag never collides with them), which removes the fifth field entirely.

Alignment matters: ``np.empty`` promises only 16-64 B, and a 32 B record straddling a 128 B boundary
gives back much of the win, so :func:`aligned_2d` places the base address deliberately.
"""
from __future__ import annotations

import numpy as np

CACHE_LINE = 128            # Apple silicon; x86 is 64 — over-aligning there is harmless

# g-hash record columns (int64 view, except G_VAL which is read/written through the float64 view).
G_KEY, G_GEN, G_VAL, G_CAME = 0, 1, 2, 3
GEN_STEP = 2                # generations advance by 2 so bit 0 stays free for the closed flag
GEN_MASK = -2               # ~1: mask the closed bit off a stamp before comparing to `gen`
GEN_WRAP = 1 << 30          # re-stamp before `gen` outgrows `ov_own_gen`'s int32 range (~5e8 plans)

# NOT packed, deliberately: the frontier heap. It was tried (32 B records, 4-ary so a node's four
# children fill one aligned line) and was byte-exact but **21% slower end to end** — a heap's sift
# path stays in the top few resident levels and its one deep access per operation walks index `size`
# by ±1, so it prefetches perfectly and had no random-access problem to fix. See `astar_kernel._hpop`.
# The lesson generalizes: pack what is probed at RANDOM, not everything with several parallel arrays.

# free-interval pool record columns, int32. The 4th column is padding, not waste: 12 B rows would
# straddle 128 B lines about 1 time in 11, 16 B rows never do (8 per line, exactly).
P_LO, P_HI, P_NXT = 0, 1, 2


def aligned_2d(rows: int, cols: int, dtype=np.int64, align: int = CACHE_LINE) -> np.ndarray:
    """A ``(rows, cols)`` C-contiguous array whose base address is a multiple of ``align``.

    Over-allocates a byte buffer and slices to the first aligned offset; the returned view keeps the
    oversized buffer alive, so no copy and no lifetime hazard."""
    itemsize = np.dtype(dtype).itemsize
    nbytes = rows * cols * itemsize
    raw = np.empty(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes].view(dtype).reshape(rows, cols)
