"""Invariants of the kernel's packed record layouts (issue #8 memory plan).

The end-to-end guarantee — compiled plans stay byte-identical to the pure-Python reference — is
gated by ``test_astar_compiled.py``. These tests pin the *reasons* that guarantee holds, so a future
refactor that quietly breaks one fails here with a precise message instead of surfacing as a rare
plan divergence:

* records are cache-line aligned and correctly strided (the reason the layout is faster at all),
* the ``float64``/``int64`` aliasing addresses the columns it claims to,
* the generation stamp stays even and is re-stamped before it can wrap into ``ov_own_gen``'s int32.
"""
from __future__ import annotations

import numpy as np
import pytest

from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar._packed import (
    CACHE_LINE, G_CAME, G_GEN, G_KEY, G_VAL, GEN_MASK, GEN_STEP, GEN_WRAP,
    P_HI, P_LO, P_NXT, aligned_2d,
)


@pytest.mark.parametrize("cols,dtype,row_bytes", [(4, np.int64, 32), (4, np.int32, 16)])
def test_aligned_2d_alignment_and_stride(cols, dtype, row_bytes):
    a = aligned_2d(1000, cols, dtype)
    assert a.ctypes.data % CACHE_LINE == 0
    assert a.strides == (row_bytes, np.dtype(dtype).itemsize)
    assert a.flags.c_contiguous
    assert CACHE_LINE % row_bytes == 0                     # rows tile lines exactly, never straddle


def test_packed_hash_float_view_aliases_the_value_column_only():
    gp = aligned_2d(64, 4)
    gpf = gp.view(np.float64)
    gp[:] = 0
    gp[9, G_KEY] = 12345
    gp[9, G_GEN] = 8
    gpf[9, G_VAL] = -3.75
    gp[9, G_CAME] = -1
    assert (gp[9, G_KEY], gp[9, G_GEN], gp[9, G_CAME]) == (12345, 8, -1)
    assert gpf[9, G_VAL] == -3.75
    assert gp[9, G_VAL] != 0                               # same memory, reinterpreted
    assert (gp[8] == 0).all() and (gp[10] == 0).all()      # no spill into neighbouring records


def test_closed_bit_and_gen_mask_do_not_collide():
    """The closed flag lives in bit 0 of the generation stamp; generations are even so the two
    never alias. Setting/clearing either must leave the other intact."""
    for gen in (GEN_STEP, 2 ** 20, GEN_WRAP - GEN_STEP):
        assert gen % 2 == 0
        stamp = gen
        assert stamp & GEN_MASK == gen and not stamp & 1
        stamp |= 1                                         # close the node
        assert stamp & GEN_MASK == gen and stamp & 1


def test_pool_row_columns_are_distinct():
    iv = aligned_2d(32, 4, np.int32)
    iv[:] = 0
    iv[5, P_LO], iv[5, P_HI], iv[5, P_NXT] = 3, 17, 9
    assert (int(iv[5, P_LO]), int(iv[5, P_HI]), int(iv[5, P_NXT])) == (3, 17, 9)
    assert (iv[4] == 0).all() and (iv[6] == 0).all()


def test_bump_gen_steps_by_two_and_stays_even():
    p = AStarPlanner()
    p._gen = 0
    seen = [p._bump_gen() for _ in range(5)]
    assert seen == [GEN_STEP * k for k in range(1, 6)]
    assert all(g % 2 == 0 for g in seen)


def test_bump_gen_restamps_before_int32_wrap():
    """Never fires in practice (~5e8 plans), so it is forced here: a silent wrap would alias stale
    slots into the live generation and corrupt a plan."""
    p = AStarPlanner()
    gp = aligned_2d(16, 4)
    gp[:, G_GEN] = 999
    p._ks_caps[4] = {"g_pack": gp}
    p._ks = {"ov_own_gen": np.full(8, 999, np.int32)}
    p._gen = GEN_WRAP - GEN_STEP

    assert p._bump_gen() == GEN_STEP
    assert (gp[:, G_GEN] == 0).all()
    assert (p._ks["ov_own_gen"] == 0).all()
    assert p._bump_gen() == 2 * GEN_STEP                   # counting resumes normally
