"""Multi-altitude (3D) njit SIPP kernel: isolated compile + per-path correctness.

These are *kernel-level* unit tests — they drive :func:`freespace_sim.planner.sipp_kernel._search`
directly on hand-built flat arrays, no planner host. That serves two purposes the host-level
``test_sipp_compiled`` equivalence tests can't:

  1. **Compile gate.** ``@njit`` compiles lazily on first *call*, not on import; Python-parsing the
     file checks nothing numba cares about. The first test here forces type inference + lowering of
     the whole 3D kernel — catching type-unification / unsupported-op / empty-array-shape bugs.
  2. **Path isolation.** Each case is a minimal, hand-computable scenario that exercises exactly one
     code path (a rung climb, a rung descent, an interval-fragmentation hover, the ground-delay fan),
     so a failure localizes to that path instead of somewhere in a congested Dallas replay.

The origin/goal are always placed *interior* to the box: a reroute neighbour outside the box makes
the kernel return ``FB_OOB`` immediately (the host's ``margin`` guarantees this in production).

NOT covered here (needs the reference oracle on congested input — see ``test_sipp_compiled``): exact
equivalence with the pure-Python reference, and the dominance staircase under genuine label competition.
"""
import math

import numpy as np
import pytest

pytest.importorskip("numba")  # kernel import does ``from numba import njit`` — skip the module if absent

from freespace_sim.planner.sipp_kernel import FB_OOB, NO_PATH, OK, _search  # noqa: E402

SQRT3 = math.sqrt(3.0)

# a generous box with the origin well inside it, so no reroute strays out (would trip FB_OOB)
QMIN, RMIN, QSPAN, RSPAN = -3, -3, 10, 10


def hex_center(q, r, R):
    return R * SQRT3 * (q + r / 2.0), R * 1.5 * r


def qr_of(q, r):
    """Level-less (iq*rspan+ir) index the kernel completes with a flight level."""
    return (q - QMIN) * RSPAN + (r - RMIN)


def _block(iv_lo, iv_hi, iv_nxt, state, c, s):
    """Split cell ``c``'s free interval containing step ``s`` — faithful copy of
    ``CompiledOccupancy._block`` so the fragmentation matches what the host pool would produce."""
    slot = c
    while slot != -1:
        a, b = int(iv_lo[slot]), int(iv_hi[slot])
        if a <= s <= b:
            if s + 1 <= b:
                if a <= s - 1:
                    iv_hi[slot] = s - 1
                    ns = state["n"]; state["n"] += 1
                    iv_lo[ns] = s + 1; iv_hi[ns] = b; iv_nxt[ns] = int(iv_nxt[slot])
                    iv_nxt[slot] = ns
                else:
                    iv_lo[slot] = s + 1
            elif a <= s - 1:
                iv_hi[slot] = s - 1
            else:
                iv_lo[slot] = s + 1
            return
        slot = int(iv_nxt[slot])


def run(*, nlevels, base=0, max_step=20,
        lane_qr, lane_lat, n_to, to_ok, c_gd,
        takeoff_steps, takeoff_cost, rung_steps, rung_cost,
        goal_cells, lf_lo, lf_hi,
        c_hold, c_lat, pitch, dt, gx, gy, R, h_off,
        blocked=(), walled=(), max_lab=5000, max_heap=5000, maxpath=200):
    """Build the kernel's flat-array inputs for one scenario and return (m, g, n_exp, status, path)."""
    NC = QSPAN * RSPAN * nlevels
    cap = NC + 64
    gen = 1

    # global pool: every cell fully free [base, max_step] as its single pre-seeded slot == cell id
    iv_lo = np.zeros(cap, np.int64)
    iv_hi = np.full(cap, max_step, np.int64)
    iv_nxt = np.full(cap, -1, np.int64)
    iv_lo[:NC] = base
    for c in walled:                                   # permanently empty (lo>hi) ⇒ no free interval,
        iv_lo[c] = 1; iv_hi[c] = 0; iv_nxt[c] = -1     # matches CompiledOccupancy.register_static_terminal
    state = {"n": NC}                                  # next free split slot (mirrors nslots)
    for c, s in blocked:
        _block(iv_lo, iv_hi, iv_nxt, state, c, s)

    # no overlay: ov_gen != gen everywhere ⇒ all cells read the global pool
    ov_lo = np.zeros(1, np.int64)
    ov_hi = np.zeros(1, np.int64)
    ov_nxt = np.full(1, -1, np.int64)
    ov_head = np.full(NC, -1, np.int64)
    ov_gen = np.zeros(NC, np.int64)

    lane_qr = np.asarray(lane_qr, np.int64)
    lane_lat = np.asarray(lane_lat, np.float64)
    n_lanes = lane_qr.size
    to_ok = np.asarray(to_ok, np.int64)
    takeoff_steps = np.asarray(takeoff_steps, np.int64)
    takeoff_cost = np.asarray(takeoff_cost, np.float64)
    rung_steps = np.asarray(rung_steps, np.int64)
    rung_cost = np.asarray(rung_cost, np.float64)

    goal_gen = np.zeros(NC, np.int64)
    for gc in goal_cells:
        goal_gen[gc] = gen
    lf_lo = np.asarray(lf_lo, np.int64)
    lf_hi = np.asarray(lf_hi, np.int64)
    lf_n = lf_lo.size

    front_head = np.full(cap, -1, np.int64)
    front_tail = np.full(cap, -1, np.int64)
    front_gen = np.zeros(cap, np.int64)

    lab_cell = np.zeros(max_lab, np.int64)
    lab_slot = np.zeros(max_lab, np.int64)
    lab_arr = np.zeros(max_lab, np.int64)
    lab_g = np.zeros(max_lab, np.float64)
    lab_par = np.full(max_lab, -1, np.int64)
    lab_next = np.full(max_lab, -1, np.int64)
    lab_prev = np.full(max_lab, -1, np.int64)
    lab_dead = np.full(max_lab, -1, np.int64)          # -1 != gen ⇒ fresh labels alive

    heap_f = np.zeros(max_heap, np.float64)
    heap_c = np.zeros(max_heap, np.int64)
    heap_n = np.zeros(max_heap, np.int64)

    out_q = np.zeros(maxpath, np.int64)
    out_r = np.zeros(maxpath, np.int64)
    out_s = np.zeros(maxpath, np.int64)
    out_L = np.zeros(maxpath, np.int64)

    m, g, n_exp, status = _search(
        iv_lo, iv_hi, iv_nxt,
        ov_lo, ov_hi, ov_nxt, ov_head, ov_gen, cap,
        QMIN, RMIN, RSPAN, QSPAN, base, max_step, nlevels,
        lane_qr, lane_lat, n_lanes, to_ok, n_to, c_gd,
        takeoff_steps, takeoff_cost, rung_steps, rung_cost,
        goal_gen, lf_lo, lf_hi, lf_n,
        c_hold, c_lat, pitch, dt, gx, gy, R, h_off,
        gen, front_head, front_tail, front_gen,
        lab_cell, lab_slot, lab_arr, lab_g, lab_par, lab_next, lab_prev, lab_dead, max_lab,
        heap_f, heap_c, heap_n, max_heap,
        out_q, out_r, out_s, out_L,
    )
    path = [(int(out_q[i]), int(out_r[i]), int(out_L[i]), int(out_s[i])) for i in range(m)][::-1] if m > 0 else []
    return m, g, n_exp, status, path


def test_single_level_straight_shot():
    """Compile gate + takeoff → 6-neighbour reroute → goal accept → reconstruction, empty airspace."""
    R = 100.0
    gx, gy = hex_center(3, 0, R)
    m, g, _n, status, path = run(
        nlevels=1,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=1, to_ok=[1], c_gd=1.0,
        takeoff_steps=[0], takeoff_cost=[0.0], rung_steps=[0], rung_cost=[0.0],
        goal_cells=[qr_of(3, 0)], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
    )
    assert status == OK
    assert path == [(0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 2), (3, 0, 0, 3)]
    assert g == pytest.approx(3 * (R * SQRT3))         # three lateral hops, no hover


def test_vertical_rung_climb():
    """NEW 3D path: takeoff only at L0, goal at L2 (same q,r) ⇒ forced climb L0→L1→L2 via rungs."""
    R = 100.0
    gx, gy = hex_center(0, 0, R)                        # goal at origin column ⇒ heuristic dist 0
    m, g, _n, status, path = run(
        nlevels=3,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=1, to_ok=[1, 0, 0], c_gd=1.0,
        takeoff_steps=[0, 1, 2], takeoff_cost=[0.0, 10.0, 20.0],
        rung_steps=[1, 1], rung_cost=[5.0, 5.0],
        goal_cells=[qr_of(0, 0) * 3 + 2], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
    )
    assert status == OK
    assert path == [(0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 2, 2)]
    assert g == pytest.approx(10.0)                     # two rungs × 5


def test_vertical_rung_descend():
    """NEW 3D path (the ``dL==0`` branch): takeoff only at L2, goal at L0 ⇒ forced descent L2→L1→L0."""
    R = 100.0
    gx, gy = hex_center(0, 0, R)
    m, g, _n, status, path = run(
        nlevels=3,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=1, to_ok=[0, 0, 1], c_gd=1.0,
        takeoff_steps=[0, 1, 2], takeoff_cost=[0.0, 10.0, 20.0],
        rung_steps=[1, 1], rung_cost=[5.0, 5.0],
        goal_cells=[qr_of(0, 0) * 3 + 0], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
    )
    assert status == OK
    assert path == [(0, 0, 2, 2), (0, 0, 1, 3), (0, 0, 0, 4)]   # takeoff to L2 lands at step 2
    assert g == pytest.approx(20.0 + 5.0 + 5.0)         # takeoff-to-L2 + two descent rungs


def test_interval_fragmentation_hover():
    """SIPP core: block (1,0) at steps 1-2 ⇒ the cell splits into [0,0]+[3,20]; the flight must
    air-hover at the origin until (1,0) frees at step 3, paying ``c_hold`` per held step."""
    R = 100.0
    gx, gy = hex_center(2, 0, R)
    c10 = qr_of(1, 0)
    m, g, _n, status, path = run(
        nlevels=1,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=1, to_ok=[1], c_gd=1.0,
        takeoff_steps=[0], takeoff_cost=[0.0], rung_steps=[0], rung_cost=[0.0],
        goal_cells=[qr_of(2, 0)], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
        blocked=[(c10, 1), (c10, 2)],
    )
    assert status == OK
    assert path == [(0, 0, 0, 0), (1, 0, 0, 3), (2, 0, 0, 4)]
    assert g == pytest.approx(2 * (3.0 * 1.0) + 2 * (R * SQRT3))   # 2-step hover + 2 lateral hops


def test_ground_delay_fan():
    """Folded takeoff enumeration: block the origin at step 0 with ``to_ok`` admitting steps 0 and 1.
    Step-0 takeoff finds no free interval → skipped; step-1 succeeds paying the cheaper ``c_gd``
    ground-delay (never air-hover) — the reason ground delay is enumerated into the start labels."""
    R = 100.0
    gx, gy = hex_center(2, 0, R)
    c00 = qr_of(0, 0)
    m, g, _n, status, path = run(
        nlevels=1,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=2, to_ok=[1, 1], c_gd=1.0,
        takeoff_steps=[0], takeoff_cost=[0.0], rung_steps=[0], rung_cost=[0.0],
        goal_cells=[qr_of(2, 0)], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
        blocked=[(c00, 0)],
    )
    assert status == OK
    assert path == [(0, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 3)]
    assert g == pytest.approx(1 * 1.0 * 1.0 + 2 * (R * SQRT3))     # 1-step ground delay + 2 lateral hops


def test_no_path_when_walled_in():
    """Heap genuinely exhausts ⇒ NO_PATH (not a crash, not a bogus route). Wall all 6 neighbours of
    the origin so the single takeoff label has no successor and the search drains without ever
    reaching the box edge (which would instead — correctly — trip FB_OOB → host fallback)."""
    R = 100.0
    gx, gy = hex_center(3, 0, R)
    neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    m, _g, _n, status, _path = run(
        nlevels=1,
        lane_qr=[qr_of(0, 0)], lane_lat=[0.0], n_to=1, to_ok=[1], c_gd=1.0,
        takeoff_steps=[0], takeoff_cost=[0.0], rung_steps=[0], rung_cost=[0.0],
        goal_cells=[qr_of(3, 0)], lf_lo=[0], lf_hi=[20],
        c_hold=3.0, c_lat=1.0, pitch=R * SQRT3, dt=1.0, gx=gx, gy=gy, R=R, h_off=0.0,
        walled=[qr_of(q, r) for q, r in neighbours],
    )
    assert status == NO_PATH
    assert m == -1
