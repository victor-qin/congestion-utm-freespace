"""Compiled (numba) space-time A* kernel for the multi-altitude hex planner (issue #8, Track B).

This is the hot path of :class:`~freespace_sim.planner.astar.AStarPlanner` — the ``while pq`` search loop
+ ``_edges`` + ``is_blocked`` (~95% of a dense plan) — lifted into a ``@njit`` function over flat arrays.
The pure-Python ``AStarPlanner`` stays the reference oracle (and the fallback); this kernel must reproduce
its exact optimal weighted cost AND its exact expansion order (so ``last_expansions`` matches and the
returned path is byte-identical among equal-cost optima). Everything geometry/terminal/commit stays in the
Python host.

**State** ``(q, r, L, step)`` — a hex cell at flight level ``L`` and time ``step`` — plus the ground state
(packed with ``L=-1``). There is no dense id for this 4-D space, so ``g``/``closed``/``came`` live in an
**open-addressing hash** in flat arrays, version-stamped by ``gen`` (bump per plan → O(1) reset). The
priority queue is a hand-rolled binary min-heap keyed ``(f, insertion_counter)`` — byte-identical to the
host's ``heapq`` ``(priority, next(counter))`` tie-break, which is what makes the expansion order match.

**Occupancy** is :class:`CompiledHexOccupancy`'s two interval pools (corridor + column, per-``(q,r,L)``
free intervals) plus this flight's cheap per-cell own-column **mark** (``ov_own_gen``); ``_blocked`` folds
them into ``is_blocked`` (foreign column → wall; own column → transparent unless a corridor covers it).
Out-of-box neighbours / hash-full / heap-full return a distinct ``FB_*`` code so the host warns precisely
and falls back to the pure-Python reference.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_SQRT3 = 1.7320508075688772
_MAGIC = np.uint64(0x9E3779B97F4A7C15)          # Fibonacci hashing multiplier

# status codes
OK = 0
NO_PATH_EMPTY = 1          # queue emptied, no goal → host maps BUDGET_EXCEEDED
NO_PATH_TRUNC = 2          # hit max_expansions → host maps SEARCH_EXHAUSTED
FB_OOB = 3                 # reroute left the occupancy box  → host reference
FB_HASH = 4                # g/closed/came hash saturated    → host reference
FB_HEAP = 5                # priority queue overflowed        → host reference
FB_MASK = 6                # search reached a ground/goal step beyond the bounded to_ok/land_ok masks →
#                            host widens the masks and re-runs (NOT a reference fallback; stays exact)


@njit(cache=True, nogil=True)
def _slot0(key, log2cap):
    h = np.uint64(key) * _MAGIC
    return np.int64(h >> np.uint64(64 - log2cap))       # high log2cap bits → well-mixed slot


@njit(cache=True, nogil=True)
def _hpush(heap_f, heap_c, heap_n, size, f, c, node):
    heap_f[size] = f; heap_c[size] = c; heap_n[size] = node
    i = size
    while i > 0:
        par = (i - 1) // 2
        if heap_f[i] < heap_f[par] or (heap_f[i] == heap_f[par] and heap_c[i] < heap_c[par]):
            tf = heap_f[i]; heap_f[i] = heap_f[par]; heap_f[par] = tf
            tc = heap_c[i]; heap_c[i] = heap_c[par]; heap_c[par] = tc
            tn = heap_n[i]; heap_n[i] = heap_n[par]; heap_n[par] = tn
            i = par
        else:
            break
    return size + 1


@njit(cache=True, nogil=True)
def _hpop(heap_f, heap_c, heap_n, size):
    node = heap_n[0]
    size -= 1
    heap_f[0] = heap_f[size]; heap_c[0] = heap_c[size]; heap_n[0] = heap_n[size]
    i = 0
    while True:
        lft = 2 * i + 1; rgt = 2 * i + 2; sm = i
        if lft < size and (heap_f[lft] < heap_f[sm] or (heap_f[lft] == heap_f[sm] and heap_c[lft] < heap_c[sm])):
            sm = lft
        if rgt < size and (heap_f[rgt] < heap_f[sm] or (heap_f[rgt] == heap_f[sm] and heap_c[rgt] < heap_c[sm])):
            sm = rgt
        if sm == i:
            break
        tf = heap_f[i]; heap_f[i] = heap_f[sm]; heap_f[sm] = tf
        tc = heap_c[i]; heap_c[i] = heap_c[sm]; heap_c[sm] = tc
        tn = heap_n[i]; heap_n[i] = heap_n[sm]; heap_n[sm] = tn
        i = sm
    return node, size


@njit(cache=True, nogil=True)
def _probe(g_key, g_gen, gen, key, cap, log2cap):
    """Linear-probe the open-addressing table for ``key``; return the slot holding it OR the first empty
    (this-generation) slot; -1 if the table is full (probe budget = cap)."""
    i = _slot0(key, log2cap)
    mask = cap - 1
    for _ in range(cap):
        if g_gen[i] != gen:
            return i                                   # empty (stale) slot → key not present
        if g_key[i] == key:
            return i                                   # found
        i = (i + 1) & mask
    return -1


@njit(cache=True, nogil=True)
def _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
           heap_f, heap_c, heap_n, size, max_heap, nkey, ng, f, ctr, st_key):
    """Relax edge into ``nkey`` at cost ``ng``, priority ``f``. Mirrors astar.py:317-322:
    push iff ``ng < g.get(nkey, inf)``; on relax, update g/came but PRESERVE the closed bit (the
    reference never reopens — with a consistent heuristic a closed node is never relaxed anyway).
    Returns ``(size, ctr, rc)`` with rc: 1 pushed, 0 no-op, -1 hash-full, -2 heap-full."""
    nslot = _probe(g_key, g_gen, gen, nkey, hash_cap, log2cap)
    if nslot < 0:
        return size, ctr, -1
    if g_gen[nslot] != gen:                             # new node this generation
        g_key[nslot] = nkey; g_gen[nslot] = gen; g_val[nslot] = ng
        g_came[nslot] = st_key; g_flag[nslot] = 0
    elif ng < g_val[nslot]:                            # relax existing (preserve g_flag/closed)
        g_val[nslot] = ng; g_came[nslot] = st_key
    else:
        return size, ctr, 0
    if size >= max_heap:
        return size, ctr, -2
    size = _hpush(heap_f, heap_c, heap_n, size, f, ctr, nkey)
    return size, ctr + 1, 1


@njit(cache=True, nogil=True)
def _blocked(q, r, L, s, qmin, rmin, qspan, rspan, n_levels,
             iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen):
    """0 = free, 1 = blocked, -1 = out-of-box. Reproduces ``occupancy.is_blocked`` via the corridor pool
    (``iv_*``) + column pool (``cv_*``) + this flight's own-column mark (``ov_own_gen[cell] == gen``):
    a FOREIGN column is a wall; an OWN column is transparent unless a corridor (fixed-lane sibling) also
    covers it; a plain cell is the corridor pool."""
    iq = q - qmin; ir = r - rmin
    if iq < 0 or iq >= qspan or ir < 0 or ir >= rspan:
        return -1
    cell = (iq * rspan + ir) * n_levels + L
    # column pool: is `cell` column-blocked at s? (blocked iff s is in no free interval)
    colb = 1
    slot = cell
    while slot != -1:
        if cv_lo[slot] <= s <= cv_hi[slot]:
            colb = 0
            break
        slot = cv_nxt[slot]
    if colb == 1 and ov_own_gen[cell] != gen:
        return 1                                       # foreign column → wall
    # corridor pool
    slot = cell
    while slot != -1:
        if iv_lo[slot] <= s <= iv_hi[slot]:
            return 0
        slot = iv_nxt[slot]
    return 1


@njit(cache=True, nogil=True)
def _h_air(q, r, L, gx, gy, R, h_off, c_lat, takeoff_cost):
    dx = R * _SQRT3 * (q + r / 2.0) - gx
    dy = R * 1.5 * r - gy
    d = np.sqrt(dx * dx + dy * dy)
    m = d - h_off
    if m < 0.0:
        m = 0.0
    return c_lat * m + takeoff_cost[L]


@njit(cache=True, nogil=True)
def _search(
    # ---- occupancy pool + per-flight overlay (CompiledHexOccupancy) ----
    iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen,
    qmin, rmin, qspan, rspan, n_levels, base, max_step,
    # ---- ground / takeoff-fan (host masks) ----
    oq, orr, lane_q, lane_r, lane_lat, n_lanes, takeoff_steps, takeoff_cost, to_ok, n_gsteps, c_gd_dt,
    # ---- air edges ----
    rung_steps, rung_cost, c_lat_pitch, c_hold_dt, vertical_edges,
    # ---- goal (host masks) ----
    goal_q, goal_r, n_goal, land_ok,
    # ---- heuristic ----
    gx, gy, R, h_off, c_lat, h_ground,
    # ---- g/closed/came open-addressing hash (version-stamped) ----
    gen, g_key, g_gen, g_val, g_came, g_flag, hash_cap, log2cap,
    # ---- heap ----
    heap_f, heap_c, heap_n, max_heap,
    # ---- output ----
    out_q, out_r, out_L, out_s, max_expansions,
):
    step_span = max_step - base + 1
    nlp1 = n_levels + 1
    iq0 = oq - qmin; ir0 = orr - rmin

    # ---- seed: start = ground ("g", oq, orr, base), g=0, f=h_ground ----
    start_key = ((iq0 * rspan + ir0) * nlp1 + 0) * step_span + 0
    slot = _probe(g_key, g_gen, gen, start_key, hash_cap, log2cap)
    if slot < 0:
        return 0, 0.0, 0, FB_HASH, -1
    g_key[slot] = start_key; g_gen[slot] = gen; g_val[slot] = 0.0; g_came[slot] = -1; g_flag[slot] = 0
    size = _hpush(heap_f, heap_c, heap_n, 0, h_ground, 0, start_key)
    ctr = 1
    n_exp = 0

    while size > 0:
        st_key, size = _hpop(heap_f, heap_c, heap_n, size)
        sslot = _probe(g_key, g_gen, gen, st_key, hash_cap, log2cap)
        if g_flag[sslot] & 1:                          # already closed → skip (lazy-deletion heap)
            continue
        g_flag[sslot] |= 1                             # close on first pop
        base_g = g_val[sslot]

        # unpack st_key → (q, r, L, step)
        sp = st_key % step_span
        rem = st_key // step_span
        Lp = rem % nlp1
        cell2 = rem // nlp1
        ir = cell2 % rspan
        iq = cell2 // rspan
        q = iq + qmin; r = ir + rmin; L = Lp - 1; step = sp + base

        if L >= 0:                                     # air state → goal test (astar.py:282-305)
            is_goal = False
            for gidx in range(n_goal):
                if q == goal_q[gidx] and r == goal_r[gidx]:
                    is_goal = True
                    break
            if is_goal and sp >= n_gsteps:             # goal reached beyond the bounded mask → widen+re-run
                return 0, 0.0, n_exp, FB_MASK, step
            if is_goal and land_ok[sp * n_levels + L]:
                m = 0                                  # reconstruct: walk came goal→start into out_*
                cur = st_key
                while cur != -1:
                    csp = cur % step_span
                    crem = cur // step_span
                    cLp = crem % nlp1
                    ccell = crem // nlp1
                    cir = ccell % rspan
                    ciq = ccell // rspan
                    out_q[m] = ciq + qmin; out_r[m] = cir + rmin
                    out_L[m] = cLp - 1; out_s[m] = csp + base
                    cslot = _probe(g_key, g_gen, gen, cur, hash_cap, log2cap)
                    cur = g_came[cslot]
                    m += 1
                return m, base_g, n_exp, OK, -1

        n_exp += 1
        if n_exp > max_expansions:
            return 0, 0.0, n_exp, NO_PATH_TRUNC, -1

        # ============ expand (inlined _edges, exact successor order) ============
        if L < 0:                                      # ---- ground state (astar.py:376-409) ----
            gi = step - base
            if gi >= n_gsteps:                          # ground step beyond the bounded mask → widen+re-run
                return 0, 0.0, n_exp, FB_MASK, step
            if step + 1 <= max_step:                    # ground-wait g→g (emitted FIRST)
                nkey = ((iq0 * rspan + ir0) * nlp1 + 0) * step_span + (step + 1 - base)
                ng = base_g + c_gd_dt
                size, ctr, rc = _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
                                       heap_f, heap_c, heap_n, size, max_heap,
                                       nkey, ng, ng + h_ground, ctr, st_key)
                if rc == -1:
                    return 0, 0.0, n_exp, FB_HASH, -1
                if rc == -2:
                    return 0, 0.0, n_exp, FB_HEAP, -1
            if gi < n_gsteps:                           # takeoff fan: for lane: for level
                for li in range(n_lanes):
                    lq = lane_q[li]; lr = lane_r[li]
                    for Lv in range(n_levels):
                        ts = step + takeoff_steps[Lv]
                        if ts > max_step:
                            continue
                        if not to_ok[gi * n_levels + Lv]:
                            continue
                        if _blocked(lq, lr, Lv, ts, qmin, rmin, qspan, rspan, n_levels,
                                    iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen) != 0:
                            continue
                        liq = lq - qmin; lir = lr - rmin
                        nkey = ((liq * rspan + lir) * nlp1 + (Lv + 1)) * step_span + (ts - base)
                        ng = base_g + takeoff_cost[Lv] + lane_lat[li]
                        hh = _h_air(lq, lr, Lv, gx, gy, R, h_off, c_lat, takeoff_cost)
                        size, ctr, rc = _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
                                               heap_f, heap_c, heap_n, size, max_heap,
                                               nkey, ng, ng + hh, ctr, st_key)
                        if rc == -1:
                            return 0, 0.0, n_exp, FB_HASH, -1
                        if rc == -2:
                            return 0, 0.0, n_exp, FB_HEAP, -1
            continue

        # ---- air state (astar.py:410-433) ----
        ns = step + 1
        if ns > max_step:
            continue
        for d in range(6):                              # 6 reroutes (AXIAL_NEIGHBORS order)
            if d == 0:
                nq = q + 1; nr = r
            elif d == 1:
                nq = q + 1; nr = r - 1
            elif d == 2:
                nq = q; nr = r - 1
            elif d == 3:
                nq = q - 1; nr = r
            elif d == 4:
                nq = q - 1; nr = r + 1
            else:
                nq = q; nr = r + 1
            b = _blocked(nq, nr, L, ns, qmin, rmin, qspan, rspan, n_levels,
                         iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen)
            if b == -1:                                  # out-of-box stray → host reference
                return 0, 0.0, n_exp, FB_OOB, (nq + 32768) * 65536 + (nr + 32768)
            if b == 1:
                continue
            niq = nq - qmin; nir = nr - rmin
            nkey = ((niq * rspan + nir) * nlp1 + (L + 1)) * step_span + (ns - base)
            ng = base_g + c_lat_pitch
            hh = _h_air(nq, nr, L, gx, gy, R, h_off, c_lat, takeoff_cost)
            size, ctr, rc = _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
                                   heap_f, heap_c, heap_n, size, max_heap, nkey, ng, ng + hh, ctr, st_key)
            if rc == -1:
                return 0, 0.0, n_exp, FB_HASH, -1
            if rc == -2:
                return 0, 0.0, n_exp, FB_HEAP, -1
        # hover (same level)
        if _blocked(q, r, L, ns, qmin, rmin, qspan, rspan, n_levels,
                    iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen) == 0:
            nkey = ((iq * rspan + ir) * nlp1 + (L + 1)) * step_span + (ns - base)
            ng = base_g + c_hold_dt
            hh = _h_air(q, r, L, gx, gy, R, h_off, c_lat, takeoff_cost)
            size, ctr, rc = _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
                                   heap_f, heap_c, heap_n, size, max_heap, nkey, ng, ng + hh, ctr, st_key)
            if rc == -1:
                return 0, 0.0, n_exp, FB_HASH, -1
            if rc == -2:
                return 0, 0.0, n_exp, FB_HEAP, -1
        # vertical layer change ±1
        if vertical_edges:
            for dL in range(-1, 2, 2):                  # -1, +1
                L2 = L + dL
                if L2 < 0 or L2 >= n_levels:
                    continue
                rung = L if dL == 1 else L2
                ts = step + rung_steps[rung]
                if ts > max_step:
                    continue
                clear = True                            # both {L, L2} clear over (step, ts]
                sk = step + 1
                while sk <= ts:
                    if _blocked(q, r, L, sk, qmin, rmin, qspan, rspan, n_levels,
                                iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen) != 0 or \
                       _blocked(q, r, L2, sk, qmin, rmin, qspan, rspan, n_levels,
                                iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, ov_own_gen, gen) != 0:
                        clear = False
                        break
                    sk += 1
                if not clear:
                    continue
                nkey = ((iq * rspan + ir) * nlp1 + (L2 + 1)) * step_span + (ts - base)
                ng = base_g + rung_cost[rung]
                hh = _h_air(q, r, L2, gx, gy, R, h_off, c_lat, takeoff_cost)
                size, ctr, rc = _relax(g_key, g_gen, g_val, g_came, g_flag, gen, hash_cap, log2cap,
                                       heap_f, heap_c, heap_n, size, max_heap, nkey, ng, ng + hh, ctr, st_key)
                if rc == -1:
                    return 0, 0.0, n_exp, FB_HASH, -1
                if rc == -2:
                    return 0, 0.0, n_exp, FB_HEAP, -1

    return 0, 0.0, n_exp, NO_PATH_EMPTY, -1
