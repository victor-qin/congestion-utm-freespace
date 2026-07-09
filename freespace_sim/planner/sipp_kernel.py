"""Compiled (numba) air-cruise kernel for cost-aware SIPP (issue #8, Track B).

This is the hot path of :class:`~freespace_sim.planner.sipp.SIPPPlanner` — the safe-interval A* over the
air lattice — lifted into a ``@njit`` function over flat arrays. The pure-Python ``SIPPPlanner`` stays
the reference oracle (and the fallback); this kernel must reproduce its exact optimal weighted cost.
Everything terminal/geometry/commit stays in the Python host.

**Occupancy** is the linked-list interval pool of :class:`CompiledOccupancy`: a cell's free intervals
are slots walked from slot ``cell`` along ``iv_nxt``; each slot is a unique frontier node id.

**Multi-label, not single-best.** Because the objective is weighted cost (``c_hold != c_gd``), a
``(cell, interval)`` is reached at several non-dominating ``(arrival, cost)`` labels (e.g. the origin via
many cheap ground-delay amounts — none dominates, since reproducing a later arrival by air-hover costs
``c_hold > c_gd``). Each search node is a *label*; the Pareto frontier per slot is a **version-stamped
linked list** of labels (unbounded — a fixed cap would overflow at the origin / hot hubs), walked for
dominance on insert. No eviction (a since-dominated label only adds a cheap compare); no per-(cell,step)
dedup and no stale-skip (pure speedups — omitting them costs extra expansions, never optimality).

Dominance (matches ``sipp._nondominated``): stored ``(t2,g2)`` dominates new ``(t,g)`` iff
``t2 <= t and g2 + (t - t2)*c_hold <= g``. Goal cells are frontier-EXEMPT (their per-step landing gate
is not interval-captured). Ground delay (the cheap ``c_gd`` lever) is enumerated into the *start labels*
by the kernel's own takeoff loop (from a host-precomputed dwell_ok/pad_clear feasibility mask), so every
in-air wait here is air-hover at ``c_hold``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_SQRT3 = 1.7320508075688772

OK = 0
NO_PATH = 1
FB_OOB = 2            # fallback: a reroute strayed outside the kernel box (rare edge-skirting geometry)
FB_CAP = 3            # fallback: label/heap capacity overflow (search too big — a hard/near-infeasible flight)


@njit(cache=True, nogil=True)   # nogil: release the GIL so a batch of plans runs on real threads (#8 Track A)
def _search(
    iv_lo, iv_hi, iv_nxt,                                            # global interval pool (slot < cap)
    ov_lo, ov_hi, ov_nxt, ov_head, ov_gen, cap,                      # per-flight overlay (slot >= cap)
    qmin, rmin, rspan, qspan, base, max_step,                        # box + step window
    lane_cell, lane_lat, n_lanes, to_ok, n_to, c_gd, climb_steps,     # takeoff lanes + ground-delay enumeration
    goal_gen, lf_lo, lf_hi, lf_n,                                     # goal cells (version-stamped) + landing ivals
    c_hold, c_lat, pitch, dt, gx, gy, R, h_off, climb_cost,         # cost + heuristic params
    gen, front_head, front_tail, front_gen,                          # per-slot sorted-by-arr staircase
    lab_cell, lab_slot, lab_arr, lab_g, lab_par, lab_next, lab_prev, lab_dead, max_lab,  # labels
    heap_f, heap_c, heap_n, max_heap,                                # binary heap
    out_q, out_r, out_s,                                             # output path buffers
):
    nlab = 0
    size = 0
    ctr = 0
    n_exp = 0
    ch_dt = c_hold * dt                                 # per-step air-hover cost (staircase key slope)

    # ---- folded takeoff enumeration: the host's `for s: for lane:` loop, in njit. Ground delay is the
    # cheap c_gd lever, so a start label at ground-step s = base+si, lane li is (lane cell, arrival
    # ts=s+climb, g = si*c_gd*dt + climb_cost + lane_lat). Order (s-major, lane-minor, ts>max_step break,
    # to_ok gate) matches the old host enumeration EXACTLY, so labels + ctr + f are byte-identical. ----
    for si in range(n_to):
        ts = base + si + climb_steps
        if ts > max_step:
            break
        if not to_ok[si]:                               # dwell_ok (terminal) / pad_clear (non-terminal) gate
            continue
        g0 = si * c_gd * dt + climb_cost                # ground-delay cost + climb (lane lateral added below)
        for li in range(n_lanes):                       # exit lanes (terminal) or the origin cell (climb-in-place)
            cell = lane_cell[li]
            sj = ov_head[cell] if ov_gen[cell] == gen else cell   # own-lane overlay, else the global pool
            slot = -1
            while sj != -1:                             # the interval (slot) whose free run contains ts
                if sj >= cap:
                    jj = sj - cap; lo = ov_lo[jj]; hi = ov_hi[jj]; nxt = ov_nxt[jj]
                else:
                    lo = iv_lo[sj]; hi = iv_hi[sj]; nxt = iv_nxt[sj]
                if lo <= ts <= hi:
                    slot = sj; break
                sj = nxt
            if slot < 0:                                # lane cell blocked at ts (own-exempt view) → no takeoff
                continue
            if nlab >= max_lab or size >= max_heap:
                return -1, 0.0, n_exp, FB_CAP
            L = nlab; nlab += 1
            g = g0 + lane_lat[li]
            lab_cell[L] = cell; lab_slot[L] = slot; lab_arr[L] = ts
            lab_g[L] = g; lab_par[L] = -1; lab_next[L] = -1
            iq = cell // rspan
            q = iq + qmin; r = cell - iq * rspan + rmin
            dxx = R * _SQRT3 * (q + r / 2.0) - gx
            dyy = R * 1.5 * r - gy
            f = g + c_lat * max(0.0, np.sqrt(dxx * dxx + dyy * dyy) - h_off) + climb_cost
            heap_f[size] = f; heap_c[size] = ctr; heap_n[size] = L; ctr += 1
            ii = size; size += 1
            while ii > 0:
                par = (ii - 1) // 2
                if heap_f[ii] < heap_f[par] or (heap_f[ii] == heap_f[par] and heap_c[ii] < heap_c[par]):
                    tf = heap_f[ii]; heap_f[ii] = heap_f[par]; heap_f[par] = tf
                    tc = heap_c[ii]; heap_c[ii] = heap_c[par]; heap_c[par] = tc
                    tn = heap_n[ii]; heap_n[ii] = heap_n[par]; heap_n[par] = tn
                    ii = par
                else:
                    break

    while size > 0:
        L = heap_n[0]
        size -= 1                                       # pop min → sift down
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

        if lab_dead[L] == gen:                          # evicted since pushed (a dominator was inserted)
            continue
        cell = lab_cell[L]; slot = lab_slot[L]; arr = lab_arr[L]; g = lab_g[L]
        iq = cell // rspan
        q = iq + qmin; r = cell - iq * rspan + rmin
        is_goal = goal_gen[cell] == gen

        if is_goal:                                     # goal acceptance within a landing-feasible run
            feasible = False
            for k in range(lf_n):
                if lf_lo[k] <= arr <= lf_hi[k]:
                    feasible = True
                    break
            if feasible:
                m = 0                                   # reconstruct: walk parents into out_* (goal→start)
                cur = L
                while cur != -1:
                    cc = lab_cell[cur]; ci = cc // rspan
                    out_q[m] = ci + qmin
                    out_r[m] = cc - ci * rspan + rmin
                    out_s[m] = lab_arr[cur]
                    cur = lab_par[cur]
                    m += 1
                return m, g, n_exp, OK

        n_exp += 1
        hh = ov_hi[slot - cap] if slot >= cap else iv_hi[slot]
        hi_c = hh if hh < max_step else max_step        # current cell free-until (how long we may hover)

        for d in range(6):                              # reroute: one successor per neighbour interval
            if d == 0:
                nq = q + 1; nr = r
            elif d == 1:
                nq = q - 1; nr = r
            elif d == 2:
                nq = q; nr = r + 1
            elif d == 3:
                nq = q; nr = r - 1
            elif d == 4:
                nq = q + 1; nr = r - 1
            else:
                nq = q - 1; nr = r + 1
            niq = nq - qmin; nir = nr - rmin
            if niq < 0 or niq >= qspan or nir < 0 or nir >= rspan:
                return -1, 0.0, n_exp, FB_OOB           # out-of-box stray → host fallback
            ncell = niq * rspan + nir
            ngoal = goal_gen[ncell] == gen
            sj = ov_head[ncell] if ov_gen[ncell] == gen else ncell   # neighbour interval chain (overlay/pool)
            while sj != -1:
                if sj >= cap:
                    jj = sj - cap; lo = ov_lo[jj]; hi = ov_hi[jj]; nxts = ov_nxt[jj]
                else:
                    lo = iv_lo[sj]; hi = iv_hi[sj]; nxts = iv_nxt[sj]
                if lo < base:
                    lo = base
                if hi > max_step:
                    hi = max_step
                if lo <= hi:
                    a = arr + 1
                    if a < lo:
                        a = lo
                    if a > hi:
                        sj = nxts                   # next interval in THIS chain (overlay or pool)
                        continue
                    if a - 1 > hi_c:                     # cannot hover here long enough (chain ascends)
                        break
                    wait = a - (arr + 1)
                    ng = g + ch_dt * wait + c_lat * pitch
                    # --- dominance on the (ncell, sj) staircase: largest stored arr2 <= a (walk tail←) ---
                    make = True
                    m = -1
                    if not ngoal:
                        if front_gen[sj] != gen:
                            front_gen[sj] = gen; front_head[sj] = -1; front_tail[sj] = -1
                        m = front_tail[sj]
                        while m != -1 and lab_arr[m] > a:
                            m = lab_prev[m]
                        if m != -1 and lab_g[m] + (a - lab_arr[m]) * ch_dt <= ng + 1e-9:
                            make = False                     # dominated by the predecessor (min staircase v)
                    if make:
                        if nlab >= max_lab or size >= max_heap:
                            return -1, 0.0, n_exp, FB_CAP
                        L2 = nlab; nlab += 1
                        lab_cell[L2] = ncell; lab_slot[L2] = sj; lab_arr[L2] = a
                        lab_g[L2] = ng; lab_par[L2] = L
                        if ngoal:
                            lab_next[L2] = -1; lab_prev[L2] = -1
                        else:
                            if m != -1 and lab_arr[m] == a:      # same arr, new is cheaper → evict it
                                pm = lab_prev[m]; nm = lab_next[m]
                                lab_dead[m] = gen
                                if pm == -1:
                                    front_head[sj] = nm
                                else:
                                    lab_next[pm] = nm
                                if nm == -1:
                                    front_tail[sj] = pm
                                else:
                                    lab_prev[nm] = pm
                                m = pm
                            nx2 = front_head[sj] if m == -1 else lab_next[m]    # splice L2 in after m
                            lab_prev[L2] = m; lab_next[L2] = nx2
                            if m == -1:
                                front_head[sj] = L2
                            else:
                                lab_next[m] = L2
                            if nx2 == -1:
                                front_tail[sj] = L2
                            else:
                                lab_prev[nx2] = L2
                            e = nx2                              # forward-evict the contiguous dominated run
                            while e != -1 and ng + (lab_arr[e] - a) * ch_dt <= lab_g[e] + 1e-9:
                                ne = lab_next[e]
                                lab_dead[e] = gen
                                lab_next[L2] = ne
                                if ne == -1:
                                    front_tail[sj] = L2
                                else:
                                    lab_prev[ne] = L2
                                e = ne
                        dxx = R * _SQRT3 * (nq + nr / 2.0) - gx
                        dyy = R * 1.5 * nr - gy
                        f = ng + c_lat * max(0.0, np.sqrt(dxx * dxx + dyy * dyy) - h_off) + climb_cost
                        heap_f[size] = f; heap_c[size] = ctr; heap_n[size] = L2; ctr += 1
                        ii = size; size += 1
                        while ii > 0:
                            par = (ii - 1) // 2
                            if heap_f[ii] < heap_f[par] or (heap_f[ii] == heap_f[par] and heap_c[ii] < heap_c[par]):
                                tf = heap_f[ii]; heap_f[ii] = heap_f[par]; heap_f[par] = tf
                                tc = heap_c[ii]; heap_c[ii] = heap_c[par]; heap_c[par] = tc
                                tn = heap_n[ii]; heap_n[ii] = heap_n[par]; heap_n[par] = tn
                                ii = par
                            else:
                                break
                sj = nxts

        if is_goal and arr + 1 <= hi_c:                 # goal-cell hover: retry the per-step landing gate
            if nlab >= max_lab or size >= max_heap:
                return -1, 0.0, n_exp, FB_CAP
            L2 = nlab; nlab += 1
            lab_cell[L2] = cell; lab_slot[L2] = slot; lab_arr[L2] = arr + 1
            lab_g[L2] = g + ch_dt; lab_par[L2] = L; lab_next[L2] = -1; lab_prev[L2] = -1
            dxx = R * _SQRT3 * (q + r / 2.0) - gx
            dyy = R * 1.5 * r - gy
            f = (g + c_hold * dt) + c_lat * max(0.0, np.sqrt(dxx * dxx + dyy * dyy) - h_off) + climb_cost
            heap_f[size] = f; heap_c[size] = ctr; heap_n[size] = L2; ctr += 1
            ii = size; size += 1
            while ii > 0:
                par = (ii - 1) // 2
                if heap_f[ii] < heap_f[par] or (heap_f[ii] == heap_f[par] and heap_c[ii] < heap_c[par]):
                    tf = heap_f[ii]; heap_f[ii] = heap_f[par]; heap_f[par] = tf
                    tc = heap_c[ii]; heap_c[ii] = heap_c[par]; heap_c[par] = tc
                    tn = heap_n[ii]; heap_n[ii] = heap_n[par]; heap_n[par] = tn
                    ii = par
                else:
                    break

    return -1, 0.0, n_exp, NO_PATH
