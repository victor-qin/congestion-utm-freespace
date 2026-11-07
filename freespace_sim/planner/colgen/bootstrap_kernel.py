"""Bounded, resumable native counterpart of the goal bootstrap Python oracle.

Only queue expansion is compiled. Canonical sink certification remains on the host;
this heuristic never replaces either restricted refinement or the exact pricing DP.
Numba is optional: callers retain the Python oracle when this module cannot load.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from . import dp_prepare
from .dp_kernel import (
    SCORE_EPS,
    RECOMPUTE_EPS,
    _fill_recent,
    _paid_visit_correction,
    _role_allows,
    _sink_may_improve,
    _state_hash,
    _visit_cost,
    _visit_hits_forbidden,
)

# Unique serials settle every tie before the compact label and finished fields.
_QUEUE_ITEM = types.Tuple((types.float64, types.int64, types.int64, types.int64, types.boolean))
_DONE, _GOAL, _YIELD, _DECLINED = range(4)


@njit(cache=True, nogil=True)
def _priority(hops, distance, paid, delay, destination):
    """Preserve the reference's order of operations, including max's first argument."""
    total = max(1, hops + distance)
    if total >= len(delay):
        return math.inf
    union_paid = paid
    if destination[total] > paid:
        union_paid = destination[total]
    return delay[total] + union_paid


@njit(cache=True, nogil=True)
def _advance(frontier, states, control, labels, topology, duals, paid_rows,
             forbidden, limits, delay, destination, screen, partials, recent, other):
    """Expand until a sink, bounded host deadline checkpoint, or exhaustion.

    Parameters
    ------------
    - frontier, states: Persistent heap and collision-chain heads for this root.
    - control, labels: Caller-owned resume counters and parent-chain arrays.
    - topology, duals, paid_rows, forbidden: Read-only packed graph and prices.
    - limits, delay, destination: Root clock, search limits and completion estimates.
    - screen: Stage-local admissible sink-bound inputs; never the queue priority.
    - partials, recent, other: Reusable exact-sum and history scratch.

    Return
    --------
    - (status, label): Pause reason and candidate label, or -1 without a candidate.
    """
    cells, parent, hops_array, paid_array, first_array, links = labels
    arc_start, arc_target, roles, remaining, destinations = topology
    cell_series, series_first, series_start, series_prefix = duals
    paid_start, paid_cell, paid_step, paid_value = paid_rows
    bits, row_steps, row_step0 = forbidden
    start, max_step, max_hops, revisit_depth, history_depth, track_first, budget, lo, hi = limits
    pops = 0
    while len(frontier) and control[0] < budget:
        if pops >= 256:
            return _YIELD, -1
        pops += 1
        bound, distance, serial, label, finished = heapq.heappop(frontier)
        if finished:
            env_start, env_length, cutoff, incumbent_state, objective = screen
            # Zero is a conservative lower bound on POSITIVE paid cost. The goal's
            # sequential paid sum is not DP's score inversion, so do not assume their
            # floating-point error is identical. Destination and delay bounds suffice.
            verdict = _sink_may_improve(
                0, hops_array[label], 0.0, env_start, env_length, delay, destination,
                objective[0], objective[1], objective[2], cutoff, incumbent_state,
            )
            if verdict == 0:
                control[4] += 1
                continue
            control[3] += 1
            return _GOAL, label
        control[0] += 1
        hops = hops_array[label]
        step = start + hops
        if hops >= max_hops or step >= max_step:
            continue
        cell = cells[label]
        for arc in range(arc_start[cell], arc_start[cell + 1]):
            neighbor = arc_target[arc]
            previous = label
            revisited = False
            for _ in range(revisit_depth):
                if previous < 0:
                    break
                if cells[previous] == neighbor:
                    revisited = True
                    break
                previous = parent[previous]
            if revisited:
                continue
            distance = remaining[neighbor]
            next_step = step + 1
            if hops + 1 + distance > max_hops or next_step + distance > max_step:
                continue
            finish = destinations[neighbor] != 0 and _role_allows(roles[arc], hops == 0, True)
            onward = _role_allows(roles[arc], hops == 0, False)
            if not (finish or onward) or _visit_hits_forbidden(
                bits, row_steps, row_step0, neighbor, next_step, lo, hi
            ):
                continue
            visit = _visit_cost(cell_series, series_first, series_start, series_prefix,
                                neighbor, next_step, lo, hi)
            correction, ok = _paid_visit_correction(
                paid_start, paid_cell, paid_step, paid_value, 0, neighbor, next_step,
                lo, hi, partials,
            )
            if not ok:
                return _DECLINED, -1
            new_paid = paid_array[label] + (visit - correction)
            if not math.isfinite(new_paid):
                return _DECLINED, -1
            child = control[1]
            if child >= len(cells):
                return _DECLINED, -1
            cells[child], parent[child] = neighbor, label
            hops_array[child], paid_array[child] = hops + 1, new_paid
            first = neighbor if hops == 0 else first_array[label]
            first_array[child] = first
            new_bound = _priority(hops + 1, distance, new_paid, delay, destination)
            if finish:
                heapq.heappush(frontier, (new_bound, np.int64(0), control[2], child, True))
                control[2] += 1
            if onward:
                n_recent = _fill_recent(child, history_depth, parent, cells, recent)
                first_key = first if track_first else -1
                # A root owns its table, so departure and origin cell are constant.
                key = _state_hash(neighbor, recent, n_recent, 0, first_key, -1, next_step)
                head = states[key] if key in states else np.int64(-1)
                occupant = head
                match = -1
                while occupant >= 0:
                    if (hops_array[occupant] == hops + 1
                            and (not track_first or first_array[occupant] == first)):
                        n_other = _fill_recent(occupant, history_depth, parent, cells, other)
                        same = n_other == n_recent
                        for i in range(n_recent):
                            if recent[i] != other[i]:
                                same = False
                                break
                        if same:
                            match = occupant
                            break
                    occupant = links[occupant]
                old_paid = math.inf if match < 0 else paid_array[match]
                if new_paid < old_paid - SCORE_EPS:
                    # Keep prior queued labels alive: the Python oracle never skips stale pops.
                    links[child] = head
                    states[key] = child
                    heapq.heappush(frontier, (new_bound, np.int64(distance), control[2], child, False))
                    control[2] += 1
            control[1] += 1
    return _DONE, -1


def goal_directed_bootstrap(
    fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
    variants, order, topology, envelopes, *, incumbent, max_labels, deadline,
    prepared_rows=None, prepared_duals=None, prepared_forbidden=None, diagnostics=None,
):
    """Run exactly the bounded Python goal search, pausing for its sink certifier.

    Parameters
    ------------
    - fg, duals, pi_f, cfg, benefit, forbidden_rows, model: Pricing subproblem.
    - variants, order, topology, envelopes: Already ranked roots and shared preparation.
    - incumbent, max_labels, deadline: Initial cutoff, total expansion cap and deadline.
    - prepared_rows, prepared_duals, prepared_forbidden: Optional shared packed inputs.
    - diagnostics (dict | None): Native-use, expansion, sink ask/skip and decline evidence.

    Return
    --------
    - result (tuple | None): (incumbent, expansion count), or None to use the oracle.

    The sink screen reuses the DP's admissible delay and unavoidable positive endpoint
    costs, with the same negative-credit allowance and recomputation slack. It uses ZERO
    for positive paid cost: this is valid regardless of accumulation roundoff, unlike
    substituting the differently associated goal sum for DP's score inversion. Thus it
    may ask more often, but never uses the queue estimate as a pruning bound. The
    envelope's first omitted scalar delay is rechecked against the current cutoff;
    monotonicity then licenses out-of-prefix skips even if an envelope was rewound.
    Only finished heap entries are screened: expansion, onward labels, dominance
    and serial order are untouched.
    """
    from . import pricing

    def decline(reason):
        """Return an explicit unsupported verdict without hiding programming errors."""
        if diagnostics is not None:
            diagnostics.update(native_used=False, decline_reason=reason)
        return None

    def completed(value, expanded):
        """Keep diagnostic counts separate from the caller's global search record."""
        if diagnostics is not None:
            diagnostics.update(native_used=True, expanded=expanded, decline_reason=None,
                               sinks_asked=int(control[3]), sinks_skipped=int(control[4]))
        return value, expanded

    if not topology.ok:
        return decline(topology.unsupported_reason)
    rows = prepared_rows if prepared_rows is not None else dp_prepare.prepare_rows(fg, cfg, topology)
    if not rows.ok:
        return decline(rows.unsupported_reason)
    packed = (prepared_duals if prepared_duals is not None
              else dp_prepare.prepare_duals(duals, fg, topology, rows))
    forbidden = (prepared_forbidden if prepared_forbidden is not None
                 else dp_prepare.prepare_forbidden(forbidden_rows, fg, rows, topology))
    if forbidden.n_unmapped:
        return decline("unmapped forbidden rows")
    destinations = pricing._destination_options(fg)
    certify = pricing._sink_certifier(
        fg, duals, pi_f, cfg, benefit, forbidden_rows, model, deadline=deadline,
    )
    offsets = duals.offsets
    recent_depth = max(2, offsets[1] - offsets[0])
    max_degree = int(np.max(np.diff(topology.arc_start), initial=0))
    capacity = 1 + max_degree * max(0, int(max_labels))
    labels = tuple(np.empty(capacity, dtype=dtype) for dtype in
                   (np.int32, np.int64, np.int32, np.float64, np.int32, np.int64))
    partials = np.empty(64, np.float64)
    recent = np.empty(recent_depth, np.int32)
    other = np.empty(recent_depth, np.int32)
    control = np.array([0, 1, 0, 0, 0], np.int64)
    for root in order:
        pricing._check_deadline(deadline)
        if control[0] >= max_labels:
            break
        departure = int(variants.departure_step[root])
        lane_raw = int(variants.lane_idx[root])
        lane = None if lane_raw < 0 else lane_raw
        cell_id = int(variants.cell[root])
        start = int(variants.start_step[root])
        # Variants already priced these exact endpoint/visit unions. Their paid CSR
        # omits zero rows; fsum of the retained rows has the same exact value.
        paid_class = int(variants.paid_class[root])
        paid_arrays = (
            variants.paid_start[paid_class:paid_class + 2],
            variants.paid_cell, variants.paid_step, variants.paid_value,
        )
        delay_bounds, _ = envelopes._delay_envelope(departure, lane)
        delay = np.asarray(delay_bounds, dtype=np.float64)
        distance = int(topology.hex_remaining[cell_id])
        destination = np.zeros(len(delay), np.float64)
        # Triangle inequality gives hops + remaining >= the root's hex distance.
        # Filling smaller totals would build endpoint claims the oracle never asks for.
        for total in range(max(1, distance), len(delay)):
            pricing._check_deadline(deadline)
            destination[total] = envelopes._destination_cost(start + total, total)
        initial_paid = float(variants.start_dual_cost[root])
        if not math.isfinite(initial_paid) or np.isnan(delay).any() or np.isnan(destination).any():
            return decline("unsupported nonfinite priority arithmetic")
        total = max(1, distance)
        bound = (math.inf if total >= len(delay)
                 else float(delay[total]) + max(initial_paid, destination[total]))
        frontier = List.empty_list(_QUEUE_ITEM)
        frontier.append((bound, distance, int(control[2]), 0, False))
        control[2] += 1
        control[1] = 1
        labels[0][0], labels[1][0], labels[2][0] = cell_id, -1, 0
        labels[3][0], labels[4][0], labels[5][0] = initial_paid, -1, -1
        states = Dict.empty(types.uint64, types.int64)
        screen_enabled = incumbent is not None and model.air_weight >= 0.0 and all(
            math.isfinite(value) for value in
            (benefit, pi_f, duals.max_negative_credit, incumbent[0], model.air_weight)
        )
        max_root_hops = min(fg.max_step - start, fg.max_air_hops)
        if screen_enabled and len(delay) <= max_root_hops:
            # A mutable envelope may have frozen its prefix against a stronger cutoff
            # before being rewound. Revalidate the original delay-only stopping test.
            # Delay bounds are nondecreasing in total hops (constant when folding is
            # not exact), so this also covers every later omitted hop.
            omitted_delay = envelopes.delay_lower_bound(departure, lane, len(delay), 0)
            omitted_bound = benefit - pi_f - omitted_delay + duals.max_negative_credit
            screen_enabled = omitted_bound < incumbent[0] - RECOMPUTE_EPS
        screen = (
            np.zeros(1, np.int32), np.array([len(delay)], np.int32),
            np.array([0.0 if incumbent is None else incumbent[0]], np.float64),
            np.array([int(screen_enabled)], np.int32),
            np.array([benefit, pi_f, duals.max_negative_credit], np.float64),
        )
        limits = (start, fg.max_step, fg.max_air_hops, offsets[1] - offsets[0], recent_depth,
                  int(bool(fg.static_walls and fg.origin_terminal is not None)),
                  int(max_labels), offsets[0], offsets[1])
        while True:
            pricing._check_deadline(deadline)
            status, label = _advance(
                frontier, states, control, labels,
                (topology.arc_start, topology.arc_target, topology.arc_roles,
                 topology.hex_remaining, topology.dest_mask),
                (packed.cell_series, packed.series_first, packed.series_start, packed.series_prefix),
                paid_arrays, (forbidden.bits, rows.n_steps, rows.step0), limits,
                delay, destination, screen, partials, recent, other,
            )
            if status == _DECLINED:
                return decline("native arithmetic or arena limit")
            if status == _DONE:
                break
            if status == _YIELD:
                continue
            path = []
            cursor = label
            while cursor >= 0:
                index = labels[0][cursor]
                path.append((int(topology.cell_q[index]), int(topology.cell_r[index])))
                cursor = labels[1][cursor]
            path = tuple(reversed(path))
            for dest_lane in destinations[path[-1]]:
                candidate = certify(incumbent, departure, lane, dest_lane,
                                    start + int(labels[2][label]), path)
                if candidate is not None:
                    return completed(candidate, int(control[0]))
    return completed(incumbent, int(control[0]))


def warm_kernel() -> bool:
    """Compile the production array signature explicitly for measurement tools.

    Parameters
    ------------
    - None: Uses a one-cell, one-expansion synthetic search with production dtypes.

    Return
    --------
    - warmed (bool): True when the resumable expansion signature is resident.

    Production workers deliberately do not call this: their first-call cache loading
    belongs in solve time. A benchmark may call it before timing and record that cost.
    """
    def readonly(array):
        """Match shared topology/duals and forbidden-array Numba mutability types."""
        array.setflags(write=False)
        return array

    labels = tuple(np.zeros(2, dtype=dtype) for dtype in
                   (np.int32, np.int64, np.int32, np.float64, np.int32, np.int64))
    labels[1][0] = -1
    frontier = List.empty_list(_QUEUE_ITEM)
    frontier.append((0., 1, 0, 0, False))
    control = np.array([0, 1, 1, 0, 0], np.int64)
    states = Dict.empty(types.uint64, types.int64)
    _advance(
        frontier, states, control, labels,
        (readonly(np.zeros(2, np.int32)), readonly(np.zeros(0, np.int32)),
         readonly(np.zeros(0, np.uint8)), readonly(np.ones(1, np.int32)),
         readonly(np.zeros(1, np.uint8))),
        (readonly(np.full(1, -1, np.int32)), readonly(np.zeros(0, np.int32)),
         readonly(np.zeros(1, np.int64)), readonly(np.zeros(0, np.float64))),
        (np.zeros(2, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
         np.zeros(0, np.float64)),
        (readonly(np.zeros(1, np.uint64)), 1, 0),
        (0, 2, 2, 0, 2, 0, 1, 0, 0),
        np.zeros(3, np.float64), np.zeros(3, np.float64),
        (np.zeros(1, np.int32), np.array([3], np.int32), np.zeros(1, np.float64),
         np.zeros(1, np.int32), np.zeros(3, np.float64)),
        np.zeros(64, np.float64), np.zeros(2, np.int32), np.zeros(2, np.int32),
    )
    return bool(_advance.signatures)
