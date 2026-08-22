"""Turn another planner's schedule into a column-generation warm start.

Colgen's own opening move is one geodesic per flight in ``seed_ladder_steps + 1``
departure-shifted copies, and the incumbent it rounds out of that is weak: measured on
``density_faa_wing_zipline`` x1500, ten iterations of `round_heuristic` against a growing
pool never once beat a schedule A* produced in 64 seconds.  Since every run at that scale
ends on a truncated MILP, the incumbent is also the floor the answer falls back to.

This module converts an accepted schedule into master columns that are **mutually
row-feasible**, which is what `solve` needs before it will take them as an incumbent
rather than as pool contents.  Two things have to happen for that:

**Translation.**  A*'s corridor centreline is continuous; a column is a lattice path plus a
departure step.  `intent_to_column` rasterises the centreline onto the axial lattice and
snaps the departure onto ``dt``.  Routes that cannot be expressed are rejected with a
reason rather than approximated -- notably a mid-flight AIR HOLD, which a column has no
way to represent at all, and a route that bows outside the flight graph's O-D ellipse.

**Repair.**  Snapping to ``dt`` is what makes translation insufficient on its own: two
flights the ledger cleared at 98.6 s and 101.4 s both round to step 25 and collide on a
cap-1 row that never existed in the original schedule.  `build` walks flights in flight-id
order -- FCFS, the same priority A* itself used -- and holds a conflicting column later
until its claims fit, so an earlier flight keeps its slot and a later one yields.

What survives at x1500: 1,493 of 1,500 placed, 0 rows over capacity.  The 7 that do not
are the genuinely inexpressible ones, and `RestrictedMaster.complete_selection` re-picks
those around the pins rather than leaving them on columns chosen for a different schedule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen.network import column_claims
from freespace_sim.planner.colgen.translate import Column, column_to_intent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from freespace_sim.config import SimConfig


def intent_to_column(intent, graph, cfg: "SimConfig", model=None):
    """Rebuild the canonical column carrying an intent's route at its departure time.

    Returns ``(column, None)`` or ``(None, reason)``.  Rejection is deliberate: a route
    that does not land on the lattice is not approximated, because a column that claims
    cells the aircraft does not fly is worse than no column.

    ``model`` fills ``Column.delay_s``, which is **the master's objective coefficient**.
    Leaving it at 0.0 makes the LP take every seeded column for free, drives the master's
    cost to 0 and closes the gap at iteration 1 -- measured, not hypothesised.  Pass the
    solve's cost model whenever the column is going anywhere near a warm start.
    """

    radius = hg.circumradius(cfg)
    cells: list[tuple[int, int]] = []
    for point, _t in intent.centerline:
        cell = hg.enu_to_axial(float(point[0]), float(point[1]), radius)
        if not cells or cells[-1] != cell:
            cells.append(cell)
    if len(cells) < 2:
        return None, "degenerate path"
    departure_step = graph.base_step + int(round(intent.ground_delay_s / cfg.dt_s))
    if not graph.base_step <= departure_step <= graph.latest_departure_step:
        return None, "departure outside graph window"

    def lane_idx(lanes, cell):
        for index, lane in enumerate(lanes):
            if lane.cell == cell:
                return index
        return None

    origin_idx = lane_idx(graph.origin_lanes, cells[0]) if graph.origin_terminal else None
    dest_idx = lane_idx(graph.dest_lanes, cells[-1]) if graph.dest_terminal else None
    if graph.origin_terminal is not None and origin_idx is None:
        return None, "origin cell is not a lane cell"
    if graph.dest_terminal is not None and dest_idx is None:
        return None, "dest cell is not a lane cell"
    if graph.origin_terminal is None and cells[0] != graph.origin_cell:
        return None, "origin cell mismatch"
    if graph.dest_terminal is None and cells[-1] != graph.dest_cell:
        return None, "dest cell mismatch"
    for before, after in zip(cells, cells[1:]):
        if hg.hex_distance(before, after) != 1:
            # An air hold shows up here: the centreline dwells in one cell and the next
            # sampled point is not a neighbour.  Nothing in the column model expresses it.
            return None, "path contains a non-neighbour hop"

    column = Column(
        flight_id=intent.request.flight_id,
        departure_step=departure_step,
        level=0,
        origin_lane_idx=origin_idx,
        dest_lane_idx=dest_idx,
        cell_path=tuple(cells),
        delay_s=0.0,
    )
    if model is None:
        return column, None
    # Costed from the column's OWN translated geometry, not from the source intent: the
    # rebuilt route is lattice-snapped, so its ground hold and en-route detour are its own
    # and generally differ from what the original planner flew.
    try:
        translated = column_to_intent(column, graph.request, cfg)
    except ValueError as exc:
        return None, f"translation rejected: {str(exc)[:50]}"
    return replace(
        column,
        delay_s=model.evaluate(
            ground_s=translated.ground_delay_s,
            air_detour_s=translated.air_detour_m / cfg.nominal_speed_mps,
        ),
    ), None


def _column_at(intent, graph, cfg: "SimConfig", model, delta: int):
    """The intent's column held ``delta`` steps later, claims computed, or a reason."""

    column, why = intent_to_column(intent, graph, cfg, model)
    if column is None:
        return None, why
    if delta:
        step = column.departure_step + delta
        if step > graph.latest_departure_step:
            return None, "shift leaves the departure window"
        column = replace(column, departure_step=step)
        # Recomputed, never carried: `delay_s` is the objective coefficient and a held
        # column is strictly more expensive than the one it was shifted from.
        translated = column_to_intent(column, graph.request, cfg)
        column = replace(
            column,
            delay_s=model.evaluate(
                ground_s=translated.ground_delay_s,
                air_detour_s=translated.air_detour_m / cfg.nominal_speed_mps,
            ),
        )
    try:
        return replace(column, claims=column_claims(column, graph, cfg)), None
    except ValueError as exc:
        return None, f"rejected: {str(exc)[:60]}"


def build(
    accepted: Mapping[int, Any],
    graphs: Mapping[int, Any],
    cfg: "SimConfig",
    model,
    row_index,
    *,
    max_shift: int = 8,
    ladder: int = 0,
) -> tuple[dict[int, list[Column]], Counter]:
    """Return ``(seed_columns, stats)`` -- a mutually row-feasible warm start.

    Flights are repaired in flight-id order, which is FCFS order, so the pass mirrors the
    priority the source planner itself used: an earlier flight keeps its slot and a later
    one holds.  A flight that cannot be placed within ``max_shift`` steps is dropped rather
    than forced; the master's `complete_selection` re-picks it around the survivors.

    ``ladder`` adds that many extra departure-shifted copies of each placed column to the
    pool.  They are deliberately NOT checked against the claim counter -- they are
    alternatives the LP may pick instead of the placed one, not members of the feasible
    set being constructed, so counting them would reserve capacity twice.
    """

    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if ladder < 0:
        raise ValueError("ladder must be non-negative")

    loads: Counter = Counter()
    seed_columns: dict[int, list[Column]] = {}
    stats: Counter = Counter()
    shifts: list[int] = []

    for flight_id in sorted(accepted):
        graph = graphs.get(flight_id)
        if graph is None:
            stats["no graph"] += 1
            continue
        placed = None
        for delta in range(max_shift + 1):
            column, why = _column_at(accepted[flight_id], graph, cfg, model, delta)
            if column is None:
                if delta == 0:
                    # Only the unshifted failure names a real inexpressibility; a later
                    # delta failing usually just means the window ran out.
                    stats[f"unconvertible: {why}"] += 1
                    break
                continue
            if any(loads[row] + 1 > row_index.cap(row) for row in column.claims):
                continue
            placed = (delta, column)
            break
        if placed is None:
            stats["dropped (no feasible shift)"] += 1
            continue
        delta, column = placed
        shifts.append(delta)
        stats["placed"] += 1
        if delta:
            stats["placed after a hold"] += 1
        for row in column.claims:
            loads[row] += 1
        columns = [column]
        for extra in range(1, ladder + 1):
            more, _why = _column_at(accepted[flight_id], graph, cfg, model, delta + extra)
            if more is not None:
                columns.append(more)
                stats["ladder columns"] += 1
        seed_columns[flight_id] = columns

    stats["total held steps"] = sum(shifts)
    stats["rows over cap"] = sum(1 for row, n in loads.items() if n > row_index.cap(row))
    stats["cost"] = sum(columns[0].delay_s for columns in seed_columns.values())
    return seed_columns, stats
