"""Per-terminal temporal pad-capacity authority — a shared vertiport's departure/arrival slot manager.

Pad capacity is a **temporal** resource: up to ``capacity`` same-hub dwells may overlap a vertiport's
shared hover column at once. This service records each committed terminal-column dwell as a time
interval per hub (fed by the ledger commit publish hook) and answers, at plan time, whether a flight's
takeoff/landing *edge* exists at a candidate time:

  * :meth:`admits` — **capacity** (step 2): fewer than ``capacity`` same-hub dwells overlap the window.
  * :meth:`column_clear` — **column activation / foreign-transit** (step 1): build the column cylinder
    and ask the ledger (the same FCL conflict check that gates commit; same-hub volumes are exempt, so
    it returns True iff a FOREIGN volume intrudes).

Unlike :class:`~freespace_sim.planner.occupancy.HexOccupancyService` this holds **no hex state** —
capacity is 1-D in time, not 3-D in cells — so it is the one authority every planner (A* today;
RRT*/MILP next) can consult at plan time. The column radius is a **per-hub constant** (asserted on
record).

There is deliberately **no "already-deployed → skip the ledger" shortcut**. It looks safe (a sibling's
column covering the window should have rejected any foreign intruder) but is **unsound**: a same-hub
flight's *own* near-hub cruise corridor (untagged) is not exempt from a *different* flight's column, yet
never conflicts with its own column — so the sibling whose column "covers" the window can be the one
whose corridor intrudes it. (Found in a Dallas replay; it converted a ground-delay into a denial.)
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from ..config import SimConfig
from ..geometry import CylinderSpec
from ..ledger import ReservationLedger
from ..types import as_terminal
from ..volumes import corridor_segment_volume, exit_radius, hover_reservation, terminal_radius


class TerminalCapacity:
    """Ledger-fed temporal capacity + column-activation authority for shared vertiport terminals.

    Push: subscribe :meth:`on_commit` to the ledger so committed dwells auto-record. Pull: holds a
    ledger reference for the lazy foreign-transit query in :meth:`column_clear`.
    """

    def __init__(self, cfg: SimConfig, ledger: ReservationLedger):
        self.cfg = cfg
        self.ledger = ledger
        self.dwells: dict[Hashable, list[tuple[float, float]]] = {}   # tid -> [(t_start, t_end), ...]
        self.radius: dict[Hashable, float] = {}                       # tid -> the hub's one column radius
        self.evicted_before: float | None = None

    # ----- maintenance (push, via ledger.subscribe) -----
    def on_commit(self, _flight_id, volumes) -> None:
        """Record every committed terminal-column cylinder as a per-hub dwell interval. A single flight
        may contribute two (origin + dest). The column radius is a hub constant — asserted here so the
        union-coverage skip in :meth:`column_clear` stays sound."""
        for v in volumes:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                r = self.radius.setdefault(v.terminal_id, v.shape.radius)
                if r != v.shape.radius:
                    raise ValueError(
                        f"terminal {v.terminal_id!r}: same-hub column radius must be constant "
                        f"({r} vs {v.shape.radius})"
                    )
                self.dwells.setdefault(v.terminal_id, []).append((v.t_start, v.t_end))

    def evict_before(self, t: float) -> None:
        """Drop dwells ending at or before ``t`` (monotonic). The caller passes the request clock: with
        ``t_departure >= t_request`` enforced and ``base = ceil(t_depart/dt)``, every future query window
        starts at ``base*dt >= t_request``, so a dwell with ``t_end <= t_request`` can never overlap one."""
        if self.evicted_before is not None and t <= self.evicted_before:
            return
        for tid in list(self.dwells):
            kept = [iv for iv in self.dwells[tid] if iv[1] > t]
            if kept:
                self.dwells[tid] = kept
            else:
                del self.dwells[tid]
        self.evicted_before = t

    def reset(self) -> None:
        self.dwells.clear()
        self.radius.clear()
        self.evicted_before = None

    # ----- queries (plan time) -----
    def admits(self, terminal_id: Hashable, t0: float, t1: float, capacity: int) -> bool:
        """Step 2 — capacity: fewer than ``capacity`` OTHER same-hub dwells overlap ``[t0, t1)``. The
        planning flight has not committed, so it is not yet in ``dwells``; ``< capacity`` means "room
        for me" (capacity 1 ⟺ the old exclusive pad)."""
        n = sum(1 for (a, b) in self.dwells.get(terminal_id, ()) if a < t1 and t0 < b)
        return n < capacity

    def column_clear(self, term, center, t0: float) -> bool:
        """Step 1 — column activation: can the column deploy at ``t0`` with no FOREIGN transit? Build the
        column cylinder (lifetime ``[t0, t0 + hover + climb)``) and ask the ledger — same-hub volumes are
        exempt (``conflict.volumes_conflict``), so a conflict means a foreign volume crosses the hub.
        Always queries; see the class docstring for why the "already-deployed" shortcut is unsound."""
        col = hover_reservation(center, t0, self.cfg, terminal_id=term.id,
                                radius=terminal_radius(term, self.cfg))
        return not self.ledger.any_conflict([col])

    def exit_clear(self, term, center, toward, t0: float) -> bool:
        """Step 1b — exit/approach lane: the corridor lane the flight flies from the column EDGE toward
        ``toward`` (origin→dest on takeoff; dest←origin on landing) is free of committed conflict over
        the dwell window ``[t0, t0 + hover + climb)``.

        This is the PRECISE (FCL) check the hex grid is too coarse to make at the shared edge. Same-hub
        SIBLING exit lanes are box↔box — NOT column-exempt (``conflict.volumes_conflict`` needs a
        cylinder) — so two flights launching the SAME direction at once collide, while DIVERGENT lanes
        (spatially disjoint) do not. A*'s ``is_blocked`` cannot draw that line: at the edge the
        corridor's hex inflation (~corridor_half + R ≈ 129 m at R=69) exceeds the spacing between
        divergent lanes (~127 m for 90°-apart launches off a 90 m column), so a grid check would
        serialize concurrent launches too. Gating the real box geometry here keeps both behaviors.

        The lane box is built EXACTLY as ``astar._build`` builds the exit lane — rooted flush at the
        column edge (:func:`volumes.exit_radius`, the one fold radius the commit also uses), one segment
        long toward ``toward``, over the column's lifetime — so this gate and the commit-time
        ``any_conflict`` agree."""
        cx, cy = float(center[0]), float(center[1])
        dx, dy = float(toward[0]) - cx, float(toward[1]) - cy
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return True                                   # degenerate (origin == dest): nothing to fly
        ux, uy = dx / n, dy / n
        exit_r = exit_radius(term, self.cfg)
        z = self.cfg.cruise_level_m
        seg = self.cfg.corridor_segment_len_m
        edge = [cx + exit_r * ux, cy + exit_r * uy, z]
        far = [cx + (exit_r + seg) * ux, cy + (exit_r + seg) * uy, z]
        t1 = t0 + self.cfg.hover_time_s + self.cfg.climb_time_s
        lane = corridor_segment_volume(edge, t0, far, t1, self.cfg, terminal_id=term.id)
        return not self.ledger.any_conflict([lane])

    def dwell_ok(self, term, center, t0: float, capacity: int, toward=None) -> bool:
        """The takeoff/landing edge exists at ``t0`` iff capacity admits AND the column is deployable
        over the dwell window ``[t0, t0 + hover + climb)`` AND — when ``toward`` (the other endpoint) is
        given — the exit/approach lane toward it is clear of committed sibling lanes (:meth:`exit_clear`).
        ``toward=None`` skips the lane check (capacity/column only)."""
        term = as_terminal(term)
        t1 = t0 + self.cfg.hover_time_s + self.cfg.climb_time_s
        return (self.admits(term.id, t0, t1, capacity)
                and self.column_clear(term, center, t0)
                and (toward is None or self.exit_clear(term, center, toward, t0)))
