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

import bisect
import math
from collections.abc import Hashable

from ..config import SimConfig
from ..conflict import volumes_conflict
from ..geometry import CylinderSpec
from ..ledger import ReservationLedger
from ..types import as_terminal
from ..volumes import corridor_segment_volume, exit_radius, hover_reservation, terminal_radius


def _merge_intervals(ivs):
    """Sort ``[(lo, hi), ...]`` and coalesce overlapping/adjacent runs into a sorted, DISJOINT list."""
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [ivs[0]]
    for lo, hi in ivs[1:]:
        plo, phi = out[-1]
        if lo <= phi:                              # overlaps or touches the previous run → extend it
            out[-1] = (plo, hi if hi > phi else phi)
        else:
            out.append((lo, hi))
    return out


def _overlaps(ivs, lo, hi) -> bool:
    """True iff ``[lo, hi)`` intersects any interval in the sorted, DISJOINT list ``ivs`` (O(log n)).
    STRICT (half-open) to match ``conflict.volumes_conflict``'s ``a.t_start < b.t_end and b.t_start <
    a.t_end`` — a volume ending exactly at ``lo`` (or starting exactly at ``hi``) does NOT overlap."""
    if not ivs:
        return False
    i = bisect.bisect_left(ivs, (hi, float("-inf")))   # first interval starting at/after hi
    return i > 0 and ivs[i - 1][1] > lo                # last one starting < hi ends strictly after lo?


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
        # foreign-transit index: per hub, the merged time intervals a FOREIGN volume sits over its column.
        # Replaces column_clear's per-step ledger scan with an O(log) lookup — built lazily/incrementally
        # from the ledger (``_ftn[tid]`` = # ledger volumes already indexed for that hub). Same answer as
        # ``not ledger.any_conflict([column])`` because the build reuses ``volumes_conflict`` verbatim.
        self._ft: dict[Hashable, list[tuple[float, float]]] = {}
        self._ftn: dict[Hashable, int] = {}
        self._ft_seen = 0                                             # ledger length at last query (shrink tripwire)

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
        for tid in list(self._ft):                      # foreign-transit index: same monotonic drop
            kept = [iv for iv in self._ft[tid] if iv[1] > t]
            if kept:
                self._ft[tid] = kept
            else:
                del self._ft[tid]
        self.evicted_before = t

    def reset(self) -> None:
        self.dwells.clear()
        self.radius.clear()
        self.evicted_before = None
        self._ft.clear()
        self._ftn.clear()
        self._ft_seen = 0

    # ----- queries (plan time) -----
    def admits(self, terminal_id: Hashable, t0: float, t1: float, capacity: int) -> bool:
        """Step 2 — capacity: fewer than ``capacity`` OTHER same-hub dwells overlap ``[t0, t1)``. The
        planning flight has not committed, so it is not yet in ``dwells``; ``< capacity`` means "room
        for me" (capacity 1 ⟺ the old exclusive pad)."""
        n = sum(1 for (a, b) in self.dwells.get(terminal_id, ()) if a < t1 and t0 < b)
        return n < capacity

    def _dwell_climb_s(self, z: float | None) -> float:
        """Climb time that sets a dwell/column window length. ``z`` is the flight's climb-to level (its
        cruise level); the committed hover column lasts ``hover + climb_time_to(z)`` (multi-altitude), so
        the gate window MUST use the same per-level climb — not the fixed preferred-plane ``climb_time_s``
        — or a level whose climb exceeds the preferred one (the top level) is under-checked and the pad
        silently over-subscribes. ``z=None`` (single-plane planners / capacity-only checks) keeps the
        preferred-plane climb, which is exactly their column length."""
        return self.cfg.climb_time_s if z is None else self.cfg.climb_time_to(z)

    def column_clear(self, term, center, t0: float, z: float | None = None) -> bool:
        """Step 1 — column activation: is the hub's column free of FOREIGN transit over the dwell window
        ``[t0, t0 + hover + climb_time_to(z))``? Exactly ``not ledger.any_conflict([column at t0])``
        (same-hub volumes exempt), served from the per-hub foreign-transit index: first bring the index
        current with any newly-committed volumes — each spatially AABB-pruned, then confirmed by the SAME
        ``volumes_conflict`` the ledger uses — then answer with an O(log) overlap query. Order-independent,
        so (unlike the rejected 'already-deployed' shortcut, see class docstring) it never misses a
        late-committed intruder or a same-hub cruise corridor. ``z`` (cruise level) sets only the QUERY
        window length via :meth:`_dwell_climb_s`; the indexed foreign intervals are level-independent (a
        foreign volume crosses the hub's spatial footprint over its own lifetime, or it does not)."""
        term = as_terminal(term)
        tid = term.id
        vols = self.ledger._vols                              # committed volumes, in commit order (same pkg)
        n = len(vols)
        if n < self._ft_seen:                                 # ledger shrank (release) → cached index is stale
            self._ft.clear(); self._ftn.clear()
        self._ft_seen = n
        start = self._ftn.get(tid, 0)
        if start < n:                                         # index the newly-committed tail for this hub
            r = terminal_radius(term, self.cfg)
            cx, cy = float(center[0]), float(center[1])
            zlo, zhi = self.cfg.ground_level_m, self.cfg.cruise_level_m
            new: list[tuple[float, float]] = []
            for idx in range(start, n):
                v = vols[idx]
                if v.terminal_id == tid:                      # same-hub + column ⇒ exempt
                    continue
                vlo, vhi = v.aabb()
                if (cx + r < vlo[0] or vhi[0] < cx - r or cy + r < vlo[1] or vhi[1] < cy - r
                        or zhi < vlo[2] or vhi[2] < zlo):      # spatial AABB miss ⇒ can't intrude the column
                    continue
                col = hover_reservation(center, v.t_start, self.cfg, terminal_id=tid, radius=r)
                if volumes_conflict(col, v):                   # exact predicate the ledger's any_conflict uses
                    new.append((v.t_start, v.t_end))
            if new:
                self._ft[tid] = _merge_intervals((self._ft.get(tid) or []) + new)
            self._ftn[tid] = n
        t1 = t0 + self.cfg.hover_time_s + self._dwell_climb_s(z)   # per-level query window (multi-altitude)
        return not _overlaps(self._ft.get(tid), t0, t1)

    def exit_clear(self, term, center, toward, t0: float, z: float | None = None) -> bool:
        """Step 1b — exit/approach lane, LEGACY path only (``fixed_exit_lanes=False``): the corridor the
        flight flies from the column EDGE toward ``toward`` (origin→dest on takeoff; dest←origin on
        landing) is free of committed conflict over the dwell window ``[t0, t0 + hover + climb)``. Only
        reached via ``dwell_ok(..., toward=...)``; the default fixed-lane path does the same job with
        exact cell occupancy in :meth:`planner.occupancy.HexOccupancyService.is_blocked` (issue #18) and
        never calls this.

        It is the PRECISE (FCL) check: same-hub SIBLING exit lanes are box↔box — NOT column-exempt
        (``conflict.volumes_conflict`` needs a cylinder) — so two flights launching the SAME direction at
        once collide, while DIVERGENT lanes (spatially disjoint) do not. The *legacy* ``is_blocked``
        could not draw that line (its ~corridor_half + R ≈ 129 m inflation exceeded the ~127 m spacing
        between 90°-apart lanes off a 90 m column, so a grid check serialized concurrent launches too);
        issue #18 fixed that for the default path by recording the sibling corridor as exact cell
        occupancy, so a fixed-lane launch sees it without this box check.

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
        # the lane sits at the flight's CHOSEN cruise level (multi-altitude); same-hub siblings at a
        # different level are vertically disjoint, so the lane check must use that level's z.
        z = self.cfg.cruise_level_m if z is None else float(z)
        seg = self.cfg.corridor_segment_len_m
        edge = [cx + exit_r * ux, cy + exit_r * uy, z]
        far = [cx + (exit_r + seg) * ux, cy + (exit_r + seg) * uy, z]
        t1 = t0 + self.cfg.hover_time_s + self._dwell_climb_s(z)
        lane = corridor_segment_volume(edge, t0, far, t1, self.cfg, terminal_id=term.id)
        return not self.ledger.any_conflict([lane])

    def dwell_ok(self, term, center, t0: float, capacity: int, toward=None, z: float | None = None) -> bool:
        """The takeoff/landing edge exists at ``t0`` iff capacity admits AND the column is deployable
        over the dwell window ``[t0, t0 + hover + climb)`` AND — when ``toward`` (the other endpoint) is
        given — the exit/approach lane toward it (at cruise level ``z``) is clear of committed sibling
        lanes (:meth:`exit_clear`). ``toward=None`` skips the lane check (capacity/column only)."""
        term = as_terminal(term)
        t1 = t0 + self.cfg.hover_time_s + self._dwell_climb_s(z)
        return (self.admits(term.id, t0, t1, capacity)
                and self.column_clear(term, center, t0, z)
                and (toward is None or self.exit_clear(term, center, toward, t0, z)))

    def dwell_ok_levels(self, term, center, t0: float, capacity: int, zs, toward=None) -> list[bool]:
        """Per-level takeoff/landing feasibility — a bool per cruise level ``z`` in ``zs`` — for BOTH
        takeoff paths (fixed-lane passes ``toward=None`` and deconflicts siblings by cell occupancy;
        the legacy path passes ``toward`` for the per-level exit-lane check).

        The dwell window is per-level (the committed column lasts ``hover + climb_time_to(z)``), so:
          * ``admits`` (capacity) is genuinely per-level — a higher level's column dwells longer;
          * ``column_clear`` (foreign transit) is level-MONOTONE — the cylinder's radius and [ground,
            ceiling] extent are level-independent, so a larger ``z`` only lengthens the time window ⇒ a
            strict superset. Probe the ledger ONCE at the longest window; if it clears, every shorter
            level clears too, and only the cheap ``admits`` varies. Re-probe per level only in the rare
            case the top window has a foreign transit (a shorter window may still clear).

        Net: 1 ledger query + N cheap ``admits`` in the common case, instead of one FCL query per level
        (the A* ground-state hot path)."""
        term = as_terminal(term)
        hover = self.cfg.hover_time_s
        col_top_ok = self.column_clear(term, center, t0, max(zs))
        return [self.admits(term.id, t0, t0 + hover + self._dwell_climb_s(z), capacity)
                and (col_top_ok or self.column_clear(term, center, t0, z))
                and (toward is None or self.exit_clear(term, center, toward, t0, z))
                for z in zs]
