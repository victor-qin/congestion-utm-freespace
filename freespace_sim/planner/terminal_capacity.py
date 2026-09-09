"""Per-terminal temporal pad-capacity authority — a shared vertiport's departure/arrival slot manager.

Pad capacity is a **temporal** resource: up to ``capacity`` same-hub dwells may overlap a vertiport's
shared hover column at once. This service records each committed terminal-column dwell as a time
interval per hub (fed by the ledger commit publish hook) and answers, at plan time, whether a flight's
takeoff/landing *edge* exists at a candidate time:

  * :meth:`admits` — **capacity** (step 2): fewer than ``capacity`` same-hub dwells overlap the window.
  * :meth:`reservation_admitted` — apply that same capacity test to the exact tagged dwell
    cylinders in a rebuilt candidate reservation.
  * :meth:`column_clear` — **column activation / foreign-transit** (step 1): build the column cylinder
    and ask the ledger (the same FCL conflict check that gates commit; same-hub volumes are exempt, so
    it returns True iff a FOREIGN volume intrudes).

Unlike :class:`~freespace_sim.planner.astar.occupancy.HexOccupancyService` this holds **no hex state** —
capacity is 1-D in time, not 3-D in cells — so it is the one authority every planner (A* and the
MILP family) can consult at plan time. The column radius is a **per-hub constant** (asserted on
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

from ..conflict import volumes_conflict
from ..config import SimConfig
from ..geometry import CylinderSpec
from ..ledger import ReservationLedger
from ..types import as_terminal
from ..volumes import corridor_segment_volume, exit_radius, hover_reservation, terminal_radius

# When True (default), :meth:`TerminalCapacity.column_clear` may short-circuit for a hub whose permanent
# terminal wall is actually registered on the ledger. That wall is the same-radius column cylinder as a
# transient dwell and spans the full [ground, ceiling] tube; its derived hex projection is wider still.
# Consequently no foreign committed volume can transit the column at any flight level, and the dynamic
# foreign-transit scan is vacuous. Requiring the concrete per-terminal registration (in addition to the
# config flag) preserves lower-level planner/ledger callers that have not installed their intended walls.
# Set False only to force the legacy scan in ``analysis/ab_column_clear.py``.
SKIP_FOREIGN_WHEN_WALLED = True


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
    ledger reference for the lazy foreign-transit query in :meth:`column_clear`. Rebuilders pass
    their exact tagged output to :meth:`reservation_admitted` before accepting a retimed candidate.
    """

    def __init__(self, cfg: SimConfig, ledger: ReservationLedger, track_removal: bool = False):
        """Bind the authority to one ledger and initialize its dwell/foreign-transit indices.

        Parameters
        ------------
        - cfg (SimConfig): supplies column radius, hover/climb timing, and the always-active flag.
        - ledger (ReservationLedger): the ledger this authority observes (push) and queries (pull).
        - track_removal (bool): when True, keep per-flight dwell rows so :meth:`on_release` can
          subtract them exactly (a dwell is a counting structure, removed by value; LNS destroy);
          off ⇒ zero bookkeeping, byte-identical behavior.

        Return
        --------
        - output (None): builds the instance; subscribe :meth:`on_commit` separately.
        """
        self.cfg = cfg
        self.ledger = ledger
        self.track_removal = track_removal
        self._rows: dict[int, list] = {}                              # fid -> [count, (tid, t0, t1), ...]
        self.dwells: dict[Hashable, list[tuple[float, float]]] = {}   # tid -> [(t_start, t_end), ...]
        self.radius: dict[Hashable, float] = {}                       # tid -> the hub's one column radius
        # Every dynamic ledger volume observed through ``on_commit``. Unlike the dwell count, this
        # includes untagged corridor boxes, so comparing it with ``ledger.n_volumes`` detects a
        # release even when commits occurred between two planner calls.
        self._n_observed_volumes = 0
        self.evicted_before: float | None = None
        # per-hub FOREIGN-TRANSIT index (lazy, incremental): tid -> merged time intervals during which
        # SOME foreign volume spatially transits the hub column ⇒ column_clear is an O(log) bisect instead
        # of a per-call ledger.any_conflict. The column footprint is level-independent (radius + [ground,
        # ceiling]), so the index is z-independent; only the query WINDOW length depends on the level.
        self._ft: dict[Hashable, list[tuple[float, float]]] = {}
        self._ftn: dict[Hashable, int] = {}                           # tid -> #ledger volumes already indexed
        self._ft_seen = 0                                             # ledger length at last query (shrink tripwire)

    # ----- maintenance (push, via ledger.subscribe) -----
    def on_commit(self, flight_id, volumes) -> None:
        """Record every committed terminal-column cylinder as a per-hub dwell interval.

        A single flight may contribute two (origin + dest). The column radius is a hub constant —
        asserted here so the union-coverage skip in :meth:`column_clear` stays sound.

        Parameters
        ------------
        - flight_id: owner id of the committed volumes; keys the removal rows when tracking.
        - volumes (Iterable[Volume4D]): committed volumes; tagged cylinders become dwells.

        Return
        --------
        - output (None): appends to ``dwells``/``_rows`` and bumps ``_n_observed_volumes``; raises
          ``ValueError`` if a hub's column radius is not constant.
        """
        self._n_observed_volumes += len(volumes)
        rows = None
        if self.track_removal:
            rows = self._rows.setdefault(flight_id, [])
            rows.append(len(volumes))
        floor = self.evicted_before
        for v in volumes:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                r = self.radius.setdefault(v.terminal_id, v.shape.radius)
                if r != v.shape.radius:
                    raise ValueError(
                        f"terminal {v.terminal_id!r}: same-hub column radius must be constant "
                        f"({r} vs {v.shape.radius})"
                    )
                if floor is not None and v.t_end <= floor:
                    continue        # already behind the eviction watermark: `evict_before` would have
                    #                 dropped it, and it can never overlap a future query window. Not
                    #                 recorded either, so `dwells` and `_rows` stay in exact
                    #                 correspondence — see on_release.
                self.dwells.setdefault(v.terminal_id, []).append((v.t_start, v.t_end))
                if rows is not None:
                    rows.append((v.terminal_id, v.t_start, v.t_end))

    def on_release(self, flight_id, volumes) -> None:
        """Reverse this flight's dwell rows and invalidate the foreign-transit cache.

        Rows absent at or before :attr:`evicted_before` were intentionally evicted; any other absence
        is drift. Never remove an arbitrary equal-valued dwell because it may belong to another
        committed flight. Keeps ``_n_observed_volumes`` aligned with the ledger.

        Parameters
        ------------
        - flight_id: owner whose recorded dwell rows are subtracted (no-op if untracked or unknown).
        - volumes: accepted for the subscriber signature; unused (recorded rows drive removal).

        Return
        --------
        - output (None): mutates ``dwells``/``_rows`` and clears the foreign-transit index; raises
          ``KeyError``/``ValueError`` on a drifted (missing) dwell row.
        """
        rows = self._rows.pop(flight_id, ())
        floor = self.evicted_before
        for row in rows:
            if isinstance(row, int):
                self._n_observed_volumes -= row
                continue
            tid, t0, t1 = row
            if floor is not None and t1 <= floor:
                continue                               # eviction already dropped this dwell
            ivs = self.dwells[tid]
            ivs.remove((t0, t1))                       # KeyError/ValueError here IS the drift signal
            if not ivs:
                del self.dwells[tid]
        self._ft.clear()
        self._ftn.clear()
        self._ft_seen = 0

    def evict_before(self, t: float) -> None:
        """Drop dwells (and foreign-transit intervals) ending at or before ``t`` (monotonic).

        The caller passes the request clock. With ``t_departure >= t_request`` enforced and
        ``base = ceil(t_depart/dt)``, every future query window starts at ``base*dt >= t_request``,
        so a dwell with ``t_end <= t_request`` can never overlap one — safe to drop.

        Parameters
        ------------
        - t (float): eviction watermark (s); dwells and intervals with ``t_end <= t`` are dropped.

        Return
        --------
        - output (None): shrinks ``dwells``/``_ft`` and advances ``evicted_before``; no-op if ``t``
          does not advance the watermark.
        """
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
        """Clear all dwells, radii, rows, and indices — return to the just-constructed state."""
        self.dwells.clear()
        self.radius.clear()
        self._rows.clear()
        self._ft.clear()
        self._ftn.clear()
        self._ft_seen = 0
        self._n_observed_volumes = 0
        self.evicted_before = None

    # ----- queries (plan time) -----
    def admits(self, terminal_id: Hashable, t0: float, t1: float, capacity: int) -> bool:
        """Step 2 — capacity: fewer than ``capacity`` OTHER same-hub dwells overlap ``[t0, t1)``.

        The planning flight has not committed, so it is not yet in ``dwells``; ``< capacity`` means
        "room for me" (capacity 1 ⟺ the old exclusive pad).

        Parameters
        ------------
        - terminal_id (Hashable): the hub whose recorded dwells are counted.
        - t0 (float): candidate dwell window start (s).
        - t1 (float): candidate dwell window end (s); overlap is half-open (``a < t1 and t0 < b``).
        - capacity (int): max concurrent same-hub dwells allowed.

        Return
        --------
        - output (bool): True iff the overlapping-dwell count is below ``capacity``.
        """
        n = sum(1 for (a, b) in self.dwells.get(terminal_id, ()) if a < t1 and t0 < b)
        return n < capacity

    def reservation_admitted(self, volumes, origin_term=None, dest_term=None) -> bool:
        """True iff every request-terminal dwell in a rebuilt reservation has pad capacity.

        Rebuilders already produced the authoritative windows, so inspect their exact tagged
        :class:`~freespace_sim.geometry.CylinderSpec` volumes rather than re-deriving timing from
        corners or flight levels. Untagged volumes, corridor boxes, and cylinders for terminal IDs
        unrelated to this request are deliberately ignored. Interval overlap remains half-open via
        :meth:`admits`, matching the ledger's time predicate.

        This is separate from geometric conflict checking: same-terminal columns are intentionally
        conflict-exempt, so ``ledger.any_conflict`` cannot detect pad over-subscription after a
        rebuild shortens a path and moves its destination dwell earlier.

        Parameters
        ------------
        - volumes (Iterable[Volume4D]): a rebuilt reservation; only tagged cylinders are inspected.
        - origin_term (Terminal | tuple | None): origin terminal, normalized via ``as_terminal``.
        - dest_term (Terminal | tuple | None): destination terminal, normalized via ``as_terminal``.

        Return
        --------
        - output (bool): True iff every dwell for a REQUESTED terminal fits capacity (True when the
          request names no terminal); raises ``ValueError`` on inconsistent terminal capacity.
        """
        normalized = tuple(
            term for term in (as_terminal(origin_term), as_terminal(dest_term))
            if term is not None
        )
        if not normalized:
            return True
        terminals = {}
        for term in normalized:
            prior = terminals.get(term.id)
            if prior is not None and prior.capacity != term.capacity:
                raise ValueError(
                    f"terminal {term.id!r}: capacity must be constant "
                    f"({prior.capacity} vs {term.capacity})"
                )
            terminals[term.id] = term
        for v in volumes:
            if v.terminal_id is None or not isinstance(v.shape, CylinderSpec):
                continue
            term = terminals.get(v.terminal_id)
            if (term is not None
                    and not self.admits(v.terminal_id, v.t_start, v.t_end, term.capacity)):
                return False
        return True

    def _window_s(self, term, center, z: float | None) -> float:
        """The column window past the pad hover — :func:`volumes.column_dwell_s`, the single owner.

        Gate and commit MUST agree; asking one function is what makes that structural rather than a
        convention two call sites happen to follow. ``z=None`` (single-plane / capacity-only) keeps
        the preferred-plane climb, which is exactly those planners' column length."""
        from ..volumes import column_dwell_s

        if z is None:
            from .hexgrid import max_lane_traverse_s

            return self.cfg.climb_time_s + max_lane_traverse_s(center, term, self.cfg)
        return column_dwell_s(center, term, self.cfg, z)

    def _dwell_climb_s(self, z: float | None) -> float:
        """Climb time that sets a dwell/column window length. ``z`` is the flight's climb-to level (its
        cruise level); the committed hover column lasts ``hover + climb_time_to(z)`` (multi-altitude), so
        the gate window MUST use the same per-level climb — not the fixed preferred-plane ``climb_time_s``
        — or a level whose climb exceeds the preferred one (the top level) is under-checked and the pad
        silently over-subscribes. ``z=None`` (single-plane planners / capacity-only checks) keeps the
        preferred-plane climb, which is exactly their column length."""
        return self.cfg.climb_time_s if z is None else self.cfg.climb_time_to(z)

    def column_clear(self, term, center, t0: float, z: float | None = None) -> bool:
        """Step 1 — column activation: is the hub's column free of FOREIGN transit over the dwell
        window ``[t0, t0 + hover + climb_time_to(z))``?

        Reproduces ``not ledger.any_conflict([column at t0, z])`` over the COMMITTED (transient)
        volumes (same-hub volumes exempt), served from the per-hub foreign-transit index: bring
        the index current with any newly-committed volumes (each spatially AABB-pruned, then
        confirmed by the SAME ``volumes_conflict`` the ledger uses — recorded as its
        ``[t_start, t_end)`` transit interval), then answer with an O(log) overlap query. The
        column footprint is level-independent, so the index is z-independent; only the query
        window length uses ``z`` (per-level climb). Order-independent, so (unlike the rejected
        'already-deployed' shortcut, see class docstring) it never misses a late-committed
        intruder or a same-hub cruise corridor. NOTE: it scans ``ledger._vols`` only, NOT the
        always-active ``_static_vols`` walls — a foreign hub's permanent wall never overlaps this
        hub's own column under the demand's hub spacing, and the commit-time ``any_conflict`` is
        the authoritative backstop regardless, so at worst this diverges on the denial REASON,
        never admitting a real conflict.

        Parameters
        ------------
        - term (Terminal | tuple): the hub, normalized via ``as_terminal``; supplies id and radius.
        - center (Vec): the hub centre (column location).
        - t0 (float): dwell window start (s).
        - z (float | None): cruise level setting the window length; None ⇒ preferred-plane climb.

        Return
        --------
        - output (bool): True iff no foreign committed volume transits the column over the window.
        """
        term = as_terminal(term)
        tid = term.id
        if (SKIP_FOREIGN_WHEN_WALLED
                and self.cfg.terminal_airspace_always_active
                and self.ledger.has_static_terminal(tid)):
            # The registered permanent cylinder exactly covers this column from ground to ceiling, while
            # the derived routing wall over-covers it at every flight level. FCFS commit therefore cannot
            # admit a foreign transit here, making the dynamic committed-tail scan below vacuous.
            return True
        vols = self.ledger._vols                              # committed volumes, commit order (same pkg)
        n = len(vols)
        if n < self._ft_seen:                                 # ledger shrank (release) → cached index stale
            self._ft.clear()
            self._ftn.clear()
        self._ft_seen = n
        start = self._ftn.get(tid, 0)
        if start < n:                                         # index the newly-committed tail for this hub
            r = terminal_radius(term, self.cfg)
            col_ref = hover_reservation(center, 0.0, self.cfg, terminal_id=tid, radius=r,
                                        climb_time_s=self._dwell_climb_s(None))
            col_bb = col_ref.flat_aabb()                      # column footprint (level-independent), flat floats
            # Reuse the ledger's per-volume AABB (cached once at commit, index-aligned with
            # _vols) + its OWN scalar broadphase, instead of recomputing v.aabb() here. This loop
            # indexes each committed volume once per querying hub (O(vols × hubs)) and the miss
            # test discards most — so the old per-idx v.aabb() dominated aabb() allocations. Same
            # six comparisons in the same order ⇒ byte-identical result
            # (see tests/test_terminal_capacity.py + test_geometry.py).
            ledger_aabb = self.ledger._aabb
            aabb_miss = self.ledger._aabb_miss
            new: list[tuple[float, float]] = []
            for idx in range(start, n):
                v = vols[idx]
                if v.terminal_id == tid:                      # same-hub + column ⇒ exempt
                    continue
                if aabb_miss(col_bb, ledger_aabb[idx]):       # spatial AABB miss ⇒ can't intrude
                    continue
                col = hover_reservation(center, v.t_start, self.cfg, terminal_id=tid, radius=r,
                                        climb_time_s=self._dwell_climb_s(None))
                if volumes_conflict(col, v):                   # exact predicate the ledger's any_conflict uses
                    new.append((v.t_start, v.t_end))
            if new:
                self._ft[tid] = _merge_intervals((self._ft.get(tid) or []) + new)
            self._ftn[tid] = n
        t1 = t0 + self.cfg.hover_time_s + self._window_s(term, center, z)
        return not _overlaps(self._ft.get(tid), t0, t1)

    def exit_clear(self, term, center, toward, t0: float, z: float | None = None) -> bool:
        """Step 1b — exit/approach lane, LEGACY path only (``fixed_exit_lanes=False``): is the
        corridor the flight flies from the column EDGE toward ``toward`` free of committed conflict
        over the dwell window ``[t0, t0 + hover + climb)``? (origin→dest on takeoff; dest←origin
        on landing.)

        Only reached via ``dwell_ok(..., toward=...)``; the default fixed-lane path does the same
        job with exact cell occupancy in
        :meth:`planner.astar.occupancy.HexOccupancyService.is_blocked` and never calls this.

        It is the PRECISE (FCL) check: same-hub SIBLING exit lanes are box↔box — NOT column-exempt
        (``conflict.volumes_conflict`` needs a cylinder) — so two flights launching the SAME
        direction at once collide, while DIVERGENT lanes (spatially disjoint) do not. The legacy
        ``is_blocked`` grid inflation cannot draw that line (its keep-out radius exceeds the spacing
        between adjacent lanes off one column, serializing concurrent launches); the default path
        instead records the sibling corridor as exact cell occupancy, so a fixed-lane launch sees
        it without this box check.

        The lane box is rooted flush at the column edge :func:`volumes.exit_radius` (the one fold
        radius the commit also uses), one segment long toward ``toward``, over the column's
        lifetime — built exactly as ``astar._build`` builds the exit lane
        (see context/figures/exit_radius.png), so this gate and the commit-time ``any_conflict``
        agree.

        Parameters
        ------------
        - term (Terminal | tuple): the hub, normalized via ``as_terminal``.
        - center (Vec): the hub centre (pad location); the lane roots at its column edge.
        - toward (Vec): the other endpoint the lane points at (dest on takeoff, origin on landing).
        - t0 (float): dwell window start (s).
        - z (float | None): the lane's cruise level; None ⇒ ``cfg.cruise_level_m``.

        Return
        --------
        - output (bool): True iff the exit-lane box has no committed conflict (True when degenerate,
          origin == dest).
        """
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
        t1 = t0 + self.cfg.hover_time_s + self._window_s(term, center, z)
        lane = corridor_segment_volume(edge, t0, far, t1, self.cfg, terminal_id=term.id)
        return not self.ledger.any_conflict([lane])

    def dwell_ok(self, term, center, t0: float, capacity: int, toward=None, z: float | None = None) -> bool:
        """The takeoff/landing edge exists at ``t0`` iff capacity admits AND the column is deployable
        AND (when ``toward`` is given) the exit lane toward it is clear.

        Conjoins :meth:`admits` (capacity), :meth:`column_clear` (foreign transit) over the dwell
        window ``[t0, t0 + hover + climb)``, and — when ``toward`` (the other endpoint) is given —
        :meth:`exit_clear` (committed sibling lanes at cruise level ``z``). ``toward=None`` skips
        the lane check (capacity/column only).

        Parameters
        ------------
        - term (Terminal | tuple): the hub, normalized via ``as_terminal``.
        - center (Vec): the hub centre (pad location).
        - t0 (float): candidate takeoff/landing time (s).
        - capacity (int): max concurrent same-hub dwells (forwarded to :meth:`admits`).
        - toward (Vec | None): the other endpoint for the lane check, or None to skip it.
        - z (float | None): cruise level for the window length and lane check.

        Return
        --------
        - output (bool): True iff capacity, column, and (optional) lane checks all pass.
        """
        term = as_terminal(term)
        t1 = t0 + self.cfg.hover_time_s + self._window_s(term, center, z)
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
        (the A* ground-state hot path).

        Parameters
        ------------
        - term (Terminal | tuple): the hub, normalized via ``as_terminal``.
        - center (Vec): the hub centre (pad location).
        - t0 (float): candidate takeoff/landing time (s).
        - capacity (int): max concurrent same-hub dwells (forwarded to :meth:`admits`).
        - zs (Sequence[float]): cruise levels to test, in caller order.
        - toward (Vec | None): the other endpoint for the legacy per-level exit-lane check, or None.

        Return
        --------
        - output (list[bool]): one feasibility flag per level in ``zs``, in the same order.
        """
        term = as_terminal(term)
        hover = self.cfg.hover_time_s
        col_top_ok = self.column_clear(term, center, t0, max(zs))
        return [self.admits(term.id, t0, t0 + hover + self._window_s(term, center, z), capacity)
                and (col_top_ok or self.column_clear(term, center, t0, z))
                and (toward is None or self.exit_clear(term, center, toward, t0, z))
                for z in zs]
