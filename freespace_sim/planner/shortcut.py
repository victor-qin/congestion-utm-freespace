"""Deterministic shortcut post-passes — drop redundant knots without losing feasibility.

After any planner returns a corner polyline, repeatedly splice out an interior knot i (replace
``i-1 → i → i+1`` with ``i-1 → i+1``) and keep the removal iff the *rebuilt* reservation stays
conflict-free against the ledger and within the detour budget — the same build-then-check contract
every planner obeys. Triangle inequality guarantees a removal never lengthens the horizontal path,
so the result is always ≤ the input length and never invents a new conflict.

The legacy strategy is a deterministic single-knot fixpoint sweep, so a hex staircase collapses
(each removal re-enables the next) without depending on random shortcut draws. ``single_knot_heading``
keeps that exact order and result, but skips a candidate rebuild when a same-heading knot can be
proved to preserve every reservation sample bit-for-bit. The experimental ``batched_turns`` strategy
instead seeds at a real 3-D heading change, jumps directly to the ends of the adjacent straight runs,
and falls back through intermediate anchors only when a maximal jump fails. Every changed candidate
still respects the *real committed obstacles*, not A*'s conservative inflated raster. Wrap a planner
with :class:`ShortcutRefiner`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from itertools import chain
from typing import Literal

import numpy as np

from ..config import SimConfig
from ..cost import endpoint_altitude_change_m, trajectory_cost
from ..ledger import ReservationLedger
from ..types import FlightRequest, IntentStatus, OperationalIntent
from ..volumes import (build_reservation_from_corners, enroute_detour_m,
                       enroute_flown_m, enroute_reference_m)
from . import iter_planner_chain
from .terminal_capacity import TerminalCapacity

_EPS = 1e-9
_HEADING_EPS = 1e-9
_ShortcutStrategy = Literal["single_knot", "single_knot_heading", "batched_turns"]


@dataclass(frozen=True)
class _Knot:
    """A surviving logical corner with an identity stable across tuple splices."""

    id: int
    point: np.ndarray


@dataclass(frozen=True)
class _ShortcutContext:
    """Arguments shared by every full-route feasibility probe in one refinement."""

    origin: object
    dest: object
    t_depart: float
    g_delay: float
    cfg: SimConfig
    ledger: ReservationLedger
    straight_horiz: float
    origin_term: object = None
    dest_term: object = None
    corridor_t0: float | None = None
    tcap: TerminalCapacity | None = None

    def rebuild(self, knots):
        return _rebuild(
            [k.point for k in knots], self.origin, self.dest, self.t_depart, self.g_delay,
            self.cfg, self.ledger, self.straight_horiz, self.origin_term, self.dest_term,
            self.corridor_t0, self.tcap,
        )


@dataclass(frozen=True)
class _ShortcutState:
    """Current logical corners plus the reservation built for the last accepted splice."""

    knots: tuple[_Knot, ...]
    built: tuple | None = None

    def contains(self, knot_id: int) -> bool:
        return any(k.id == knot_id for k in self.knots)


def _rebuild(corners, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
             origin_term=None, dest_term=None, corridor_t0=None, tcap=None):
    """Resample corners, then apply detour, ledger, and terminal-capacity gates.

    Returns (volumes, centerline, cum_horiz, cum_dz) or None if it busts the detour budget or
    overlaps a committed reservation or over-subscribes a terminal. This is the feasibility oracle
    all shortcut strategies consult.
    ``origin_term``/``dest_term`` preserve the inner A*'s terminal tags through the rebuild;
    ``corridor_t0`` anchors the corridor at the inner planner's VERIFIED first-cruise stamp.
    """
    volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
        corners, origin, dest, t_depart, g_delay, cfg, origin_term=origin_term, dest_term=dest_term,
        corridor_t0=corridor_t0,
    )
    # Both sides span lane → lane via the same helper metrics uses (issue #50), so the gate enforces
    # exactly the ratio the caller will report as air_detour_m / stretch.
    flown = enroute_flown_m([p for p, _ in centerline], origin, dest, origin_term, dest_term, cfg)
    if straight_horiz > _EPS and flown / straight_horiz > cfg.max_detour_factor:
        return None
    if ledger.any_conflict(volumes):
        return None
    # Same-terminal dwell cylinders are deliberately exempt from geometric ledger conflicts, so the
    # interval-count authority must separately admit the exact rebuilt windows after every retiming.
    if tcap is not None and not tcap.reservation_admitted(volumes, origin_term, dest_term):
        return None
    return volumes, centerline, cum_horiz, cum_dz


def shortcut_corners(corners, origin, dest, t_depart, g_delay, cfg: SimConfig,
                     ledger: ReservationLedger, origin_term=None, dest_term=None, corridor_t0=None,
                     tcap: TerminalCapacity | None = None, skip_exact_heading: bool = False):
    """Greedily drop interior knots whose removal stays conflict-free; return simplified corners.

    Deterministic single-knot fixpoint: sweep interior knots front-to-back, remove any whose removal
    rebuilds conflict-free, and repeat full sweeps until one removes nothing. Endpoints (the climb-top
    and descent-top) are never dropped. If the input path is itself infeasible to rebuild, it is
    returned unchanged (the caller keeps the planner's verified original). With
    ``skip_exact_heading=True``, a same-heading knot is removed without a probe only when
    :func:`_merge_preserves_resampling` proves the before/after subsegments are byte-identical.
    """
    corners = [np.asarray(c, float) for c in corners]
    if len(corners) <= 2:
        return corners
    straight_horiz = enroute_reference_m(origin, dest, origin_term, dest_term, cfg)
    if _rebuild(corners, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
                origin_term, dest_term, corridor_t0, tcap) is None:
        return corners
    changed = True
    while changed and len(corners) > 2:
        changed = False
        i = 1
        while i < len(corners) - 1:
            cand = corners[:i] + corners[i + 1:]
            exact_heading = skip_exact_heading and _merge_preserves_resampling(
                corners[i - 1], corners[i], corners[i + 1], cfg.corridor_segment_len_m)
            if exact_heading or _rebuild(
                    cand, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
                    origin_term, dest_term, corridor_t0, tcap) is not None:
                corners = cand           # removed knot i; re-test the same index (list shifted)
                changed = True
            else:
                i += 1
    return corners


def _same_heading_3d(prev, knot, nxt) -> bool:
    """Whether the two incident XYZ legs point along the same ray, at any scale.

    The relative cross-product tolerance is scale-independent. Reversals fail the positive-dot gate,
    while a degenerate leg is conservatively classified as a heading change.
    """
    incoming = np.asarray(knot, float) - np.asarray(prev, float)
    outgoing = np.asarray(nxt, float) - np.asarray(knot, float)
    ni = float(np.linalg.norm(incoming))
    no = float(np.linalg.norm(outgoing))
    if ni <= _EPS or no <= _EPS:
        return False
    return (float(np.dot(incoming, outgoing)) > 0.0
            and float(np.linalg.norm(np.cross(incoming, outgoing)))
            <= _HEADING_EPS * ni * no)


def _segment_count(a, b, segment_len_m: float) -> int:
    """Subdivision count used by ``build_reservation_from_corners`` for one chord."""
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    dz = float(b[2]) - float(a[2])
    return max(1, math.ceil(math.sqrt(dx * dx + dy * dy + dz * dz) / segment_len_m))


def _segment_partition(a, b, segment_len_m: float):
    """Yield the exact scalar subsegment signature emitted by the reservation builder.

    Equality of these signatures means both routes feed byte-identical ``sa``/``sb`` values into
    timing, terminal tagging, volume construction, and metrics. This deliberately mirrors the
    builder's arithmetic instead of treating mathematical collinearity as byte equality. Streaming
    avoids materializing a long run's complete partition during every progressive merge.
    """
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    dx, dy, dz = bx - ax, by - ay, bz - az
    total = math.sqrt(dx * dx + dy * dy + dz * dz)
    nsub = max(1, math.ceil(total / segment_len_m))
    for k in range(1, nsub + 1):
        f0, f1 = (k - 1) / nsub, k / nsub
        yield (
            ax + f0 * dx, ay + f0 * dy, az + f0 * dz,
            ax + f1 * dx, ay + f1 * dy, az + f1 * dz,
        )


def _merge_preserves_resampling(prev, knot, nxt, segment_len_m: float) -> bool:
    """Whether dropping a same-heading knot preserves its rebuilt reservation byte-for-byte.

    Heading equality by itself is insufficient because every logical chord is independently
    resampled. Merging arbitrary collinear chords can move box boundaries and their timestamps.
    """
    if not _same_heading_3d(prev, knot, nxt):
        return False
    before_count = (_segment_count(prev, knot, segment_len_m)
                    + _segment_count(knot, nxt, segment_len_m))
    if before_count != _segment_count(prev, nxt, segment_len_m):
        return False
    before = chain(
        _segment_partition(prev, knot, segment_len_m),
        _segment_partition(knot, nxt, segment_len_m),
    )
    after = _segment_partition(prev, nxt, segment_len_m)
    return all(
        struct.pack("=6d", *old) == struct.pack("=6d", *new)
        for old, new in zip(before, after, strict=True)
    )


def _turn_ids(knots: tuple[_Knot, ...]) -> list[int]:
    """Stable IDs of the current 3-D turns, deterministically left-to-right."""
    return [
        knots[i].id
        for i in range(1, len(knots) - 1)
        if not _same_heading_3d(knots[i - 1].point, knots[i].point, knots[i + 1].point)
    ]


def _incoming_run_start(knots: tuple[_Knot, ...], turn_index: int) -> int:
    """Index of A for the maximal same-heading incoming run A…E→F."""
    start = turn_index - 1
    while start > 0 and _same_heading_3d(
            knots[start - 1].point, knots[start].point, knots[start + 1].point):
        start -= 1
    return start


def _outgoing_run_end(knots: tuple[_Knot, ...], turn_index: int) -> int:
    """Index of I for F→G…I, evaluated before any splice changes the heading at G."""
    end = turn_index + 1
    while end < len(knots) - 1 and _same_heading_3d(
            knots[end - 1].point, knots[end].point, knots[end + 1].point):
        end += 1
    return end


def _knot_index(knots: tuple[_Knot, ...], knot_id: int) -> int | None:
    """Current tuple index for a stable knot ID, or None if a prior splice removed it."""
    return next((i for i, knot in enumerate(knots) if knot.id == knot_id), None)


def _splice_between(state: _ShortcutState, left_id: int,
                    right_id: int) -> tuple[_Knot, ...] | None:
    """Remove all knots strictly between surviving endpoints; endpoints themselves are immutable."""
    left = _knot_index(state.knots, left_id)
    right = _knot_index(state.knots, right_id)
    if left is None or right is None or right <= left + 1:
        return None
    return state.knots[:left + 1] + state.knots[right:]


def _try_splice(state: _ShortcutState, left_id: int, right_id: int,
                context: _ShortcutContext) -> _ShortcutState | None:
    """Build and fully check one chord; a rejection never mutates ``state``."""
    candidate = _splice_between(state, left_id, right_id)
    if candidate is None:
        return None
    built = context.rebuild(candidate)
    if built is None:
        return None
    return _ShortcutState(candidate, built)


def _grow_one_turn(state: _ShortcutState, turn_id: int,
                   context: _ShortcutContext) -> _ShortcutState:
    """Seed E→G, batch A→G and A→I, then recover intermediate anchors on failures.

    Failed probes are deliberately non-pruning WITHIN a side: spatial and temporal feasibility are
    non-monotone in chord length (a shorter chord is differently oriented AND lands earlier, which
    re-times every downstream volume), so a rejected A→G never stops D→G, C→G, B→G.

    ACROSS sides the search is deliberately gated: the outgoing phase runs only once some left anchor
    was accepted, so a turn whose every incoming probe fails is left alone rather than re-probed as
    E→H / E→I. Those chords could succeed — but the legacy sweep cannot reach them either (it must
    drop F before G becomes a corner), and spending the probes there costs most on exactly the turns
    that have already proved expensive.

    The run endpoints and fallback order come from the immutable pre-splice snapshot; stable IDs
    resolve them against the progressively shortened current state.
    """
    snapshot = state
    turn_index = _knot_index(snapshot.knots, turn_id)
    if turn_index is None or turn_index <= 0 or turn_index >= len(snapshot.knots) - 1:
        return state

    incoming_start = _incoming_run_start(snapshot.knots, turn_index)
    outgoing_end = _outgoing_run_end(snapshot.knots, turn_index)
    local_left = snapshot.knots[turn_index - 1]   # E
    right = snapshot.knots[turn_index + 1]        # G
    far_left = snapshot.knots[incoming_start]     # A
    far_right = snapshot.knots[outgoing_end]      # I

    best = state
    accepted_left_id: int | None = None
    far_left_accepted = False

    # Local seed E→G. Its failure says nothing about the differently-oriented A→G chord.
    seeded = _try_splice(best, local_left.id, right.id, context)
    if seeded is not None:
        best = seeded
        accepted_left_id = local_left.id

    # Common-case batch: jump directly over the maximal incoming straight run.
    if far_left.id != local_left.id:
        maximal = _try_splice(best, far_left.id, right.id, context)
        if maximal is not None:
            best = maximal
            accepted_left_id = far_left.id
            far_left_accepted = True

    # Maximal failed: recover D→G, C→G, B→G. Keep probing after failures. A is already ruled out
    # (the splice it would produce is the one just rejected), and the slice is empty when the
    # incoming run is a single leg — exactly the case where there was no batch to attempt.
    if not far_left_accepted:
        intermediate_left = snapshot.knots[incoming_start + 1:turn_index - 1]
        for left in reversed(intermediate_left):
            if not best.contains(left.id):
                continue
            recovered = _try_splice(best, left.id, right.id, context)
            if recovered is not None:
                best = recovered
                accepted_left_id = left.id

    if accepted_left_id is None or far_right.id == right.id:
        return best

    # Common-case destination batch. I was captured before the incoming splice altered G's heading.
    maximal = _try_splice(best, accepted_left_id, far_right.id, context)
    if maximal is not None:
        return maximal

    # Maximal failed: recover →H, then successively farther endpoints. I itself was already tested.
    for candidate_right in snapshot.knots[turn_index + 2:outgoing_end]:
        if not best.contains(candidate_right.id):
            continue
        recovered = _try_splice(best, accepted_left_id, candidate_right.id, context)
        if recovered is not None:
            best = recovered
    return best


def _shortcut_turn_seeded(corners, had_holds: bool,
                          context: _ShortcutContext) -> _ShortcutState | None:
    """Turn-seeded maximal-run shortcut with stable IDs and full restart after each change.

    The accepted inner intent is already verified. A no-turn route therefore performs zero rebuilds.
    When repeated positions were collapsed, the hold-free baseline is rebuilt first to preserve the
    legacy rule that a load-bearing hold may not be silently discarded.
    """
    state = _ShortcutState(tuple(
        _Knot(i, np.asarray(point, float)) for i, point in enumerate(corners)
    ))
    if not _turn_ids(state.knots):
        return state

    if had_holds:
        baseline = context.rebuild(state.knots)
        if baseline is None:
            return None
        state = replace(state, built=baseline)

    while True:
        for turn_id in _turn_ids(state.knots):
            grown = _grow_one_turn(state, turn_id, context)
            if len(grown.knots) < len(state.knots):
                state = grown
                break                         # recompute every heading/index from the new tuple
        else:
            return state


def _terminal_capacity_for(planner, ledger) -> TerminalCapacity | None:
    """Find a capacity authority already brought current by the inner A*/MILP plan.

    Reuse avoids a second ledger subscription/index. Asks each planner in the wrapper chain via the
    optional ``capacity_authority(ledger)`` member (see the ``Planner`` Protocol) and takes the first
    that answers; planners holding no authority simply do not implement it.

    This used to reach into ``_tcap`` plus whichever of ``_tcap_ledger`` / ``_svc_ledger`` the family
    happened to use. That was a silent-failure shape, not just a style one: renaming any of those
    three private names — all of them internal to planners this module does not own — made every
    lookup return None, and returning None here does not raise, it makes ``plan`` hand back the
    UNREFINED inner intent for every terminal flight. ``astar_shortcut`` would quietly become bare
    ``astar``. Named method ⇒ a rename is a grep away, and ``test_shortcut_reuses_the_inner_capacity_
    authority`` pins that the reuse actually happens.
    """
    for p in iter_planner_chain(planner):
        get = getattr(p, "capacity_authority", None)
        if get is None:
            continue
        tcap = get(ledger)
        if isinstance(tcap, TerminalCapacity):
            return tcap
    return None


class ShortcutRefiner:
    """Wrap a planner with the legacy, exact-heading-skip, or batched-turn simplifier.

    A pure post-process: it never makes a plan worse (rebuild is re-verified against the ledger and
    terminal capacity, and the result is returned only when its cost is ≤ the original). Collapses
    mid-air holds first, since the rebuild re-times at nominal speed — if a hold was load-bearing for
    temporal deconfliction the rebuilt path will conflict and the original is kept.
    """

    def __init__(self, inner, label: str | None = None,
                 strategy: _ShortcutStrategy = "single_knot"):
        if strategy not in ("single_knot", "single_knot_heading", "batched_turns"):
            raise ValueError(f"unknown shortcut strategy: {strategy!r}")
        self.inner = inner
        self.label = label
        self.strategy = strategy

    def plan(self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig) -> OperationalIntent:
        intent = self.inner.plan(req, ledger, cfg)
        return refine_intent(intent, ledger, cfg, tcap=_terminal_capacity_for(self.inner, ledger),
                             strategy=self.strategy, label=self.label)


def refine_intent(intent: OperationalIntent, ledger: ReservationLedger, cfg: SimConfig, *,
                  tcap: TerminalCapacity | None, strategy: _ShortcutStrategy = "single_knot",
                  label: str | None = None) -> OperationalIntent:
    """Refine an intent whose volumes are NOT (or no longer) committed to ``ledger``.

    Split out of :meth:`ShortcutRefiner.plan` so a caller holding an already-planned intent can
    refine it without re-running the search — LNS's deferred arms release a repaired flight, call
    this, and re-commit. The contract is the wrapper's: never worse than ``intent``, and the returned
    geometry is what was conflict-checked. ``tcap`` is the caller's, because only the caller knows
    which planner already brought a capacity authority current for this ledger.
    """
    req = intent.request
    # Keep the legacy wrapper's exact three-point behavior. The experimental strategy fixes the
    # E-F-G blind spot without silently changing astar_shortcut's A/B baseline.
    min_centerline = 2 if strategy == "batched_turns" else 3
    if (not intent.accepted or not intent.centerline
            or len(intent.centerline) <= min_centerline):
        return intent

    corners: list[np.ndarray] = []
    for p, _ in intent.centerline:                 # collapse repeated positions (holds)
        p = np.asarray(p, float)
        if not corners or not np.allclose(p, corners[-1]):
            corners.append(p)
    had_holds = len(corners) < len(intent.centerline)
    if len(corners) <= 2:
        return intent

    g_delay = intent.ground_delay_s
    # Read the takeoff time off the committed origin column rather than inverting the centerline.
    # The old inverse subtracted only the climb, but issue #52 made centerline[0] land at
    # `takeoff + climb + Lane.steps*dt`, so the recovered departure came out LATE by the traverse
    # (measured 15 s on a 180 m hub: 12 s of lane traverse plus 3 s of pre-existing takeoff-step
    # rounding, which the old inverse also dropped) and every rebuilt volume shifted with it —
    # leaving the origin column unreserved while the drone was still in it. volumes[0].t_start IS
    # the takeoff time in both builders, so this needs no knowledge of which lane was taken.
    t_depart = intent.volumes[0].t_start - g_delay
    # Anchor the rebuilt corridor at the inner planner's VERIFIED first-cruise stamp: a refiner
    # re-times splices, not the takeoff. Re-deriving the start inside the rebuild mixes its
    # continuous clock (climb_time_to + WORST lane) with A*'s quantised stamp (climb_steps*dt +
    # CHOSEN lane's steps) — measured -3..+1 s on every rebuilt volume, lane-dependent.
    t_first = float(intent.centerline[0][1])
    ot, dt = req.origin_terminal, req.dest_terminal
    straight = enroute_reference_m(req.origin, req.dest, ot, dt, cfg)
    # Without the same interval authority the inner terminal-aware planner consulted, a retimed
    # shortcut cannot prove pad capacity. Refuse only the refinement; the verified inner intent is
    # still a valid fallback. Non-terminal routes need no capacity service.
    if (ot is not None or dt is not None) and tcap is None:
        return intent

    if strategy == "batched_turns":
        context = _ShortcutContext(
            req.origin, req.dest, t_depart, g_delay, cfg, ledger, straight,
            ot, dt, t_first, tcap,
        )
        state = _shortcut_turn_seeded(corners, had_holds, context)
        if (state is None or len(state.knots) >= len(corners) or state.built is None):
            return intent
        simplified = [k.point for k in state.knots]
        built = state.built                   # already rebuilt + fully checked; do not duplicate it
    else:
        simplified = shortcut_corners(
            corners, req.origin, req.dest, t_depart, g_delay, cfg, ledger,
            ot, dt, corridor_t0=t_first, tcap=tcap,
            skip_exact_heading=strategy == "single_knot_heading",
        )
        if len(simplified) >= len(corners):
            return intent                              # nothing removed
        built = _rebuild(
            simplified, req.origin, req.dest, t_depart, g_delay, cfg, ledger, straight,
            ot, dt, corridor_t0=t_first, tcap=tcap,
        )
        if built is None:
            return intent

    volumes, centerline, cum_horiz, cum_dz = built
    # Lattice overhead: decrement by what the sweep actually removed rather than inheriting the
    # inner planner's figure (which would over-report staircase on an already-straightened path).
    # Splicing out a redundant knot removes hex staircase before it removes any real berth, so
    # attributing the removal to overhead first is the right first-order split — approximate for
    # the refiner, exact for bare A*. Floored at 0 and monotone, so it can never invent overhead.
    inner_horiz = float(np.linalg.norm(np.diff(np.array(corners)[:, :2], axis=0), axis=1).sum())
    removed = max(0.0, inner_horiz - cum_horiz)
    refined = OperationalIntent(
        request=req, status=IntentStatus.ACCEPTED, volumes=volumes, centerline=centerline,
        ground_delay_s=g_delay, air_hold_s=0.0,
        air_detour_m=enroute_detour_m(
            enroute_flown_m([p for p, _ in centerline], req.origin, req.dest, ot, dt, cfg),
            straight),                                                              # issue #50
        lattice_overhead_m=max(0.0, intent.lattice_overhead_m - removed),
        altitude_change_m=endpoint_altitude_change_m(
            float(np.asarray(centerline[0][0])[2]), float(np.asarray(centerline[-1][0])[2]),
            cum_dz, cfg),
        planner=label or f"{intent.planner}+sc",
    )
    refined.cost = trajectory_cost(refined, cfg)
    return refined if refined.cost <= intent.cost + _EPS else intent
