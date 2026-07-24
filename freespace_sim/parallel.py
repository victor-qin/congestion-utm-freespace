"""Track A parallel infrastructure (issue #8) — read-envelopes for speculative planning.

This module carries the *validation geometry* for the speculative-planning / ordered-commit
parallel sim: a :class:`PlanEnvelope` summarises everything one plan **read** — every occupancy
cell the search probed (recorded as a (q, r, L, step) bbox by the compiled kernel / the
``_RecordingOcc`` reference shim in ``planner.astar``) plus the origin/destination hub discs whose
capacity/mask state the host consulted. The exact-mode commit test is then: *if no interleaved
commit's volumes intersect the envelope, the sequential planner would have produced a byte-identical
intent* — a plan is a deterministic function of its read set.

Conventions:
  * The cell bbox is conservative in z (levels are recorded but the meters conversion ignores them —
    committed columns span the tube anyway); xy + the recorded time window are the discriminators.
  * The time window is RECORDED, not predicted (``[t_request, max_step*dt + hover tail]``); the
    spatial-only rule applies to *pre-plan prediction* tubes (Phase 3), never to this validation.
  * Hub discs cover the host-side reads the cell bbox can't see: ``TerminalCapacity`` dwell/transit
    queries, the compiled path's takeoff/landing masks, and ``col_owners`` overlay marks.

Phase 2 adds the worker pool (``ParallelConfig`` / ``run_parallel``) on top of these primitives.
"""

from __future__ import annotations

import os
import pickle   # IPC only: coordinator ↔ worker processes WE spawn (the same transport
#                 multiprocessing.Connection.send uses internally) — never untrusted/persisted data;
#                 saved runs stay parquet+json (runs.py) by design.
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection

from .config import SimConfig
from .planner import hexgrid as hg

# int64 sentinels for an empty (never-probed) bbox: min slots start HUGE, max slots start -HUGE,
# so ``bbox[0] > bbox[1]`` ⇔ no probe happened. Shared with the kernel's read_bbox array.
BBOX_HUGE = 1 << 62


def env_pad_m(cfg: SimConfig) -> float:
    """Meters a probed cell's influence extends past its centre: the occupancy rasterisation
    inflation (``occupancy.HexOccupancyService`` — ``infl_blocked = corridor_width/2 + R``,
    ``infl_pad = hover_radius + R``). A volume can affect a cell iff it comes within this of the
    cell centre, so padding the cell bbox by it makes the meters envelope a superset of every
    volume that could have changed any probe. Both terms already include the circumradius R —
    do NOT add R again (it only inflates the false-dirty rate)."""
    R = hg.circumradius(cfg)
    return max(cfg.corridor_width_m / 2.0 + R, cfg.effective_hover_radius_m + R)


def cell_bbox_to_aabb(cell_bbox, cfg: SimConfig):
    """The world-xy AABB ``(xmin, ymin, xmax, ymax)`` covering hex cells ``qmin..qmax × rmin..rmax``
    (slots 0-3 of an 8-slot read bbox), padded by :func:`env_pad_m`.

    ``x = R·√3·(q + r/2)`` depends on BOTH q and r, so evaluate all four (q, r) corners and take
    min/max; ``y = R·1.5·r`` is monotone in r alone."""
    qmin, qmax, rmin, rmax = cell_bbox[0], cell_bbox[1], cell_bbox[2], cell_bbox[3]
    R = hg.circumradius(cfg)
    xs = [R * hg.SQRT3 * (q + r / 2.0) for q in (qmin, qmax) for r in (rmin, rmax)]
    ys = [R * 1.5 * rmin, R * 1.5 * rmax]
    pad = env_pad_m(cfg)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


@dataclass(frozen=True)
class PlanEnvelope:
    """Everything one plan read, summarised for the exact-mode commit test.

    ``cell_bbox`` — (qmin, qmax, rmin, rmax, Lmin, Lmax, smin, smax) over every occupancy probe,
    or ``None`` if the search never probed a cell (e.g. an immediate denial). ``xy`` is its
    meters conversion (via :func:`cell_bbox_to_aabb`), precomputed once at build. ``hub_reads``
    are ``(cx, cy, radius)`` discs for the o/d hub state the host consulted (gated by the same
    global time window). ``unbounded`` marks a plan whose read set cannot be bounded (a truncated
    SEARCH_EXHAUSTED denial) — the coordinator must treat it as always-dirty."""

    cell_bbox: tuple | None
    xy: tuple | None                       # (xmin, ymin, xmax, ymax) or None
    hub_reads: tuple                       # ((cx, cy, radius), ...)
    t_lo: float
    t_hi: float
    unbounded: bool = False


def _disc_hits_aabb(cx: float, cy: float, radius: float, a) -> bool:
    """Does the xy disc intersect the volume AABB ``(xmin, ymin, zmin, xmax, ymax, zmax)``?
    Clamp the centre into the box; compare the residual to the radius (scalar hot path)."""
    dx = (a[0] - cx) if cx < a[0] else (cx - a[3]) if cx > a[3] else 0.0
    dy = (a[1] - cy) if cy < a[1] else (cy - a[4]) if cy > a[4] else 0.0
    return dx * dx + dy * dy <= radius * radius


def envelope_intersects(env: PlanEnvelope, commits) -> bool:
    """True iff any committed volume ``(flat_aabb, t_start, t_end)`` in ``commits`` intersects the
    envelope — i.e. the speculation is DIRTY and exact mode must replan. The time window is the
    plan's *recorded* reach (kept deliberately — measured, not predicted); within it, a volume is a
    hit if it overlaps the probed-cell xy box or any consulted hub disc."""
    if env.unbounded:
        return True
    xy = env.xy
    for a, t0, t1 in commits:
        if t1 <= env.t_lo or t0 >= env.t_hi:
            continue                                       # outside everything the plan could read
        if xy is not None and not (a[3] < xy[0] or xy[2] < a[0]
                                   or a[4] < xy[1] or xy[3] < a[1]):
            return True
        for (cx, cy, rad) in env.hub_reads:
            if _disc_hits_aabb(cx, cy, rad, a):
                return True
    return False


# ======================================================================================
# Phase 2 — worker pool + coordinator (speculative planning, ordered commit)
# ======================================================================================


@dataclass
class ParallelConfig:
    """Knobs for the speculative parallel sim (``sim.run(parallel=...)``).

    ``mode``:
      * ``"exact"`` (default) — results are byte-identical to the sequential run (modulo
        ``solve_time_s``): a speculation commits only when NO interleaved commit intersects its
        recorded read-envelope; otherwise the coordinator replans it serially. ``n_workers`` /
        ``window`` are pure performance knobs — they cannot change results.
      * ``"relaxed"`` — a valid FCFS-class allocation: a speculation commits whenever its volumes
        are still conflict-free against the full ledger (the interleaved obstacles it never saw
        merely make its plan equal-or-cheaper than sequential's). With ``pin_prefixes`` (default)
        each flight k plans against exactly the first ``max(0, k - window)`` commits, making
        results a pure function of (scenario, config) — but note ``window`` thereby becomes a
        RESULT-AFFECTING parameter, part of the allocation semantics.

    Eager re-speculation (``max_respec`` > 0) re-dispatches a dirtied pending result to a worker
    immediately instead of stalling the commit frontier for a serial replan. Its trigger depends on
    whether the result is already back when the dirtying commit lands — wall-clock timing. Exact
    mode tolerates that provably (clean-envelope validation forces the sequential answer no matter
    which prefix a worker saw), so it is ON there; in relaxed+pinned mode it would leak timing into
    results, so it is DISABLED (relaxed dirty-rates are tiny — 0.6–5% measured at practical windows
    on dallas_full — so frontier serial replans stay cheap). relaxed+unpinned keeps it on and
    accepts nondeterminism.

    ``run_parallel`` writes a ``stats`` dict (serial replans, re-specs, canary count, dirty rate)
    onto the instance after the run."""

    n_workers: int = max(1, (os.cpu_count() or 4) - 2)
    window: int | None = None              # None → 4 × n_workers
    mode: str = "exact"                    # "exact" | "relaxed"
    pin_prefixes: bool = True              # relaxed only: deterministic snapshot prefixes
    max_respec: int = 2                    # eager re-speculation retries per flight (0 = off)
    # ---- Phase 3: spatial prediction + adaptation (scheduling hints — NEVER validation). All
    # three are auto-disabled under relaxed+pinned, where dispatch order / window feed the pinned
    # snapshot semantics and must stay fixed (see run_parallel). ----
    predictive_dispatch: bool = True       # prefer fresh flights whose xy tube avoids in-flight work
    tube_margin_m: float | None = None     # predictive tube pad (None → env_pad_m); SPATIAL-ONLY
    adaptive_window: bool = False          # shrink/grow the live window on the dirty-rate EMA
    worker_kernel_log2: int | None = None  # workers' adaptive g-hash floor (regrow keeps any value
    #                                        exact). Measured on dallas_full @ 8 workers: no plan-time
    #                                        difference vs the ceiling (the concurrency slowdown is
    #                                        broader memory-system contention, not hash footprint) —
    #                                        so default None = planner default; knob kept for study.
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("exact", "relaxed"):
            raise ValueError(f"ParallelConfig.mode must be 'exact' or 'relaxed', got {self.mode!r}")
        if self.n_workers < 1:
            raise ValueError("ParallelConfig.n_workers must be >= 1")

    @property
    def resolved_window(self) -> int:
        return self.window if self.window is not None else 4 * self.n_workers

    @property
    def eager_enabled(self) -> bool:
        # exact: timing-safe by the clean-envelope theorem; relaxed+pinned: would break determinism
        return self.max_respec > 0 and (self.mode == "exact" or not self.pin_prefixes)


#: Planners the parallel path supports in v1: every plan must come from an envelope-recording A*
#: (bare, reference oracle, or wrapped by the shortcut refiner, whose probes stay inside the inner
#: A*'s hull — the convex-hull lemma). The MILP family optimizes outside any recorded read set.
PARALLEL_PLANNERS = ("astar", "astar_ref", "astar_shortcut")


def spatial_tube(req, cfg: SimConfig, margin_m: float):
    """The pre-plan prediction tube for a flight: the xy-AABB spanning origin→dest ⊕ ``margin_m``.
    SPATIAL-ONLY by design (issue #8 directive): under density, queueing delay makes departure /
    arrival times unpredictable before planning, so the tube deliberately carries no time axis.
    A scheduling HINT only — dispatch preference, never validation."""
    ox, oy = float(req.origin[0]), float(req.origin[1])
    dx, dy = float(req.dest[0]), float(req.dest[1])
    return (min(ox, dx) - margin_m, min(oy, dy) - margin_m,
            max(ox, dx) + margin_m, max(oy, dy) + margin_m)


def _tubes_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def predicted_overlap_density(reqs, cfg: SimConfig, margin_m: float | None = None) -> float:
    """Fraction of flight pairs in ``reqs`` whose spatial tubes overlap — a cheap pre-run predictor
    of speculation interference (drives window sizing / bench expectations). O(n²) scalar tests:
    intended for a bounded lookahead slice, not a whole scenario."""
    m = margin_m if margin_m is not None else env_pad_m(cfg)
    tubes = [spatial_tube(r, cfg, m) for r in reqs]
    n = len(tubes)
    if n < 2:
        return 0.0
    hits = sum(_tubes_overlap(tubes[i], tubes[j]) for i in range(n) for j in range(i + 1, n))
    return hits / (n * (n - 1) / 2)


class AdaptiveWindow:
    """Clamp-bounded live-window controller: shrink while the frontier dirty-rate EMA runs hot
    (speculation is being wasted), grow back while it runs cold. Observations arrive in commit
    order. Only a THROUGHPUT knob: exact-mode results are window-invariant by construction, and
    relaxed+pinned mode keeps its semantic ``window`` fixed (this controller is disabled there —
    see ``run_parallel``)."""

    def __init__(self, lo: int, hi: int, alpha: float = 0.15,
                 shrink_at: float = 0.35, grow_at: float = 0.10):
        self.lo, self.hi = max(1, lo), max(1, hi)
        self.w = self.hi
        self.ema = 0.0
        self.alpha, self.shrink_at, self.grow_at = alpha, shrink_at, grow_at

    def observe(self, dirty: bool) -> None:
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * (1.0 if dirty else 0.0)
        if self.ema > self.shrink_at:
            self.w = max(self.lo, self.w - 1)
        elif self.ema < self.grow_at:
            self.w = min(self.hi, self.w + 1)


def _iter_astar(planner):
    """Every ``AStarPlanner`` reachable via the ``inner``/``warm_planner`` chain — local copy of
    ``sim._astar_planners`` (kept here to avoid a sim ↔ parallel import cycle)."""
    from .planner.astar import AStarPlanner
    out, seen, stack = [], set(), [planner]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, AStarPlanner):
            out.append(p)
        stack.extend((getattr(p, "inner", None), getattr(p, "warm_planner", None)))
    return out


def _flat_aabb_t(vol):
    """(flat_aabb, t_start, t_end) — the commit-log entry the envelope test consumes."""
    from .ledger import ReservationLedger
    return (ReservationLedger._flat_aabb(vol), float(vol.t_start), float(vol.t_end))


def _worker_main(conn, cfg, planner_name, static_terms, mode, pin, telemetry, kernel_log2):
    """Worker process: a replica ledger + its own planner stack, kept in sync by the coordinator's
    delta stream through the EXISTING subscribe machinery (``replica.commit`` fires the occupancy /
    TerminalCapacity hooks exactly as the live sim does — planners bind to whatever ledger they're
    handed). One delta arrives per COMMITTED FLIGHT SLOT (denials included, with empty volumes), so
    the applied-delta count IS the flight-index prefix the plan saw.

    Messages (FIFO per pipe — every delta the coordinator sent before an assign is drained before
    that assign is processed, so ``applied`` ≥ the assign's pinned prefix by construction):
      ("delta", fid, volumes)         → exact/unpinned: absorb IMMEDIATELY (idle-time sync, off the
                                        plan critical path); pinned: buffer, applied ≤ P at assign
      ("assign", k, req, P, floor)    → sync, plan, reply ("result", k, intent, env, P_used, tele)
      ("stop",)                       → exit

    ``telemetry``: a per-worker observer collector captures this plan's ``on_deny`` rows; the fresh
    rows ride the result message and the coordinator merges them IN COMMIT ORDER (discarding rows of
    superseded speculations), so the master streams match the sequential run's."""
    from .ledger import ReservationLedger
    from .planner import get_planner
    from .uss import _warn_if_terminal_dropped

    replica = ReservationLedger(cfg)
    for center, term in static_terms:
        replica.register_static_terminal(center, term)
    # ONE planner instance shared across USS ids: planner state is ledger-derived (occupancy /
    # capacity absorb the same replica either way), so instance identity cannot change any plan —
    # while per-USS instances would DOUBLE the per-delta absorb cost (every subscribed service pays
    # on_commit per commit) and the JIT warm. The byte-parity tests gate this equivalence.
    planner = get_planner(planner_name)
    tele = None
    if telemetry:
        from .telemetry import TelemetryCollector
        tele = TelemetryCollector()
    for a in _iter_astar(planner):
        a.record_envelope = True
        if kernel_log2 is not None:
            a.kernel_log2_min = kernel_log2             # small adaptive hash floor (contention relief)
        if tele is not None:
            a._tele = tele

    fresh_prefix = mode == "exact" or not pin               # freshest-prefix semantics?
    deltas: list = []                                       # pinned mode only: buffered, applied ≤ P
    applied = 0
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        tag = msg[0]
        if tag == "stop":
            return
        if tag == "delta":
            # exact / unpinned: absorb IMMEDIATELY — the coordinator streams deltas while this
            # worker idles in recv(), so occupancy maintenance runs off the plan critical path
            # (validation makes any prefix correct, and fresher prefixes mean fewer replans).
            # pinned relaxed: buffer — only the assign's pinned prefix P may ever be applied.
            if fresh_prefix:
                if msg[2]:
                    replica.commit(msg[1], msg[2])
                applied += 1
            else:
                deltas.append((msg[1], msg[2]))
            continue
        _, k, req, P, floor = msg                           # "assign"
        if not fresh_prefix:
            target = min(P, len(deltas))
            while applied < target:
                fid, vols = deltas[applied]
                if vols:
                    replica.commit(fid, vols)
                deltas[applied] = None                      # free absorbed volumes
                applied += 1
        for a in _iter_astar(planner):
            a.evict_floor = floor                           # frontier clock: never evict past it
        try:
            n_filed = len(tele.filed_volumes) if tele is not None else 0
            n_conf = len(tele.conflict_events) if tele is not None else 0
            t0 = time.monotonic()
            intent = planner.plan(req, replica, cfg)
            intent.solve_time_s = time.monotonic() - t0
            _warn_if_terminal_dropped(req, intent)
            env = next((a.last_envelope for a in _iter_astar(planner)
                        if a.last_envelope is not None), None)
            tele_rows = (None if tele is None else
                         (tele.filed_volumes[n_filed:], tele.conflict_events[n_conf:]))
            conn.send(("result", k, intent, env, applied, tele_rows))
        except Exception:
            conn.send(("error", k, traceback.format_exc()))


def run_parallel(scenario, cfg, pcfg: ParallelConfig, ledger, dss, planner_name,
                 static_terms, status, report, collector=None):
    """Coordinator: dispatch speculative plans to worker processes, validate + commit strictly in
    scenario (FCFS) order against the authoritative ledger, stream commit deltas back out, and
    eagerly re-dispatch dirtied pending results. Returns the intents in scenario order.

    Delta fan-out is LAZY per worker (an outbox flushed right before that worker's next assign):
    an eager broadcast to a busy worker can fill the pipe and block the coordinator mid-run,
    stalling every healthy worker behind one slow plan. Each delta is pickled once
    (``send_bytes``); a worker only ever receives deltas while idle, so the flush never blocks.

    Phase-3 scheduling hints (never validation): ``predictive_dispatch`` prefers a fresh flight
    whose SPATIAL tube avoids everything currently in flight (the frontier flight is always taken
    first — liveness); ``adaptive_window`` shrinks the live window while the dirty-rate EMA runs
    hot. ``collector`` (telemetry) receives worker ``on_deny`` rows merged in commit order; serial
    replans write into it directly."""
    import multiprocessing as mp

    from .planner import get_planner

    events = scenario.events
    total = len(events)
    if total == 0:
        return []
    W = pcfg.resolved_window

    # BEFORE spawn: constructing the serial replan lane runs the numba warm — on a cold cache the
    # parent compiles once and workers load from disk instead of racing to compile (cache stampede).
    serial = get_planner(planner_name)
    for a in _iter_astar(serial):
        if collector is not None:
            a._tele = collector                             # serial replans feed the master directly

    ctx = mp.get_context("spawn")
    workers = []                                            # (process, conn)
    for _ in range(pcfg.n_workers):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker_main,
                           args=(child, cfg, planner_name, static_terms,
                                 pcfg.mode, pcfg.pin_prefixes, collector is not None,
                                 pcfg.worker_kernel_log2),
                           daemon=True)
        proc.start()
        child.close()
        workers.append((proc, parent))
    conn_to_w = {conn: w for w, (_p, conn) in enumerate(workers)}

    results: list = [None] * total
    pending: dict[int, tuple] = {}          # k -> (intent, env, P_used, tele_rows)
    retries: dict[int, int] = {}
    respec_q: deque[int] = deque()
    fresh: deque[int] = deque()             # undispatched fresh flights, ascending (predictive pick)
    idle: deque[int] = deque(range(len(workers)))
    busy: dict[int, int] = {}               # worker idx -> flight k
    outbox: list[list[bytes]] = [[] for _ in workers]       # lazy per-worker delta blobs
    commit_log: list[list] = []             # per FLIGHT slot: [(flat_aabb, t0, t1), ...] ([] = denial)
    next_commit = 0
    cursor_box = [0]                        # next fresh flight index not yet in `fresh`
    n_serial = n_respec = n_canary = n_dirty = n_deferred = 0
    # coordinator wall accounting (issue #8 Phase E/F): is the serial commit floor binding, or is the
    # coordinator idle waiting on straggler workers? t_commit = time inside the ordered-commit block
    # (dss.commit + occupancy hooks + delta broadcast, all serial); t_wait = time blocked in
    # connection.wait for any worker to return. t_commit ≫ t_wait ⇒ shrink the floor; t_wait ≫
    # t_commit ⇒ the bottleneck is worker plan variance, not the coordinator.
    t_commit = t_wait = 0.0
    # live window: adaptive only where it is a pure throughput knob (exact / unpinned relaxed) —
    # in relaxed+pinned mode W is SEMANTIC (part of the pinned prefixes) and must stay fixed.
    adapt = (AdaptiveWindow(lo=pcfg.n_workers, hi=W)
             if pcfg.adaptive_window and (pcfg.mode == "exact" or not pcfg.pin_prefixes) else None)
    # Predictive dispatch REORDERS fresh assignments. A pinned-relaxed worker that has already
    # absorbed a bigger prefix (for a later flight) cannot shrink its replica back for an earlier
    # one — the pinned snapshot would silently include extra commits, timing-dependently. So, like
    # eager re-spec, predictive dispatch is disabled under relaxed+pinned; in-order per-worker
    # streams keep each worker's applied prefix monotone and the pinned semantics exact.
    predictive = pcfg.predictive_dispatch and (pcfg.mode == "exact" or not pcfg.pin_prefixes)
    tube_m = pcfg.tube_margin_m if pcfg.tube_margin_m is not None else env_pad_m(cfg)
    tubes: dict[int, tuple] = {}

    def _tube(k: int):
        t = tubes.get(k)
        if t is None:
            t = tubes[k] = spatial_tube(events[k].request, cfg, tube_m)
        return t

    def _flush(widx: int):
        conn = workers[widx][1]
        for blob in outbox[widx]:
            conn.send_bytes(blob)
        outbox[widx].clear()

    def _assign(widx: int, k: int):
        _flush(widx)                                        # flush deltas FIRST (FIFO ⇒ applied ≥ P)
        conn = workers[widx][1]
        req = events[k].request
        P = max(0, k - W) if (pcfg.mode == "relaxed" and pcfg.pin_prefixes) else len(commit_log)
        floor = events[next_commit].request.t_request
        conn.send(("assign", k, req, P, floor))
        busy[widx] = k

    def _pick_fresh():
        """Next fresh flight to dispatch. Frontier first, unconditionally (liveness: commits can
        never pass an undispatched frontier). Otherwise, with predictive dispatch, prefer the first
        of the next few candidates whose spatial tube misses every in-flight speculation — dispatch
        REORDERING only; the commit order is untouchable."""
        nonlocal n_deferred
        if not predictive or fresh[0] == next_commit or len(fresh) == 1:
            return fresh.popleft()
        inflight = [_tube(j) for j in busy.values()]
        inflight += [_tube(j) for j in pending]
        inflight += [_tube(j) for j in respec_q]
        for i, k in enumerate(list(fresh)[:8]):             # bounded lookahead
            if all(not _tubes_overlap(_tube(k), t) for t in inflight):
                if i:
                    n_deferred += 1
                del fresh[i]
                return k
        n_deferred += 1
        return fresh.popleft()                              # all overlap → no starvation, take oldest

    def _dispatch():
        W_live = adapt.w if adapt is not None else W
        while cursor_box[0] < total and len(fresh) < 16 and cursor_box[0] - next_commit < W_live:
            fresh.append(cursor_box[0])
            cursor_box[0] += 1
        while idle:
            if respec_q:
                _assign(idle.popleft(), respec_q.popleft())
            elif fresh:                     # admitted under the window at refill (only tightens after)
                _assign(idle.popleft(), _pick_fresh())
            else:
                break

    try:
        _dispatch()
        while next_commit < total:
            # ---- commit everything ready at the frontier, strictly in order ----
            _tc0 = time.monotonic()
            while next_commit in pending:
                k = next_commit
                intent, env, P_used, tele_rows = pending.pop(k)
                interleaved = [ab for slot in commit_log[P_used:] for ab in slot]
                if pcfg.mode == "exact":
                    dirty = env is None or env.unbounded or (
                        bool(interleaved) and envelope_intersects(env, interleaved))
                else:
                    dirty = bool(intent.accepted and intent.volumes
                                 and ledger.any_conflict(intent.volumes))
                if adapt is not None:
                    adapt.observe(dirty)
                if dirty:
                    n_dirty += 1
                    n_serial += 1
                    t0 = time.monotonic()
                    intent = serial.plan(events[k].request, ledger, cfg)   # writes master telemetry
                    intent.solve_time_s = time.monotonic() - t0
                elif collector is not None and tele_rows is not None:
                    # clean speculation: merge its worker-side on_deny rows, in commit order
                    collector.filed_volumes.extend(tele_rows[0])
                    collector.conflict_events.extend(tele_rows[1])
                was_accepted = intent.accepted
                committed = dss.commit(intent)
                if pcfg.mode == "exact" and not dirty and was_accepted and not committed:
                    # A clean envelope certifies byte-equality with sequential, whose own filed-
                    # corridor check passed — the mechanism re-check can then never fail. If it
                    # does, the envelope under-recorded a read: an instrumentation BUG, surfaced
                    # loudly (the run stays conflict-free — the backstop did its job).
                    n_canary += 1
                    warnings.warn(
                        f"parallel exact-mode canary: clean speculation for flight "
                        f"{intent.request.flight_id} was rejected at commit — the read-envelope "
                        f"under-recorded a read (soundness bug); result stays conflict-free.",
                        RuntimeWarning, stacklevel=2)
                results[k] = intent
                slot = ([_flat_aabb_t(v) for v in intent.volumes]
                        if committed and intent.volumes else [])
                commit_log.append(slot)
                if slot:
                    blob = pickle.dumps(("delta", intent.request.flight_id, intent.volumes),
                                        protocol=pickle.HIGHEST_PROTOCOL)
                else:                                       # denial: empty slot keeps prefixes aligned
                    blob = pickle.dumps(("delta", intent.request.flight_id, None),
                                        protocol=pickle.HIGHEST_PROTOCOL)
                for box in outbox:
                    box.append(blob)
                # Stream the backlog to IDLE workers now: they are parked in recv(), so they absorb
                # while idling and the occupancy sync leaves the plan critical path. (A transient
                # send-block against a slowly-absorbing idle worker is bounded by its absorb rate —
                # and that absorb is exactly the work we want done before its next assign.) Busy
                # workers keep the lazy outbox, flushed right before their next assign.
                for w in idle:
                    _flush(w)
                next_commit += 1
                status(next_commit, events[k].request, intent)
                if report:
                    report(next_commit, total, intent)
                # ---- eager re-speculation: a fresh commit dirties waiting results ----
                if slot and pcfg.eager_enabled:
                    for j in [j for j, (_i, e, _p, _t) in pending.items()
                              if retries.get(j, 0) < pcfg.max_respec
                              and (e is None or envelope_intersects(e, slot))]:
                        del pending[j]
                        retries[j] = retries.get(j, 0) + 1
                        respec_q.append(j)
                        n_respec += 1
            t_commit += time.monotonic() - _tc0
            if next_commit >= total:
                break
            _dispatch()
            # ---- wait for at least one worker result ----
            busy_conns = [workers[w][1] for w in busy]
            if not busy_conns:
                # frontier not pending and nothing in flight ⇒ it must be queued for re-spec;
                # dispatch made no progress only if there are no idle workers — impossible here.
                continue
            _tw0 = time.monotonic()
            ready = mp_connection.wait(busy_conns)
            t_wait += time.monotonic() - _tw0
            for conn in ready:
                widx = conn_to_w[conn]
                try:
                    msg = conn.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"parallel worker {widx} died (EOF) while planning flight "
                        f"{busy.get(widx)}") from exc
                busy.pop(widx, None)
                idle.append(widx)
                if msg[0] == "error":
                    raise RuntimeError(
                        f"parallel worker {widx} raised while planning flight {msg[1]}:\n{msg[2]}")
                _tag, k, intent, env, P_used, tele_rows = msg
                pending[k] = (intent, env, P_used, tele_rows)
            _dispatch()
    finally:
        for _proc, conn in workers:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for proc, conn in workers:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
            conn.close()

    pcfg.stats = {
        "mode": pcfg.mode, "n_workers": pcfg.n_workers, "window": W,
        "n_flights": total, "n_dirty": n_dirty, "n_serial_replans": n_serial,
        "n_respec": n_respec, "n_canary": n_canary, "n_deferred": n_deferred,
        "dirty_rate": n_dirty / total, "predictive": predictive,
        "final_window": adapt.w if adapt is not None else W,
        "t_commit_s": t_commit, "t_wait_s": t_wait,
    }
    return results
