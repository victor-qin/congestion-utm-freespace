"""The LNS delay ruler: what each flight would have cost flying ALONE.

``delay(fid) = incumbent cost - unimpeded cost`` is what seeds the agent-based destroy operator
and orders the ``premium`` repair, so every movable flight needs one plan against a world holding
nothing but the run's always-active terminal walls. That is a few hundred to a few thousand A*
searches, and it is the whole of ``LNSState``'s construction cost.

**Why it parallelises exactly.** The ruler ledger is never committed to, so plan *i* cannot observe
plan *j*: each is a pure function of ``(request, cfg, static_terms)``. Sharding it across processes
therefore cannot change a single cost — unlike the speculative parallel sim (``freespace_sim.parallel``),
which needs read-envelope validation precisely because its plans DO see each other's commits. The
worker count is a pure throughput knob, pinned by ``tests/test_lns.py``.

The decision to spawn is made from a measured prefix rather than a flight-count threshold: worker
startup is a fixed ~0.4 s (imports + the cached numba warm) while per-flight plan cost varies by an
order of magnitude across scenarios, so "is the remaining work worth a pool?" is the question that
actually transfers between machines and configs.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from multiprocessing import connection as mp_connection

log = logging.getLogger("freespace_sim.lns")

#: Flights planned in-process before deciding whether a pool pays for itself.
_PROBE_N = 24
#: Projected remaining sequential seconds below which the pool is not worth standing up. Well above
#: the measured ~0.4 s spawn so a marginal case stays sequential rather than paying for a wash.
_MIN_PARALLEL_S = 4.0
#: Adaptive g-hash/heap floor for a ruler planner (see `_new_ruler`).
_RULER_LOG2 = 18


def resolve_workers(n_workers: int | None) -> int:
    """``None`` -> a default pool size; anything else passes through (1 = in-process).

    Capped like ``ParallelConfig.n_workers``, but for a different reason: there is no ordered-commit
    stall here and each worker's ruler ledger is empty, so the cap is about the ~130 MB of imports
    and occupancy pools a worker costs, not about contention.
    """
    if n_workers is not None:
        return max(1, int(n_workers))
    return min(8, max(1, (os.cpu_count() or 4) - 2))


def _new_ruler(cfg, static_terms, shortcut: bool = False):
    """A planner + the walls-only ledger it rules against.

    ``shortcut=True`` wraps the ruler in a :class:`ShortcutRefiner`, which is REQUIRED whenever the
    incumbent being ruled was itself refined: ``delay()`` subtracts this cost from the incumbent's,
    so a bare-A* ruler against a shortcut schedule makes the premium NEGATIVE (measured on
    dallas_hub_2uss x1800: 76/451 flights) and ``max(0.0, ...)`` then clamps them to a silent 0 —
    invisible to the agent-based destroy seed, last in the ``premium`` repair order.

    Always A*, even under ``repair_planner="sipp"``: the ruler must match the INCUMBENT, which at
    build time is the FCFS baseline (`LNSState` pins that with ``_REPRODUCIBLE_PLANNERS``). SIPP
    repairs arrive later and are exact for the same cost function, so an A* ruler stays a lower
    bound for them — while a SIPP ruler would put the ruler and the baseline in different
    currencies, which is the failure above.

    One planner per ledger (the A* services bind to whichever ledger they first see). The refiner
    only ever READS the ledger (``any_conflict``), so the shard-purity that lets this parallelise
    holds, and ``evict_floor = 0.0`` freezes the eviction watermark so the flights may be planned in
    any order — which is what lets a worker take an arbitrary shard.
    """
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.astar import AStarPlanner

    free = ReservationLedger(cfg)
    for center, term in static_terms:
        free.register_static_terminal(center, term)
    # `kernel_log2_min` starts the g-hash/heap at ~15 MB instead of the ~470 MB ceiling that
    # `max_expansions` implies. Exact either way (overflow grows x4 and re-runs — see
    # `AStarPlanner._kernel_state`), and a ruler search never overflows it: the world is EMPTY, so
    # every flight flies its geodesic. Without this a pool of 8 rulers reserved several GB to hold
    # hash tables for searches that expand a few hundred nodes.
    planner = AStarPlanner(kernel_log2_min=_RULER_LOG2)
    planner.evict_floor = 0.0
    if shortcut:
        from freespace_sim.planner.shortcut import ShortcutRefiner

        return ShortcutRefiner(planner, label="astar_sc"), free
    return planner, free


def _plan_shard(cfg, static_terms, requests, planner=None, free=None, shortcut: bool = False):
    """``[(flight_id, cost | None, denial_reason | None), ...]`` — cost is None on a denial."""
    if planner is None:
        planner, free = _new_ruler(cfg, static_terms, shortcut)
    out = []
    for req in requests:
        u = planner.plan(req, free, cfg)
        out.append((req.flight_id, float(u.cost) if u.accepted else None,
                    None if u.accepted else u.denial_reason))
    return out


def _worker_main(conn, cfg, static_terms, requests, shortcut=False):
    """Worker process: build a private ruler, plan the shard, send it back, exit."""
    try:
        conn.send(_plan_shard(cfg, static_terms, requests, shortcut=shortcut))
    finally:
        conn.close()


def _finish_processes(procs, timeout=5.0) -> None:
    """Reap every started worker, escalating from join to terminate to kill.

    One shared deadline per phase keeps a broken pool from multiplying ``timeout`` by its worker
    count. Processes whose ``start`` failed have no pid and require no OS cleanup.
    """
    started = [proc for proc in procs if proc.pid is not None]
    deadline = time.monotonic() + timeout
    for proc in started:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [proc for proc in started if proc.is_alive()]
    for proc in alive:
        proc.terminate()
    deadline = time.monotonic() + timeout
    for proc in alive:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [proc for proc in alive if proc.is_alive()]
    for proc in alive:
        proc.kill()
    for proc in alive:
        proc.join()


def unimpeded_costs(cfg, static_terms, requests, *, n_workers=1, log_every=1000,
                    shortcut: bool = False):
    """Unimpeded cost per request, as ``[(flight_id, cost | None, denial_reason | None), ...]`` in
    the order ``requests`` was given — regardless of how the work was sharded, so a caller's log
    lines and warnings stay in flight order.

    ``n_workers <= 1`` plans in-process. Otherwise a probe prefix is planned in-process first and
    the pool is only spawned if the projected remainder justifies it; a worker that dies has its
    shard replanned in-process, loudly, rather than leaving the ruler short a flight.
    """
    n = len(requests)
    if n == 0:
        return []
    planner, free = _new_ruler(cfg, static_terms, shortcut)
    if n_workers <= 1:
        return _sequential(cfg, static_terms, requests, planner, free, log_every, 0)

    t0 = time.monotonic()
    probe = _plan_shard(cfg, static_terms, requests[:_PROBE_N], planner, free, shortcut)
    rate = (time.monotonic() - t0) / max(1, len(probe))
    rest = requests[len(probe):]
    projected = rate * len(rest)
    if projected < _MIN_PARALLEL_S:
        log.info("lns: unimpeded baseline sequential (%d flights, ~%.1fs projected)", n, projected)
        return probe + _sequential(cfg, static_terms, rest, planner, free, log_every, len(probe))

    W = min(n_workers, len(rest))
    log.info("lns: unimpeded baseline on %d workers (%d flights, ~%.0fs sequential)",
             W, n, projected)
    shards = [rest[w::W] for w in range(W)]         # round-robin: adjacent flights are the same
    #                                                 delivery's legs, so a contiguous split would
    #                                                 hand one worker a whole slow region
    conns, procs = [], []
    by_worker: list = [None] * W
    pool_error = None
    try:
        ctx = mp.get_context("spawn")
        for shard in shards:
            parent, child = ctx.Pipe(duplex=False)
            conns.append(parent)
            try:
                proc = ctx.Process(target=_worker_main,
                                   args=(child, cfg, static_terms, shard, shortcut), daemon=True)
                procs.append(proc)
                proc.start()
            finally:
                child.close()

        index = {conn: w for w, conn in enumerate(conns)}
        waiting = list(conns)
        while waiting:
            for conn in mp_connection.wait(waiting):
                waiting.remove(conn)
                w = index[conn]
                try:
                    by_worker[w] = conn.recv()
                except EOFError:                    # died before sending — exitcode says why
                    by_worker[w] = None
    except Exception as exc:
        # Pool construction/IPC is an optimization. Replan the whole remainder in-process rather
        # than returning a partial ruler; the finally below first tears down any workers that did
        # start. KeyboardInterrupt/SystemExit still propagate after the same cleanup.
        pool_error = exc
    finally:
        for conn in conns:
            conn.close()
        _finish_processes(procs)

    if pool_error is not None:
        log.warning("lns: unimpeded worker pool failed (%s) — replanning %d flights in-process",
                    pool_error, len(rest))
        return probe + _sequential(cfg, static_terms, rest, planner, free,
                                   log_every, len(probe))

    for w, rows in enumerate(by_worker):
        if rows is None:
            log.warning("lns: unimpeded worker %d died (exit %s) — replanning its %d flights "
                        "in-process", w, procs[w].exitcode, len(shards[w]))
            by_worker[w] = _plan_shard(cfg, static_terms, shards[w], planner, free, shortcut)

    out = probe + [None] * len(rest)                # un-stripe back into request order
    for w, rows in enumerate(by_worker):
        out[len(probe) + w::W] = rows
    return out


def _sequential(cfg, static_terms, requests, planner, free, log_every, offset):
    rows = []
    for k, req in enumerate(requests):
        u = planner.plan(req, free, cfg)
        rows.append((req.flight_id, float(u.cost) if u.accepted else None,
                     None if u.accepted else u.denial_reason))
        if log_every and (offset + k + 1) % log_every == 0:
            log.info("lns: unimpeded baseline %d", offset + k + 1)
    return rows
