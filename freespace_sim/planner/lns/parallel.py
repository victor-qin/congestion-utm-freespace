"""Parallel LNS — DROP-LNS's destroy/repair operation parallelism, as processes.

Chan et al., *Anytime MAPF using Operation Parallelism in LNS* (arXiv:2402.01961,
``context/drop-lns-parallel-2024.pdf``). The paper runs a main thread plus m worker threads over
a shared best-known solution ``P_min`` guarded by two mutexes; each worker takes a private copy
of ``P_min``, destroys, repairs, and writes back under the lock.

Two things make the port here different from the paper, and both are simplifications:

**1. The private copy is a persistent replica synced by delta.** The paper's ``P`` is a list of
paths, so copying it is free. Ours is a ``ReservationLedger`` plus an occupancy service stack
plus a claim index — building one from scratch is the O(schedule) rebuild that PR #109 measured
at 3.74 s, 94% of an iteration, and then eliminated. So a worker builds its replica ONCE and is
afterwards told "the incumbent moved" by a compacted diff, which costs O(the flights that
changed). ``LNSState.apply_delta`` is that operation; it rides the ``release_many`` /
``subscribe_release`` machinery #109 already built.

**2. There are no mutexes, because there is no shared memory.** A single-threaded coordinator IS
``M_main`` and ``M_task``. It owns the incumbent — as its own ``LNSState`` over the caller's
ledger, so the write-back is just ``apply_delta`` and the seed selection shares one implementation
with the workers (see the note above ``_Changelog``) — and it never plans. Workers are
spawned processes on one duplex ``Pipe`` each — the transport ``freespace_sim.parallel`` and
``colgen.pricing_pool`` both established, and for the same reasons (``fork`` inherits the numba
runtime and thread state; ``mp.Pool`` silently respawns a dead worker with the original initargs;
``ProcessPoolExecutor`` deadlocks on ``max_tasks_per_child``). Zero lock contention is not a
detail: synchronization overhead is exactly what the paper measures degrading DROP at 16 threads.

Processes rather than threads even though ``astar.kernel`` is ``@njit(nogil=True)``: the Python
host around ``plan()`` was measured at ~31% of profiled self-time in the mask build alone, and
``hexgrid._RANGE_CACHE`` is a process-global ``OrderedDict`` with a check-then-act eviction.

**The always-rewind rule.** A worker never keeps its own accept — it reports the repair and
immediately restores its replica. So ``applied_version`` is always exactly a
coordinator-blessed version and a sync diff is always well-defined, with no three-way merge
anywhere. It costs one extra ``release_many`` plus k commits on the ~21% of tasks that accept.

Modes:

* ``sync``   — a barrier per round; the best of m results is applied (paper Eq. 3). Deterministic,
  and uses the same seeded task-selection stream as the sequential loop. Effective widths of zero
  or one stay in-process; a private replica cannot add concurrency in that case.
* ``drop``   — asynchronous; a result is applied as soon as it lands. Nondeterministic by
  construction (completion order is wall clock). Built on the same worker and protocol.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass, replace
from multiprocessing import connection as mp_connection

import numpy as np

from freespace_sim import verify
from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.lns.neighborhood import (
    AdaptiveSelector,
    _select_most_delayed,
    agent_based_neighborhood,
    map_based_neighborhood,
    random_neighborhood,
)
from freespace_sim.planner.lns.state import LNSState
from freespace_sim.planner.lns.solver import (
    assert_incumbent_ok,
    _build_lns_state,
    _effective_search_workers,
    _finalize_lns_result,
    _trajectory_row,
    _validate_lns_config,
    run_lns,
)
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

_GRACEFUL_STOP_S = 0.5        # one window shared by ALL workers (per-worker joins cost 15 s
_CLOSE_JOIN_TIMEOUT_S = 5.0   # after a single loss — see colgen.pricing_pool.close)


# ====================================================================== messages
@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs to build a replica and run tasks. Must be picklable.

    Every field that changes what a repair is ALLOWED to do lives here, because a worker that
    silently differs from the coordinator's belief is the failure mode with no symptom: dropping
    ``turnaround_s`` disarms the paired-return anchor guard, and ``verify`` checks 4D conflicts
    only, so the run still reports ``verified``.
    """

    neighborhood_size: int
    accept_epsilon: float
    repair_order: str
    max_walks: int
    map_max_cells: int
    turnaround_s: float | None
    frozen_flight_ids: frozenset
    movable_uss_ids: frozenset | None
    incremental_release: bool
    kernel_log2_min: int | None
    pair_closed_neighborhood: bool = False
    record_envelope: bool = True
    # Answer-neutral (see LNSConfig.window_bytes), but still shipped: it is per-planner state, so a
    # worker left on the default would run a different cache configuration than the one measured.
    window_bytes: int | None = None
    repair_planner: str = "astar"      # registry NAME; a planner object is not picklable


@dataclass
class TaskResult:
    """One destroy/repair operation, as reported home. ``base_version`` is the incumbent version
    the worker's replica held when it started — the coordinator's staleness test reads it."""

    rnd: int
    slot: int
    worker: int
    base_version: int
    op: str
    victims: tuple
    new_intents: dict
    cost_old: float
    cost_new: float
    envelopes: tuple
    reason: str

    @property
    def improved(self) -> bool:
        return self.reason == "improved"

    @property
    def improvement(self) -> float:
        return self.cost_old - self.cost_new if self.improved else 0.0


class WorkerLost(RuntimeError):
    """A worker died mid-round. Raised rather than absorbed: an OOM-killed worker that is merely
    waited on HANGS the run, which is the documented failure mode of every pool in this repo."""


class WorkerStartTimeout(WorkerLost):
    """The wall-clock budget expired while workers were starting."""


# ====================================================================== coordinator helpers
# The coordinator holds ONE ``LNSState`` on the CALLER's ledger rather than a hand-rolled
# incumbent dict plus a shim context. That is a deliberate departure from the first sketch of this
# design, and it deletes three problems rather than solving them:
#
# * ``tabu`` is a serial recurrence that ``_select_most_delayed`` mutates, so the seeds must be
#   chosen centrally (m workers with private tabus would all pick the same most-delayed flight and
#   run m copies of one neighborhood). Choosing them off a real ``LNSState`` means the coordinator
#   and the workers share ONE implementation of ``movable_ids`` and ``delay`` — a shim reproducing
#   them is a second implementation that can drift, silently changing which flights get destroyed.
# * ``run_lns``'s contract is that the caller's ledger is mutated in place and the returned intents
#   supersede the input list. With the state on that ledger, ``apply_delta`` IS the write-back:
#   incremental, exact, and already covered by the transaction tests. A coordinator that owned only
#   data would have to reconstruct it at the end, and would have to remember to take the ledger over
#   first — ``ledger.commit`` fires the FCFS run's observers, so a late write-back re-absorbs into
#   services that are about to be discarded.
# * ``final_intents`` and the closing ``verify`` replay work unchanged.
#
# The cost is one extra resident state (the coordinator's), which holds the caller's ledger that
# exists anyway. The coordinator still never PLANS: its ``repair_planner`` is constructed but never
# used, and the A* services bind lazily, so it never subscribes to the ledger at all.


class _Changelog:
    """Versioned record of accepted repairs, so a worker at version v can be caught up in
    O(flights changed since v) rather than O(schedule).

    Entries are compacted on read — a flight touched three times since v only needs its latest
    intent — which is what keeps a sync diff small even for a worker that has fallen far behind.
    """

    def __init__(self) -> None:
        self.version = 0
        self._entries: list[tuple[int, dict, dict]] = []      # (version, new_by_fid, old_by_fid)

    def record(self, changes: dict, olds: dict) -> int:
        self.version += 1
        self._entries.append((self.version, dict(changes), dict(olds)))
        return self.version

    def diff_since(self, base_version: int) -> dict:
        out: dict = {}
        for v, changes, _olds in self._entries:
            if v > base_version:
                for fid, intent in changes.items():
                    # Last write wins AND last touch determines replay order. Plain ``update``
                    # replaces an existing value without moving its insertion position, which can
                    # change the ledger layout and break one-worker parity after an A, B, A history.
                    out.pop(fid, None)
                    out[fid] = intent
        return out

    def touched_since(self, base_version: int) -> set:
        return {fid for v, changes, _ in self._entries if v > base_version for fid in changes}

    def boxes_since(self, base_version: int):
        """Every volume REMOVED or ADDED since ``base_version``, as the envelope test's triples.

        Removed as well as added: added-only would be enough for feasibility (deleting an obstacle
        cannot create a conflict) but not for the stronger claim the merge rests on — that the
        stale plan is the plan the worker would have produced against the CURRENT incumbent.
        """
        from freespace_sim.parallel import _flat_aabb_t

        out = []
        for v, changes, olds in self._entries:
            if v <= base_version:
                continue
            for it in changes.values():
                out.extend(_flat_aabb_t(vol) for vol in it.volumes)
            for it in olds.values():
                out.extend(_flat_aabb_t(vol) for vol in it.volumes)
        return out

    def revert_to(self, base_version: int) -> dict:
        """The intents that ``base_version`` held, for every fid changed since — i.e. what it takes
        to walk the incumbent back to the state a stale worker planned against.

        The FIRST recorded ``old`` per fid is the one at ``base_version``: later entries' olds are
        intermediate values this walk-back skips over.
        """
        out: dict = {}
        for v, _changes, olds in self._entries:
            if v <= base_version:
                continue
            for fid, it in olds.items():
                out.setdefault(fid, it)
        return out

    def trim(self, min_live_version: int) -> None:
        """Drop entries no worker can still need."""
        self._entries = [e for e in self._entries if e[0] > min_live_version]


# ====================================================================== worker
def _worker_main(conn, cfg: SimConfig, intents: list, static_terms: tuple,
                 unimpeded_cost: dict, spec: WorkerSpec, index: int) -> None:
    """One worker process: a private replica, then destroy/repair tasks against it.

    ``unimpeded_cost`` keeps its ``None``s: ``LNSState.__init__`` is the single owner of the
    "ruler denied this flight -> treat as undelayed" rule, so the coordinator and every worker
    resolve it identically by construction rather than by agreement.
    """
    try:
        state = LNSState.replica(
            cfg, intents,
            static_terms=static_terms,
            unimpeded_cost=unimpeded_cost,
            turnaround_s=spec.turnaround_s,
            frozen_flight_ids=spec.frozen_flight_ids,
            movable_uss_ids=spec.movable_uss_ids,
            incremental_release=spec.incremental_release,
            kernel_log2_min=spec.kernel_log2_min,
            record_envelope=spec.record_envelope,
            window_bytes=spec.window_bytes,
            repair_planner_name=spec.repair_planner,
        )
    except BaseException:                                       # noqa: BLE001 - reported home
        import traceback
        conn.send(("ready", os.getpid(), traceback.format_exc()))
        return
    conn.send(("ready", os.getpid(), None))

    applied_version = 0
    tabu: set[int] = set()          # only reached when the coordinator does NOT supply a seed
    # Scoped INSIDE the worker: the parent's filter (solver.run_lns) cannot reach a spawned
    # process, and the incremental_release=False reference path warns once per iteration.
    # Scoped rather than global for the same reason the parent scopes it — a caller that wants
    # to see the warning afterwards must still be able to.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ReservationLedger shrank")
        while True:
            msg = conn.recv()
            kind = msg[0]
            if kind == "stop":
                return
            if kind == "sync":
                _, version, changes = msg
                state.apply_delta(changes)
                applied_version = version
                continue
            if kind != "task":
                raise RuntimeError(f"lns worker got an unknown message {kind!r}")

            _, rnd, slot, op, seed_fid, rng_state = msg
            rng = np.random.default_rng()
            # The generator STATE, not a seed: the coordinator already consumed one draw for the
            # ALNS roulette from this same stream, and a freshly seeded generator would start a
            # draw earlier — enough to change every victim set. See run_lns_parallel.
            rng.bit_generator.state = rng_state
            state.rng = rng

            if op == "agent":
                victims = agent_based_neighborhood(state, spec.neighborhood_size, tabu,
                                                   spec.max_walks, seed_fid=seed_fid)
            elif op == "map":
                victims = map_based_neighborhood(state, spec.neighborhood_size,
                                                 spec.map_max_cells)
            else:
                victims = random_neighborhood(state, spec.neighborhood_size)

            if spec.pair_closed_neighborhood:
                victims = state.close_over_pairs(victims)
            if not victims:
                # The sequential loop short-circuits here with reason="empty" and never calls
                # try_repair. Falling through would compute cost_new = 0.0, fail the strict
                # 0.0 < 0.0 - eps test, and report "no_improvement" — a parity break with
                # nothing to do with parallelism. Reachable whenever contention_cells() is empty.
                conn.send(("result", rnd, slot, index, applied_version, op,
                           (), {}, 0.0, 0.0, (), "empty"))
                continue

            out = state.try_repair(victims, rng, spec.accept_epsilon,
                                   order_mode=spec.repair_order, report_only=True)
            conn.send(("result", rnd, slot, index, applied_version, op,
                       tuple(sorted(victims)), out.new_intents, out.cost_old, out.cost_new,
                       out.envelopes, out.reason))


# ====================================================================== pool
class LNSWorkerPool:
    """m spawned workers, one duplex pipe each, modelled on ``colgen.pricing_pool.PricingPool``.

    The details copied from it are all failure modes it already paid for: closing the parent's
    copy of the child end (else EOF never fires and worker death is invisible), waiting on
    ``proc.sentinel`` alongside the pipe (else an OOM-killed worker hangs the run instead of
    failing it), and one SHARED grace window at teardown (per-worker joins made teardown after a
    single loss take 15 s).
    """

    def __init__(self, cfg: SimConfig, intents: list, static_terms: tuple, unimpeded_cost: dict,
                 spec: WorkerSpec, n_workers: int) -> None:
        self._cfg = cfg
        self._intents = intents
        self._static_terms = static_terms
        self._unimpeded_cost = unimpeded_cost
        self._spec = spec
        self.n_workers = int(n_workers)
        self._procs: list = []
        self._conns: list = []
        self.worker_version: list[int] = []
        self.spawn_s = 0.0

    # ---------------------------------------------------------------- lifecycle
    def start(self, *, deadline: float | None = None) -> "LNSWorkerPool":
        ctx = mp.get_context("spawn")   # never fork: it inherits the numba runtime + thread state
        t0 = time.monotonic()
        try:
            for i in range(self.n_workers):
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerStartTimeout("lns worker startup exhausted the wall-clock budget")
                parent, child = ctx.Pipe()
                p = ctx.Process(
                    target=_worker_main,
                    args=(child, self._cfg, self._intents, self._static_terms,
                          self._unimpeded_cost, self._spec, i),
                    daemon=True,
                )
                try:
                    p.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()    # without closing this parent-side copy, EOF never fires
                self._procs.append(p)
                self._conns.append(parent)

            for i, (conn, proc) in enumerate(zip(self._conns, self._procs)):
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                ready = mp_connection.wait([conn, proc.sentinel], timeout=timeout)
                if not ready:
                    raise WorkerStartTimeout(
                        f"lns worker {i} did not become ready before the wall-clock budget expired")
                if conn not in ready:
                    raise WorkerLost(
                        f"lns worker {i} died before reporting ready (exit {proc.exitcode})")
                try:
                    msg = conn.recv()
                except EOFError as exc:
                    raise WorkerLost(f"lns worker {i} closed its pipe before reporting ready") from exc
                if len(msg) != 3 or msg[0] != "ready":
                    raise WorkerLost(f"lns worker {i} sent an invalid startup message {msg!r}")
                _, pid, err = msg
                if err:
                    raise WorkerLost(f"lns worker {i} failed to build its replica:\n{err}")
                log.debug("lns worker %d ready (pid %d)", i, pid)

            self.worker_version = [0] * self.n_workers
            self.spawn_s = time.monotonic() - t0
            log.info("lns: %d workers up in %.1fs", self.n_workers, self.spawn_s)
            return self
        except BaseException:
            self.spawn_s = time.monotonic() - t0
            self.close()
            raise

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("stop",))
            except (OSError, BrokenPipeError, ValueError):
                pass
        deadline = time.monotonic() + _GRACEFUL_STOP_S      # ONE window for all of them
        for p in self._procs:
            try:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            except (AssertionError, OSError, ValueError):
                pass
        for p in self._procs:
            try:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
            except (AssertionError, OSError, ValueError):
                pass
        for conn in self._conns:
            try:
                conn.close()
            except (OSError, ValueError):
                pass
        self._procs, self._conns, self.worker_version = [], [], []

    def __enter__(self) -> "LNSWorkerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- traffic
    def sync(self, worker: int, changelog: _Changelog) -> None:
        """Bring one worker up to the current incumbent version, if it is behind."""
        base = self.worker_version[worker]
        if base == changelog.version:
            return
        diff = changelog.diff_since(base)
        self._conns[worker].send(("sync", changelog.version, diff))
        self.worker_version[worker] = changelog.version

    def sync_all(self, changelog: _Changelog) -> None:
        for w in range(self.n_workers):
            self.sync(w, changelog)

    def dispatch(self, worker: int, rnd: int, slot: int, op: str, seed_fid, rng_state) -> None:
        self._conns[worker].send(("task", rnd, slot, op, seed_fid, rng_state))

    def collect(self, n: int, timeout: float | None = None) -> list[TaskResult]:
        """Wait for ``n`` results. Waits on every worker's SENTINEL as well as its pipe, so a
        worker that dies surfaces as ``WorkerLost`` instead of an indefinite block."""
        sentinels = {p.sentinel: i for i, p in enumerate(self._procs)}
        by_conn = {c: i for i, c in enumerate(self._conns)}
        out: list[TaskResult] = []
        while len(out) < n:
            ready = mp_connection.wait(list(self._conns) + list(sentinels), timeout=timeout)
            if not ready:
                raise WorkerLost(f"lns pool timed out with {len(out)}/{n} results")
            for obj in ready:
                if obj in sentinels:
                    i = sentinels[obj]
                    raise WorkerLost(
                        f"lns worker {i} died (exit {self._procs[i].exitcode}) with "
                        f"{len(out)}/{n} results in — an OOM kill looks exactly like this")
                try:
                    msg = obj.recv()
                except EOFError as exc:
                    raise WorkerLost(f"lns worker {by_conn[obj]} closed its pipe") from exc
                (_, rnd, slot, worker, base_version, op, victims, new_intents,
                 cost_old, cost_new, envelopes, reason) = msg
                out.append(TaskResult(rnd, slot, worker, base_version, op, victims,
                                      new_intents, cost_old, cost_new, envelopes, reason))
                if len(out) >= n:
                    break
        return out


# ====================================================================== coordinator
def _pick_task(state, lns, selector, tabu, i):
    """Choose the operator and (for ``agent``) the seed for global task index ``i``.

    Returns ``(op, seed_fid, rng_state)``. Two things here are load-bearing:

    * The seed is chosen by the COORDINATOR. ``tabu`` is a serial recurrence that
      ``_select_most_delayed`` mutates, so m workers with private copies would every one of them
      pick the same most-delayed flight — the pool would run m duplicates of one neighborhood and
      look perfectly healthy doing it.
    * What travels is the generator STATE, not the seed. ``AdaptiveSelector.pick`` consumes one
      draw from this very stream before the destroy operator reads it, so a worker re-seeding from
      ``(seed, i)`` would start a draw earlier — enough to change every victim set.
    """
    rng = np.random.default_rng(np.random.SeedSequence([lns.seed, i]))
    if lns.adaptive:
        op = selector.pick(rng)
    else:
        op = lns.operators[int(rng.integers(len(lns.operators)))]
    seed_fid = _select_most_delayed(state, tabu) if op == "agent" else None
    return op, seed_fid, rng.bit_generator.state


def _out_of_budget(lns, t0) -> bool:
    return lns.time_limit_s is not None and time.monotonic() - t0 > lns.time_limit_s


def _stale_overwrite(state, changelog, result, accept_epsilon):
    """Return ``(combined_delta, net_gain)`` when a stale whole solution still wins.

    If the worker improved its base by R and intervening commits improved it by S, replacing the
    incumbent with the worker's solution realizes R-S. The combined delta makes that replacement
    one ledger transaction even when the same victim changed in both solutions.
    """
    reverts = changelog.revert_to(result.base_version)
    intervening_gain = sum(
        float(intent.cost) - float(state.incumbent[fid].cost)
        for fid, intent in reverts.items()
    )
    net_gain = result.improvement - intervening_gain
    if net_gain <= max(0.0, float(accept_epsilon)):
        return None, net_gain
    return {**reverts, **result.new_intents}, net_gain


def _loop_sync(state, pool, lns, selector, tabu, changelog, t0, trajectory, cost_before):
    """SYNC-LNS: a barrier per round, then apply the single best of m results (paper Eq. 3).

    Deterministic — the slot order is fixed and every decision is a pure function of it — which is
    keeps task selection deterministic for a fixed worker count and seed.

    Note the shape this gives at a FIXED ITERATION budget: m workers consume m tasks per round and
    m-1 of them are discarded, so quality per iteration FALLS as m rises. That is the paper's own
    SYNC behaviour, not a bug — SYNC buys wall clock, not iterations, and the paper reports quality
    against a fixed time budget for exactly this reason. DROP is the mode that converts throughput
    into accepted improvements.
    """
    m = pool.n_workers
    n_iter = n_accepted = n_not_selected = 0
    rnd = 0
    while rnd < lns.max_iterations:
        if _out_of_budget(lns, t0):
            break
        n_slots = min(m, lns.max_iterations - rnd)
        pool.sync_all(changelog)
        changelog.trim(min(pool.worker_version))
        for slot in range(n_slots):
            op, seed_fid, rng_state = _pick_task(state, lns, selector, tabu, rnd + slot)
            pool.dispatch(slot, rnd, slot, op, seed_fid, rng_state)

        results = sorted(pool.collect(n_slots), key=lambda r: r.slot)
        n_iter += n_slots

        winner = None
        for r in results:
            if r.improved and (winner is None or r.improvement > winner.improvement):
                winner = r
        incumbent_cursor = state.total_cost
        if winner is not None:
            olds = {f: state.incumbent[f] for f in winner.new_intents}
            state.apply_delta(winner.new_intents)
            changelog.record(winner.new_intents, olds)
            n_accepted += 1

        for r in results:
            is_winner = r is winner
            # Only the APPLIED result earns credit; everything else decays — which is exactly what
            # the sequential loop does when it feeds `out.improvement == 0.0` on a rejection.
            if lns.adaptive:
                selector.update(r.op, r.improvement if is_winner else 0.0)
            reason = r.reason
            if r.improved and not is_winner:
                reason = "not_selected"
                n_not_selected += 1
            incumbent_after = state.total_cost if is_winner else incumbent_cursor
            trajectory.append(_trajectory_row(
                r.rnd + r.slot, r.op, r.victims, is_winner, reason,
                r.cost_old, r.cost_new, incumbent_after, time.monotonic() - t0,
                incumbent_before=incumbent_cursor,
            ))
            incumbent_cursor = incumbent_after

        _maybe_verify(state, lns, n_accepted, winner is not None)
        _maybe_log(
            lns, "sync", m, n_iter, n_accepted, state, selector, cost_before,
            previous_iter=n_iter - n_slots,
        )
        rnd += n_slots
    return {"n_iter": n_iter, "n_accepted": n_accepted, "n_not_selected": n_not_selected,
            "n_dirty": 0, "n_overwrite": 0}


def _loop_drop(state, pool, lns, selector, tabu, changelog, t0, trajectory, cost_before):
    """DROP-LNS: no barrier. A result is applied the moment it lands and its worker is
    re-dispatched, so no worker ever idles on another (paper Fig. 1a).

    Results now arrive against a STALE ``base_version``, and the accept rule gains three cases
    beyond the sequential one:

    * **clean** — nothing committed since the worker's base touched its read set, so the plan is
      exactly the plan it would have produced against the current incumbent. MERGE it, keeping both
      workers' improvements. The paper has no such case; it always discards or overwrites. Soundness
      is Track A's argument: a repaired path that occupies (cell, step) must have READ it to confirm
      it free, so it is inside the recorded envelope, and a non-intersecting commit cannot conflict.
    * **overwrite** — dirty, but the worker's whole solution still beats the incumbent
      (paper Alg. 2 line 23). Walk the interleaved flights back to the worker's base and apply its
      repair; the result is exactly the worker's own replica, hence feasible by construction.
    * **discard** — dirty and not better.

    Nondeterministic by construction: which of these fires depends on completion order, i.e. on the
    wall clock. Rows carry ``base_version`` so a run stays auditable.
    """
    m = pool.n_workers
    n_iter = n_accepted = n_clean = n_dirty = n_overwrite = 0
    n_stale_victims = n_stale_cost = 0
    next_i = 0
    inflight: set[int] = set()

    def _dispatch(w: int) -> bool:
        nonlocal next_i
        if next_i >= lns.max_iterations or _out_of_budget(lns, t0):
            return False
        i = next_i
        next_i += 1
        op, seed_fid, rng_state = _pick_task(state, lns, selector, tabu, i)
        pool.sync(w, changelog)          # bring it current, THEN give it work
        pool.dispatch(w, i, 0, op, seed_fid, rng_state)
        inflight.add(w)
        return True

    for w in range(m):
        _dispatch(w)

    while inflight:
        r = pool.collect(1)[0]
        inflight.discard(r.worker)
        n_iter += 1
        incumbent_before = state.total_cost
        applied, reason = False, r.reason
        changes: dict = {}
        realized_improvement = 0.0

        if r.improved:
            stale_by = changelog.version - r.base_version
            if stale_by == 0:
                applied = True
                changes = r.new_intents
                realized_improvement = r.improvement
            else:
                victims_changed = bool(
                    set(r.victims) & changelog.touched_since(r.base_version))
                read_set_clean = not victims_changed and _read_set_is_clean(
                    r.envelopes, changelog.boxes_since(r.base_version))
                if read_set_clean:
                    applied, reason = True, "improved"
                    changes = r.new_intents
                    realized_improvement = r.improvement
                    n_clean += 1
                else:
                    if not victims_changed:
                        n_dirty += 1
                    overwrite, net_gain = _stale_overwrite(
                        state, changelog, r, lns.accept_epsilon)
                    if overwrite is not None:
                        applied, reason = True, "overwrite"
                        changes = overwrite
                        realized_improvement = net_gain
                        n_overwrite += 1
                    elif victims_changed:
                        reason = "stale_victims"
                        n_stale_victims += 1
                    else:
                        reason = "stale"
                        n_stale_cost += 1

        if applied:
            olds = {f: state.incumbent[f] for f in changes}
            state.apply_delta(changes)
            changelog.record(changes, olds)
            n_accepted += 1

        if lns.adaptive:
            selector.update(r.op, realized_improvement if applied else 0.0)
        row = _trajectory_row(
            n_iter - 1, r.op, r.victims, applied, reason,
            r.cost_old, r.cost_new, state.total_cost, time.monotonic() - t0,
            incumbent_before=incumbent_before,
            audit={
                "dispatch_iter": r.rnd,
                "base_version": r.base_version,
                "worker": r.worker,
            },
        )
        trajectory.append(row)

        _maybe_verify(state, lns, n_accepted, applied)
        _maybe_log(
            lns, "drop", m, n_iter, n_accepted, state, selector, cost_before,
            previous_iter=n_iter - 1,
        )
        changelog.trim(min(pool.worker_version) if pool.worker_version else 0)
        _dispatch(r.worker)

    return {"n_iter": n_iter, "n_accepted": n_accepted, "n_not_selected": 0,
            # Split deliberately: "the victims overlapped" and "the read set was dirty" are
            # different phenomena with different fixes. Victim overlap says the neighborhoods
            # collided (a seed-diversity problem); a dirty read set says the repairs collided in
            # SPACE (the envelope test's own resolution). Collapsing them hides which one binds.
            "n_stale_victims": n_stale_victims, "n_stale_cost": n_stale_cost,
            "n_dirty": n_dirty, "n_overwrite": n_overwrite, "n_clean_merge": n_clean}


def _finish_stats(stats: dict, n_iter: int) -> dict:
    """Derive the rates the modes must be judged on.

    ``dirty_rate`` is first-class because it is the design's main empirical risk: the destroy
    heuristics are contention-seeking by construction, so neighborhoods may cluster at the same hub
    mouths and most stale results would then fail the read-set test. If it approaches 1, DROP
    degrades to the paper's own behaviour — correct, just less parallel-efficient.
    """
    out = dict(stats)
    n = max(1, n_iter)
    stale_eligible = stats.get("n_clean_merge", 0) + stats.get("n_dirty", 0)
    out["dirty_rate"] = (stats.get("n_dirty", 0) / stale_eligible) if stale_eligible else 0.0
    out["discard_rate"] = (stats.get("n_stale_victims", 0) + stats.get("n_stale_cost", 0)) / n
    out["accept_rate"] = stats.get("n_accepted", 0) / n
    return out


def _read_set_is_clean(envelopes, boxes) -> bool:
    """Did anything committed since the worker's base touch what its repair READ?

    A ``None`` envelope is ALWAYS dirty: the planner resets ``last_envelope`` per plan and only
    ``_mk_envelope`` sets it, so None means "read set unknown", not "read nothing".
    """
    from freespace_sim.parallel import envelope_intersects

    if not boxes:
        return True
    for env in envelopes:
        if env is None or envelope_intersects(env, boxes):
            return False
    return True


def _maybe_verify(state, lns, n_accepted, just_applied) -> None:
    """Independent conflict replay. Only the coordinator can do this — it needs the whole intent
    list, and no worker holds the blessed incumbent."""
    if not (just_applied and lns.verify_every and n_accepted % lns.verify_every == 0):
        return
    assert_incumbent_ok(state)


def _maybe_log(
    lns, mode, m, n_iter, n_accepted, state, selector, cost_before, *, previous_iter
) -> None:
    if (not lns.log_every
            or n_iter // lns.log_every == previous_iter // lns.log_every):
        return
    log.info("lns[%s x%d] %d/%d: cost %.0f (%.2f%% below start), %d accepted, weights %s",
             mode, m, n_iter, lns.max_iterations, state.total_cost,
             100.0 * (cost_before - state.total_cost) / max(1e-9, cost_before),
             n_accepted, {k: round(v, 3) for k, v in selector.weights.items()})


def _run_and_close_pool(pool, *, deadline, execute):
    """Start a pool, execute its loop, and close every replica before returning.

    Startup timeout is the one non-fatal pool outcome: the configured wall budget was consumed
    before search began, so the valid incumbent is returned with zero started workers. Every other
    startup/loop failure propagates after the same teardown.
    """
    try:
        try:
            pool.start(deadline=deadline)
        except WorkerStartTimeout:
            log.info("lns: wall-clock budget expired while starting search workers")
            return None, pool.spawn_s, 0
        return execute(pool), pool.spawn_s, pool.n_workers
    finally:
        pool.close()


def run_lns_parallel(
    cfg: SimConfig,
    ledger: ReservationLedger,
    intents: list[OperationalIntent],
    lns,
    *,
    static_terms: tuple | None = None,
    turnaround_s: float | None = None,
):
    """DROP-LNS over a committed schedule. Same contract as ``run_lns``; see that docstring.

    ``sync`` runs a barrier per round and applies the best of m results (paper Eq. 3). An effective
    width below two delegates to the sequential engine instead of building a replica that cannot
    run concurrently.
    """
    lns = _validate_lns_config(lns)
    pool_workers = _effective_search_workers(lns)
    if pool_workers <= 1:
        return run_lns(
            cfg, ledger, intents, replace(lns, search_workers=1),
            static_terms=static_terms, turnaround_s=turnaround_s,
        )

    t0 = time.monotonic()
    state = _build_lns_state(
        cfg, ledger, intents, lns,
        static_terms=static_terms, turnaround_s=turnaround_s,
        maintain_claim_index=False,
    )
    static_terms = state.static_terms
    init_s = time.monotonic() - t0
    cost_before = state.total_cost

    spec = WorkerSpec(
        neighborhood_size=lns.neighborhood_size,
        accept_epsilon=lns.accept_epsilon,
        repair_order=lns.repair_order,
        max_walks=lns.max_walks,
        map_max_cells=lns.map_max_cells,
        turnaround_s=turnaround_s,
        frozen_flight_ids=lns.frozen_flight_ids,
        movable_uss_ids=lns.movable_uss_ids,
        incremental_release=lns.incremental_release,
        kernel_log2_min=lns.worker_kernel_log2,
        pair_closed_neighborhood=lns.pair_closed_neighborhood,
        record_envelope=lns.parallel_mode == "drop" and pool_workers > 1,
        window_bytes=lns.window_bytes,
        repair_planner=lns.repair_planner,
    )

    selector = AdaptiveSelector(tuple(lns.operators), lns.gamma)
    tabu: set[int] = set()
    changelog = _Changelog()
    trajectory: list[dict] = []
    n_accepted = 0
    n_iter = 0
    stats: dict = {
        "n_iter": 0, "n_accepted": 0, "n_not_selected": 0,
        "n_stale_victims": 0, "n_stale_cost": 0,
        "n_dirty": 0, "n_overwrite": 0, "n_clean_merge": 0,
    }
    pool_spawn_s = 0.0
    started_workers = 0
    try:
        if not _out_of_budget(lns, t0):
            pool = LNSWorkerPool(
                cfg, state.final_intents(), static_terms,
                dict(state._unimp_cost), spec, pool_workers,
            )
            deadline = (
                None if lns.time_limit_s is None or math.isinf(lns.time_limit_s)
                else t0 + lns.time_limit_s
            )
            loop = _loop_sync if lns.parallel_mode == "sync" else _loop_drop
            completed, pool_spawn_s, started_workers = _run_and_close_pool(
                pool,
                deadline=deadline,
                execute=lambda active: loop(
                    state, active, lns, selector, tabu, changelog,
                    t0, trajectory, cost_before,
                ),
            )
            if completed is not None:
                stats = completed
                n_iter, n_accepted = stats["n_iter"], stats["n_accepted"]

        return _finalize_lns_result(
            state, trajectory, cost_before, n_iter, n_accepted, t0, init_s, selector,
            repair_planner_name=lns.repair_planner,
            search_workers=started_workers,
            parallel_mode=lns.parallel_mode,
            pool_spawn_s=pool_spawn_s,
            parallel_stats=_finish_stats(stats, n_iter),
        )
    except BaseException:
        log.exception("lns aborted; detaching repair-planner subscribers before propagating")
        raise
    finally:
        ledger.detach_subscribers()
