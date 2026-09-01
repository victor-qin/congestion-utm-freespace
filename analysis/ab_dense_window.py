"""Paired parity/throughput harness for final compiled A* and its independent reference oracle.

The pool-less compiled planner cannot run with its dense window disabled: the window IS the dynamic
occupancy image. The old interval-pool off arm therefore no longer exists. This harness now compares
the pure-Python reference search with the final compiled-window search on real plans against a real
committed schedule. It measures end-to-end compiled acceleration, not the window's isolated speedup;
the value of retaining it is the paired identity gate under realistic single- and multi-process load.

Method — the two things that make this comparison honest:

* **Paired and interleaved.** Both arms replan the SAME flights against the SAME ledger, in
  alternating passes (A B A B ...), and the reported time is each arm's median pass. A box shared
  with other work drifts over minutes; alternating passes cancel the drift instead of assigning it
  to whichever arm ran second.
* **Identity checked, not assumed.** Every plan's accepted flag, cost, volumes and node-expansion
  count are compared between arms. A speedup that changed an answer is not a speedup, and
  expansions are the sharper test — two different searches almost never expand the same count.

The ledger is a finished FCFS schedule, so every plan sees a full-density world (this is the LNS
repair's situation, not the FCFS run's early flights). Pass ``--cache`` to reuse the baseline
pickle the LNS probes build, which is what makes running this at ``density_faa`` scale affordable.

``--procs N`` runs the identical A/B in N concurrent processes behind a barrier, which is the
measurement that matters for parallel LNS: ``search_workers=8`` puts eight replicas on cores that
share 12 MB of L2, and ``_packed`` measured that regime rewarding a cache-footprint fix far more
than a solo run did (2.5x solo, 3.1x at 8 processes). Every process does the same work, so the
  compiled speedup at N procs against its speedup at 1 is the contention-specific part of the win.

Usage:
    uv run python analysis/ab_dense_window.py --scenario metro_2uss --demand-duration 600
    uv run python analysis/ab_dense_window.py --scenario density_faa_wing_zipline \\
        --demand-duration 1800 --cache /tmp/faa_full_baseline.pkl --flights 120
    uv run python analysis/ab_dense_window.py --cache /tmp/faa_full_baseline.pkl \\
        --scenario density_faa_wing_zipline --demand-duration 1800 --flights 60 --procs 1,4,8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import statistics
import time
import warnings

import numpy as np

from freespace_sim import sim
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.scenarios import get_scenario
from freespace_sim.scenarios.spec import with_overrides


def _rebuild(cfg, intents, static_terms) -> ReservationLedger:
    """A ledger holding exactly the accepted intents' own ``Volume4D`` objects — the recipe
    ``LNSState.replica`` and ``verify`` use."""
    led = ReservationLedger(cfg)
    for center, term in static_terms:
        led.register_static_terminal(center, term)
    for it in intents:
        if it.accepted and it.volumes:
            led.commit(it.request.flight_id, it.volumes)
    return led


def _sig(it) -> tuple:
    """Everything the compiled path must reproduce from the independent reference."""
    if not it.accepted:
        return (False, it.denial_reason if hasattr(it, "denial_reason") else None)
    return (True, round(float(it.cost), 12),
            tuple((v.flat_aabb(), round(float(v.t_start), 12), round(float(v.t_end), 12),
                   v.terminal_id) for v in it.volumes))


def _pass(planner, requests, ledger, cfg):
    t = time.perf_counter()
    rows = [(_sig(planner.plan(r, ledger, cfg)), planner.last_expansions) for r in requests]
    return time.perf_counter() - t, rows


def _make_arms(window_bytes, kernel_log2):
    arms = {
        "reference": AStarPlanner(compiled=False),
        "compiled": AStarPlanner(window_bytes=window_bytes, kernel_log2_min=kernel_log2),
    }
    # A pass ends on the latest request and the next pass restarts at the earliest one. Occupancy,
    # pad-capacity, and terminal-capacity eviction are monotone, so without a floor later passes
    # silently plan against a partially-evicted schedule instead of repeating the same workload.
    for planner in arms.values():
        planner.evict_floor = 0.0
    return arms


def _paired_pass(arms, requests, ledger, cfg, *, before_arm=None):
    """Time both arms and reject the pass unless every plan is identical."""
    elapsed = {}
    rows = {}
    for name, planner in arms.items():
        if before_arm is not None:
            before_arm()
        elapsed[name], rows[name] = _pass(planner, requests, ledger, cfg)

    reference = rows["reference"]
    compiled = rows["compiled"]
    if reference != compiled:
        n_plans = max(len(reference), len(compiled))
        n_diff = sum(
            i >= len(reference) or i >= len(compiled) or reference[i] != compiled[i]
            for i in range(n_plans)
        )
        raise RuntimeError(
            f"DIVERGENCE: compiled A* changed {n_diff} of {n_plans} plans; timing is invalid"
        )
    return elapsed


def _timed_passes(arms, requests, ledger, cfg, passes, *, before_arm=None):
    """Run paired passes and return each arm's median, synchronizing before each arm if requested."""
    samples = {name: [] for name in arms}
    for _ in range(passes):
        elapsed = _paired_pass(
            arms, requests, ledger, cfg, before_arm=before_arm,
        )
        for name, duration in elapsed.items():
            samples[name].append(duration)
    return {name: statistics.median(durations) for name, durations in samples.items()}


_BARRIER = None


def _init(barrier):
    global _BARRIER
    _BARRIER = barrier


def _child(job):
    """One concurrent A/B. Unpickling, rebuilding, and warming happen before the first timed-arm
    barrier; every later arm has its own barrier so all processes measure the same contention."""
    args, _rank = job
    warnings.filterwarnings("ignore")
    spec = with_overrides(get_scenario(args.scenario),
                          demand_duration_s=args.demand_duration, horizon_s=args.horizon)
    cfg = spec.config()
    with open(args.cache, "rb") as fh:              # self-produced cache; see the LNS probes
        intents, static_terms = pickle.load(fh)
    ledger = _rebuild(cfg, intents, static_terms)
    requests = [it.request for it in intents if it.accepted]
    if args.flights:
        requests = requests[:args.flights]
    arms = _make_arms(args.window_bytes, args.kernel_log2)
    for p in arms.values():
        _pass(p, requests[:1], ledger, cfg)
    # A barrier before EVERY arm prevents faster children from starting a later arm after slower
    # siblings have fallen behind or exited. Medians match the solo harness's stated methodology.
    return _timed_passes(
        arms, requests, ledger, cfg, args.passes, before_arm=_BARRIER.wait,
    )


def fanout(args, nprocs):
    """Run the A/B in ``nprocs`` concurrent processes; return each one's per-arm median pass."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(nprocs)
    with ctx.Pool(nprocs, initializer=_init, initargs=(barrier,)) as pool:
        return pool.map(_child, [(args, r) for r in range(nprocs)], chunksize=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="metro_2uss")
    ap.add_argument("--demand-duration", type=float, default=600.0)
    ap.add_argument("--horizon", type=float, default=7200.0)
    ap.add_argument("--cache", default=None, help="baseline pickle (intents, static_terms) to reuse")
    ap.add_argument("--flights", type=int, default=0, help="replan only the first N (0 = all)")
    ap.add_argument("--passes", type=int, default=3, help="alternating passes per arm")
    ap.add_argument("--window-bytes", type=int, default=2 << 20)
    ap.add_argument("--kernel-log2", type=int, default=None,
                    help="AStarPlanner.kernel_log2_min — set to the LNS worker's value (e.g. 18) to "
                         "measure under the cache pressure a parallel worker actually runs at")
    ap.add_argument("--procs", default=None,
                    help="comma-separated process counts to fan the A/B out over (needs --cache)")
    args = ap.parse_args()

    if args.procs:
        if not args.cache or not os.path.exists(args.cache):
            raise SystemExit("--procs needs a prebuilt --cache: a child must not replan the baseline")
        print(f"{'procs':>5} {'ref s':>8} {'jit s':>8} {'speedup':>8}   (median over processes)")
        print("-" * 42)
        for n in [int(x) for x in args.procs.split(",")]:
            rows = fanout(args, n)
            reference = statistics.median(r["reference"] for r in rows)
            compiled = statistics.median(r["compiled"] for r in rows)
            print(f"{n:>5} {reference:>8.2f} {compiled:>8.2f} "
                  f"{reference / compiled:>7.3f}x", flush=True)
        return

    spec = with_overrides(get_scenario(args.scenario),
                          demand_duration_s=args.demand_duration, horizon_s=args.horizon)
    cfg = spec.config()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if args.cache and os.path.exists(args.cache):
            with open(args.cache, "rb") as fh:          # self-produced cache; see the LNS probes
                intents, static_terms = pickle.load(fh)
            print(f"baseline from {args.cache}: {len(intents)} legs", flush=True)
        else:
            t = time.monotonic()
            res = sim.run(cfg, demand=spec.demand_model(), planner_name="astar", progress=False)
            intents, static_terms = res.intents, res.ledger.static_terminals()
            print(f"baseline: {len(intents)} legs in {time.monotonic() - t:.0f}s", flush=True)
            if args.cache:
                with open(args.cache, "wb") as fh:
                    pickle.dump((intents, static_terms), fh, protocol=5)

        t = time.monotonic()
        ledger = _rebuild(cfg, intents, static_terms)
        print(f"ledger rebuilt: {ledger.n_volumes:,} volumes in {time.monotonic() - t:.0f}s",
              flush=True)

        requests = [it.request for it in intents if it.accepted]
        if args.flights:
            requests = requests[:args.flights]

        arms = _make_arms(args.window_bytes, args.kernel_log2)
        times = {k: [] for k in arms}
        # One untimed warm pass per arm: the first plan absorbs the whole ledger into that planner's
        # occupancy services, which is ledger-sized and has nothing to do with the window.
        for p in arms.values():
            _pass(p, requests[:1], ledger, cfg)
        for i in range(args.passes):
            elapsed = _paired_pass(arms, requests, ledger, cfg)
            for name, dt in elapsed.items():
                times[name].append(dt)
                print(f"  pass {i + 1} {name:>9}: {dt:8.2f}s  "
                      f"({1e3 * dt / len(requests):6.1f} ms/plan)",
                      flush=True)

        med = {k: statistics.median(v) for k, v in times.items()}
        p_compiled = arms["compiled"]
        st = (p_compiled._ks["win_stats"] if p_compiled._ks is not None
              else np.zeros(2, np.int64))
        print(f"\n{len(requests)} plans, {args.passes} passes, "
              f"scenario={args.scenario} @ {args.demand_duration:.0f}s demand")
        print(f"  reference  : {med['reference']:8.2f}s   "
              f"({1e3 * med['reference'] / len(requests):6.2f} ms/plan)")
        print(f"  compiled   : {med['compiled']:8.2f}s   "
              f"({1e3 * med['compiled'] / len(requests):6.2f} ms/plan)")
        print(f"  SPEEDUP    : {med['reference'] / med['compiled']:8.3f}x")
        print(f"  identical  : True   (0 of {len(requests)} plans differ)")
        print(f"  window     : hit {st[0]:,} / miss {st[1]:,} "
              f"({st[0] / max(1, st[0] + st[1]):.3%} hit), "
              f"peak {p_compiled._win_bytes_peak / 1e3:,.0f} kB, "
              f"{p_compiled._win_off} plans with no window, "
              f"{p_compiled._win_painted:,} cells painted")
        arena = p_compiled._cocc._arena
        live_keys = int(np.count_nonzero(arena.length))
        print(f"  arena      : {arena.n_claims:,} claims in {live_keys:,} slabs, "
              f"tail {int(arena.tail[0]):,}, garbage {int(arena.garbage[0]):,}, "
              f"{arena.nbytes() / 1e6:.1f} MB allocated")


if __name__ == "__main__":
    main()
