"""Paired A/B for the per-plan dense occupancy window (``planner.astar.window``).

The window is a pure cache of the interval pools ``kernel._blocked`` otherwise walks, so the only
questions are *is it faster* and *is it still byte-identical*. This measures both, on real plans
against a real committed schedule rather than a synthetic probe stream.

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
window's speedup at N procs against its speedup at 1 is the contention-specific part of the win.

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
    """Everything about a plan that a window could conceivably move."""
    if not it.accepted:
        return (False, it.denial_reason if hasattr(it, "denial_reason") else None)
    return (True, round(float(it.cost), 12),
            tuple((v.flat_aabb(), round(float(v.t_start), 12), round(float(v.t_end), 12),
                   v.terminal_id) for v in it.volumes))


def _pass(planner, requests, ledger, cfg):
    t = time.perf_counter()
    rows = [(_sig(planner.plan(r, ledger, cfg)), planner.last_expansions) for r in requests]
    return time.perf_counter() - t, rows


def _paired_pass(arms, requests, ledger, cfg):
    """Time both arms and reject the pass unless every plan is identical."""
    elapsed = {}
    rows = {}
    for name, planner in arms.items():
        elapsed[name], rows[name] = _pass(planner, requests, ledger, cfg)

    off = rows["off"]
    on = rows["on"]
    if off != on:
        n_plans = max(len(off), len(on))
        n_diff = sum(
            i >= len(off) or i >= len(on) or off[i] != on[i]
            for i in range(n_plans)
        )
        raise RuntimeError(
            f"DIVERGENCE: window changed {n_diff} of {n_plans} plans; timing is invalid"
        )
    return elapsed


_BARRIER = None


def _init(barrier):
    global _BARRIER
    _BARRIER = barrier


def _child(job):
    """One concurrent A/B. Everything before the barrier — unpickling the baseline, rebuilding the
    ledger, and the untimed warm pass that absorbs it into each planner — is setup whose cost must
    not land inside anyone's measurement, so the barrier sits after it."""
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
    arms = {"off": AStarPlanner(window_bytes=0, kernel_log2_min=args.kernel_log2),
            "on": AStarPlanner(window_bytes=args.window_bytes, kernel_log2_min=args.kernel_log2)}
    for p in arms.values():
        _pass(p, requests[:1], ledger, cfg)
    _BARRIER.wait()
    # Alternating passes, reported as each arm's BEST. Under a fan-out the processes drift out of
    # step as they finish, so late passes are progressively less contended; the minimum is the pass
    # that ran with the fan-out most fully in flight, which is the regime being measured.
    out = {name: float("inf") for name in arms}
    for _ in range(args.passes):
        elapsed = _paired_pass(arms, requests, ledger, cfg)
        for name, duration in elapsed.items():
            out[name] = min(out[name], duration)
    return out


def fanout(args, nprocs):
    """Run the A/B in ``nprocs`` concurrent processes; return each one's per-arm best pass."""
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
        print(f"{'procs':>5} {'off s':>8} {'on s':>8} {'speedup':>8}   (median over processes)")
        print("-" * 42)
        for n in [int(x) for x in args.procs.split(",")]:
            rows = fanout(args, n)
            off = statistics.median(r["off"] for r in rows)
            on = statistics.median(r["on"] for r in rows)
            print(f"{n:>5} {off:>8.2f} {on:>8.2f} {off / on:>7.3f}x", flush=True)
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

        arms = {"off": AStarPlanner(window_bytes=0, kernel_log2_min=args.kernel_log2),
                "on": AStarPlanner(window_bytes=args.window_bytes,
                                   kernel_log2_min=args.kernel_log2)}
        times = {k: [] for k in arms}
        rows = {}
        # One untimed warm pass per arm: the first plan absorbs the whole ledger into that planner's
        # occupancy services, which is ledger-sized and has nothing to do with the window.
        for name, p in arms.items():
            _pass(p, requests[:1], ledger, cfg)
        for i in range(args.passes):
            for name, p in arms.items():
                dt, r = _pass(p, requests, ledger, cfg)
                times[name].append(dt)
                rows[name] = r
                print(f"  pass {i + 1} {name:>3}: {dt:8.2f}s  ({1e3 * dt / len(requests):6.1f} ms/plan)",
                      flush=True)

        same = rows["off"] == rows["on"]
        n_diff = sum(1 for a, b in zip(rows["off"], rows["on"]) if a != b)
        med = {k: statistics.median(v) for k, v in times.items()}
        p_on = arms["on"]
        st = p_on._ks["win_stats"] if p_on._ks is not None else np.zeros(2, np.int64)
        print(f"\n{len(requests)} plans, {args.passes} passes, "
              f"scenario={args.scenario} @ {args.demand_duration:.0f}s demand")
        print(f"  window off : {med['off']:8.2f}s   ({1e3 * med['off'] / len(requests):6.2f} ms/plan)")
        print(f"  window on  : {med['on']:8.2f}s   ({1e3 * med['on'] / len(requests):6.2f} ms/plan)")
        print(f"  SPEEDUP    : {med['off'] / med['on']:8.3f}x")
        print(f"  identical  : {same}   ({n_diff} of {len(rows['off'])} plans differ)")
        print(f"  window     : hit {st[0]:,} / miss {st[1]:,} "
              f"({st[0] / max(1, st[0] + st[1]):.3%} hit), peak {p_on._win_bytes_peak / 1e3:,.0f} kB, "
              f"{p_on._win_off} plans with no window, {p_on._win_painted:,} cells painted")
        print(f"  pools      : corridor {arms['on']._cocc.corr.nslots:,} slots, "
              f"column {arms['on']._cocc.col.nslots:,} (NC={arms['on']._cocc.NC:,}) — "
              f"{(arms['on']._cocc.corr.iv.nbytes + arms['on']._cocc.col.iv.nbytes) / 1e6:.0f} MB")
        if not same:
            raise SystemExit("DIVERGENCE: the window changed a plan — this is a bug, not a result")


if __name__ == "__main__":
    main()
