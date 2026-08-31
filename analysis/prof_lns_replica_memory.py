"""Stage 0 gate for parallel LNS: what does a worker's private replica actually cost?

A DROP-LNS worker holds a full private ``LNSState`` — its own ``ReservationLedger``, its own
``AStarPlanner`` with the occupancy/terminal-capacity services, and its own claim index. That is
the design's whole premise (a delta-sync is O(changed) only because the replica is persistent),
and it is also its whole risk: memory is LINEAR in workers. The colgen pricing pool went
3.9 -> 12.5 GB at 4 workers -> 22.7 GB at 8 and is *still* defaulted off for exactly this reason
(``tests/test_experiment_run.py`` pins ``n_pricing_workers == 0``).

This script measures the real thing rather than estimating it: it spawns m workers, each of which
builds a replica of the same schedule, runs one representative repair to materialize lazy planner
state, and then sits still while the parent reads the summed RSS of the whole process tree.

**It reuses ``sweep_pricing_workers._tree_rss_mib`` deliberately.**
``getrusage(RUSAGE_CHILDREN).ru_maxrss`` is the largest SINGLE child by POSIX definition, never
the sum, so it reads flat however many workers run — that artifact is what once made a
linearly-scaling pool look free.

Usage (guard the main: the repo is spawn-only, so module-level work would die in the child):

    uv run python analysis/prof_lns_replica_memory.py --demand-duration 120 --horizon 1500
    uv run python analysis/prof_lns_replica_memory.py --demand-duration 300 --horizon 1500 \
        --workers 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freespace_sim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:      # a harness that measures the wrong tree measures nothing
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.lns.state import LNSState  # noqa: E402

# Everything below is PARENT-ONLY and is imported lazily inside the functions that use it.
# `spawn` re-imports this module in every child, so a module-level `from analysis.
# sweep_pricing_workers import ...` would drag the colgen solver into all m workers and inflate
# the very number this script exists to measure. The child needs LNSState and nothing else.


def _warm_replica(state: LNSState) -> None:
    """Materialize the planner state retained by a live search worker."""
    victims = state.movable_ids()[:8]
    if victims:
        state.try_repair(
            victims, state.rng, math.inf, order_mode="premium", report_only=True,
        )


def _worker_main(conn, cfg, intents, static_terms, unimp, kernel_log2):
    """Build one replica, report, then hold it alive until told to stop.

    Holding it is the point: the parent's measurement is only meaningful while every replica is
    resident, which is exactly the steady state of a running pool.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ReservationLedger shrank")
            t0 = time.monotonic()
            state = LNSState.replica(cfg, intents, static_terms=static_terms,
                                     unimpeded_cost=unimp, kernel_log2_min=kernel_log2)
            _warm_replica(state)
            build_s = time.monotonic() - t0
        conn.send(("ready", os.getpid(), build_s, state.ledger.n_volumes, None))
    except BaseException as exc:                                   # noqa: BLE001 - reported home
        conn.send(("ready", os.getpid(), 0.0, 0, f"{type(exc).__name__}: {exc}"))
        return
    while True:
        if conn.recv()[0] == "stop":
            return


def _measure(n_workers, cfg, intents, static_terms, unimp, kernel_log2):
    """Spawn n replicas, wait for all of them, and read the tree's RSS at full occupancy."""
    from analysis.sweep_pricing_workers import _tree_rss_mib   # parent-only (see imports note)

    ctx = mp.get_context("spawn")          # never fork: it inherits the numba runtime + threads
    chans, procs = [], []
    t0 = time.monotonic()
    for _ in range(n_workers):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_worker_main,
                        args=(child, cfg, intents, static_terms, unimp, kernel_log2),
                        daemon=True)
        p.start()
        child.close()                      # else EOF never fires and a dead worker is invisible
        chans.append(parent)
        procs.append(p)
    builds, errors = [], []
    for c in chans:
        _, _pid, build_s, n_vol, err = c.recv()
        builds.append(build_s)
        if err:
            errors.append(err)
    spawn_s = time.monotonic() - t0
    rss = _tree_rss_mib()                  # every replica resident: the steady state of a pool
    for c in chans:
        c.send(("stop",))
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    return {
        "n_workers": n_workers,
        "tree_rss_mib": round(rss, 1),
        "spawn_and_build_s": round(spawn_s, 2),
        "replica_build_s_max": round(max(builds), 2) if builds else 0.0,
        "errors": errors,
    }


def main() -> None:
    from analysis.sweep_pricing_workers import _tree_rss_mib   # parent-only (see imports note)
    from freespace_sim import sim
    from freespace_sim.scenarios import get_scenario
    from freespace_sim.scenarios.spec import with_overrides

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--demand-duration", type=float, default=120.0)
    ap.add_argument("--horizon", type=float, default=1500.0)
    ap.add_argument("--workers", default="1,2,4,8",
                    help="comma-separated replica counts to measure")
    ap.add_argument("--kernel-log2", type=int, default=None,
                    help="AStarPlanner.kernel_log2_min for the replicas. The ruler pass measured "
                         "473 -> 214 MB/worker at 18; oversized kernel arrays were also measured "
                         "to slow CONCURRENT plans ~1.75x at 8 workers (astar.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    spec = with_overrides(spec, demand_duration_s=args.demand_duration, horizon_s=args.horizon)
    cfg, demand = spec.config(), spec.demand_model()

    t0 = time.monotonic()
    res = sim.run(cfg, demand=demand, planner_name="astar", progress=False)
    base_wall = time.monotonic() - t0
    print(f"baseline: {len(res.intents)} legs, {len(res.accepted)} accepted, "
          f"{res.ledger.n_volumes} volumes, {base_wall:.0f}s", flush=True)

    rss_bare = _tree_rss_mib()
    t0 = time.monotonic()
    base = LNSState(cfg, res.ledger, res.intents, static_terms=res.ledger.static_terminals())
    init_s = time.monotonic() - t0
    rss_one = _tree_rss_mib()
    intents = base.final_intents()
    static_terms = base.static_terms
    unimp = dict(base._unimp_cost)         # the ruler result: computed once, broadcast to workers
    print(f"in-process LNSState: init {init_s:.1f}s, tree RSS {rss_bare:.0f} -> {rss_one:.0f} MiB "
          f"({len(base.movable_ids())} movable)", flush=True)

    rows = []
    for token in args.workers.split(","):
        m = int(token.strip())
        if m <= 0:
            continue
        row = _measure(m, cfg, intents, static_terms, unimp, args.kernel_log2)
        row["marginal_mib_per_worker"] = round((row["tree_rss_mib"] - rss_one) / m, 1)
        rows.append(row)
        print(f"  m={m:<3} tree RSS {row['tree_rss_mib']:>8.1f} MiB  "
              f"(+{row['marginal_mib_per_worker']:.0f}/worker)  "
              f"spawn+build {row['spawn_and_build_s']:.1f}s  "
              f"slowest replica {row['replica_build_s_max']:.1f}s"
              + (f"  ERRORS {row['errors']}" if row["errors"] else ""), flush=True)

    payload = {
        "scenario": args.scenario,
        "demand_duration_s": args.demand_duration,
        "horizon_s": args.horizon,
        "n_legs": len(res.intents),
        "n_volumes": res.ledger.n_volumes,
        "n_movable": len(base.movable_ids()),
        "kernel_log2": args.kernel_log2,
        "rss_bare_mib": round(rss_bare, 1),
        "rss_one_state_mib": round(rss_one, 1),
        "lns_init_s": round(init_s, 2),
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":       # REQUIRED: spawn re-imports this module in every child
    main()
