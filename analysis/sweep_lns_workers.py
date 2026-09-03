"""Is parallel LNS worth using, and where does it start to be?

Sweeps ``search_workers`` x ``parallel_mode`` over one scenario cut and reports the paper's
effectiveness metrics (Chan et al. §5.2) beside the ones this world actually needs.

**Compare at matched WALL, not matched iterations.** A parallel arm given the same task count as
the sequential loop is not a fairer comparison, it is a worse one: SYNC discards m-1 of every m
results by construction, so at equal tasks it looks catastrophic for a reason that has nothing to
do with whether parallelism pays. ``--iters`` therefore takes a per-arm list, so each arm can be
given the tasks it needs to fill the same wall. ``--time-limit`` is the other way to do it and is
the one the paper uses (a fixed budget T), at the cost of reproducibility.

**The verdict depends on SCALE and on BUDGET — do not read it off the 120 s cut, and do not quote
a single number.** Improvement per second of loop wall, normalised to sequential:

    120 s cut (290 legs, 26k volumes)        FULL density_faa (4,636 legs, 427k volumes)
    sequential 2.46%   19.3 s 3.11 t/s 1.00x   sequential 2000 it 2.06% 2000.1 s 1.00 t/s 1.00x
    drop m=4   1.99%   15.1 s 7.97 t/s 1.03x   drop m=4   2000 it 1.56%  677.4 s 2.95 t/s 2.25x
    drop m=8   1.90%   19.7 s 9.14 t/s 0.76x   drop m=8   2000 it 1.14%  435.8 s 4.59 t/s 2.55x
    sync m=4   1.68%   21.9 s 5.49 t/s 0.60x   drop m=4   6000 it 2.64% 1721.3 s 3.49 t/s 1.49x
                                               drop m=8   6000 it 2.11% 1046.1 s 5.74 t/s 1.96x

Matched on QUALITY rather than task count: m=8 reaches the sequential 2000-task schedule in
**1.91x less wall** (2.11% in 1,046 s vs 2.06% in 2,000 s), and m=4 beats it outright (2.64% in
1,721 s). At 290 legs m=4 is break-even and m=8 LOSES; at 4,636 both win and the ordering reverses.
Per-task cost grows with the schedule (3.11 -> 1.00 tasks/s), giving the pool more to hide, and
staleness falls relative to it (discarded-stale 3.4% at full scale vs ~10% on the cut). The rate
also FALLS as the budget grows (m=8: 2.55x at 2000 tasks, 1.96x at 6000) because LNS has
diminishing returns. DROP is nondeterministic, so read arms across reruns. See lns_plan.md §8.

Usage (guard the main: the repo is spawn-only):

    uv run python analysis/sweep_lns_workers.py --demand-duration 120 --horizon 1500 \
        --arms "1:sync:60,4:drop:120,8:drop:180" --out /tmp/lns_sweep_120.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freespace_sim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:      # a harness that measures the wrong tree measures nothing
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from analysis.sweep_pricing_workers import _tree_rss_mib  # noqa: E402
from freespace_sim import sim, verify  # noqa: E402
from freespace_sim.planner.lns import LNSConfig, run_lns  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402
from freespace_sim.scenarios.spec import with_overrides  # noqa: E402


def _parse_arms(text: str):
    """``"1:sync:60,4:drop:120"`` -> [(1, "sync", 60), (4, "drop", 120)]."""
    arms = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad arm {token!r} — want workers:mode:iterations")
        arms.append((int(parts[0]), parts[1], int(parts[2])))
    return arms


def _relative_rate(rate: float, base_rate: float) -> float | None:
    """Return a speedup only when the sequential arm established a non-zero baseline."""
    return rate / base_rate if base_rate > 0.0 else None


def _worker_metadata(requested_workers: int, result) -> dict:
    """Keep the requested width while reporting the pool width that could actually run."""
    return {
        "requested_workers": int(requested_workers),
        "workers": int(result.search_workers),
        "mode": result.parallel_mode,
    }


def _sequential_baseline(rows: list[dict]) -> dict | None:
    """Find the explicitly requested in-process arm, not another arm capped into that engine."""
    return next((row for row in rows
                 if row["mode"] == "sequential" and row["requested_workers"] == 1), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--demand-duration", type=float, default=120.0)
    ap.add_argument("--horizon", type=float, default=1500.0)
    ap.add_argument("--arms", default="1:sync:60,2:drop:80,4:drop:120,8:drop:180,4:sync:120",
                    help="comma-separated workers:mode:iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neighborhood", type=int, default=8)
    ap.add_argument("--repair-planner", default="astar",
                    choices=["astar", "astar_ref", "sipp", "sipp_ref"],
                    help="planner that repairs a destroyed neighborhood; recorded in every row so a "
                         "worker sweep and a planner A/B cannot be confused for each other")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="wall budget per arm. The paper's way to make arms comparable (fixed T); "
                         "costs reproducibility, so iterations stay the default")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = with_overrides(get_scenario(args.scenario),
                          demand_duration_s=args.demand_duration, horizon_s=args.horizon)
    cfg, demand = spec.config(), spec.demand_model()

    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="astar compiled kernel FALLBACK")
        warnings.filterwarnings("ignore", message="ReservationLedger shrank")
        for i, (m, mode, iters) in enumerate(_parse_arms(args.arms)):
            # A fresh baseline per arm: LNSState consumes the (ledger, intents) pair in place, so
            # they cannot be reused across arms.
            res = sim.run(cfg, demand=demand, planner_name="astar", progress=False)
            if i == 0:
                print(f"baseline: {len(res.intents)} legs, {len(res.accepted)} accepted, "
                      f"{res.ledger.n_volumes} volumes", flush=True)
            static_terms = res.ledger.static_terminals()
            lns = LNSConfig(seed=args.seed, neighborhood_size=args.neighborhood, log_every=0,
                            repair_planner=args.repair_planner,
                            max_iterations=iters, search_workers=m, parallel_mode=mode,
                            time_limit_s=args.time_limit)
            out = run_lns(res.config, res.ledger, res.intents, lns)
            rss = _tree_rss_mib()          # after close(); peak lives in the run itself
            bad = verify.find_interflight_conflict(out.intents, res.config,
                                                   static_terminals=static_terms)
            loop_s = out.wall_s - out.init_wall_s
            st = dict(out.parallel_stats)
            row = {
                **_worker_metadata(m, out), "iterations": iters,
                # Recorded per row so a worker sweep and a planner A/B can never be confused for
                # each other after the fact. Off the RESULT, i.e. what actually repaired.
                "repair_planner": out.repair_planner,
                "n_accepted": out.n_accepted, "n_iterations": out.n_iterations,
                "cost_before": out.cost_before, "cost_after": out.cost_after,
                "improvement_pct": 100.0 * (out.cost_before - out.cost_after)
                / max(1e-9, out.cost_before),
                "loop_s": loop_s, "init_s": out.init_wall_s, "wall_s": out.wall_s,
                "pool_spawn_s": out.pool_spawn_s,
                "tasks_per_s": out.n_iterations / max(1e-9, loop_s),
                "auc": out.auc, "npo": out.npo,
                "verified": bool(out.verified), "replay_clean": bad is None,
                "tree_rss_mib": round(rss, 1),
                # The deny-rate every density readout must carry (context/lns_plan.md §5).
                "n_denied": sum(1 for it in out.intents if not it.accepted),
                **st,
            }
            rows.append(row)
            requested_note = (
                "" if row["workers"] == row["requested_workers"]
                else f" ({row['requested_workers']} requested)"
            )
            print(f"{row['mode']:<10} m={row['workers']}{requested_note} "
                  f"it={iters:>4}  {out.n_accepted:>3} acc  "
                  f"{out.cost_after:.0f} ({row['improvement_pct']:5.2f}%)  "
                  f"loop {loop_s:6.1f}s  {row['tasks_per_s']:5.2f} task/s  "
                  f"rss {rss:6.0f} MiB  verified={out.verified} ok={bad is None}  "
                  f"| clean={st.get('n_clean_merge', 0)} dirty={st.get('n_dirty', 0)} "
                  f"ovw={st.get('n_overwrite', 0)} "
                  f"stale={st.get('n_stale_victims', 0) + st.get('n_stale_cost', 0)} "
                  f"notsel={st.get('n_not_selected', 0)}", flush=True)

    if rows:
        seq = _sequential_baseline(rows)
        if seq:
            print("\nvs sequential (per second of loop wall):")
            base_rate = seq["improvement_pct"] / max(1e-9, seq["loop_s"])
            for r in rows:
                rate = r["improvement_pct"] / max(1e-9, r["loop_s"])
                relative = _relative_rate(rate, base_rate)
                comparison = "n/a" if relative is None else f"{relative:.2f}x"
                requested_note = (
                    "" if r["workers"] == r["requested_workers"]
                    else f"/{r['requested_workers']} requested"
                )
                print(f"  {r['mode']:<10} m={r['workers']}{requested_note}: {comparison:>6} "
                      f"({rate:.4f} %/s vs {base_rate:.4f})")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"scenario": args.scenario, "demand_duration_s": args.demand_duration,
                       "horizon_s": args.horizon, "seed": args.seed,
                       "neighborhood": args.neighborhood, "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":       # REQUIRED: spawn re-imports this module in every child
    main()
