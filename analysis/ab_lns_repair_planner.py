"""Paired A/B: which planner should repair a destroyed LNS neighborhood, A* or SIPP?

LNS repair plans against a nearly-full ledger, which is SIPP's winning regime — but LNS is also the
most commit-heavy workload in the repo (a rejected iteration pays release+recommit TWICE), and SIPP
keeps a fourth ledger-subscribed structure on terminal legs where compiled A* keeps three. This
script measures the trade directly. Design record: context/sipp_lns_plan.md §9.

Arms run STRICTLY SEQUENTIALLY in one process, sharing one baseline schedule, because the deliverable
is a wall-clock comparison: two arms racing for cores would measure contention, not planners. The
baseline is A* either way (`analysis/run_lns.py` does the same), so both arms start from the identical
incumbent and the comparison is paired per-flight.

    uv run python analysis/ab_lns_repair_planner.py --demand-duration 600 --horizon 6000 \
        --iterations 300 --neighborhood 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim.planner.lns import LNSConfig             # noqa: E402
from freespace_sim.planner.lns.solver import run_lns_on_result  # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402
from freespace_sim import sim                               # noqa: E402

#: Set by the patched `_new_repair_planner` below so the arm can read the planner's own counters
#: (SIPP's `_sfb` fallbacks, A*'s `_fb`) after the run. `LNSResult` carries neither, and the state
#: object is not returned — but a planner-choice A/B that cannot see the fallback rate is measuring
#: something it cannot name.
_LAST_PLANNER: list = []


def _fresh_baseline(spec, demand, cfg):
    """Re-run FCFS so each arm gets its OWN (intents, ledger) pair.

    Not an optimisation to skip: `run_lns_on_result` takes the ledger over and mutates it in place,
    so the second arm cannot be handed the first arm's result. Deterministic in `cfg.seed`, so the two
    baselines are identical by construction — asserted below rather than assumed.
    """
    return sim.run(cfg, demand=demand, planner_name="astar", progress=False,
                   return_anchor="nominal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--demand-duration", type=float, default=None)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--neighborhood", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--verify-every", type=int, default=0)
    ap.add_argument("--arms", default="astar,sipp")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    if (args.demand_duration is None) != (args.horizon is None):
        ap.error("--demand-duration and --horizon must be set together (spec.py contract)")

    overrides = {}
    if args.demand_duration is not None:
        overrides["demand_duration_s"] = args.demand_duration
        overrides["horizon_s"] = args.horizon
    spec = with_overrides(get_scenario(args.scenario), **overrides)
    cfg = spec.config()
    demand = spec.demand_model()

    from freespace_sim.planner.lns import state as lns_state

    _orig_new = lns_state._new_repair_planner

    def _capturing_new(name, **kw):
        planner = _orig_new(name, **kw)
        _LAST_PLANNER[:] = [planner]
        return planner

    lns_state._new_repair_planner = _capturing_new

    rows = []
    for arm in (a.strip() for a in args.arms.split(",") if a.strip()):
        t0 = time.monotonic()
        res = _fresh_baseline(spec, demand, cfg)
        base_s = time.monotonic() - t0
        n_legs = sum(1 for it in res.intents if it.accepted)
        base_cost = float(sum(it.cost for it in res.intents if it.accepted))

        lns_cfg = LNSConfig(
            seed=args.seed,
            max_iterations=args.iterations,
            neighborhood_size=args.neighborhood,
            repair_planner=arm,
            time_limit_s=args.time_limit,
            verify_every=args.verify_every,
            unimpeded_workers=None,     # the one-off ruler may fan out; it is identical across arms
            log_every=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="astar compiled kernel FALLBACK")
            warnings.filterwarnings("ignore", message="sipp compiled kernel FALLBACK")
            out = run_lns_on_result(res, demand, lns_cfg, return_anchor="nominal")
        s = out.summary()
        loop_s = max(1e-9, s["wall_s"] - s["init_wall_s"])
        # SIPP kernel fallbacks land in A*'s search, so a high rate would silently erase the
        # plan-side advantage this whole comparison is about. Read it, don't assume it is zero.
        fb = getattr(_LAST_PLANNER[0], "_sfb", None) if _LAST_PLANNER else None
        rows.append(dict(
            arm=arm, n_legs=n_legs, baseline_cost=base_cost, baseline_wall_s=base_s,
            cost_after=s["cost_after"], improvement_pct=s["improvement_pct"],
            n_accepted=s["n_accepted"], n_iterations=s["n_iterations"],
            loop_s=loop_s, init_s=s["init_wall_s"],
            t_plan_s=s["t_plan_s"], t_ledger_s=s["t_ledger_s"],
            t_other_s=loop_s - s["t_plan_s"] - s["t_ledger_s"],
            iters_per_s=s["n_iterations"] / loop_s,
            release_subs=s["n_release_subs"], commit_subs=s["n_commit_subs"],
            verified=s["verified"], fallbacks=fb,
            astar_fallbacks=getattr(_LAST_PLANNER[0], "_fb", None) if _LAST_PLANNER else None,
        ))
        print(f"[{arm}] {n_legs} legs, base {base_cost:.0f} -> {s['cost_after']:.0f} "
              f"({s['improvement_pct']:.2f}%), {s['n_accepted']}/{s['n_iterations']} accepted, "
              f"loop {loop_s:.0f}s = plan {s['t_plan_s']:.0f}s + ledger {s['t_ledger_s']:.0f}s "
              f"+ other {loop_s - s['t_plan_s'] - s['t_ledger_s']:.0f}s, "
              f"subs {s['n_release_subs']}, fb {fb}, verified={s['verified']}", flush=True)

    # Every arm re-derives the baseline from the same seeded cfg; if they disagree the comparison is
    # not paired and nothing below means anything.
    assert len({round(r["baseline_cost"], 6) for r in rows}) <= 1, \
        f"arms saw different baselines: {[r['baseline_cost'] for r in rows]}"

    print("\n arm    legs  iters  loop_s  it/s   plan_s  ledger_s other_s  impr%   acc  subs")
    for r in rows:
        print(f" {r['arm']:<6} {r['n_legs']:>5} {r['n_iterations']:>6} {r['loop_s']:>7.1f} "
              f"{r['iters_per_s']:>5.2f} {r['t_plan_s']:>7.1f} {r['t_ledger_s']:>8.1f} "
              f"{r['t_other_s']:>7.1f} {r['improvement_pct']:>6.2f} {r['n_accepted']:>5} "
              f"{r['release_subs']:>5}")
    if len(rows) == 2:
        a, b = rows
        print(f"\n{b['arm']} vs {a['arm']}: loop {a['loop_s'] / max(1e-9, b['loop_s']):.2f}x, "
              f"plan {a['t_plan_s'] / max(1e-9, b['t_plan_s']):.2f}x, "
              f"ledger {a['t_ledger_s'] / max(1e-9, b['t_ledger_s']):.2f}x, "
              f"improvement {b['improvement_pct'] - a['improvement_pct']:+.2f} pp")

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(scenario=args.scenario, iterations=args.iterations,
                 neighborhood=args.neighborhood, seed=args.seed,
                 demand_duration_s=args.demand_duration, horizon_s=args.horizon, arms=rows),
            indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
