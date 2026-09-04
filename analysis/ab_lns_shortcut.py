"""A/B the four placements of the shortcut refiner inside an LNS destroy/repair transaction.

Every arm starts from the SAME FCFS baseline (--baseline), run once and deep-copied per arm, so the
only difference between arms is where the refiner runs. The arms:

    a0_control    LNS with bare-A* repairs, then ONE shortcut sweep over the final incumbent.
                  The "bolt it on at the end" pipeline, and the attribution floor.
    a1_inter      repair planner IS a ShortcutRefiner: victim i is cut before it is committed, so
                  victim i+1 plans around the tightened corridor.
    a2_deferred   bare PP loop, then cut all k BEFORE the accept test.
    a3_post       bare PP loop, accept on un-cut cost, then cut only the winner.

Arms 1-3 refine the ruler too (LNSConfig.shortcut_ruler defaults on with the arm); a0 does not,
because its incumbent stays un-refined for the whole run.

--baseline MATTERS, and pairs with the arm. Under `astar` an in-loop arm collects a one-off geometry
win on iteration 1 that the baseline was simply never given, which inflates everything it books to
"the search". Under `astar_shortcut` the incumbent already carries tight geometry from t=0, the
delay premiums that pick victims are in the refined currency, and an in-loop cut is finally being
asked the real question: does re-cutting a repaired flight beat leaving it? Note `astar_shortcut`
+ arm `a0_control` is the deliberately biased cell — bare repairs produce looser geometry than the
incumbent they must beat, so it should reject far more.

    uv run python analysis/ab_lns_shortcut.py --demand-duration 600 --horizon 3000 \
        --iterations 300 --out /tmp/ab.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time

import numpy as np

from freespace_sim import sim, verify
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.lns import LNSConfig
from freespace_sim.planner.lns.solver import run_lns_on_result
from freespace_sim.planner.shortcut import _terminal_capacity_for, refine_intent
from freespace_sim.scenarios import get_scenario
from freespace_sim.scenarios.spec import with_overrides

ARMS = ("a0_control", "a1_inter", "a2_deferred", "a3_post")
_ARM_TO_LNS = {"a0_control": "none", "a1_inter": "interleaved",
               "a2_deferred": "deferred", "a3_post": "post_accept"}


def _polish(intents, ledger, cfg) -> tuple[list, int, float]:
    """Shortcut every accepted intent of a FINISHED schedule, in flight order.

    a0's whole shortcut budget, and the same operation a2/a3 apply per transaction: release the
    flight so it stops conflicting with itself, refine against everyone else, re-commit.

    The throwaway plan is not optional. ``refine_intent`` REFUSES a terminal flight when it has no
    capacity authority (it cannot prove pad capacity for a retimed path), and ``capacity_authority``
    only answers for a ledger the planner has actually planned against — so without this bind every
    hub flight silently comes back unrefined, which on density_faa is all of them. Returns
    (intents, n_shortened, seconds).
    """
    t0 = time.perf_counter()
    planner = AStarPlanner()
    planner.evict_floor = 0.0
    first = next((it.request for it in intents if it.accepted), None)
    if first is None:
        return list(intents), 0, time.perf_counter() - t0
    planner.plan(first, ledger, cfg)                 # bind the occupancy + capacity services only
    tcap = _terminal_capacity_for(planner, ledger)
    out, n = [], 0
    for it in intents:
        if not it.accepted or len(it.centerline) <= 3:
            out.append(it)
            continue
        fid = it.request.flight_id
        ledger.release_many([fid])
        try:
            refined = refine_intent(it, ledger, cfg, tcap=tcap)
        except BaseException:
            ledger.commit(fid, it.volumes)
            raise
        ledger.commit(fid, refined.volumes)
        n += refined is not it
        out.append(refined)
    return out, n, time.perf_counter() - t0


def _row(arm, res, lns_res, cfg, static_terms, polish=None):
    acc = [i for i in (polish[0] if polish else lns_res.intents) if i.accepted]
    cost_final = float(sum(i.cost for i in acc))
    bad = verify.find_interflight_conflict(
        polish[0] if polish else lns_res.intents, cfg, static_terminals=static_terms)
    return {
        "arm": arm,
        "cost_final_abs": cost_final,
        "cost_baseline": lns_res.cost_before,
        "cost_lns": lns_res.cost_after,
        "cost_final": cost_final,
        "improvement_pct": 100.0 * (lns_res.cost_before - cost_final) / max(1e-9, lns_res.cost_before),
        "lns_only_pct": 100.0 * (lns_res.cost_before - lns_res.cost_after) / max(1e-9, lns_res.cost_before),
        "n_iterations": lns_res.n_iterations,
        "n_accepted": lns_res.n_accepted,
        "wall_s": lns_res.wall_s,
        "init_wall_s": lns_res.init_wall_s,
        "polish_s": polish[2] if polish else 0.0,
        "polish_shortened": polish[1] if polish else 0,
        "mean_detour_m": float(np.mean([i.air_detour_m for i in acc])),
        "conflict_free": bad is None,
        "reasons": _reason_counts(lns_res.trajectory),
        "trajectory": [(r["iter"], r["incumbent_cost"], r["wall_s"]) for r in lns_res.trajectory],
    }


def _reason_counts(traj) -> dict:
    out: dict[str, int] = {}
    for r in traj:
        out[r["reason"]] = out.get(r["reason"], 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--demand-duration", type=float, default=None)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--neighborhood", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--baseline", default="astar", choices=["astar", "astar_shortcut"],
                    help="FCFS planner for the seed schedule. 'astar_shortcut' is the COHERENT "
                         "pipeline for an in-loop arm: the incumbent already carries tight geometry, "
                         "so a repair that cuts is compared like-with-like instead of collecting a "
                         "one-off win the baseline was never given. Costs across baselines are only "
                         "comparable in ABSOLUTE terms — the %% column has a different denominator.")
    ap.add_argument("--unimpeded-workers", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    spec = get_scenario(args.scenario)
    ov = {}
    if args.demand_duration is not None:
        ov["demand_duration_s"] = args.demand_duration
    if args.horizon is not None:
        ov["horizon_s"] = args.horizon
    if ov:
        spec = with_overrides(spec, **ov)
    cfg = spec.config()

    t0 = time.perf_counter()
    base = sim.run(cfg, demand=spec.demand_model(), planner_name=args.baseline)
    base_wall = time.perf_counter() - t0
    n_acc = sum(1 for i in base.intents if i.accepted)
    print(f"baseline[{args.baseline}]: {n_acc} accepted of {len(base.intents)} in {base_wall:.1f}s, "
          f"cost {sum(i.cost for i in base.intents if i.accepted):,.0f}")

    rows = []
    for arm in args.arms.split(","):
        # Deep-copy the WHOLE result: LNS mutates the ledger in place and detaches its subscribers,
        # so arms sharing one baseline object would each start from the previous arm's schedule.
        res = copy.deepcopy(base)
        lns = LNSConfig(
            seed=args.seed, max_iterations=args.iterations,
            neighborhood_size=args.neighborhood, log_every=max(1, args.iterations // 6),
            unimpeded_workers=args.unimpeded_workers, time_limit_s=args.time_limit,
            shortcut_arm=_ARM_TO_LNS[arm],
            shortcut_ruler=(args.baseline == "astar_shortcut" or None),
        )
        t = time.perf_counter()
        out = run_lns_on_result(res, spec.demand_model(), lns)
        # `out.intents` is the improved schedule; `res.intents` is the stale FCFS baseline the
        # ledger no longer holds. Polishing the latter measures a schedule that does not exist.
        #
        # EVERY arm is polished, not just a0. LNS only ever refines flights it DESTROYS, so an
        # in-loop arm reaches the ~half of the fleet that got picked and leaves the rest on baseline
        # geometry — measured on the 120 s cut, a0's closing polish alone was worth 31.6% while the
        # in-loop arms had reached 13%. Polishing all arms makes the comparison about PLACEMENT
        # (does cutting during the search help the search?) instead of about coverage.
        polish = _polish(out.intents, res.ledger, cfg)
        row = _row(arm, res, out, cfg, out_static(res), polish)
        row["arm_wall_s"] = time.perf_counter() - t
        rows.append(row)
        print(f"{arm:<12} base {row['cost_baseline']:>11,.0f} -> lns {row['cost_lns']:>11,.0f} "
              f"({row['lns_only_pct']:5.2f}%) -> +polish {row['cost_final']:>11,.0f} "
              f"({row['improvement_pct']:5.2f}%)  acc {row['n_accepted']:>4}/{row['n_iterations']} "
              f" cut {row['polish_shortened']:>4}  {row['arm_wall_s']:>7.1f}s "
              f" clean={row['conflict_free']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"scenario": args.scenario, "n_accepted_baseline": n_acc,
                       "baseline": args.baseline, "baseline_wall_s": base_wall,
                       "iterations": args.iterations, "neighborhood": args.neighborhood,
                       "seed": args.seed, "rows": rows}, fh)
        print(f"wrote {args.out}")


def out_static(res):
    return tuple(res.ledger.static_terminals())


if __name__ == "__main__":
    main()
