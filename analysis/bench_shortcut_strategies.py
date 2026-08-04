"""Benchmark legacy, exact-heading-skip, and batched A* shortcut refiners.

The workload is one deterministic 42-flight request set. Every planner/repetition runs in a fresh,
kernel-warmed child process; medians are reported. Raw identity ignores only ``solve_time_s``. The
changed-plan count also normalizes the intentional planner label so it measures
trajectory/reservation changes, not branding.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import statistics
import time
import traceback
from collections import defaultdict

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import shortcut as shortcut_mod
from freespace_sim.sim import run


PLANNERS = ("astar_shortcut", "astar_heading_shortcut", "astar_batched_shortcut")


def _intent_bytes(intent, *, normalize_planner: bool = False) -> bytes:
    normalized = copy.deepcopy(intent)
    normalized.solve_time_s = 0.0
    if normalize_planner:
        normalized.planner = "<planner>"
    return pickle.dumps(normalized, protocol=pickle.HIGHEST_PROTOCOL)


def _run_once(cfg, requests, planner_name):
    counters = defaultdict(float)
    original_rebuild = shortcut_mod._rebuild
    original_merge = shortcut_mod._merge_preserves_resampling
    original_any_conflict = ReservationLedger.any_conflict

    def counted_rebuild(*args, **kwargs):
        counters["rebuild_calls"] += 1
        start = time.perf_counter()
        result = original_rebuild(*args, **kwargs)
        counters["rebuild_wall_s"] += time.perf_counter() - start
        counters["rebuild_success" if result is not None else "rebuild_failure"] += 1
        if result is not None:
            counters["rebuilt_volumes"] += len(result[0])
        return result

    def counted_merge(*args, **kwargs):
        counters["heading_checks"] += 1
        result = original_merge(*args, **kwargs)
        counters["heading_skips" if result else "heading_fallbacks"] += 1
        return result

    def counted_any_conflict(self, volumes):
        counters["ledger_conflict_calls"] += 1
        return original_any_conflict(self, volumes)

    shortcut_mod._rebuild = counted_rebuild
    shortcut_mod._merge_preserves_resampling = counted_merge
    ReservationLedger.any_conflict = counted_any_conflict
    try:
        start = time.perf_counter()
        result = run(cfg, requests=copy.deepcopy(requests), planner_name=planner_name)
        wall = time.perf_counter() - start
    finally:
        shortcut_mod._rebuild = original_rebuild
        shortcut_mod._merge_preserves_resampling = original_merge
        ReservationLedger.any_conflict = original_any_conflict

    accepted = result.accepted

    return {
        "wall_s": wall,
        "verified": result.verified,
        "accepted": len(result.accepted),
        "denied": len(result.denied),
        "rebuild_calls": int(counters["rebuild_calls"]),
        "rebuild_success": int(counters["rebuild_success"]),
        "rebuild_failure": int(counters["rebuild_failure"]),
        "rebuild_wall_s": counters["rebuild_wall_s"],
        "heading_checks": int(counters["heading_checks"]),
        "heading_skips": int(counters["heading_skips"]),
        "heading_fallbacks": int(counters["heading_fallbacks"]),
        "rebuilt_volumes": int(counters["rebuilt_volumes"]),
        "ledger_conflict_calls": int(counters["ledger_conflict_calls"]),
        "committed_volumes": result.ledger.n_volumes,
        "mean_cost": statistics.fmean(i.cost for i in accepted),
        "mean_detour_m": statistics.fmean(i.air_detour_m for i in accepted),
        "mean_altitude_m": statistics.fmean(i.altitude_change_m for i in accepted),
        "mean_ground_delay_s": statistics.fmean(i.ground_delay_s for i in accepted),
        "mean_arrival_s": statistics.fmean(i.centerline[-1][1] for i in accepted),
        "intent_bytes": [_intent_bytes(intent) for intent in result.intents],
        "plan_bytes": [_intent_bytes(intent, normalize_planner=True) for intent in result.intents],
    }


def _worker(conn, cfg, requests, planner_name):
    """Warm one fresh process, measure one strategy, and return a serializable result."""
    try:
        run(cfg, requests=copy.deepcopy(requests[:1]), planner_name="astar")
        conn.send(("ok", _run_once(cfg, requests, planner_name)))
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _run_fresh_process(ctx, cfg, requests, planner_name):
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(child, cfg, requests, planner_name))
    process.start()
    child.close()
    status, payload = parent.recv()
    parent.close()
    process.join()
    if status != "ok" or process.exitcode != 0:
        raise RuntimeError(
            f"benchmark child for {planner_name} failed with exit {process.exitcode}:\n{payload}"
        )
    return payload


def _median(rows, key):
    return statistics.median(row[key] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    cfg = SimConfig(
        region_size_m=(4000.0, 4000.0),
        horizon_s=600.0,
        lam_per_hour=420.0,
        seed=7,
        max_ground_delay_s=180.0,
    )
    generated = UniformPoissonDemand().generate(cfg, np.random.default_rng(cfg.seed))
    if len(generated) < 42:
        raise RuntimeError(f"benchmark needs 42 requests; deterministic demand produced {len(generated)}")
    requests = generated[:42]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    samples = {planner: [] for planner in PLANNERS}
    for repetition in range(args.repetitions):
        order = PLANNERS[repetition % len(PLANNERS):] + PLANNERS[:repetition % len(PLANNERS)]
        for planner in order:
            samples[planner].append(_run_fresh_process(ctx, cfg, requests, planner))

    baseline_rows = samples["astar_shortcut"]
    changed_by_planner = {}
    for planner in PLANNERS:
        changed = [
            sum(a != b for a, b in zip(base["plan_bytes"], candidate["plan_bytes"], strict=True))
            for base, candidate in zip(baseline_rows, samples[planner], strict=True)
        ]
        if len(set(changed)) != 1:
            raise RuntimeError(f"{planner} changed-plan count varied across repetitions: {changed}")
        changed_by_planner[planner] = changed[0]

    heading_identical = all(
        heading["intent_bytes"] == legacy["intent_bytes"]
        for legacy, heading in zip(
            baseline_rows, samples["astar_heading_shortcut"], strict=True)
    )
    if not heading_identical:
        raise AssertionError("astar_heading_shortcut diverged from astar_shortcut intent bytes")

    print(f"workload: {len(requests)} flights, {args.repetitions} repetitions, medians below")
    print(
        f"{'planner':<28} {'rebuilds':>9} {'ok/fail':>11} {'rebuild ms':>12} "
        f"{'wall ms':>10} {'verified':>9} {'changed':>8} {'heading skips':>14}"
    )
    for planner in PLANNERS:
        rows = samples[planner]
        rebuilds = int(_median(rows, "rebuild_calls"))
        successes = int(_median(rows, "rebuild_success"))
        failures = int(_median(rows, "rebuild_failure"))
        print(
            f"{planner:<28} {rebuilds:>9} {f'{successes}/{failures}':>11} "
            f"{1000 * _median(rows, 'rebuild_wall_s'):>12.2f} "
            f"{1000 * _median(rows, 'wall_s'):>10.2f} "
            f"{str(all(row['verified'] for row in rows)):>9} "
            f"{changed_by_planner[planner]:>8} "
            f"{int(_median(rows, 'heading_skips')):>14}"
        )

    print("\noutcome and workload medians")
    print(
        f"{'planner':<28} {'accept/deny':>11} {'cost':>10} {'detour m':>10} "
        f"{'altitude m':>11} {'delay s':>9} {'arrival s':>10} {'ledger q':>9} "
        f"{'rebuilt vols':>13} {'committed':>10}"
    )
    for planner in PLANNERS:
        rows = samples[planner]
        accepted = int(_median(rows, "accepted"))
        denied = int(_median(rows, "denied"))
        print(
            f"{planner:<28} {f'{accepted}/{denied}':>11} "
            f"{_median(rows, 'mean_cost'):>10.2f} "
            f"{_median(rows, 'mean_detour_m'):>10.2f} "
            f"{_median(rows, 'mean_altitude_m'):>11.2f} "
            f"{_median(rows, 'mean_ground_delay_s'):>9.2f} "
            f"{_median(rows, 'mean_arrival_s'):>10.2f} "
            f"{int(_median(rows, 'ledger_conflict_calls')):>9} "
            f"{int(_median(rows, 'rebuilt_volumes')):>13} "
            f"{int(_median(rows, 'committed_volumes')):>10}"
        )

    print("heading vs legacy intent bytes: IDENTICAL (all repetitions)")
    legacy_calls = _median(samples["astar_shortcut"], "rebuild_calls")
    heading_calls = _median(samples["astar_heading_shortcut"], "rebuild_calls")
    batched_calls = _median(samples["astar_batched_shortcut"], "rebuild_calls")
    print(f"heading rebuild reduction: {(legacy_calls - heading_calls) / legacy_calls:.1%}")
    print(f"batched rebuild reduction: {(legacy_calls - batched_calls) / legacy_calls:.1%}")


if __name__ == "__main__":
    main()
