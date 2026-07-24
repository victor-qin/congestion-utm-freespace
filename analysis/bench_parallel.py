"""Parallel-sim bench (issue #8 Track A): wall-time + dirty/replan/re-spec rates vs the serial run.

Runs one scenario slice sequentially, then under each (n_workers, window, mode) combination, and
prints speedup + coordinator stats per row. In exact mode every parallel run is ASSERTED
byte-identical to the sequential baseline (summary + per-flight outcomes) — divergence is a bug,
not a data point. Deterministic (named scenario + seed) so before/after rows are comparable.

Usage::

    uv run python analysis/bench_parallel.py --scenario dallas_full --horizon 300 \
        --workers 4 8 --windows 16 32 --modes exact relaxed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim.parallel import ParallelConfig            # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402
from freespace_sim.sim import run                            # noqa: E402


def _outcome_key(res):
    return [(i.request.flight_id, i.status.value, i.denial_reason.value,
             round(i.cost, 9) if i.accepted else None, i.ground_delay_s)
            for i in res.intents]


def _delay_stats(res, cfg) -> dict:
    """Allocation quality, not wall time: relaxed mode is a VALID FCFS-class allocation but not the
    sequential one, so its acceptance count and delay levers can differ. Split ground vs air so a
    ground→air shift (the weighted-A* concern) is visible, not just a total."""
    from freespace_sim import metrics
    acc = [i for i in res.intents if i.accepted]
    n_acc = len(acc)
    ground = [i.ground_delay_s for i in acc]
    air = [i.air_hold_s + i.air_detour_m / cfg.nominal_speed_mps for i in acc]   # loiter + detour
    total = [metrics.total_delay_s(i, cfg) for i in acc]
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {"n_acc": n_acc, "n_den": len(res.intents) - n_acc,
            "mean_total": mean(total), "mean_ground": mean(ground), "mean_air": mean(air)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="dallas_full")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--windows", type=int, nargs="+", default=[None], metavar="W",
                    help="window sizes (default: 4×workers)")
    ap.add_argument("--modes", nargs="+", choices=("exact", "relaxed"), default=["exact"])
    ap.add_argument("--no-predictive", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    args = ap.parse_args(argv)

    spec = get_scenario(args.scenario)
    over = {"horizon_s": args.horizon}
    if args.lam is not None:
        over["lam_per_hour"] = args.lam
    spec = with_overrides(spec, **over)
    cfg, demand = spec.config(), spec.demand_model()

    print(f"scenario={args.scenario} λ={cfg.lam_per_hour}/h horizon={cfg.horizon_s}s "
          f"planner={cfg.planner}", flush=True)
    t0 = time.monotonic()
    seq = run(cfg, demand=spec.demand_model(), progress=True)
    t_seq = time.monotonic() - t0
    key_seq = _outcome_key(seq)
    dseq = _delay_stats(seq, cfg)
    print(f"sequential: {len(seq.intents)} flights, {t_seq:.1f}s "
          f"({1000 * t_seq / max(1, len(seq.intents)):.0f} ms/flight), verified={seq.verified}")
    print(f"  allocation: accepted={dseq['n_acc']} denied={dseq['n_den']}  "
          f"mean delay total={dseq['mean_total']:.1f}s (ground={dseq['mean_ground']:.1f} "
          f"air={dseq['mean_air']:.1f})\n")

    hdr = (f"{'mode':>8} {'N':>3} {'W':>4} {'wall_s':>7} {'speedup':>8} {'dirty%':>7}"
           f" {'serial':>7} {'respec':>7} {'commit_s':>9} {'wait_s':>8} {'exact?':>7}"
           f" | {'den':>4} {'delayΔ%':>8} {'grndΔ':>7} {'airΔ':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for mode in args.modes:
        for n in args.workers:
            for w in args.windows:
                pc = ParallelConfig(n_workers=n, window=w, mode=mode,
                                    predictive_dispatch=not args.no_predictive,
                                    adaptive_window=args.adaptive)
                t0 = time.monotonic()
                par = run(cfg, demand=spec.demand_model(), parallel=pc, progress=True)
                wall = time.monotonic() - t0
                s = pc.stats
                same = _outcome_key(par) == key_seq
                if mode == "exact":
                    assert same, "EXACT-MODE DIVERGENCE — read-envelope soundness bug"
                    assert s["n_canary"] == 0, "mechanism backstop fired in exact mode"
                assert par.verified
                dp = _delay_stats(par, cfg)
                # allocation deltas vs sequential (exact ⇒ all zero; relaxed ⇒ the tradeoff to judge)
                d_tot = 100 * (dp["mean_total"] - dseq["mean_total"]) / max(1e-9, dseq["mean_total"])
                d_grd = dp["mean_ground"] - dseq["mean_ground"]
                d_air = dp["mean_air"] - dseq["mean_air"]
                print(f"{mode:>8} {n:>3} {s['window']:>4} {wall:>7.1f} {t_seq / wall:>7.2f}x"
                      f" {100 * s['dirty_rate']:>6.1f}% {s['n_serial_replans']:>7}"
                      f" {s['n_respec']:>7} {s.get('t_commit_s', 0):>9.1f} {s.get('t_wait_s', 0):>8.1f}"
                      f" {'yes' if same else 'NO':>7} | {dp['n_den']:>4} {d_tot:>+7.1f}%"
                      f" {d_grd:>+7.1f} {d_air:>+7.1f}", flush=True)


if __name__ == "__main__":
    main()
