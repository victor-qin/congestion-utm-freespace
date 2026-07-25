"""Phase 0 viability probe for issue #8 Track A — speculative planning with ordered commit.

Replays a SAVED run (no re-simulation) and asks: had flights been planned speculatively in a
pipeline window of W requests, how often would a speculation have been dirtied by an interleaved
commit — i.e. how often does the parallel sim pay a replan? Two proxies, one per validation mode:

  relaxed proxy   space-time AABB overlap between flight k's committed volumes and the volumes of
                  the accepted flights among the W requests before it. A superset of the true
                  ``any_conflict`` (AABB is the ledger's own broadphase, minus the FCL narrowphase),
                  so it UPPER-BOUNDS the relaxed-mode replan rate.
  exact proxy     SPATIAL-ONLY overlap (design directive: prediction carries no time axis — under
                  density, delay makes times unpredictable) between the window's volumes and k's
                  hull tube: the xy-AABB of k's flown centerline padded by the occupancy inflation.
                  A proxy for the recorded read-envelope: pessimistic in time (none), optimistic in
                  space (the real search probes beyond its final hull), so treat it as a CENTRAL
                  ESTIMATE, not a bound. Denied flights count as always-dirty in exact mode (a
                  budget denial's read set spans everything reachable).

Caveats: reservations.parquet holds ACCEPTED flights only, so denied flights contribute no
obstacles (correct — they commit nothing) and no envelope (hence the pessimistic denial rule).

Usage::

    uv run python analysis/parallel_viability.py --run results/<folder> [--windows 4 8 16 32 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freespace_sim.ledger import ReservationLedger          # noqa: E402  (_flat_aabb reuse)
from freespace_sim.planner import hexgrid as hg             # noqa: E402
from freespace_sim.runs import load_run                     # noqa: E402


def _term_id(t):
    """Terminal id from a load_run round-tripped terminal (tuple) / Terminal / None."""
    if t is None:
        return None
    return t[0] if isinstance(t, tuple) else getattr(t, "id", None)


class _Flight:
    """Per-flight precomputed geometry: volume AABBs+windows, union box, spatial hull tube."""

    __slots__ = ("fid", "accepted", "tids", "aabbs", "t0", "t1", "ubox", "ut0", "ut1", "tube")

    def __init__(self, intent, margin: float):
        req = intent.request
        self.fid = req.flight_id
        self.accepted = intent.accepted
        self.tids = {x for x in (_term_id(req.origin_terminal), _term_id(req.dest_terminal)) if x}
        if not intent.accepted or not intent.volumes:
            self.aabbs = None
            self.tube = None
            return
        rows = [ReservationLedger._flat_aabb(v) for v in intent.volumes]
        self.aabbs = np.asarray(rows, float)                          # (n, 6)
        self.t0 = np.asarray([v.t_start for v in intent.volumes], float)
        self.t1 = np.asarray([v.t_end for v in intent.volumes], float)
        self.ubox = (*self.aabbs[:, :3].min(0), *self.aabbs[:, 3:].max(0))
        self.ut0, self.ut1 = float(self.t0.min()), float(self.t1.max())
        # spatial hull tube: xy-AABB of the flown centerline, padded like the occupancy raster
        pts = np.asarray([p for p, _ in (intent.centerline or [])], float)
        xy = pts[:, :2] if len(pts) else self.aabbs[:, [0, 1]]
        self.tube = (xy[:, 0].min() - margin, xy[:, 1].min() - margin,
                     xy[:, 0].max() + margin, xy[:, 1].max() + margin)


def _boxes_overlap_st(a: _Flight, b: _Flight) -> bool:
    """Any space-time AABB overlap between two flights' volume sets (relaxed-mode proxy)."""
    if (a.ubox[3] < b.ubox[0] or b.ubox[3] < a.ubox[0] or a.ubox[4] < b.ubox[1]
            or b.ubox[4] < a.ubox[1] or a.ubox[5] < b.ubox[2] or b.ubox[5] < a.ubox[2]):
        return False
    if a.ut1 <= b.ut0 or b.ut1 <= a.ut0:
        return False
    A, B = a.aabbs, b.aabbs                                           # (n,6) x (m,6), broadcast
    sep = (
        (A[:, None, 3] < B[None, :, 0]) | (B[None, :, 3] < A[:, None, 0])
        | (A[:, None, 4] < B[None, :, 1]) | (B[None, :, 4] < A[:, None, 1])
        | (A[:, None, 5] < B[None, :, 2]) | (B[None, :, 5] < A[:, None, 2])
    )
    t_sep = (a.t1[:, None] <= b.t0[None, :]) | (b.t1[None, :] <= a.t0[:, None])
    return bool(np.any(~sep & ~t_sep))


def _tube_hit(k: _Flight, j: _Flight) -> bool:
    """Spatial-only: any of j's volumes xy-overlapping k's hull tube (exact-mode proxy)."""
    tb = k.tube
    if j.ubox[3] < tb[0] or tb[2] < j.ubox[0] or j.ubox[4] < tb[1] or tb[3] < j.ubox[1]:
        return False
    B = j.aabbs
    return bool(np.any(~((B[:, 3] < tb[0]) | (tb[2] < B[:, 0])
                         | (B[:, 4] < tb[1]) | (tb[3] < B[:, 1]))))


def replay_rates(flights: list[_Flight], windows: tuple[int, ...]) -> list[dict]:
    out = []
    for W in windows:
        n = len(flights)
        rel = ex = ex_denied = samehub = 0
        for k in range(n):
            fk = flights[k]
            lo = max(0, k - W)
            js = [flights[j] for j in range(lo, k) if flights[j].accepted]
            if not fk.accepted:
                ex += 1                      # exact mode: denial envelope ~unbounded → replan
                ex_denied += 1
                continue                     # relaxed mode: snapshot denials are kept as-is
            if any(_boxes_overlap_st(fk, fj) for fj in js):
                rel += 1
            hits = [fj for fj in js if _tube_hit(fk, fj)]
            if hits:
                ex += 1
                if any(fk.tids & fj.tids for fj in hits):
                    samehub += 1
        out.append({"W": W, "n": n, "relaxed": rel / n, "exact": ex / n,
                    "exact_denied_share": ex_denied / n,
                    "samehub_share": samehub / max(1, ex - ex_denied)})
    return out


def _amdahl(r: float, workers: int, W: int) -> tuple[float, float]:
    """(serial-replan model, eager-re-spec model) projected speedups for dirty rate r."""
    n_eff = max(1, min(workers, W))
    return 1.0 / (r + (1.0 - r) / n_eff), n_eff / (1.0 + r)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="saved run folder (results/<...>)")
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--margin", type=float, default=None,
                    help="tube pad, m (default: occupancy inflation max(corr/2, hover_r) + R)")
    args = ap.parse_args(argv)

    run = load_run(args.run)
    cfg = run.config
    margin = args.margin if args.margin is not None else (
        max(cfg.corridor_width_m / 2.0, cfg.effective_hover_radius_m) + hg.circumradius(cfg))
    ordered = sorted(run.intents, key=lambda i: (i.request.t_request, i.request.flight_id))
    flights = [_Flight(i, margin) for i in ordered]
    n_acc = sum(f.accepted for f in flights)
    print(f"run: {args.run}\nflights: {len(flights)} ({n_acc} accepted)  planner: {cfg.planner}  "
          f"lambda: {getattr(cfg, 'arrival_rate_per_hr', '?')}  tube margin: {margin:.0f} m\n")
    hdr = (f"{'W':>4} {'relaxed%':>9} {'exact%':>8} {'denied%':>8} {'samehub%':>9}"
           f" {'S4 ser/eag':>12} {'S8 ser/eag':>12}")
    print(hdr + "\n" + "-" * len(hdr))
    for row in replay_rates(flights, tuple(args.windows)):
        s4 = _amdahl(row["exact"], 4, row["W"])
        s8 = _amdahl(row["exact"], 8, row["W"])
        print(f"{row['W']:>4} {100 * row['relaxed']:>8.1f}% {100 * row['exact']:>7.1f}%"
              f" {100 * row['exact_denied_share']:>7.1f}% {100 * row['samehub_share']:>8.1f}%"
              f" {s4[0]:>5.1f}/{s4[1]:<5.1f} {s8[0]:>5.1f}/{s8[1]:<5.1f}")
    print("\nrelaxed% upper-bounds the relaxed-mode replan rate (AABB ⊇ narrowphase);"
          "\nexact% is a central estimate for exact mode (spatial-only hull tube; denials pessimistic)."
          "\nS<N> ser = serial replans 1/(r+(1-r)/N); eag = eager re-spec N/(1+r).")


if __name__ == "__main__":
    main()
