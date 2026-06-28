"""Prototype: nogil-parallel optimistic batching for FCFS planning (#8 Track A).

Measures the two things that decide whether parallel FCFS pays off:
  (1) **parallel speedup** — N compiled SIPP workers (sharing the read-only occupancy, each with its own
      kernel state) search a batch on N real threads, vs the same batch planned sequentially. Works only
      because `_search` is `@njit(nogil=True)` — it releases the GIL so the searches actually overlap.
  (2) **optimistic re-plan rate vs density** — plan the batch against the frozen ledger snapshot, then
      commit serially re-checking `any_conflict`; a flight that conflicts with an already-committed
      batch-mate must be re-planned. That re-plan rate is the cost that erodes batching under congestion,
      and is exactly what we want to chart against density.

    uv run --extra compiled python -m analysis.bench_batch
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.planner.sipp import SIPPPlanner
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.types import FlightRequest, vec
from freespace_sim.uss import USS


def _short_reqs(W, n, rmin, rmax, horizon, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        o = rng.uniform([0, 0], [W, W])
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(rmin, rmax)
        d = np.clip(o + rad * np.array([np.cos(ang), np.sin(ang)]), 0, W)
        out.append(FlightRequest(i, vec(o[0], o[1], 0), vec(d[0], d[1], 0), float(rng.uniform(0, horizon))))
    return sorted(out, key=lambda r: (r.t_request, r.flight_id))


def main():
    W, N, K = 24000.0, 4, 48
    allreqs = _short_reqs(W, 3200, 4000.0, 8000.0, 3600.0, 0)
    print(f"N workers={N}, batch K={K}, short 4-8km flights in {int(W / 1000)}km box (Dallas-shaped)")
    for warm in (600, 1400, 2400):
        cfg = SimConfig(region_size_m=(W, W), lam_per_hour=600.0, horizon_s=3600.0, seed=0)
        sc = scenario_from_requests(allreqs[:warm + K])
        led = ReservationLedger(cfg)
        dss = DSS(ledger=led, mechanism=FCFSMechanism())
        master = get_planner("sipp")
        usses = {u: USS(u, dss, cfg, master) for u in sc.uss_ids}
        evs = sc.events
        for ev in evs[:warm]:
            usses[ev.request.uss_id].handle_request(ev.request)
        batch = [ev.request for ev in evs[warm:warm + K]]
        if len(batch) < K:
            continue
        vols0 = led.n_volumes
        master.plan(batch[0], led, cfg)                       # sync master occupancy + JIT-compile
        workers = [SIPPPlanner(compiled=True) for _ in range(N)]
        for w in workers:
            w.share_occupancy_from(master)
            w.plan(batch[0], led, cfg)                        # borrow occ + allocate this worker's state

        t = time.monotonic()                                  # sequential (no commit → ledger frozen)
        for f in batch:
            master.plan(f, led, cfg)
        t_seq = time.monotonic() - t

        chunks = [batch[i::N] for i in range(N)]              # parallel: one worker per thread
        def run_chunk(wi):
            for f in chunks[wi]:
                workers[wi].plan(f, led, cfg)
        t = time.monotonic()
        with ThreadPoolExecutor(N) as ex:
            list(ex.map(run_chunk, range(N)))
        t_par = time.monotonic() - t

        snap = [master.plan(f, led, cfg) for f in batch]      # optimistic: all vs the frozen snapshot
        replans = 0
        for f, p in zip(batch, snap):
            if not p.accepted:
                continue
            if led.any_conflict(p.volumes):                   # conflicts a batch-mate just committed
                replans += 1
                p = master.plan(f, led, cfg)                  # re-plan against the now-current ledger
            if p.accepted:
                led.commit(f.flight_id, p.volumes)
        print(f"  warm={warm} vols={vols0}: seq={t_seq * 1000:5.0f}ms par={t_par * 1000:5.0f}ms "
              f"speedup={t_seq / t_par:.2f}x (N={N}) | optimistic re-plans={replans}/{K} "
              f"({100 * replans / K:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
