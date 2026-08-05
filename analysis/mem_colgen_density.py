"""Measure column-generation **memory** on a real density scenario, first N flights.

Everything tuned so far was measured on ``colgen_test`` — an 8 km x 8 km acceptance
miniature whose flights are ~7 hops long.  The density scenarios are 60 km x 30 km with a
16 km hub radius, so a flight is an order of magnitude longer and the label pool scales
roughly with ``hops^2``.  The two constants that bound the compiled DP's workspace
(``label_limit_max``, ``max_candidates``) were sized against the miniature; this script
exists to find out what they cost on a real graph before either is trusted at scale.

What it reports, and why each number is here:

* **Peak RSS** (``ru_maxrss``, bytes on macOS) — the number that decides whether a cluster
  task fits in its per-core memory share.  Sampled as a time series too, so the peak can be
  attributed to a stage rather than just quoted.
* **The per-flight label ladder.** ``dp_kernel.search_dag`` starts at ``64 * n_cells``
  labels and multiplies by 4 on overflow up to ``label_limit_max``.  Each rung allocates
  ``limit * (32 + 4*depth)`` bytes of label pool plus ``2*limit`` state slots at
  ``20 + 4*depth`` each — i.e. **~108 B per label at depth 3**, so the 1<<23 ceiling is
  ~0.9 GB of *transient* workspace for one flight.  Instrumenting the ladder shows how many
  flights climb it and how high.
* **Graph geometry per flight** (corridor cells, departure variants, time layers), because
  that is what predicts the ladder.

The instrumentation wraps ``dp_kernel._search_dag`` (the njit dispatcher, looked up as a
module global by ``search_dag``) rather than ``search_dag`` itself, so the observed array
shapes are the ones actually handed to the kernel — a rung that is allocated and then
discarded on overflow is still counted, which is the point.

Usage:
    uv run python analysis/mem_colgen_density.py
    uv run python analysis/mem_colgen_density.py --flights 100 --time-limit 600
    uv run python analysis/mem_colgen_density.py --scenario density_faa_wing_zipline_amazon
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import freespace_sim  # noqa: E402

# The A/B harness lesson (analysis/ab_column_clear.py): a script under analysis/ puts its own
# directory on sys.path[0], and an installed freespace_sim would then win silently.  Assert the
# tree under measurement rather than trusting the import.
assert Path(freespace_sim.__file__).resolve().is_relative_to(REPO_ROOT), (
    f"imported freespace_sim from {freespace_sim.__file__}, expected a tree under {REPO_ROOT}"
)

from freespace_sim.planner.colgen import dp_kernel, solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

_MB = 1024.0 * 1024.0
# ru_maxrss is bytes on macOS, kilobytes on Linux.
_MAXRSS_SCALE = 1.0 if sys.platform == "darwin" else 1024.0


# ------------------------------------------------------------------------------- RSS sampling


def _current_rss_bytes() -> float:
    """Current RSS via ``ps`` (KB).  psutil is not a dependency of this project."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return float(out) * 1024.0 if out else 0.0


def _tree_rss_bytes() -> tuple[float, int]:
    """RSS summed over this process and every descendant, plus the process count.

    Under parallel pricing the label pools live in *worker* processes, so the parent's own
    RSS understates the machine's requirement by roughly n_workers times the pool size --
    which is exactly the number that decides whether a cluster task fits.  One ``ps`` call
    and a walk down the ppid tree is enough and costs ~5 ms.
    """
    out = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,rss="], capture_output=True, text=True, check=False
    ).stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, float] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, kb = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb * 1024.0
    total, n, stack = 0.0, 0, [os.getpid()]
    while stack:
        pid = stack.pop()
        if pid in rss:
            total += rss[pid]
            n += 1
        stack.extend(children.get(pid, ()))
    return total, n


def _peak_rss_bytes() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_SCALE


class _RssSampler:
    """Background RSS time series, so the peak can be attributed to a stage."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, int]] = []   # (t, tree_rss_bytes, n_procs)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            total, n = _tree_rss_bytes()
            self.samples.append((time.monotonic() - self._t0, total, n))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_RssSampler":
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def peak(self) -> float:
        return max((r for _, r, _ in self.samples), default=0.0)

    def max_procs(self) -> int:
        return max((n for _, _, n in self.samples), default=0)


# ------------------------------------------------------------- kernel-workspace instrumentation


class _KernelProbe:
    """Record every label-pool rung ``search_dag`` allocates, with the bytes it costs.

    ``search_dag`` looks ``_search_dag`` up as a module global on each call, so replacing that
    attribute intercepts the rungs without touching the retry logic itself.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._orig = dp_kernel._search_dag

    def install(self) -> None:
        dp_kernel._search_dag = self._wrapper

    def restore(self) -> None:
        dp_kernel._search_dag = self._orig

    def _wrapper(self, *args):
        # Exactly two 2-D int32 arrays reach the kernel — label_recent (limit, depth) and
        # state_recent (state_cap, depth) — and state_cap >= 2*limit by construction, so the
        # smaller first dimension identifies the label pool without hard-coding an arg index.
        recents = sorted(
            (a for a in args if isinstance(a, np.ndarray) and a.ndim == 2 and a.dtype == np.int32),
            key=lambda a: a.shape[0],
        )
        assert len(recents) == 2, f"expected label_recent + state_recent, found {len(recents)}"
        limit, depth = recents[0].shape
        state_cap = dp_kernel._next_pow2(2 * limit)
        label_bytes = limit * (6 * 4 + 8 + 4 * depth)
        state_bytes = state_cap * (5 * 4 + 4 * depth)
        started = time.perf_counter()
        result = self._orig(*args)
        status, n_labels, n_cand, _remaining = result
        self.calls.append({
            "limit": limit,
            "depth": depth,
            "state_cap": state_cap,
            "workspace_bytes": label_bytes + state_bytes,
            "n_labels": int(n_labels),
            "n_candidates": int(n_cand),
            "status": int(status),
            "wall_s": time.perf_counter() - started,
        })
        return result


class _PriceProbe:
    """Per-flight graph geometry.  ``solver.py`` binds ``price_flight`` at import, so the
    patch has to land on ``solver_mod``, not on ``pricing``."""

    def __init__(self, kernel_probe: "_KernelProbe", n_expected: int, progress: bool = True) -> None:
        self.flights: list[dict] = []
        self.progress = progress
        self.retained: str = "(not sampled)"
        self._kernel = kernel_probe
        self._n_expected = n_expected
        self._orig = solver_mod.price_flight

    def install(self) -> None:
        solver_mod.price_flight = self._wrapper

    def restore(self) -> None:
        solver_mod.price_flight = self._orig

    def _wrapper(self, fg, *args, **kwargs):
        started = time.perf_counter()
        rss_before = _current_rss_bytes()
        kernel_calls_before = len(self._kernel.calls)
        try:
            return self._orig(fg, *args, **kwargs)
        finally:
            rungs = self._kernel.calls[kernel_calls_before:]
            record = {
                "flight_id": getattr(getattr(fg, "request", None), "flight_id", None),
                "levels": len(getattr(fg, "levels", ()) or ()),
                "min_step": getattr(fg, "min_step", None),
                "max_step": getattr(fg, "max_step", None),
                "shortest_hops": getattr(fg, "shortest_hops", None),
                "rss_before": rss_before,
                "rss_after": _current_rss_bytes(),
                "wall_s": time.perf_counter() - started,
                "n_rungs": len(rungs),
                "peak_workspace": max((r["workspace_bytes"] for r in rungs), default=0),
                "peak_labels": max((r["n_labels"] for r in rungs), default=0),
            }
            self.flights.append(record)
            if len(self.flights) == self._n_expected:
                # Walk NOW, not after solve() returns: `graphs` is a local inside
                # ColGenSolver.solve, so by the time the caller sees the result every
                # FlightGraph is already unreachable and a post-hoc walk finds nothing.
                self.retained = _retained_report()
            if self.progress:
                # Streamed so a run that is killed or runs long still leaves the per-flight
                # ladder on record — the whole point of the measurement.
                print(
                    f"  [{len(self.flights):>4}] fid={record['flight_id']:<8} "
                    f"{record['wall_s']:7.2f}s hops={record['shortest_hops']:<4} "
                    f"rungs={record['n_rungs']} labels={record['peak_labels']:>10,} "
                    f"ws={record['peak_workspace'] / _MB:6.1f}MB "
                    f"rss={record['rss_after'] / _MB:7.1f}MB",
                    flush=True,
                )


# --------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--flights", type=int, default=100, help="first N requests by (t_request, flight_id)")
    ap.add_argument("--time-limit", type=float, default=600.0, help="ColGenParams.time_limit_s")
    ap.add_argument("--max-iterations", type=int, default=1,
                    help="1 = fixed work (one pricing sweep), so memory is comparable run to run")
    ap.add_argument("--sample-interval", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 (default) = sequential sweep; N = fan the pricing sweep across N "
                         "processes via colgen.pricing_pool")
    ap.add_argument("--max-tasks-per-child", type=int, default=4,
                    help="recycle each worker after this many flights, returning its arena to "
                         "the OS; 0 = never recycle")
    ap.add_argument("--start-method", default=None, choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--benefit-m", type=float, default=None,
                    help="ColGenParams.M; must exceed the largest achievable column cost. "
                         "Oversizing it inflates the Lagrangian bound by M per LP-uncovered flight")
    ap.add_argument("--chunksize", type=int, default=1,
                    help="flights handed to a worker per dispatch (imap_unordered chunksize)")
    ap.add_argument("--objective", default="total_delay",
                    choices=["total_delay", "total_cost"],
                    help="total_delay sums ground and excess-air seconds unweighted; "
                         "total_cost weights them by the config dials (1:3)")
    ap.add_argument("--label-limit-max", type=int, default=None,
                    help="override dp_kernel.search_dag's label ceiling (default 1<<23); "
                         "pricing.py calls search_dag positionally through the module, so the "
                         "override is installed as a module-attribute wrapper")
    args = ap.parse_args()

    if args.label_limit_max is not None:
        _orig_search_dag = dp_kernel.search_dag

        def _search_dag_with_ceiling(*a, **kw):
            kw.setdefault("label_limit_max", args.label_limit_max)
            return _orig_search_dag(*a, **kw)

        dp_kernel.search_dag = _search_dag_with_ceiling
        print(f"label ceiling OVERRIDDEN: {args.label_limit_max:,} "
              f"(default {1 << 23:,})")

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    assert demand is not None, f"scenario {args.scenario} has no demand model"

    print(f"scenario  : {args.scenario}")
    print(f"region    : {cfg.region_size_m[0]:.0f} x {cfg.region_size_m[1]:.0f} m")
    print(f"levels    : {cfg.flight_levels_m}")
    print(f"horizon   : {cfg.horizon_s:.0f} s   demand window {cfg.demand_duration_s:.0f} s")
    print(f"lam/hour  : {cfg.lam_per_hour}")
    print(f"kernel    : {'ACTIVE' if dp_kernel is not None else 'ABSENT'}")

    t_gen = time.perf_counter()
    all_requests = demand.generate(cfg, np.random.default_rng(cfg.seed))
    t_gen = time.perf_counter() - t_gen
    ordered = sorted(all_requests, key=lambda r: (r.t_request, r.flight_id))
    requests = ordered[: args.flights]
    print(f"demand    : {len(all_requests)} requests generated in {t_gen:.1f}s; "
          f"taking first {len(requests)}")
    if requests:
        span = requests[-1].t_request - requests[0].t_request
        print(f"            t_request span {span:.1f}s  "
              f"({requests[0].t_request:.1f} .. {requests[-1].t_request:.1f})")

    # Always-active terminals are permanent infrastructure: sim.run registers EVERY placed hub,
    # not just the ones the sampled flights touch.  Registering only the sampled subset would
    # understate the capacity-row count and therefore the memory.
    static_terms = list(demand.terminals(cfg)) if cfg.terminal_airspace_always_active else []
    print(f"terminals : {len(static_terms)} static hub walls (all placed hubs)")

    params = ColGenParams(
        max_iterations=args.max_iterations,
        time_limit_s=args.time_limit,
        objective=args.objective,
        **({} if args.benefit_m is None else {"M": args.benefit_m}),
    )
    print(f"params    : max_iterations={params.max_iterations} time_limit_s={params.time_limit_s} "
          f"detour_slack_hops={params.detour_slack_hops} objective={params.objective}")
    print()

    pool_cfg = None
    if args.workers > 0:
        def _progress(done, total, flight_id, wall_s, kstats):
            # Streamed so a long sweep is legible while it runs, and so a run that is
            # killed still leaves its per-flight record on disk.
            print(f"  [{done:>4}/{total}] fid={flight_id:<7} {wall_s:8.2f}s "
                  f"labels={kstats.get('labels', 0):>10,} "
                  f"status={kstats.get('status', '-')} "
                  f"attempts={kstats.get('attempts', 0)} "
                  f"fallback={kstats.get('reference_fallback', False)}", flush=True)

        pool_cfg = ParallelPricingConfig(
            n_workers=args.workers,
            max_tasks_per_child=args.max_tasks_per_child or None,
            start_method=args.start_method,
            chunksize=args.chunksize,
            on_progress=_progress,
        )
        print(f"parallel  : {pool_cfg.n_workers} workers  "
              f"max_tasks_per_child={pool_cfg.max_tasks_per_child}  "
              f"start_method={pool_cfg.start_method or 'platform default'}")
        print(f"            chunksize={pool_cfg.chunksize}")
        print()

    # In parallel mode price_flight runs in the WORKERS, so these parent-side patches never
    # fire and the per-flight ladder detail is unavailable — deliberately, rather than
    # silently reporting an empty ladder as if nothing climbed.
    kprobe = _KernelProbe()
    pprobe = _PriceProbe(kprobe, len(requests))
    if pool_cfg is None:
        kprobe.install()
        pprobe.install()
    rss_start, _ = _tree_rss_bytes()
    try:
        with _RssSampler(args.sample_interval) as sampler:
            t0 = time.perf_counter()
            def _iter_line(st):
                # Streamed so a converging solve is legible while it runs -- the bound
                # and the gap are what say whether more iterations are worth buying,
                # and they were previously discarded if the run was killed.
                print(
                    f"  ITER {st['iteration']:>3}  elapsed={st['elapsed_s']:8.1f}s"
                    f"  cost_ub={st['cost_upper_bound']:13.2f}"
                    f"  cost_lb={st['cost_lower_bound']:13.2f}"
                    f"  lp_gap={st['lp_gap']:.6f}"
                    f"  cols={st['columns']:>6} (+{st['columns_added']})",
                    flush=True,
                )
                print(
                    f"           rc: n+={st['rc_n_positive']:>5} sum={st['rc_sum']:12.2f}"
                    f" max={st['rc_max']:9.2f} p90={st['rc_p90']:8.2f} p50={st['rc_p50']:8.2f}"
                    f"   duals: L2={st['dual_l2']:11.2f} Linf={st['dual_linf']:9.2f}"
                    f" nnz={st['dual_nonzero']:>6}"
                    f"   raw_ub={st['raw_upper_bound']:.2f}",
                    flush=True,
                )
                stages = st.get("stage_s", {})
                counts = st.get("stage_n", {})
                if stages:
                    parts = " ".join(
                        f"{k}={v:.2f}s/{counts.get(k, 0)}"
                        for k, v in sorted(stages.items(), key=lambda kv: -kv[1])
                    )
                    print(
                        f"           master: {parts}"
                        f"  lazy_rows={st['lazy_rows_added']}/{st['lazy_row_rounds']}r",
                        flush=True,
                    )
                print(
                    f"           coverage: uncovered={st['n_uncovered']:>4}"
                    f" rc~M={st['n_rc_near_M']:>4} overlap={st['n_overlap']:>4}"
                    f"   max_column_cost={st['max_column_cost']:.2f}",
                    flush=True,
                )

            result = ColGenSolver().solve(
                requests, cfg, static_terms, params,
                parallel=pool_cfg, on_iteration=_iter_line,
            )
            wall = time.perf_counter() - t0
    finally:
        if pool_cfg is None:
            kprobe.restore()
            pprobe.restore()

    stats = result.stats
    peak_rusage = _peak_rss_bytes()
    print("=" * 78)
    print(f"wall            : {wall:.2f}s")
    print(f"termination     : {stats.get('termination_reason')}  iterations={stats.get('iterations')}")
    print(f"gap_metric      : {stats.get('gap_metric', '?')}"
          f"   lp_gap={stats.get('lp_gap')}")
    print(f"                  (revenue={stats.get('lp_gap_revenue')}"
          f"  cost={stats.get('lp_gap_cost')})")
    print(f"IP gap (eq 11)  : {stats.get('ip_gap_revenue')}"
          f"   ip_objective={stats.get('ip_objective')}  ip_status={stats.get('ip_status')}")
    print(f"selected        : {stats.get('selected_flights')}/{len(requests)}")
    print(f"objective       : {stats.get('objective')}")
    print(f"columns         : {len(result.columns)}")
    print(f"backend         : {stats.get('backend')}")
    print()
    print(f"RSS at start    : {rss_start / _MB:8.1f} MB")
    print(f"RSS peak (tree) : {sampler.peak() / _MB:8.1f} MB   <-- headline "
          f"(parent + all workers, max {sampler.max_procs()} processes)")
    print(f"RSS peak parent : {peak_rusage / _MB:8.1f} MB   (ru_maxrss, this process only)")
    print(f"RSS growth      : {(sampler.peak() - rss_start) / _MB:8.1f} MB")
    if pool_cfg is not None:
        stats_par = {k: v for k, v in stats.items() if k.startswith("parallel_")}
        print(f"worker processes: {stats_par.get('parallel_worker_processes')} distinct pids "
              f"for {len(requests)} flights "
              f"(recycling {'ON' if pool_cfg.max_tasks_per_child else 'OFF'})")
        print(f"worker peak RSS : "
              f"{stats_par.get('parallel_worker_peak_rss_bytes', 0) / _MB:8.1f} MB "
              f"(max over workers, not their simultaneous total)")
        print(f"tasks discarded : {stats_par.get('parallel_tasks_discarded')} "
              f"(completed past the first timeout, dropped to keep the sweep deterministic)")
        sweep_wall = stats_par.get("parallel_sweep_wall_s", 0.0)
        task_total = stats_par.get("parallel_task_wall_total_s", 0.0)
        task_max = stats_par.get("parallel_task_wall_max_s", 0.0)
        n_w = max(1, pool_cfg.n_workers)
        print(f"sweep makespan  : {sweep_wall:8.2f}s   "
              f"(solve wall {wall:.2f}s -> {wall - sweep_wall:.2f}s is SERIAL: graph build, "
              f"seeding, heuristic, master LP/IP)")
        print(f"useful work     : {task_total:8.2f}s summed over tasks "
              f"= what the sequential sweep costs")
        print(f"efficiency      : {task_total / (sweep_wall * n_w) if sweep_wall else 0:8.2%} "
              f"of the {n_w} worker-slots kept busy")
        for fid, wall_s, peak_b, kstats in stats_par.get("parallel_slowest_flights", ())[:6]:
            print(f"    slow fid={fid:<7} {wall_s:8.2f}s  peak={peak_b / _MB:7.0f}MB  {kstats}")
        print(f"straggler       : {task_max:8.2f}s longest single task "
              f"= hard floor under the makespan "
              f"(max achievable sweep speedup {task_total / task_max if task_max else 0:.1f}x)")
    # Printed for both modes: once the sweep is parallel these stages ARE the wall clock, so
    # the breakdown says what to attack next rather than leaving "serial" as one opaque bar.
    graph_s = stats.get("graph_build_elapsed_s", 0.0)
    seed_s = stats.get("seed_elapsed_s", 0.0)
    to_master_s = stats.get("time_to_master_s", 0.0)
    greedy_s = stats.get("initial_greedy_elapsed_s", 0.0)
    named = graph_s + seed_s + to_master_s + greedy_s
    print(f"serial stages   : graph_build {graph_s:.2f}s  seeding {seed_s:.2f}s  "
          f"initial_greedy {greedy_s:.2f}s  time_to_master {to_master_s:.2f}s")
    print(f"                  {named:.2f}s named; the rest of the serial time is the "
          f"master LP/IP re-solves and _canonical_column in the parent")
    print()

    if kprobe.calls:
        by_limit: dict[int, int] = {}
        for c in kprobe.calls:
            by_limit[c["limit"]] = by_limit.get(c["limit"], 0) + 1
        worst = max(kprobe.calls, key=lambda c: c["workspace_bytes"])
        print(f"kernel invocations : {len(kprobe.calls)} "
              f"(incl. re-runs after overflow)")
        print(f"largest workspace  : {worst['workspace_bytes'] / _MB:.1f} MB "
              f"(limit={worst['limit']:,} depth={worst['depth']} "
              f"state_cap={worst['state_cap']:,})")
        print(f"peak labels used   : {max(c['n_labels'] for c in kprobe.calls):,}")
        print("status histogram   : "
              + ", ".join(f"{dp_kernel.STATUS_NAMES.get(s, s)}={n}" for s, n in sorted(
                  ((s, sum(1 for c in kprobe.calls if c['status'] == s))
                   for s in {c['status'] for c in kprobe.calls}))))
        print("  label-pool rungs (limit -> invocations, workspace):")
        for limit in sorted(by_limit):
            depth = next(c["depth"] for c in kprobe.calls if c["limit"] == limit)
            state_cap = dp_kernel._next_pow2(2 * limit)
            mb = (limit * (32 + 4 * depth) + state_cap * (20 + 4 * depth)) / _MB
            print(f"    {limit:>12,}  ->  {by_limit[limit]:>4} calls   {mb:8.1f} MB each")
    else:
        print("kernel invocations : 0  (no flight reached the compiled path)")
    print()

    if pprobe.flights:
        hops = [f["shortest_hops"] for f in pprobe.flights if f["shortest_hops"] is not None]
        steps = [f["max_step"] - f["min_step"] for f in pprobe.flights
                 if f["max_step"] is not None and f["min_step"] is not None]
        print(f"priced flights  : {len(pprobe.flights)}")
        if hops:
            print(f"shortest_hops   : min={min(hops)} median={sorted(hops)[len(hops) // 2]} max={max(hops)}")
        if steps:
            print(f"time layers     : min={min(steps)} median={sorted(steps)[len(steps) // 2]} max={max(steps)}")
        # The label ladder re-runs the WHOLE search at each rung, so a flight that climbs 4
        # rungs pays for 4 searches to get 1 answer.  This histogram is the size of that waste.
        rung_hist: dict[int, int] = {}
        for f in pprobe.flights:
            rung_hist[f["n_rungs"]] = rung_hist.get(f["n_rungs"], 0) + 1
        print("  label-ladder rungs climbed per flight (each rung = a full re-search):")
        for n in sorted(rung_hist):
            print(f"    {n} rung(s): {rung_hist[n]:>4} flights")
        wasted = sum(
            r["wall_s"] for r in kprobe.calls
            if r["status"] in (dp_kernel.FB_LABEL_OVERFLOW, dp_kernel.FB_HASH_FULL)
        )
        print(f"  wall spent on rungs that overflowed and were discarded: {wasted:.2f}s "
              f"({100.0 * wasted / wall:.1f}% of the solve)")
        slowest = sorted(pprobe.flights, key=lambda f: -f["wall_s"])[:5]
        print("  slowest priced flights:")
        for f in slowest:
            print(f"    fid={f['flight_id']:<8} {f['wall_s']:7.2f}s  hops={f['shortest_hops']} "
                  f"layers={(f['max_step'] or 0) - (f['min_step'] or 0)}  "
                  f"rss {f['rss_before'] / _MB:.0f} -> {f['rss_after'] / _MB:.0f} MB")

    print()
    print("live solver state, sampled during the LAST priced flight:")
    print(pprobe.retained)

    gc.collect()
    after = _current_rss_bytes()
    print(f"RSS after solve returns + gc.collect(): {after / _MB:8.1f} MB")
    print(f"  -> {(after - rss_start) / _MB:8.1f} MB is held after every FlightGraph is "
          f"unreachable (allocator retention / fragmentation, not live solver state)")
    return 0


_TRACKED = (
    "FlightGraph", "PreparedTopology", "PreparedDuals", "PreparedVariants",
    "_FlightSearchCache", "_LazyForbiddenHops", "_LazyCellCatalog",
)


def _retained_report() -> str:
    """Size the state the solver *holds*, as opposed to the peak it *touches*.

    Peak RSS cannot answer "does this scale to 4636 flights?", because it is a high-water
    mark: one flight's 864 MB label pool raises it forever even though the pool is freed
    immediately.  What scales with flight count is what stays reachable — ``solve`` keeps
    ``graphs: dict[int, FlightGraph]`` for the whole run so later CG iterations can reuse
    each graph, and each of those carries a lazily-grown arc cache plus the cached
    ``PreparedTopology``.  A gc walk is the honest way to size it: those objects are
    reachable only through ``solve``'s locals.
    """
    from collections import Counter

    gc.collect()
    counts: Counter = Counter()
    array_bytes: Counter = Counter()
    container_len: Counter = Counter()
    seen: set[int] = set()

    for obj in gc.get_objects():
        cls = type(obj).__name__
        if cls not in _TRACKED:
            continue
        counts[cls] += 1
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:                      # properties that compute or raise on access
                continue
            if isinstance(value, np.ndarray) and id(value) not in seen:
                seen.add(id(value))
                array_bytes[cls] += value.nbytes
            elif isinstance(value, (dict, set, frozenset, tuple)) and id(value) not in seen:
                seen.add(id(value))
                container_len[cls] += len(value)

    lines = [f"  RSS at sample time: {_current_rss_bytes() / _MB:.1f} MB"]
    for cls in sorted(counts):
        lines.append(
            f"    {cls:<22} {counts[cls]:>6} objects  "
            f"{array_bytes[cls] / _MB:8.1f} MB arrays  "
            f"{container_len[cls]:>12,} container entries"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
