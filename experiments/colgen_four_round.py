"""Profile four full-cohort colgen rounds on a fixed captured input.

Replays the archived 1,537-flight density_faa_wing_zipline seed-0 cohort (requests, config,
params and static terminals from a pricing capture) through the real solver for exactly four
rounds (12 CPU pricing workers and 4 Gurobi threads unless overridden), then stops before
the final IP. Each round's callback record (``iteration_wall_s``, ``stage_s``) and each
sweep's outer pricing time are saved, which is what the per-round "how much can faster
pricing shorten a round" breakdown reads. ``--rows-only`` sets ``iteration_ip_search=False``
so the per-round IP step adds its eager rows without searching. Round sums exclude startup;
``--profile`` adds timer wrappers inside the pricing workers.

    python -m experiments.colgen_four_round --capture CAPTURE.pkl --out results/four_round/X \
        --profile [--rows-only] [--workers N] [--gurobi-threads N] [--lock FILE]
"""

from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
from dataclasses import fields, replace
import fcntl
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TIMERS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
PENDING: dict[str, float] = {}


class PricingPrefixComplete(Exception):
    """End an analysis capture after four pricing sweeps, before further IP work."""


def timed(owner: Any, name: str, label: str) -> None:
    """Wrap a Python call boundary with an inclusive timer and call count."""
    original = getattr(owner, name)

    @functools.wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        """Preserve the call and record elapsed time even when it raises."""
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TIMERS[label][0] += time.perf_counter() - start
            TIMERS[label][1] += 1

    setattr(owner, name, observed)


def instrument(*, timing: bool = True) -> None:
    """Install analysis-only inclusive timers in the current process.

    Parameters
    ------------
    - timing (bool): include intrusive call timers, otherwise record only resources.

    Return
    --------
    - output (None): timer wrappers preserve arguments, results and exceptions.
    """
    from freespace_sim.planner.colgen import bootstrap_kernel, dp_kernel, dp_prepare
    from freespace_sim.planner.colgen import pricing, pricing_pool, solver

    for owner, name, label in (
        (pricing, 'seed_column', 'seed'),
        (solver, 'seed_column', 'parent_seed'),
        (solver, '_add_departure_ladder', 'parent_ladder'),
        (dp_prepare, 'prepare_topology', 'cold_topology'),
        (dp_prepare, 'prepare_rows', 'cold_rows'),
        (dp_prepare, 'prepare_duals', 'price_pack'),
        (dp_prepare, '_prepare_origin_data', 'origin_data'),
        (dp_prepare, 'prepare_variants', 'variants'),
        (dp_prepare.CompletionEnvelopes, '__init__', 'completion_init'),
        (dp_prepare.CompletionEnvelopes, 'delay_lower_bound', 'delay_scalar'),
        (dp_kernel, '_price_dag', 'native_dp'),
        (bootstrap_kernel, '_advance', 'native_goal'),
    ):
        if timing:
            timed(owner, name, label)
    original_load = pricing_pool._load_sweep_state

    def load(*args: Any, **kwargs: Any) -> Any:
        """Attribute the once-per-worker snapshot load to its first task."""
        start = time.perf_counter()
        result = original_load(*args, **kwargs)
        PENDING['snapshot_s'] = time.perf_counter() - start
        return result

    pricing_pool._load_sweep_state = load
    original_one = pricing_pool._price_one

    def one(*args: Any, **kwargs: Any) -> Any:
        """Attach nonfunctional stage measurements to an existing result record."""
        before = {key: value.copy() for key, value in TIMERS.items()}
        result = original_one(*args, **kwargs)
        record = result[6]
        record['experiment_profile'] = {
            key: [value[0] - before.get(key, [0, 0])[0],
                  value[1] - before.get(key, [0, 0])[1]]
            for key, value in TIMERS.items()
        }
        record['experiment_profile'].update({key: [value, 1] for key, value in PENDING.items()})
        PENDING.clear()
        record['experiment_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        graph = pricing_pool._WORKER['graphs'][int(args[1])]
        cache = getattr(graph._search_cache, 'cpu_completion', None)
        if cache is not None:
            values = getattr(cache, 'delay_values', getattr(cache, 'air_values', {}))
            record['experiment_cache'] = dict(
                keys=len(values), scalars=sum(map(len, values.values())),
                hits=getattr(cache, 'hits', None), misses=getattr(cache, 'misses', None),
                scalar_hits=getattr(cache, 'scalar_hits', None),
                scalar_misses=getattr(cache, 'scalar_misses', None),
            )
        return result

    pricing_pool._price_one = one


def worker_main(*args: Any, **kwargs: Any) -> None:
    """Install profiling in a spawned worker before its ordinary entry point."""
    sys.path.insert(0, os.environ['CPU_EXPERIMENT_SOURCE'])
    from freespace_sim.planner.colgen import pricing_pool
    profile = os.environ.get('CPU_EXPERIMENT_PROFILE') == '1'
    if profile or os.environ.get('CPU_EXPERIMENT_RESOURCES') == '1':
        instrument(timing=profile)
    pricing_pool._worker_main(*args, **kwargs)


def main() -> None:
    """Run a full-cohort solve or identical-input pricing replay under one lock.

    Parameters
    ------------
    - None: CLI selects source, capture, output, mode and optional profiling.

    Return
    --------
    - output (None): saves exact inputs/results, timing and source provenance.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('solve', 'replay'), default='solve')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--resources', action='store_true')
    parser.add_argument('--lock', type=Path, default=None,
                        help='File to flock before measuring, e.g. a machine-wide benchmark lock.')
    parser.add_argument('--workers', type=int, default=12,
                        help='Pricing worker processes (n_pricing_workers).')
    parser.add_argument('--gurobi-threads', type=int, default=4,
                        help='Gurobi Threads for the master LP and IPs.')
    parser.add_argument('--rows-only', action='store_true',
                        help='Per-round step adds eager rows only (iteration_ip_search=False).')
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    harness_bytes = Path(__file__).read_bytes()
    (args.out / 'harness_used.py').write_bytes(harness_bytes)
    sys.path.insert(0, str(args.source))
    os.environ['CPU_EXPERIMENT_SOURCE'] = str(args.source)
    os.environ['CPU_EXPERIMENT_PROFILE'] = str(int(args.profile))
    os.environ['CPU_EXPERIMENT_RESOURCES'] = str(int(args.resources))
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMBA_NUM_THREADS='1',
                      COLGEN_GUROBI_THREADS=str(args.gurobi_threads))
    from freespace_sim.planner.colgen import solver, dp_kernel, bootstrap_kernel, pricing_pool
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.network import build_flight_graph
    from freespace_sim.planner.colgen.pricing import DualView, seed_column
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.sim import run
    from analysis.colgen.common import encode, digest, source_hashes

    assert args.source in Path(solver.__file__).resolve().parents
    payload = pickle.loads(args.capture.read_bytes())
    requests, cfg = payload['requests'], payload['cfg']
    assert len(requests) == 1537
    params = ColGenParams(**{f.name: getattr(payload['params'], f.name)
                            for f in fields(ColGenParams) if hasattr(payload['params'], f.name)})
    params = replace(params, solver='gurobi', pricing_device='cpu', pricing_gpu_backend=None,
                     n_pricing_workers=args.workers, max_iterations=4,
                     iteration_ip_search=not args.rows_only)
    # The per-round step, and so --rows-only, only runs with a positive per-round IP limit.
    assert params.iteration_ip_time_limit_s > 0, params.iteration_ip_time_limit_s
    rounds: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    original_sweep, original_solve = solver.price_sweep, solver.ColGenSolver.solve
    initial: dict[str, Any] = {}

    def observe_sweep(order: Any, reqs: Any, graphs: Any, config: Any, par: Any,
                      catalog: Any, duals: Any, view: Any, flight_duals: Any,
                      known: Any, **kwargs: Any) -> Any:
        """Capture inputs before the unmodified sweep; serialization follows the solve."""
        snap = dict(duals=dict(duals), flight_duals=dict(flight_duals), known=dict(known),
                    order=list(order), heuristic_only=kwargs.get('heuristic_only', False))
        if not initial:
            initial['catalog'] = catalog
            initial['seed_state'] = {
                fid: dict(seed_columns=g._search_cache.seed_columns,
                          seed_model=g._search_cache.seed_model,
                          seed_search_complete=g._search_cache.seed_search_complete,
                          seed_delay_certified=g._search_cache.seed_delay_certified)
                for fid, g in graphs.items()
            }
        start = time.perf_counter()
        result = original_sweep(order, reqs, graphs, config, par, catalog, duals, view,
                                flight_duals, known, **kwargs)
        snap.update(outer_s=time.perf_counter()-start, sweep_s=result.wall_s,
                    pool_setup_s=result.pool_setup_s, flight_ids=result.flight_ids,
                    reduced_costs=result.reduced_costs, columns=result.columns,
                    records=result.flight_records, complete=result.complete,
                    certified=result.certified, kernel_counters=dict(result.kernel_counters))
        snapshots.append(snap)
        print(json.dumps(dict(event='pricing', round=len(snapshots), outer_s=snap['outer_s'],
                              complete=result.complete, flights=len(result.flight_ids))), flush=True)
        if args.mode == 'solve' and len(snapshots) == 4:
            assert result.complete and result.certified and len(result.flight_ids) == 1537
        return result

    def observe_solve(self: Any, *positional: Any, **kwargs: Any) -> Any:
        """Save scalar round diagnostics while preserving an existing callback."""
        callback = kwargs.get('on_iteration')

        def record(value: dict[str, Any]) -> None:
            """Exclude the live master from captured per-round scalar metadata."""
            rounds.append({k: v for k, v in value.items()
                           if isinstance(v, (str, int, float, bool, type(None))) or k == 'stage_s'})
            print(json.dumps(dict(event='round', round=value['iteration'],
                                  elapsed_s=value.get('elapsed_s'))), flush=True)
            if callback is not None:
                callback(value)
            if args.mode == 'solve' and len(rounds) == 4:
                raise PricingPrefixComplete

        kwargs['on_iteration'] = record
        return original_solve(self, *positional, **kwargs)

    # --lock serializes with other benchmarks on a shared machine; a job owning its node
    # needs none.
    with contextlib.ExitStack() as stack:
        if args.lock is not None:
            lock = stack.enter_context(args.lock.open('a'))
            print(f'waiting for benchmark lock {args.lock}', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.perf_counter()
        dp_kernel.warm_kernel()
        bootstrap_kernel.warm_kernel()
        warmup_s = time.perf_counter() - start
        if args.profile:
            instrument()
        if args.profile or args.resources:
            pricing_pool._worker_main = worker_main
        solver.price_sweep, solver.ColGenSolver.solve = observe_sweep, observe_solve
        start = time.perf_counter()
        if args.mode == 'solve':
            try:
                result = run(cfg, requests=requests, planner_name='colgen', planner_params=params,
                             static_terminals=payload['static_terminals'], progress=False)
            except PricingPrefixComplete:
                outcome = dict(pricing_prefix_complete=True, final_ip_skipped=True,
                               stop_reason='analysis_four_pricing_sweeps', stats={})
            else:
                outcome = dict(verified=result.verified, accepted=len(result.accepted),
                               denied=len(result.denied), stats=result.planner_stats)
        else:
            catalog = payload['initial']['catalog']
            graphs = {r.flight_id: build_flight_graph(r, cfg, catalog, params) for r in requests}
            seed_start = time.perf_counter()
            for graph in graphs.values():
                try:
                    seed = seed_column(graph, cfg, model=cost_model(cfg, params))
                    solver._canonical_column(seed, graph, cfg)
                except ValueError:
                    pass
            pool_options = ({'seed_graphs': graphs}
                            if 'seed_graphs' in inspect.signature(pricing_pool.PricingPool).parameters
                            else {})
            pool = pricing_pool.PricingPool(requests, cfg, params, catalog, **pool_options)
            try:
                seed_s = time.perf_counter() - seed_start
                for snap in payload['snapshots']:
                    observe_sweep(snap['order'], requests, graphs, cfg, params, catalog,
                                  snap['duals'], DualView(snap['duals'], cfg), snap['flight_duals'],
                                  snap['known'], pool=pool, heuristic_only=snap['heuristic_only'])
            finally:
                pool.close()
            outcome = dict(seed_stage_s=seed_s, replay=True)
        elapsed = time.perf_counter() - start
    summary = dict(source=str(args.source), source_sha256=source_hashes(args.source),
                   capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                   harness_sha256=hashlib.sha256(harness_bytes).hexdigest(),
                   profile=args.profile, resources=args.resources,
                   mode=args.mode, params=encode(params),
                   requests_sha256=digest(requests), warmup_s=warmup_s, elapsed_s=elapsed,
                   rounds=rounds, parent_timers=dict(TIMERS), **outcome)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    (args.out/'rounds.pkl').write_bytes(pickle.dumps(dict(requests=requests, cfg=cfg, params=params,
            static_terminals=payload['static_terminals'], initial=initial, snapshots=snapshots),
            protocol=pickle.HIGHEST_PROTOCOL))
    print(json.dumps(dict(event='complete', elapsed_s=elapsed, rounds=len(snapshots))), flush=True)


if __name__ == '__main__':
    # Hashed-key iteration order must match the archived runs. Re-executing once pins the seed
    # however the module is launched, without relying on the caller to set PYTHONHASHSEED.
    if os.environ.get('PYTHONHASHSEED') != '0':
        # orig_argv keeps the exact invocation (interpreter flags, -m or a file path).
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]],
                  {**os.environ, 'PYTHONHASHSEED': '0'})
    main()
