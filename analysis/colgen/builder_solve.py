"""Direct production hard88 solver comparison with configurable synchronous preparation workers.

Adapted from parallel_profile.py; run under builder_locked.py with PYTHONPATH selecting the base
or candidate source. Timed run and all solver/numeric settings otherwise remain unchanged.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
from dataclasses import fields, replace
import json
from pathlib import Path
import pstats
import sys
import time
from typing import Any
from unittest.mock import patch

from analysis.colgen.common import digest, encode, source_hashes, write_json
from analysis.colgen import gpu_validation
from analysis.colgen.platform_replay import load_capture
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.solver import ColGenSolver
from freespace_sim.sim import run


def main(argv: list[str] | None = None) -> int:
    """Measure one fresh GPU or CPU solve with unchanged iteration callbacks.

    Parameters
    ------------
    - argv (list[str] | None): backend, capture, output, and optional owner profiling.

    Return
    --------
    - output (int): zero for a verified completed solve; artifacts preserve incomplete results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--direct', action='store_true',
                        help='Time the production solver without per-batch validation file writes; '
                             'GPU full-CPU fallback and worker-device guards remain enabled')
    parser.add_argument("--prepare-workers", type=int, default=12)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Choose a fresh output directory')
    root = Path(__file__).resolve().parents[2]
    selected = json.loads((root / 'analysis/colgen/arrival_union_results.json').read_text())
    ids = [r['flight_id'] for r in selected['hard88']['after']['rows']]
    rounds: list[dict[str, Any]] = []
    original = ColGenSolver.solve

    def observed(self: ColGenSolver, *positional: Any, **kwargs: Any) -> Any:
        """Keep the solver callback's own sweep timer; do not sum overlapping GPU timers."""
        callback = kwargs.get('on_iteration')

        def record(value: dict[str, Any]) -> None:
            """Save scalars and stage accounting without retaining the master or dual snapshot."""
            rounds.append({k: v for k, v in value.items()
                           if isinstance(v, (str, int, float, bool, type(None)))
                           or k == 'stage_s'})
            gpu_validation.save_report(args.out / 'rounds.json', {'rounds': rounds})
            if callback is not None:
                callback(value)
        kwargs['on_iteration'] = record
        return original(self, *positional, **kwargs)

    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    try:
        with patch.object(ColGenSolver, 'solve', observed):
            if profiler is not None:
                profiler.enable()
            if args.device == 'gpu' and not args.direct:
                status = gpu_validation.main([
                    'solve', '--capture', str(args.capture), '--out', str(args.out),
                    '--flight-ids', *map(str, ids), '--integer-bounds', '--approach-bounds',
                    '--prepare-workers', str(args.prepare_workers), '--prepare-batch-size', '12',
                    '--prepare-cache-bytes', '178956970', '--batch-size', '1',
                    '--max-labels', '250000', '--chunk-size', '2048', '--budget', '900',
                ])
            else:
                captured, capture_hash = load_capture(args.capture, ids)
                by_id = {r.flight_id: r for r in captured['requests']}
                requests = [by_id[i] for i in ids]
                params = ColGenParams(**{f.name: getattr(captured['params'], f.name)
                                        for f in fields(ColGenParams)
                                        if hasattr(captured['params'], f.name)})
                params = replace(params, solver='gurobi', pricing_device='cpu',
                                 pricing_gpu_backend=None, n_pricing_workers=12,
                                 time_limit_s=900.0, ip_time_limit_s=60.0, ip_reserve_s=65.0)
                if args.device == 'gpu':
                    params = replace(params, pricing_device='gpu', pricing_gpu_backend='metal',
                                     pricing_cpu_recompute=True, n_pricing_workers=0,
                                     pricing_gpu_integer_bounds=True, pricing_gpu_approach_bounds=True,
                                     pricing_gpu_prepare_workers=args.prepare_workers, pricing_gpu_prepare_batch_size=12,
                                     pricing_gpu_prepare_cache_bytes=178956970,
                                     pricing_gpu_batch_size=1, pricing_gpu_max_labels=250000,
                                     pricing_gpu_chunk_size=2048)
                args.out.mkdir(parents=True)
                write_json(args.out / 'manifest.json', {
                    'capture_sha256': capture_hash, 'flight_ids': ids,
                    'requests_sha256': digest(gpu_validation.serializable(requests)),
                    'config_sha256': digest(gpu_validation.serializable(captured['cfg'])),
                    'static_terminals_sha256': digest(gpu_validation.serializable(captured['static_terminals'])),
                    'run_params': encode(params), 'source_sha256': source_hashes(Path(run.__code__.co_filename).resolve().parents[1]),
                })
                with ExitStack() as guards:
                    if args.device == 'gpu':
                        from freespace_sim.planner.colgen import gpu_prepare_pool, pricing, pricing_pool, solver
                        from freespace_sim.planner.colgen.gpu_pricing import _Batch

                        def forbid_cpu(*positional: Any, **kwargs: Any) -> Any:
                            """Reject full CPU pricing and CPU per-prune numeric certification."""
                            raise AssertionError('CPU pricing or numeric certification in GPU benchmark')

                        guards.enter_context(patch.object(gpu_prepare_pool, '_worker_main',
                                                          gpu_validation.cpu_only_preparation_worker))
                        for module in (pricing, pricing_pool, solver):
                            guards.enter_context(patch.object(module, 'price_flight', forbid_cpu))
                        for name in ('_best_column', '_best_column_compiled'):
                            guards.enter_context(patch.object(pricing, name, forbid_cpu))
                        guards.enter_context(patch.object(_Batch, 'bound', forbid_cpu))
                    solve_started = time.perf_counter()
                    result = run(captured['cfg'], requests=requests, planner_name='colgen',
                                 planner_params=params, static_terminals=captured['static_terminals'],
                                 progress=False)
                    elapsed = time.perf_counter() - solve_started
                stats = result.planner_stats
                passed = (result.verified and len(result.accepted) == len(ids) and not result.denied
                          and stats['termination_reason'] == 'lp_gap'
                          and stats.get('kernel_fell_back', 0) == 0
                          and stats['pricing_search_complete'] and stats['pricing_bound_certified'])
                gpu_validation.save_report(args.out / 'planner_stats.json', stats)
                write_json(args.out / 'result.json', {'elapsed_s': elapsed, 'passed': passed,
                           'verified': result.verified, 'accepted': len(result.accepted),
                           'denied': len(result.denied), 'device': args.device,
                           'cpu_full_pricing_guard': args.device == 'gpu',
                           'cpu_numeric_verifier_guard': args.device == 'gpu',
                           'timing_scope': 'Production run, including iteration callback; no extra batch audits'})
                status = 0 if passed else 1
    finally:
        if profiler is not None:
            profiler.disable()
            args.out.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.out / 'owner.prof'))
            stats = pstats.Stats(profiler)
            functions = [dict(file=f, line=line, function=name, primitive_calls=v[0],
                              calls=v[1], self_s=v[2], cumulative_s=v[3])
                         for (f, line, name), v in stats.stats.items()]
            write_json(args.out / 'profile.json', {
                'scope': 'Owner process only; worker CPU stages are recorded in pricing telemetry.',
                'profile_wall_s': time.perf_counter() - started,
                'total_calls': stats.total_calls,
                'by_self': sorted(functions, key=lambda x: x['self_s'], reverse=True)[:40],
                'by_cumulative': sorted(functions, key=lambda x: x['cumulative_s'], reverse=True)[:40],
            })
    print(json.dumps({'device': args.device, 'profiled': args.profile, 'status': status,
                      'rounds': [{k: r.get(k) for k in ('iteration', 'sweep_s', 'iteration_wall_s')}
                                 for r in rounds]}), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
