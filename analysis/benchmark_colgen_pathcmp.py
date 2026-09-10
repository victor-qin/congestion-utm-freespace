"""Matched frozen-source pricing replays for native path-comparison changes.

Single-flight timing follows one untimed solve per fresh graph; every measured
repeat must preserve bit-exact output and work-count hashes. Sweep timing includes
12-worker startup/preparation and runs only the captured first-LP pricing sweep.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time


def encode(value):
    """Serialize exact float bits and unordered claim sets deterministically."""
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((encode(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def digest(value):
    """Hash the exact canonical representation rather than pickle implementation details."""
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


def write(path, value):
    """Persist readable records after every completed bounded unit of work."""
    path.write_text(json.dumps(value, indent=2) + '\n')


def signature(answer, search):
    """Separate identity, claims, numeric output, and search work for parity review."""
    rc, column = answer
    counts = {k: search[k] for k in ('n_labels', 'attempts', 'bootstrap_labels',
        'bootstrap_goal_labels', 'bootstrap_dp_labels', 'status', 'declined') if k in search}
    identity = None if column is None else (column.flight_id, column.departure_step,
        column.level, column.origin_lane_idx, column.dest_lane_idx, column.cell_path)
    return dict(output_sha256=digest(answer), identity_sha256=digest(identity),
                claims_sha256=digest(None if column is None else column.claims),
                cost_rc_sha256=digest((rc, None if column is None else column.delay_s)),
                work_sha256=digest(counts), work=counts, rc=rc,
                cost=None if column is None else column.delay_s,
                returned_column=column is not None,
                claim_count=0 if column is None else len(column.claims))


def main():
    """Run one explicit source and compare its artifacts with a separate baseline.

    Parameters
    ------------
    - CLI arguments: Frozen source, trusted capture, output, mode and repeat settings.

    Return
    --------
    - None: Persist timings, exact signatures, provenance and optional parity checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--mode', choices=('single', 'sweep'), default='single')
    parser.add_argument('--flights', type=int, nargs='+', default=[934, 2692, 1520, 1600, 1640, 2282])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--instrument', action='store_true',
                        help='Run a separate instrumented call after measured trials.')
    parser.add_argument('--budget', type=float, default=1800)
    parser.add_argument('--flight-budget', type=float, default=120)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh output directory')
    args.out.mkdir(parents=True)
    harness_source = Path(__file__).read_bytes()
    (args.out/'harness_used.py').write_bytes(harness_source)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMBA_NUM_THREADS='1', COLGEN_GUROBI_THREADS='4')
    from freespace_sim.planner.colgen import dp_kernel, pricing, pricing_pool
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.planner.colgen.params import ColGenParams
    assert source in Path(dp_kernel.__file__).resolve().parents
    with args.capture.open('rb') as handle:
        captured = pickle.load(handle)  # Explicit trusted local first-LP capture.
    old = captured['params']
    values = {f.name: getattr(old, f.name) for f in fields(ColGenParams) if hasattr(old, f.name)}
    values.update(max_air_overrun_hops=6, bootstrap_method='astar', columns_per_flight=1,
                  n_pricing_workers=12)
    params = ColGenParams(**values)
    cfg = captured['cfg']
    catalog = StaticTerminalCatalog(captured['static_terminals'], cfg)
    manifest = dict(source_root=str(source), capture=str(args.capture.resolve()),
        harness_sha256=hashlib.sha256(harness_source).hexdigest(),
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(), params=encode(params),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((source/'freespace_sim').rglob('*.py'))},
        mode=args.mode, repeats=args.repeats, workers=12 if args.mode == 'sweep' else 0,
        scope='Frozen captured first-LP duals; six extra hops; hybrid bootstrap; no master/IP solve')
    write(args.out/'manifest.json', manifest)
    write(args.out/'status.json', dict(state='running'))
    started_all = time.perf_counter()
    records = []
    try:
        if args.mode == 'single':
            dp_kernel.warm_kernel()
            import inspect
            profile = None
            if args.instrument:
                from profile_native_dp import NativeProfile
                profile = NativeProfile(dp_kernel)
            native = dp_kernel._price_dag
            arg_names = list(inspect.signature(native.py_func).parameters)
            native_s = 0.0
            native_calls = 0
            arena_bytes = 0
            def measured(*a):
                nonlocal native_s, native_calls, arena_bytes
                started = time.perf_counter()
                try:
                    return native(*a)
                finally:
                    native_s += time.perf_counter()-started
                    native_calls += 1
                    arena_bytes = max(arena_bytes, sum(v.nbytes for n, v in zip(arg_names, a)
                        if n.startswith('label_') and hasattr(v, 'nbytes')))
            for fid in args.flights:
                req = next(r for r in captured['requests'] if r.flight_id == fid)
                graph = build_flight_graph(req, cfg, catalog, params)
                view = pricing.DualView(captured['duals'], cfg)
                def run():
                    pricing.clear_search_record()
                    return pricing.price_flight(graph, view, captured['flight_duals'][fid], cfg,
                        params, known_column=captured['known_columns'].get(fid),
                        deadline=time.monotonic()+args.flight_budget)
                warm_start = time.perf_counter()
                warm_answer = run()
                warm_s = time.perf_counter()-warm_start
                expected = signature(warm_answer, pricing.last_search_record())
                trials = []
                for trial in range(args.repeats):
                    native_s = 0.0
                    native_calls = arena_bytes = 0
                    dp_kernel._price_dag = measured
                    start = time.perf_counter()
                    try:
                        answer = run()
                    finally:
                        elapsed = time.perf_counter()-start
                        dp_kernel._price_dag = native
                    observed = signature(answer, pricing.last_search_record())
                    assert observed['output_sha256'] == expected['output_sha256'], (fid, 'warm output drift')
                    # First solve can grow graph budgets; work parity is checked between warm trials.
                    if trials:
                        assert observed['work_sha256'] == trials[0]['signature']['work_sha256']
                    trials.append(dict(trial=trial, wall_s=elapsed, native_s=native_s,
                                       native_calls=native_calls, label_arena_bytes=arena_bytes,
                                       signature=observed, search=pricing.last_search_record()))
                record = dict(flight_id=fid, untimed_first_solve_s=warm_s,
                    median_wall_s=statistics.median(t['wall_s'] for t in trials),
                    median_native_s=statistics.median(t['native_s'] for t in trials), trials=trials)
                if profile is not None:
                    profile.counts.fill(0)
                    profile.native_s = profile.compile_s = 0.0
                    profile.max_labels = profile.label_capacity_bytes = profile.state_capacity_bytes = 0
                    with profile.installed():
                        instrumented_answer = run()
                    assert digest(instrumented_answer) == expected['output_sha256']
                    record['instrumentation'] = profile.report()
                records.append(record)
                write(args.out/'records.json', records)
                print(json.dumps({k:v for k,v in record.items() if k not in ('trials', 'instrumentation')}), flush=True)
        else:
            start = time.perf_counter()
            result = pricing_pool.price_sweep(captured['pricing_order'], captured['requests'], {},
                cfg, params, catalog, captured['duals'], pricing.DualView(captured['duals'], cfg),
                captured['flight_duals'], captured['known_columns'],
                deadline=time.monotonic()+args.budget)
            elapsed = time.perf_counter()-start
            by_flight = {r['flight_id']:r for r in result.flight_records}
            records = [dict(flight_id=fid, signature=signature((rc, col), by_flight.get(fid, {})),
                            search=by_flight.get(fid, {}))
                       for fid,rc,col in zip(result.flight_ids, result.reduced_costs, result.columns, strict=True)]
            write(args.out/'records.json', records)
            with (args.out/'columns.pkl').open('wb') as handle:
                pickle.dump(dict(zip(result.flight_ids, result.columns, strict=True)), handle)
            summary = dict(complete=result.complete, flights=len(records), wall_s=elapsed,
                           task_total_s=result.task_total_s, pool_setup_s=result.pool_setup_s,
                           kernel_counters=dict(result.kernel_counters), timeout_flight_id=result.timeout_flight_id)
            write(args.out/'sweep.json', summary)
            print(json.dumps(summary), flush=True)
            assert result.complete and len(records) == len(captured['requests'])
        if args.reference:
            baseline = {r['flight_id']:r for r in json.loads((args.reference/'records.json').read_text())}
            mismatches = []
            for record in records:
                expected = baseline[record['flight_id']]
                a = expected['trials'][0]['signature'] if args.mode == 'single' else expected['signature']
                b = record['trials'][0]['signature'] if args.mode == 'single' else record['signature']
                changed = [key for key in ('output_sha256','identity_sha256','claims_sha256',
                                          'cost_rc_sha256','work_sha256') if a[key] != b[key]]
                if changed:
                    mismatches.append(dict(flight_id=record['flight_id'], changed=changed, expected=a, actual=b))
            write(args.out/'parity.json', dict(passed=not mismatches, compared=len(records), mismatches=mismatches))
            assert not mismatches, f'{len(mismatches)} flights differ; inspect parity.json'
        write(args.out/'status.json', dict(state='complete', wall_s=time.perf_counter()-started_all))
    except BaseException as exc:
        write(args.out/'status.json', dict(state='failed', error=repr(exc), wall_s=time.perf_counter()-started_all))
        raise


if __name__ == '__main__':
    main()
