"""Warm captured round-11 goal profiling; run arms serially in isolated processes.

Uninstrumented repeats establish timings. A separate profiled call attributes inclusive
and exclusive host/native phases; its wrapper overhead is reported, never hidden.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
from functools import wraps
import importlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

from benchmark_captured_sweep import digest, signature, write

HERE = Path(__file__).resolve().parent
SOURCES = {
    'baseline': '/private/tmp/colgen-prep-goal-baseline-20260910',
    'prep': '/private/tmp/colgen-prep-only-20260910',
    'combined': '/private/tmp/colgen-prep-goal-combined-20260910',
}


class PhaseProfile:
    """Temporary host wrappers with nested accounting and no production source changes."""

    def __init__(self, pricing, prepare, native):
        self.pricing, self.prepare, self.native = pricing, prepare, native
        self.inclusive, self.exclusive = defaultdict(float), defaultdict(float)
        self.calls = Counter()
        self.stack = []
        self.restores = []
        self.goal_data = []

    def timed(self, name, function, *args, **kwargs):
        """Attribute child time once while retaining each enclosing inclusive duration."""
        inside_goal = any(frame[0] == 'goal' for frame in self.stack)
        key = name if name == 'goal' else name + ('_in_goal' if inside_goal else '_outside_goal')
        frame = [key, 0.0]
        self.stack.append(frame)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            self.stack.pop()
            self.inclusive[key] += elapsed
            self.exclusive[key] += elapsed - frame[1]
            self.calls[key] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def patch(self, owner, name, wrapper):
        """Install one temporary wrapper and retain the exact original callable."""
        original = getattr(owner, name)
        self.restores.append((owner, name, original))
        setattr(owner, name, wrapper(original))

    def measured(self, name):
        """Create a method-compatible wrapper for one phase."""
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                return self.timed(name, function, *args, **kwargs)
            return wrapped
        return decorate

    def measured_in_sink(self, name):
        """Measure screening internals only while the original sink callback is active."""
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                if any(frame[0].startswith('canonical_sink_') for frame in self.stack):
                    return self.timed(name, function, *args, **kwargs)
                return function(*args, **kwargs)
            return wrapped
        return decorate

    def __enter__(self):
        def goal(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                variants, order = args[7:9]
                root_rows = []
                for root in order:
                    paid_class = int(variants.paid_class[root])
                    lo, hi = variants.paid_start[paid_class:paid_class + 2]
                    cells = variants.paid_cell[lo:hi]
                    root_rows.append(dict(root=int(root), paid_rows=int(hi - lo),
                                          paid_cells=len(set(cells.tolist()))))
                result = self.timed('goal', function, *args, **kwargs)
                self.goal_data.append(dict(
                    entry=kwargs.get('incumbent'), result=result, roots=root_rows,
                    expanded=int(self.pricing._LAST_SEARCH.get('n_labels', 0)),
                ))
                return result
            return wrapped

        def factory(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                return self.measured('canonical_sink')(function(*args, **kwargs))
            return wrapped

        self.patch(self.pricing, '_goal_directed_bootstrap', goal)
        self.patch(self.prepare, 'prepare_duals', self.measured('pack_duals'))
        self.patch(self.prepare.CompletionEnvelopes, '_destination_cost', self.measured('destination_cost'))
        self.patch(self.pricing, '_sink_certifier', factory)
        for name in ('_path_claims', '_path_delay_s', '_canonical_candidate'):
            self.patch(self.pricing, name, self.measured_in_sink(name.removeprefix('_')))
        self.patch(self.pricing.DualView, 'claim_cost', self.measured_in_sink('claim_cost'))
        if self.native is not None:
            self.patch(self.native, '_advance', self.measured('native_advance'))
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)

    def result(self):
        """Serialize after pricing so hashing itself does not enter phase timings."""
        return dict(inclusive_s=dict(self.inclusive), exclusive_s=dict(self.exclusive),
                    calls=dict(self.calls), goal=[dict(
                        entry_sha256=digest(item['entry']), output_sha256=digest(item['result']),
                        expanded=item['expanded'], roots=item['roots'],
                    ) for item in self.goal_data])


def run_arm(args):
    """Profile warm captured caller inputs through one frozen production source.

    Parameters
    ------------
    - args: Source, capture, baseline selection records, repeats and output locations.

    Return
    --------
    - result (dict): Exact signatures, ordinary timings and separately instrumented phases.
    """
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    from freespace_sim.planner.colgen import dp_kernel, dp_prepare, pricing
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.planner.colgen.params import ColGenParams

    assert source in Path(pricing.__file__).resolve().parents
    with args.capture.open('rb') as handle:
        captured = pickle.load(handle)  # Trusted local capture, never external pickle input.
    assert captured['round'] == 11
    params = ColGenParams(**{f.name: getattr(captured['params'], f.name)
                            for f in fields(ColGenParams) if hasattr(captured['params'], f.name)})
    rows = json.loads(args.selection_records.read_text())
    selected = sorted(rows, key=lambda r: (-r['search'].get('bootstrap_goal_s', 0.), r['flight_id']))[:args.top]
    requests = {r.flight_id: r for r in captured['requests']}
    cfg = captured['cfg']
    catalog = StaticTerminalCatalog(captured['static_terminals'], cfg)
    view = pricing.DualView(captured['duals'], cfg)
    try:
        native = importlib.import_module('freespace_sim.planner.colgen.bootstrap_kernel')
    except ModuleNotFoundError as exc:
        if exc.name != 'freespace_sim.planner.colgen.bootstrap_kernel':
            raise
        native = None
    warm_started = time.perf_counter()
    dp_kernel.warm_kernel()
    if native is not None:
        native.warm_kernel()
    warm_s = time.perf_counter() - warm_started
    result = dict(source_root=str(source), captured_round=11,
                  capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                  selection_records=str(args.selection_records.resolve()), params_sha256=digest(params),
                  source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((source / 'freespace_sim').rglob('*.py'))},
                  warm_s=warm_s, records=[], scope=(
                      'Same captured per-flight request, duals, known column and policy; fresh graph per '
                      'flight, one untimed solve, ordinary warm repeats, then one instrumented solve. '
                      'Each solve has a fresh invocation-local preparation and a new generous deadline.'))
    for chosen in selected:
        fid = chosen['flight_id']
        graph = build_flight_graph(requests[fid], cfg, catalog, params)

        def solve():
            pricing.clear_search_record()
            started = time.perf_counter()
            answer = pricing.price_flight(
                graph, view, captured['flight_duals'][fid], cfg, params,
                known_column=captured['known_columns'].get(fid),
                heuristic_only=captured.get('heuristic_only', False),
                deadline=time.monotonic() + args.budget,
            )
            elapsed = time.perf_counter() - started
            search = pricing.last_search_record()
            return dict(wall_s=elapsed, search=search, signature=signature(answer, search))

        warm = solve()
        ordinary = [solve() for _ in range(args.repeats)]
        with PhaseProfile(pricing, dp_prepare, native) as profile:
            measured = solve()
        measured['phases'] = profile.result()
        all_runs = [*ordinary, measured]
        exact_keys = ('output_sha256', 'identity_sha256', 'claims_sha256', 'cost_rc_sha256', 'work_sha256')
        parity = all(all(run['signature'][key] == ordinary[0]['signature'][key] for key in exact_keys)
                     for run in all_runs)
        if not parity:
            raise AssertionError(f'Profiling or warm-cache state changed flight {fid}')
        median_wall = statistics.median(r['wall_s'] for r in ordinary)
        item = dict(flight_id=fid, selected_cold_goal_s=chosen['search'].get('bootstrap_goal_s'),
                    input_sha256=digest((requests[fid], captured['flight_duals'][fid],
                                         captured['known_columns'].get(fid))),
                    warmup=warm, ordinary=ordinary, profiled=measured, within_arm_exact=True,
                    median_wall_s=median_wall,
                    median_goal_s=statistics.median(r['search'].get('bootstrap_goal_s', 0.) for r in ordinary),
                    instrumentation_wall_ratio=measured['wall_s'] / median_wall)
        result['records'].append(item)
        write(args.out / 'profile.json', result)
        print(json.dumps(dict(flight_id=fid, median_wall_s=item['median_wall_s'],
                              median_goal_s=item['median_goal_s'], phases=measured['phases'])), flush=True)
    return result


def main():
    """Run one arm, or orchestrate all three serially and require exact parity.

    Parameters
    ------------
    - CLI arguments: Optional source root, trusted capture, selection record and output paths.

    Return
    --------
    - status (int): Zero after successful execution and exact cross-arm checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--capture', type=Path, default=HERE / 'baseline_capture/pricing_captures/round_0011.pkl')
    parser.add_argument('--selection-records', type=Path, default=HERE / 'replay_matrix/03_forward_baseline_round_0011/records.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--budget', type=float, default=30.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.source_root is not None:
        run_arm(args)
        return 0
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    arms = {}
    for name, source in SOURCES.items():
        command = [sys.executable, str(Path(__file__).resolve()), '--source-root', source,
                   '--capture', str(args.capture), '--selection-records', str(args.selection_records),
                   '--out', str(args.out / name), '--top', str(args.top), '--repeats', str(args.repeats),
                   '--budget', str(args.budget)]
        with (args.out / f'{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=300 + args.top * (args.repeats + 2) * args.budget)
        arms[name] = json.loads((args.out / name / 'profile.json').read_text())
    baseline = {r['flight_id']: r for r in arms['baseline']['records']}
    parity = []
    keys = ('output_sha256', 'identity_sha256', 'claims_sha256', 'cost_rc_sha256', 'work_sha256')
    for arm, data in arms.items():
        for row in data['records']:
            expected = baseline[row['flight_id']]['ordinary'][0]['signature']
            actual = row['ordinary'][0]['signature']
            changed = [key for key in keys if expected[key] != actual[key]]
            original = baseline[row['flight_id']]
            if row['input_sha256'] != original['input_sha256']:
                changed.append('caller_input')
            a_goals = [{k: v for k, v in g.items() if k != 'roots'}
                       for g in row['profiled']['phases']['goal']]
            b_goals = [{k: v for k, v in g.items() if k != 'roots'}
                       for g in original['profiled']['phases']['goal']]
            if a_goals != b_goals:
                changed.append('goal_input_output_work')
            parity.append(dict(arm=arm, flight_id=row['flight_id'], changed=changed))
    write(args.out / 'comparison.json', dict(passed=all(not p['changed'] for p in parity),
                                            parity=parity, arms=arms,
                                            timing_note='Nested phase times must not be added. Instrumented '
                                            'host callbacks add overhead; ordinary timings are authoritative.'))
    if any(p['changed'] for p in parity):
        raise AssertionError('Cross-arm output/work mismatch')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
