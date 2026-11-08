"""Cold-builder ablations on saved legal inputs; all sustained work holds the shared lock."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path('/Users/victor/Documents/Research/congestion-utm-freespace')
CHECKOUT = Path(__file__).resolve().parents[2]
CAPTURES = [ROOT / '.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl',
            ROOT / '.agent/runs/colgen-capture-20261105-211249/pricing_captures/round_0011.pkl']
STAGES = ('corridor', 'roles', 'geometry', 'endpoint')


def restore_stages(enabled: set[str]) -> None:
    """Restore disabled builders from the frozen base without changing production flags.

    Parameters
    ------------
    - enabled (set[str]): builders retained from the candidate source.

    Return
    --------
    - output (None): replaces only disabled entry points with their base definitions.
    """
    from freespace_sim.planner.colgen import network, gpu_bounds, gpu_endpoint
    targets = [('corridor', network, '_LazyCorridorCells', '_within_budget'),
               ('roles', network, None, '_static_hop_allowed_roles'),
               ('geometry', gpu_bounds, 'CompletionBounds', '_build_geometry'),
               ('endpoint', gpu_endpoint, 'EndpointBounds', '__init__')]
    for stage, module, cls, name in targets:
        if stage in enabled:
            continue
        path = CHECKOUT / '.agent/builder-base' / Path(module.__file__).relative_to(
            Path(module.__file__).parents[3])
        tree = ast.parse(path.read_text())
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree
        node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), module.__dict__, namespace)
        setattr(getattr(module, cls) if cls else module, name, namespace[name])


def fingerprint(value: object) -> str:
    """Hash exact array bytes and canonical unordered memberships outside measured intervals.

    Parameters
    ------------
    - value (object): data-only preparation payload component.

    Return
    --------
    - output (str): SHA256 digest independent of set/dictionary insertion order.
    """
    import numpy as np
    h = hashlib.sha256()
    def add(v: object) -> None:
        """Append typed array bytes or recursively normalized values."""
        if isinstance(v, np.ndarray):
            h.update(str((v.dtype.str, v.shape)).encode())
            h.update(v.tobytes())
        elif is_dataclass(v):
            for f in fields(v):
                h.update(f.name.encode())
                add(getattr(v, f.name))
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                add(k)
                add(v[k])
        elif isinstance(v, (set, frozenset)):
            for item in sorted(v, key=repr):
                add(item)
        elif isinstance(v, (tuple, list)):
            h.update(str(len(v)).encode())
            for item in v:
                add(item)
        else:
            h.update(repr(v).encode())
        h.update(b'\0')
    add(value)
    return h.hexdigest()


def main() -> None:
    """Prepare cold/warm/changed/restored inputs and save exact component fingerprints.

    Parameters
    ------------
    - None: command-line arguments select the source, stages and output.

    Return
    --------
    - output (None): writes per-phase timings and comparison evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stages', nargs='*', choices=STAGES, default=list(STAGES))
    parser.add_argument('--source', type=Path, default=CHECKOUT)
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--early-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a fresh output')
    sys.path.insert(0, str(args.source.resolve()))
    with (ROOT / '.agent/gpu-followups-20261106/metal-benchmark.lock').open('a') as lock:
        print('Waiting for shared lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('Acquired shared lock', flush=True)
        from analysis.colgen.platform_replay import load_capture
        from freespace_sim.planner.colgen import pricing
        from freespace_sim.planner.colgen.gpu_prepared import prepare_flight, export_flight
        from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
        from freespace_sim.planner.colgen.params import ColGenParams
        from freespace_sim.planner.colgen.pricing_platform import PricingOptions, PricingProblem
        if args.source.resolve() == CHECKOUT:
            restore_stages(set(args.stages))
        ids = ([0, 2, 32, 34, 276, 278, 320, 322, 480, 806, 1638, 1646, 2526] if args.controls else
               json.loads((CHECKOUT / 'analysis/colgen/gpu_hard88_measurements.json').read_text())['cohort'])
        started = time.perf_counter()
        loaded = [load_capture(path, ids) for path in CAPTURES]
        captures, hashes = zip(*loaded)
        first = captures[0]
        params = ColGenParams(**{f.name: getattr(first['params'], f.name) for f in fields(ColGenParams)
                                if hasattr(first['params'], f.name)})
        requests = {r.flight_id: r for r in first['requests']}
        for other in captures[1:]:
            assert other['cfg'] == first['cfg']
            assert fingerprint(other['static_terminals']) == fingerprint(first['static_terminals'])
            by_id = {r.flight_id: r for r in other['requests']}
            assert all(fingerprint(requests[fid]) == fingerprint(by_id[fid]) for fid in ids)
        catalog = StaticTerminalCatalog(first['static_terminals'], first['cfg'])
        graphs = {fid: build_flight_graph(requests[fid], first['cfg'], catalog, params) for fid in ids}
        views = [pricing.DualView(c['duals'], c['cfg']) for c in captures]
        report = dict(source=str(args.source.resolve()), stages=args.stages, capture_sha256=hashes,
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(pricing.__file__).parent.glob('*.py')},
                      setup_s=time.perf_counter()-started, flight_ids=ids, phases=[])
        phases = [('cold', 0), ('warm', 0)]
        if not args.early_only:
            phases += [('changed', 1), ('changed_warm', 1), ('restored', 0)]
        def forbid(*a, **kw):
            raise AssertionError('Full CPU pricing forbidden')
        with ExitStack() as guards:
            for name in ('price_flight', '_best_column', '_best_column_compiled'):
                guards.enter_context(patch.object(pricing, name, forbid))
            for label, index in phases:
                elapsed = 0.0
                rows = []
                for fid in ids:
                    c = captures[index]
                    p = PricingProblem(graphs[fid], first['cfg'], params, views[index],
                                       c['flight_duals'].get(fid, 0.), c['known_columns'].get(fid))
                    start = time.perf_counter()
                    result = prepare_flight(p, views[index], PricingOptions(), time.monotonic()+180)
                    wall = time.perf_counter()-start
                    elapsed += wall
                    data = export_flight(result, include_topology=True)
                    components = {name: fingerprint(getattr(data, name))
                                  for name in ('fixed', 'variants', 'bounds', 'column', 'negative', 'status')}
                    membership_size = {}
                    if data.endpoint is not None:
                        components['endpoint'] = fingerprint({k: v for k, v in data.endpoint.items() if k != 'stats'})
                        components['complete_membership'] = fingerprint(result.endpoint.membership.arrivals)
                        membership = result.endpoint.membership.arrivals
                        unique = {id(array): array for array in membership.values()}
                        membership_size = dict(rows=len(membership), unique_arrays=len(unique),
                            unique_array_bytes=sum(a.nbytes for a in unique.values()),
                            logical_array_bytes=sum(a.nbytes for a in membership.values()))
                        del membership, unique
                    rows.append(dict(flight_id=fid, elapsed_s=wall, status=result.status, fingerprints=components,
                                     membership_size=membership_size,
                                     canonical_endpoint_cache_entries=len(graphs[fid]._search_cache.endpoint_claims),
                                     canonical_endpoint_cache_claims=sum(len(claims) for claims in
                                         graphs[fid]._search_cache.endpoint_claims.values())))
                    del result, data
                report['phases'].append(dict(label=label, prepare_s=elapsed, rows=rows,
                    rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2)+'\n')
                print(label, elapsed, flush=True)


if __name__ == '__main__':
    main()
