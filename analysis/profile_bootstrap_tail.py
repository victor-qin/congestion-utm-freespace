"""Bounded bootstrap-only probes for captured hard pricing subproblems.

No unrestricted exact search runs here. Each call has its own short deadline, and
restricted-DP declines raise instead of entering the Python fallback.
"""
from dataclasses import fields, replace
import argparse
import json
from pathlib import Path
import pickle
import time

from freespace_sim.planner.colgen import dp_kernel, pricing
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams


def main():
    """Compare initial cutoff, first goal, exhaustion and restricted DP safely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flights', type=int, nargs='+', default=[2282, 1640])
    parser.add_argument('--seconds', type=float, default=20.0)
    args = parser.parse_args()
    with args.capture.open('rb') as handle:
        capture = pickle.load(handle)
    old = capture['params']
    params = ColGenParams(**{f.name: getattr(old, f.name) for f in fields(ColGenParams)
                             if hasattr(old, f.name)})
    cfg = capture['cfg']
    catalog = StaticTerminalCatalog(capture['static_terminals'], cfg)
    dp_kernel.warm_kernel()
    real_compiled = pricing._best_column_compiled
    def compiled_only(*a, **kw):
        result = real_compiled(*a, **kw)
        if isinstance(result, pricing.Declined):
            raise RuntimeError(f'restricted DP declined: {result}')
        return result
    pricing._best_column_compiled = compiled_only
    records = []
    for flight in args.flights:
        request = next(r for r in capture['requests'] if r.flight_id == flight)
        astar_incumbent = None
        for method in ('astar', 'dp', 'astar_then_dp'):
            graph = build_flight_graph(request, cfg, catalog, params)
            view = pricing.DualView(capture['duals'], cfg)
            original = pricing._bootstrap_incumbent
            record = dict(flight_id=flight, method=method)
            def observed(*a, **kw):
                record['initial_rc'] = None if kw['incumbent'] is None else kw['incumbent'][0]
                if method == 'astar_then_dp' and astar_incumbent is not None:
                    kw['incumbent'] = astar_incumbent
                before = time.perf_counter()
                result = original(*a, **kw)
                record['bootstrap_s'] = time.perf_counter() - before
                record['bootstrap_result_rc'] = None if result is None else result[0]
                return result
            real_goal = pricing._goal_directed_bootstrap
            def observed_goal(*a, **kw):
                variants, order = a[7], a[8]
                record['selected_roots'] = [dict(departure=int(variants.departure_step[i]),
                    lane=int(variants.lane_idx[i])) for i in order]
                result = real_goal(*a, **kw)
                if result is not None:
                    record['goal_result'] = dict(rc=result[0], departure=result[1].departure_step,
                                                lane=result[1].origin_lane_idx, hops=len(result[1].cell_path)-1)
                return result
            pricing._goal_directed_bootstrap = observed_goal
            pricing._bootstrap_incumbent = observed
            pricing.clear_search_record()
            started = time.perf_counter()
            try:
                answer = pricing.price_flight(graph, view, capture['flight_duals'][flight], cfg,
                    replace(params, bootstrap_method='astar' if method == 'astar' else 'dp'),
                    known_column=capture['known_columns'].get(flight), heuristic_only=True,
                    require_improving=False, deadline=time.monotonic() + args.seconds)
                record.update(rc=answer[0], returned_column=answer[1] is not None, timeout=False)
                if method == 'astar' and answer[1] is not None:
                    astar_incumbent = answer
            except pricing.PricingTimeout:
                record['timeout'] = True
            except RuntimeError as exc:
                record['declined'] = str(exc)
            finally:
                pricing._bootstrap_incumbent = original
                pricing._goal_directed_bootstrap = real_goal
            record.update(wall_s=time.perf_counter()-started, search=pricing.last_search_record())
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
