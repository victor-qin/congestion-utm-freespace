"""Sequential fresh-graph DP/A* bootstrap replay of trusted first-sweep captures."""
from dataclasses import fields
import json
from pathlib import Path
import pickle
import time

from freespace_sim.planner.colgen import dp_kernel, pricing
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams


def main():
    """Compare exact pricing RC and unprofiled time after native kernel warming."""
    root = Path(__file__).resolve().parents[1]
    with (root / 'analysis/colgen_pricing_profile_20260909/subproblems.pkl').open('rb') as handle:
        capture = pickle.load(handle)
    old = capture['params']
    values = {f.name: getattr(old, f.name) for f in fields(ColGenParams) if hasattr(old, f.name)}
    cfg = capture['cfg']
    catalog = StaticTerminalCatalog(capture['static_terminals'], cfg)
    dp_kernel.warm_kernel()
    records = []
    for flight in (934, 2692, 1520, 1600):
        request = next(r for r in capture['requests'] if r.flight_id == flight)
        answers = []
        for method in ('dp', 'astar'):
            params = ColGenParams(**{**values, 'bootstrap_method': method})
            graph = build_flight_graph(request, cfg, catalog, params)
            view = pricing.DualView(capture['duals'], cfg)
            pricing.clear_search_record()
            started = time.perf_counter()
            try:
                answer = pricing.price_flight(graph, view, capture['flight_duals'][flight], cfg,
                    params, known_column=capture['known_columns'].get(flight),
                    deadline=time.monotonic() + 180)
                record = dict(flight_id=flight, method=method, wall_s=time.perf_counter()-started,
                              rc=answer[0], returned_column=answer[1] is not None,
                              search=pricing.last_search_record(), timeout=False)
                answers.append(answer)
            except pricing.PricingTimeout:
                record = dict(flight_id=flight, method=method, wall_s=time.perf_counter()-started,
                              timeout=True, search=pricing.last_search_record())
                answers.append(None)
            records.append(record)
            print(json.dumps(record), flush=True)
            (root/'analysis/colgen_bootstrap_pair_20260910/results.json').write_text(json.dumps(records, indent=2)+'\n')
        if all(a is not None for a in answers):
            delta = abs(answers[0][0] - answers[1][0])
            print(json.dumps(dict(flight_id=flight, rc_delta=delta, same_column=answers[0][1]==answers[1][1])), flush=True)
            assert delta <= 1e-8, (flight, answers)


if __name__ == '__main__':
    main()
