"""Isolate nominal seeding costs on a fixed subset of the saved FAA input.

Runs no pricing or IP solve and changes no production behavior. The timed pass
is unprofiled; a separate, smaller pass collects call attribution with cProfile.
"""

from collections import defaultdict
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time

import numpy as np

from analysis.benchmark_colgen_lns import jsonable
from freespace_sim.planner.colgen import pricing, solver
from freespace_sim.planner.colgen.master import RestrictedMaster
from freespace_sim.planner.colgen.network import RowIndex, StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.scenarios import get_scenario, with_overrides


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "analysis/colgen_astar_centered_1200s_20260909/density_faa_wing_zipline_seed0_iteration_ip_eager"
OUT = ROOT / "analysis/colgen_seed_profile_20260909"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=int, default=128, help="0 selects every flight")
    parser.add_argument("--profile-flights", type=int, default=32, help="0 skips cProfile")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.flights < 0 or args.profile_flights < 0:
        parser.error("flight counts must be nonnegative")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((CASE / "inputs.json").read_text())
    spec = with_overrides(get_scenario("density_faa_wing_zipline"), planner="colgen",
                          seed=0, demand_duration_s=1200.0)
    cfg, demand = spec.config(), spec.demand_model()
    requests = sorted((r for r in demand.generate(cfg, np.random.default_rng(0))
                       if r.paired_outbound_id is None), key=lambda r: r.flight_id)
    terminals = list(demand.terminals(cfg))
    assert jsonable(requests) == saved["requests"]
    assert jsonable(terminals) == saved["static_terminals"]
    assert jsonable(cfg) == saved["config"]
    params = ColGenParams(**saved["params"])
    model = cost_model(cfg, params)
    count = min(args.flights or len(requests), len(requests))
    selected = [requests[i] for i in np.linspace(0, len(requests) - 1, count, dtype=int)]
    counts = defaultdict(int)
    original_path, original_fallback = pricing._shortest_cell_path, pricing._best_column

    def path(*args, **kwargs):
        counts["spatial_search_calls"] += 1
        return original_path(*args, **kwargs)

    def fallback(*args, **kwargs):
        counts["seed_dag_fallbacks"] += 1
        return original_fallback(*args, **kwargs)

    pricing._shortest_cell_path, pricing._best_column = path, fallback

    def run(batch):
        stages = defaultdict(float)
        start = time.perf_counter()
        catalog = StaticTerminalCatalog(terminals, cfg)
        graphs = [build_flight_graph(r, cfg, catalog, params) for r in batch]
        stages["graph_setup_s"] = time.perf_counter() - start
        rows = RowIndex()
        for graph in graphs:
            for tid, capacity in graph.terminal_capacities.items():
                rows.register_terminal(tid, capacity)
        master = RestrictedMaster([r.flight_id for r in batch], rows, params, seed=0)
        for graph in graphs:
            start = time.perf_counter()
            seed = pricing.seed_column(graph, cfg, model=model)
            stages["nominal_route_search_and_certification_s"] += time.perf_counter() - start
            start = time.perf_counter()
            seed = solver._canonical_column(seed, graph, cfg)
            master.add_column(seed)
            stages["nominal_recheck_and_insertion_s"] += time.perf_counter() - start
            start = time.perf_counter()
            solver._add_departure_ladder(master, seed, graph, cfg, model,
                                         params.seed_ladder_steps, params.seed_ladder_stride)
            stages["delayed_copies_certification_and_insertion_s"] += time.perf_counter() - start
        return dict(stages), len(master.columns)

    # Warm import/JIT paths on an unrelated small pass, then use fresh graphs.
    run(selected[:1])
    counts.clear()
    stages, columns = run(selected)
    result = dict(flights=len(selected), columns=columns, stages=stages,
                  counts={k: counts[k] for k in ("spatial_search_calls", "seed_dag_fallbacks")},
                  flight_ids=[r.flight_id for r in selected],
                  source_input=str(CASE / "inputs.json"),
                  note=("Unprofiled serial initializer only; "
                        + ("all flights" if count == len(requests) else "evenly spaced subset")
                        + "; no pricing or IP."))
    (out / "timings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "flight_ids"}, indent=2), flush=True)
    if args.profile_flights:
        profile_count = min(args.profile_flights, len(selected))
        subset = [selected[i] for i in np.linspace(0, len(selected) - 1, profile_count, dtype=int)]
        with cProfile.Profile() as profile:
            run(subset)
        profile.dump_stats(str(out / "calls.prof"))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
        (out / "calls.txt").write_text(stream.getvalue())
        print(stream.getvalue(), flush=True)


if __name__ == "__main__":
    main()
