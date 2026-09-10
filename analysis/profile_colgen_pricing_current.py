"""Capture the first production pricing sweep, then replay individual subproblems.

Capture stops immediately after the sweep: this is a pricing diagnostic, not a
completed simulation or IP benchmark. Pickles are local outputs of this script.
"""

import argparse
from collections import defaultdict
import cProfile
from dataclasses import fields
import io
import hashlib
import json
import os
from pathlib import Path
import pickle
import pstats
import time
import sys

# Select source before imports so a shared editable install cannot change the capture.
ROOT = Path(__file__).resolve().parents[1]
source_parser = argparse.ArgumentParser(add_help=False)
source_parser.add_argument("--source-root", type=Path, default=ROOT)
source_args, _ = source_parser.parse_known_args()
SOURCE_ROOT = source_args.source_root.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

from analysis.benchmark_colgen_lns import jsonable  # noqa: E402
from freespace_sim.planner.colgen import dp_kernel, dp_prepare, pricing, solver  # noqa: E402
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.scenarios import get_scenario, with_overrides  # noqa: E402

assert SOURCE_ROOT in Path(solver.__file__).resolve().parents


CASE = ROOT / "analysis/colgen_astar_centered_1200s_20260909/density_faa_wing_zipline_seed0_iteration_ip_eager"
OUT = ROOT / "analysis/colgen_pricing_profile_20260909"


def dump(path, data):
    path.write_text(json.dumps(jsonable(data), indent=2) + "\n")


class Captured(BaseException):
    pass


def capture(*, inputs_only=False, output=OUT, case=CASE):
    """Capture first-LP inputs, optionally pricing once for the legacy diagnostic."""
    if (output / "subproblems.pkl").exists():
        raise FileExistsError(f"Refusing to overwrite {output / 'subproblems.pkl'}")
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((case / "inputs.json").read_text())
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
    original = solver.price_sweep
    from freespace_sim.planner.colgen.master import RestrictedMaster
    original_lp = RestrictedMaster.solve_lp
    first_lp = {}

    def measured_lp(master):
        result = original_lp(master)
        first_lp.update(lp_objective=result[0], lp_column_count=len(master.columns),
                        gurobi_threads=getattr(master.backend, "threads", None))
        return result

    RestrictedMaster.solve_lp = measured_lp
    started = time.perf_counter()

    def observed(order, reqs, graphs, config, policy, catalog, duals, view, flight_duals,
                 known_columns, **kwargs):
        snapshot = dict(requests=list(reqs), cfg=config, params=policy,
                        static_terminals=terminals, duals=duals, flight_duals=flight_duals,
                        known_columns=known_columns)
        snapshot["pricing_order"] = list(order)
        snapshot["first_lp"] = dict(first_lp)
        with (output / "subproblems.pkl").open("wb") as handle:
            pickle.dump(snapshot, handle)
        if inputs_only:
            reference = json.loads((case / "iterations.json").read_text())[0]
            checks = dict(lp_objective_matches=bool(np.isclose(
                first_lp["lp_objective"], reference["lp_objective"], rtol=0, atol=1e-6)),
                lp_column_count_matches=first_lp["lp_column_count"] == reference["lp_column_count"])
            metadata = dict(first_lp=first_lp, reference_first_lp={key: reference[key]
                for key in ("lp_objective", "lp_column_count")}, checks=checks,
                requests=len(reqs), pricing_flights=len(order), params=policy,
                captured_after_s=time.perf_counter() - started, reference_case=str(case),
                source_root=str(SOURCE_ROOT), source_sha256={str(p.relative_to(SOURCE_ROOT)):
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((SOURCE_ROOT / "freespace_sim").rglob("*.py"))},
                snapshot_sha256=hashlib.sha256((output / "subproblems.pkl").read_bytes()).hexdigest(),
                note="Stopped before price_sweep; no pricing or per-round/final IP executed.")
            dump(output / "capture_summary.json", metadata)
            print(json.dumps(jsonable(metadata), sort_keys=True), flush=True)
            assert all(checks.values()), "First LP differs from corridor benchmark"
            raise Captured
        print(f"Pricing starting after {time.perf_counter() - started:.2f}s; workers=12", flush=True)
        result = original(order, reqs, graphs, config, policy, catalog, duals, view,
                          flight_duals, known_columns, **kwargs)
        records = [dict(r, reduced_cost=rc, returned_column=c is not None)
                   for r, rc, c in zip(result.flight_records, result.reduced_costs,
                                       result.columns, strict=True)]
        dump(output / "flight_records.json", records)
        with (output / "priced_columns.pkl").open("wb") as handle:
            pickle.dump(dict(zip(result.flight_ids, result.columns, strict=True)), handle)
        summary = dict(complete=result.complete, flights=len(records), wall_s=result.wall_s,
                       task_total_s=result.task_total_s, pool_setup_s=result.pool_setup_s,
                       counters=dict(result.kernel_counters),
                       positive_columns=sum(c is not None and rc > 1e-9
                                            for c, rc in zip(result.columns, result.reduced_costs)),
                       rc_sum=sum(max(0.0, rc) for rc in result.reduced_costs),
                       note="First production sweep only; no per-round or final IP run.")
        dump(output / "sweep_summary.json", summary)
        print(json.dumps(summary), flush=True)
        raise Captured

    solver.price_sweep = observed
    try:
        solver.ColGenSolver().solve(requests, cfg, terminals, params)
    except Captured:
        pass
    finally:
        solver.price_sweep = original
        RestrictedMaster.solve_lp = original_lp


def replay(flight_id, overrun, profile, *, bootstrap_method=None, output=None):
    # This is exclusively our own locally generated capture, never an external pickle.
    with (OUT / "subproblems.pkl").open("rb") as handle:
        captured = pickle.load(handle)
    request = next(r for r in captured["requests"] if r.flight_id == flight_id)
    cfg = captured["cfg"]
    # Older trusted captures contain slotted params without subsequently added fields.
    # Reconstruct from present fields so new options acquire their declared defaults.
    old_params = captured["params"]
    values = {field.name: getattr(old_params, field.name)
              for field in fields(ColGenParams) if hasattr(old_params, field.name)}
    values["max_air_overrun_hops"] = overrun
    if bootstrap_method is not None:
        values["bootstrap_method"] = bootstrap_method
    params = ColGenParams(**values)
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = ROOT / "analysis" / f"colgen_pricing_replay_{stamp}_{params.bootstrap_method}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Replay output: {output}", flush=True)
    catalog = StaticTerminalCatalog(captured["static_terminals"], cfg)
    graph = build_flight_graph(request, cfg, catalog, params)
    view = pricing.DualView(captured["duals"], cfg)
    known = captured["known_columns"].get(flight_id)
    pi = captured["flight_duals"][flight_id]
    # Direct wall probes are authoritative for native-vs-Python attribution.
    # cProfile can misattribute Numba work to unrelated Python frames here.
    timings = defaultdict(float)
    originals = []

    def probe(owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[label] += time.perf_counter() - started

        originals.append((owner, name, original))
        setattr(owner, name, measured)

    for owner, name, label in (
        (dp_kernel, "_price_dag", "native_dp_s"),
        (dp_kernel, "price_dag", "dp_host_inclusive_s"),
        (dp_prepare, "prepared_for", "static_preparation_s"),
        (dp_prepare, "prepare_duals", "dual_preparation_s"),
        (dp_prepare, "prepare_variants", "root_preparation_s"),
        (pricing, "_canonical_candidate", "candidate_certification_s"),
    ):
        probe(owner, name, label)
    results = []
    for state in ("cold", "warm"):
        timings.clear()
        pricing.clear_search_record()
        started = time.perf_counter()
        try:
            rc, column = pricing.price_flight(graph, view, pi, cfg, params, known_column=known,
                                             deadline=time.monotonic() + 120.0)
            info = dict(flight_id=flight_id, overrun=overrun, cache=state,
                        bootstrap_method=params.bootstrap_method,
                        wall_s=time.perf_counter() - started, rc=rc,
                        returned_column=column is not None, timeout=False,
                        search=pricing.last_search_record(), timed_stages=dict(timings))
        except pricing.PricingTimeout:
            info = dict(flight_id=flight_id, overrun=overrun, cache=state,
                        bootstrap_method=params.bootstrap_method,
                        wall_s=time.perf_counter() - started, timeout=True,
                        search=pricing.last_search_record(), timed_stages=dict(timings))
        results.append(info)
        print(json.dumps(info), flush=True)
        dump(output / f"flight_{flight_id}_overrun_{overrun}.json", results)
        if info["timeout"]:
            break
    if profile and not results[-1]["timeout"]:
        with cProfile.Profile() as prof:
            pricing.price_flight(graph, view, pi, cfg, params, known_column=known,
                                 deadline=time.monotonic() + 120.0)
        prof.dump_stats(str(output / f"flight_{flight_id}.prof"))
        stream = io.StringIO()
        pstats.Stats(prof, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
        (output / f"flight_{flight_id}_calls.txt").write_text(stream.getvalue())
        print(stream.getvalue(), flush=True)
    for owner, name, original in reversed(originals):
        setattr(owner, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("mode", choices=("capture", "capture-inputs", "replay"))
    parser.add_argument("--flight", type=int)
    parser.add_argument("--overrun", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--bootstrap-method", choices=("dp", "astar"))
    parser.add_argument("--out", type=Path,
                        help="Replay output directory; default creates a timestamped directory.")
    parser.add_argument("--case", type=Path,
                        help="Benchmark case supplying exact inputs and first-LP reference")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    os.environ["COLGEN_GUROBI_THREADS"] = "4"
    if args.mode != "capture-inputs":
        dp_kernel.warm_kernel()
    if args.mode in {"capture", "capture-inputs"}:
        if args.mode == "capture-inputs" and (args.out is None or args.case is None):
            parser.error("capture-inputs requires --out and --case")
        capture(inputs_only=args.mode == "capture-inputs", output=args.out or OUT,
                case=args.case or CASE)
    elif args.flight is None:
        parser.error("replay requires --flight")
    else:
        replay(args.flight, args.overrun, args.profile,
               bootstrap_method=args.bootstrap_method, output=args.out)


if __name__ == "__main__":
    main()
