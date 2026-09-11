"""Paired Gurobi simulations comparing rounding with incumbent-preserving LNS.

Each arm imports only its explicit source tree and runs in a fresh process. Inputs
are regenerated from fixed seeds and fingerprinted, including every static wall.
The parent alternates arm order and never overlaps measured simulations.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tarfile
import time
import traceback


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()


def flight_summary(records):
    """Summarize worker load, labels, and timing without discarding raw fields."""
    def distribution(values):
        values = sorted(values)
        if not values:
            return {"count": 0, "total": 0.0}
        def percentile(q):
            position = (len(values) - 1) * q
            low = int(position)
            return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
        return dict(count=len(values), total=math.fsum(values), max=values[-1],
                    p50=percentile(.5), p90=percentile(.9), p95=percentile(.95), p99=percentile(.99))
    workers = {}
    for record in records:
        key = str(record.get("worker")) + ":" + str(record.get("pid"))
        worker = workers.setdefault(key, dict(records=0, task_s=0.0))
        worker["records"] += 1
        worker["task_s"] += record.get("task_s", 0.0)
    fields = {key for record in records for key, value in record.items()
              if (key.endswith("_s") or "labels" in key) and isinstance(value, (int, float))}
    metrics = {key: distribution([float(r[key]) for r in records if key in r
                                and isinstance(r[key], (int, float))]) for key in sorted(fields)}
    metrics["non_bootstrap_task_s"] = distribution([
        max(0.0, r["task_s"] - r["bootstrap_s"]) for r in records
        if "task_s" in r and "bootstrap_s" in r])
    return dict(records=len(records), unique_flights=len({r["flight_id"] for r in records}),
                priced=sum(bool(r.get("priced")) for r in records), workers=workers, metrics=metrics,
                timing_note="non_bootstrap_task_s includes main search and task overhead")


def persist_flights(out, sequence, payload):
    """Atomically preserve every callback flight row and verify the stored count."""
    records = jsonable(payload.get("sweep_flight_records", ()))
    directory = out / "flight_records"
    directory.mkdir(exist_ok=True)
    path = directory / f"round_{sequence:04d}.jsonl.gz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with gzip.open(temporary, "rt") as handle:
        assert sum(1 for _ in handle) == len(records), "Incomplete flight diagnostics"
    temporary.replace(path)
    summary = flight_summary(records)
    sweep_s = payload.get("sweep_s", 0.0)
    worker_totals = [worker["task_s"] for worker in summary["workers"].values()]
    summary.update(sweep_s=sweep_s,
                   observed_worker_utilization=(sum(worker_totals) / (sweep_s * len(worker_totals))
                                                if sweep_s and worker_totals else None),
                   max_worker_task_s=max(worker_totals, default=0.0),
                   iteration=payload.get("iteration"), file=str(path.relative_to(out)),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dump(directory / f"round_{sequence:04d}.summary.json", summary)
    return summary


def child(args):
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import freespace_sim
    from freespace_sim.cost import trajectory_cost
    from freespace_sim.metrics import total_delay_s
    from freespace_sim.planner.colgen import dp_kernel
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.scenarios import get_scenario, with_overrides
    from freespace_sim.sim import run

    assert root in Path(freespace_sim.__file__).resolve().parents
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Warm the primitives and pricing through an actual small simulation.
    # Feasibility-search compilation is not exercised by this warmup; record the
    # signatures so runs that need that separate kernel can be identified.
    warm_started = time.perf_counter()
    dp_kernel.warm_kernel()
    warm_spec = with_overrides(get_scenario("colgen_test"), planner="colgen", seed=0,
                               horizon_s=600.0, demand_duration_s=4.0)
    warm_cfg = warm_spec.config()
    warm_demand = warm_spec.demand_model()
    warm_requests = warm_demand.generate(warm_cfg, np.random.default_rng(0))
    warm_result = run(
        warm_cfg, requests=warm_requests, demand=warm_demand, progress=False,
        planner_params=ColGenParams(solver="gurobi", n_pricing_workers=0,
                                   max_iterations=2, time_limit_s=300.0,
                                   ip_time_limit_s=15.0),
    )
    assert warm_result.verified
    warm_s = time.perf_counter() - warm_started
    warm_signatures = {
        "pricing": len(dp_kernel._price_dag.signatures),
        "feasible": len(dp_kernel._feasible_dag.signatures),
    }
    assert warm_signatures["pricing"] > 0, "Warmup did not compile pricing"
    del warm_result
    print(f"WARMED {warm_s:.3f}s", flush=True)

    overrides = {"planner": "colgen", "seed": args.seed}
    if args.demand_seconds is not None:
        overrides["demand_duration_s"] = args.demand_seconds
    spec = with_overrides(get_scenario(args.scenario), **overrides)
    cfg = spec.config()
    demand = spec.demand_model()
    generated = sorted(demand.generate(cfg, np.random.default_rng(args.seed)),
                       key=lambda request: request.flight_id)
    # Generate the original paired stream first so the outbound-only ablation
    # preserves every outbound ID, coordinate and clock exactly. Turning returns
    # off in the generator can renumber IDs or change legacy random draws.
    eligible = ([r for r in generated if r.paired_outbound_id is None]
                if args.outbound_only else generated)
    if args.outbound_only:
        assert len(eligible) < len(generated), "Scenario has no identified return legs"
        assert all(r.origin_terminal is not None and r.dest_terminal is None
                   for r in eligible), "Outbound-only benchmark requires hub deliveries"
    requests = eligible if args.flights == 0 else eligible[:args.flights]
    static_terms = list(demand.terminals(cfg))
    experimental = {}
    if args.columns_per_flight != 1:
        experimental["columns_per_flight"] = args.columns_per_flight
    if args.bootstrap_method is not None:
        experimental["bootstrap_method"] = args.bootstrap_method
    params = ColGenParams(
        max_air_overrun_hops=args.overrun, **experimental,
        solver="gurobi", n_pricing_workers=args.workers, max_iterations=args.iterations,
        time_limit_s=args.budget, ip_time_limit_s=args.ip_budget,
        ip_reserve_s=args.ip_reserve,
        ip_gap=args.gap, lp_gap=args.lp_gap, gap_metric="cost",
        iteration_ip_time_limit_s=(args.iteration_ip_budget
                                   if args.version.startswith("iteration_ip") else 0.0),
        iteration_ip_eager=args.version != "iteration_ip",
        cheap_pricing=args.version == "iteration_ip_cheap",
        exact_pricing_interval=args.exact_pricing_interval,
        warm_start_planner="astar" if args.version in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"} else None,
        seed_nominal_routes=args.version not in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"},
        seed_ladder_steps=0 if args.version in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"} else 20,
        warm_start_max_shift_steps=0 if args.version in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"} else 8,
        provided_seed_ladder_steps=10 if args.version == "iteration_ip_astar_centered" else 0,
        lns_destroy_flights=(0 if args.version == "baseline"
                            or args.version.startswith("iteration_ip") else args.destroy),
    )
    inputs = {"config": cfg, "params": params, "requests": requests,
              "static_terminals": static_terms, "return_anchor": "nominal"}
    if args.outbound_only:
        inputs["return_anchor"] = "not applicable: outbound-only request subset"
        inputs["demand_selection"] = "paired_outbound_id is None; original IDs and clocks"
    # The two arms differ only in the heuristic mode, excluded from this hash.
    comparable = dict(inputs)
    comparable["params"] = dataclasses.replace(params, lns_destroy_flights=0)
    if hasattr(params, "iteration_ip_time_limit_s"):
        comparable["params"] = dataclasses.replace(comparable["params"], iteration_ip_time_limit_s=0)
    comparable["params"] = dataclasses.replace(
        comparable["params"], cheap_pricing=False, iteration_ip_eager=True,
        warm_start_planner=None, seed_nominal_routes=True, seed_ladder_steps=20,
        warm_start_max_shift_steps=8, provided_seed_ladder_steps=0)
    input_sha = digest(comparable)
    demand_config_sha = digest({key: value for key, value in inputs.items() if key != "params"})
    algorithm_params_sha = digest(params)
    dump(out / "measurement.json", dict(status="running", demand_config_sha=demand_config_sha,
         algorithm_params_sha=algorithm_params_sha, input_sha=input_sha, source_root=str(root),
         diagnostic_scope="All completed iteration callbacks; interrupted in-progress sweeps have no callback"))
    dump(out / "inputs.json", inputs)
    print(f"SIMULATION {len(requests)} flights, {len(static_terms)} terminals", flush=True)

    import freespace_sim.planner.colgen.solver as solver_module
    from freespace_sim.planner.colgen.solver import ColGenSolver

    # Benchmark-only ablation: preserve nominal seeds, the departure ladder,
    # pricing bootstrap, and legal ground waits. Skip only the initial greedy
    # ground-shift schedule. The solver's existing first-LP rounding fallback
    # then supplies LNS's initial incumbent. No production default is changed.
    original_initial = solver_module._initial_feasible_selection
    initialization = (dict(enabled=False, n_seeds=0, covered=0, cost=0.0, elapsed_s=0.0)
                      if not params.seed_nominal_routes else {})

    def measured_initial(seeds, *positional, **kwargs):
        started = time.perf_counter()
        selected = ({} if args.version == "no_ground_bootstrap" else
                    original_initial(seeds, *positional, **kwargs))
        initialization.update(
            enabled=args.version != "no_ground_bootstrap", n_seeds=len(seeds),
            covered=len(selected), cost=math.fsum(c.delay_s for c in selected.values()),
            elapsed_s=time.perf_counter() - started,
        )
        print("INITIALIZATION " + json.dumps(initialization, sort_keys=True), flush=True)
        return selected

    solver_module._initial_feasible_selection = measured_initial
    # Observe both the FCFS source schedule and its column conversion. All
    # timing remains inside the outer simulation clock, and every dropped
    # import is reported. The provided-only arm makes no extra time shifts.
    astar_seed = {}
    if args.version in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"}:
        import freespace_sim.sim as sim_module
        import freespace_sim.planner.colgen.warm_start as warm_module
        original_source_run = sim_module.run
        original_build = warm_module.build

        def measured_source_run(*positional, **kwargs):
            print("FCFS ASTAR starting", flush=True)
            source_started = time.perf_counter()
            source = original_source_run(*positional, **kwargs)
            source_wall = time.perf_counter() - source_started
            source_cost = math.fsum(
                cfg.cost_ground_delay_per_s * intent.ground_delay_s
                + cfg.cost_air_hold_per_s * intent.air_hold_s
                + cfg.cost_air_lateral_per_s * intent.air_detour_m / cfg.nominal_speed_mps
                for intent in source.accepted
            )
            astar_seed.update(
                source_wall_s=source_wall, source_accepted=len(source.accepted),
                source_denied=len(source.denied), source_verified=source.verified,
                source_cost=source_cost,
                source_penalized_cost=source_cost + params.M * len(source.denied),
                source_air_hold_flights=sum(i.air_hold_s > 1e-9 for i in source.accepted),
            )
            trajectories = [
                dict(flight_id=i.request.flight_id, accepted=i.accepted,
                     denial_reason=i.denial_reason.value, ground_delay_s=i.ground_delay_s,
                     air_hold_s=i.air_hold_s, air_detour_m=i.air_detour_m,
                     altitude_change_m=i.altitude_change_m, centerline=i.centerline)
                for i in source.intents
            ]
            with gzip.open(out / "astar_trajectories.json.gz", "wt") as handle:
                json.dump(jsonable(trajectories), handle, sort_keys=True)
            dump(out / "astar_seed.json", astar_seed)
            print("FCFS ASTAR " + json.dumps(astar_seed, sort_keys=True), flush=True)
            return source

        def measured_build(*positional, **kwargs):
            conversion_started = time.perf_counter()
            columns, conversion_stats = original_build(*positional, **kwargs)
            astar_seed.update(
                conversion_wall_s=time.perf_counter() - conversion_started,
                conversion_stats=dict(conversion_stats),
                converted_flights=len(columns),
                converted_columns=sum(len(v) for v in columns.values()),
                max_shift_steps=kwargs.get("max_shift", 8),
                require_joint_feasibility=kwargs.get("require_joint_feasibility", True),
            )
            dump(out / "astar_seed.json", astar_seed)
            print("ASTAR CONVERSION " + json.dumps(astar_seed, sort_keys=True), flush=True)
            return columns, conversion_stats

        sim_module.run = measured_source_run
        warm_module.build = measured_build
    column_trace = None
    if args.capture_columns:
        # Instrument every source arm with the same helper archived by the parent.
        tracer_spec = importlib.util.spec_from_file_location(
            "benchmark_column_trace", Path(__file__).with_name("trace.py"))
        tracer_module = importlib.util.module_from_spec(tracer_spec)
        tracer_spec.loader.exec_module(tracer_module)
        column_trace = tracer_module.ColumnTrace(out)
        column_trace.install()
    iteration_records = []
    original_solve = ColGenSolver.solve

    def traced_solve(self, *positional, **kwargs):
        original_callback = kwargs.get("on_iteration")

        def record(payload):
            keep = {key: value for key, value in payload.items()
                    if key not in {"master", "x", "lp_x", "lp_columns", "capacity_duals",
                                   "sweep_flight_records"}}
            # Retain only serializable diagnostic fields, never the live model.
            keep = {key: value for key, value in keep.items()
                    if isinstance(value, (int, float, str, bool, dict, tuple, list, type(None)))}
            keep["simulation_elapsed_s"] = time.perf_counter() - start
            diagnostics_started = time.perf_counter()
            keep["flight_diagnostics"] = persist_flights(out, len(iteration_records) + 1, payload)
            keep["flight_diagnostics_write_s"] = time.perf_counter() - diagnostics_started
            iteration_records.append(keep)
            dump(out / "iterations.json", iteration_records)
            if column_trace is not None:
                column_trace.iteration(payload)
            print(f"ITER {payload.get('iteration')} elapsed={payload.get('elapsed_s',0):.2f} "
                  f"cost={payload.get('heuristic_cost')} gap={payload.get('heuristic_gap_cost')} "
                  f"stages={payload.get('stage_s')}", flush=True)
            if original_callback is not None:
                original_callback(payload)

        kwargs["on_iteration"] = record
        return original_solve(self, *positional, **kwargs)

    ColGenSolver.solve = traced_solve
    start = time.perf_counter()
    result = run(cfg, requests=requests, demand=demand, progress=False,
                 static_terminals=static_terms, planner_params=params)
    wall_s = time.perf_counter() - start
    if column_trace is not None:
        column_trace.finish()
    dump(out / "iterations.json", iteration_records)
    stats = result.planner_stats
    accepted = result.accepted
    delays = [total_delay_s(intent, cfg) for intent in accepted]
    costs = [trajectory_cost(intent, cfg) for intent in accepted]
    total_cost = math.fsum(costs)
    congestion_cost = math.fsum(
        cfg.cost_ground_delay_per_s * intent.ground_delay_s
        + cfg.cost_air_hold_per_s * intent.air_hold_s
        + cfg.cost_air_lateral_per_s * intent.air_detour_m / cfg.nominal_speed_mps
        for intent in accepted
    )
    total_delay = math.fsum(delays)
    penalized_cost = congestion_cost + params.M * len(result.denied)
    # Compute the same global certificate for both versions. The original ip_gap
    # has the reviewed reporting bug and is deliberately not used for comparison.
    lower_bound = max(0.0, len(requests) * params.M - float(stats["upper_bound"]))
    global_gap = ((penalized_cost - lower_bound) / max(1.0, abs(penalized_cost))
                  if math.isfinite(lower_bound) else math.inf)
    trajectories = [{
        "flight_id": intent.request.flight_id, "accepted": intent.accepted,
        "denial_reason": intent.denial_reason.value,
        "ground_delay_s": intent.ground_delay_s, "air_hold_s": intent.air_hold_s,
        "air_detour_m": intent.air_detour_m, "altitude_change_m": intent.altitude_change_m,
        "cost": trajectory_cost(intent, cfg) if intent.accepted else None,
        "delay_s": total_delay_s(intent, cfg) if intent.accepted else None,
        "centerline": intent.centerline,
    } for intent in sorted(result.intents, key=lambda intent: intent.request.flight_id)]
    with gzip.open(out / "trajectories.json.gz", "wt") as handle:
        json.dump(jsonable(trajectories), handle, sort_keys=True)
    denied_reasons = {}
    for intent in result.denied:
        reason = intent.denial_reason.value
        denied_reasons[reason] = denied_reasons.get(reason, 0) + 1
    row = {
        "version": args.version, "scenario": args.scenario, "seed": args.seed,
        "heuristic_mode": (args.version if args.version.startswith("iteration_ip") else
                           "rounding" if args.version == "baseline" else "lns"),
        "heuristic_stage_s": math.fsum(rec.get("stage_s", {}).get(stage, 0)
            for rec in iteration_records for stage in ("round_heuristic", "lns_heuristic")),
        "ground_bootstrap": initialization,
        "astar_seed": astar_seed,
        "demand_duration_s": cfg.effective_demand_duration_s,
        "heuristic_cost": stats.get("heuristic_cost"),
        "initial_heuristic_cost": stats.get("initial_heuristic_delay_s"),
        "ip_skipped": stats.get("ip_skipped"),
        "ip_reserve_s": params.effective_ip_reserve_s,
        "ip_setup_s": stats.get("ip_setup_s"),
        "restricted_ip_gap": stats.get("restricted_ip_gap"),
        "gurobi_threads": stats.get("gurobi_threads"),
        "lns_destroy_flights": params.lns_destroy_flights,
        "iteration_ip_time_limit_s": getattr(params, "iteration_ip_time_limit_s", 0),
        "iteration_ip_calls": stats.get("iteration_ip_calls", 0),
        "iteration_ip_wall_s": stats.get("iteration_ip_wall_s", 0),
        "n_generated": len(generated), "n_requests": len(requests),
        "outbound_only": args.outbound_only, "n_eligible": len(eligible),
        "n_return_requests": sum(r.paired_outbound_id is not None for r in requests),
        "requests_sha": digest(requests), "generated_requests_sha": digest(generated),
        "n_static_terminals": len(static_terms), "input_sha": input_sha,
        "demand_config_sha": demand_config_sha, "algorithm_params_sha": algorithm_params_sha,
        "trajectory_sha": digest(trajectories), "source_root": str(root),
        "verified": result.verified, "accepted": len(accepted), "denied": len(result.denied),
        "denial_reasons": denied_reasons,
        "total_delay_s": total_delay,
        "mean_delay_s": total_delay / len(accepted) if accepted else None,
        "p95_delay_s": float(np.quantile(delays, 0.95)) if delays else None,
        "ground_delay_s": math.fsum(intent.ground_delay_s for intent in accepted),
        "air_detour_m": math.fsum(intent.air_detour_m for intent in accepted),
        "filed_cost": total_cost, "congestion_cost": congestion_cost,
        "mandatory_altitude_cost": total_cost - congestion_cost,
        "penalized_congestion_cost": penalized_cost,
        "solver_objective": stats["objective"],
        "congestion_cost_minus_solver_objective": congestion_cost - stats["objective"],
        "global_cost_gap": max(0.0, global_gap),
        "simulation_wall_s": wall_s, "solver_wall_s": stats["elapsed_s"],
        "pricing_wall_s": stats["pricing_wall_s"], "ip_wall_s": stats["ip_elapsed_s"],
        "iterations": stats["iterations"], "columns": stats["n_columns"],
        "termination": stats["termination_reason"], "ip_status": stats["ip_status"],
        "pricing_sweeps_completed": stats["pricing_sweeps_completed"],
        "kernel_fell_back": stats["kernel_fell_back"],
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
            (2**20 if sys.platform == "darwin" else 1024),
        "warmup_s": warm_s, "warm_kernel_signatures": warm_signatures,
    }
    dump(out / "planner_stats.json", stats)
    dump(out / "result.json", row)
    dump(out / "measurement.json", dict(status="complete", demand_config_sha=demand_config_sha,
         algorithm_params_sha=algorithm_params_sha, input_sha=input_sha, source_root=str(root),
         rounds_persisted=len(iteration_records),
         flight_records=sum(r["flight_diagnostics"]["records"] for r in iteration_records)))
    print(json.dumps(jsonable(row), sort_keys=True), flush=True)


def parent(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    roots = {"baseline": str(Path(args.baseline).resolve()),
             "fixed": str(Path(args.fixed).resolve())}
    if "no_ground_bootstrap" in args.versions:
        roots["no_ground_bootstrap"] = roots["fixed"]
    if "iteration_ip" in args.versions:
        roots["iteration_ip"] = roots["fixed"]
    for version in ("iteration_ip_eager", "iteration_ip_cheap", "iteration_ip_astar_seed", "iteration_ip_astar_centered"):
        if version in args.versions:
            roots[version] = roots["fixed"]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONHASHSEED="0", COLGEN_GUROBI_THREADS=str(args.gurobi_threads))
    cases = [(scenario, seed, 0 if scenario == "colgen_test" else args.density_flights)
             for scenario in args.scenarios
             for seed in args.seeds]
    manifest = {
        "roots": roots, "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "numba", "python-fcl", "gurobipy")},
        "overrun": args.overrun, "columns_per_flight": args.columns_per_flight,
        "bootstrap_method": args.bootstrap_method,
        "lp_gap": args.lp_gap, "gurobi_threads": args.gurobi_threads,
        "exact_pricing_interval": args.exact_pricing_interval,
        "iterations": args.iterations, "solver_time_limit_s": args.budget,
        "demand_duration_override_s": args.demand_seconds,
        "outbound_only": args.outbound_only,
        "demand_selection": ("Generate original paired demand, then retain outbound requests "
                             "with identical IDs, coordinates and clocks"
                             if args.outbound_only else "All generated requests"),
        "versions": args.versions,
        "ablation": "no_ground_bootstrap skips only _initial_feasible_selection; first-LP "
                    "rounding fallback, seed ladder and pricing bootstrap remain enabled. "
                    "iteration_ip_eager forces complete bindable-row materialization before "
                    "each per-round IP, preserving its separate search budget. iteration_ip_cheap "
                    "uses restricted root pricing, exact sweeps every configured interval, "
                    "on stagnation, and on the last round; only exact sweeps certify bounds. "
                    "iteration_ip_astar_seed uses only FCFS A* imports with no shift repair, "
                    "no nominal seeds, no departure ladder, and no nominal greedy pass. "
                    "iteration_ip_astar_centered adds up to 10 legal departure steps on each "
                    "side of each imported A* column. All individually valid imports remain "
                    "in the pool; the IP resolves joint conflicts without greedy import repair.",
        "ip_time_limit_s": args.ip_budget, "workers": args.workers, "cases": cases,
        "iteration_ip_time_limit_s": args.iteration_ip_budget,
        "ip_reserve_s": args.ip_reserve, "capture_columns": args.capture_columns,
        "solver": "gurobi", "ip_gap": args.gap, "destroy": args.destroy,
        "scope": "Actual sim.run, including batch filing and final conflict verification",
        "timing": "Warmup and demand generation excluded; measured arms never overlap",
        "return_anchor": "nominal (the existing column-generation support)",
        "rss": "Peak of the whole fresh process, including warmup; not incremental simulation RSS",
        "first_version": args.first_version,
        "cost": "Congestion cost excludes mandatory climb/descent; filed_cost includes it",
    }
    for version, root in roots.items():
        manifest[f"{version}_commit"] = (subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            if (Path(root) / ".git").exists() else None)
        frozen_manifest = Path(root) / "source-manifest.json"
        if frozen_manifest.exists():
            manifest[f"{version}_frozen_manifest_sha256"] = hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
        manifest[f"{version}_source_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((Path(root) / "freespace_sim").rglob("*.py"))
        }
    patch = (subprocess.check_output(["git", "diff", "--", "freespace_sim/planner/colgen"],
                                    cwd=roots["fixed"])
             if (Path(roots["fixed"]) / ".git").exists() else b"")
    (out / "changes.patch").write_bytes(patch)
    manifest["patch_sha256"] = hashlib.sha256(patch).hexdigest()
    manifest["harness_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    (out / "harness_used.py").write_bytes(script.read_bytes())
    if args.capture_columns:
        tracer = script.with_name("trace.py")
        (out / "colgen_trace_used.py").write_bytes(tracer.read_bytes())
        manifest["tracer_sha256"] = hashlib.sha256(tracer.read_bytes()).hexdigest()
    with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
        for version, root in roots.items():
            for relative in manifest[f"{version}_source_sha256"]:
                archive.add(Path(root) / relative, arcname=f"{version}/{relative}")
    manifest["source_snapshot_sha256"] = hashlib.sha256((out / "source_snapshot.tar.gz").read_bytes()).hexdigest()
    manifest["flight_diagnostics"] = "Per completed callback: flight_records/round_NNNN.jsonl.gz plus summary; all raw fields retained"
    dump(out / "manifest.json", manifest)
    rows = []
    for case_index, (scenario, seed, flights) in enumerate(cases):
        baseline_first = (case_index % 2 == 0) == (args.first_version == "baseline")
        order = list(args.versions) if baseline_first else list(reversed(args.versions))
        for version in order:
            case_out = out / f"{scenario}_seed{seed}_{version}"
            case_out.mkdir(exist_ok=True)
            print(f"START {len(rows) + 1}/{len(cases) * len(order)} "
                  f"{scenario} seed={seed} {version}", flush=True)
            command = [sys.executable, str(script), "child", "--root", roots[version],
                       "--version", version, "--scenario", scenario, "--seed", str(seed),
                       "--flights", str(flights), "--iterations", str(args.iterations),
                       "--budget", str(args.budget), "--out", str(case_out),
                       "--overrun", str(args.overrun),
                       "--columns-per-flight", str(args.columns_per_flight),
                       "--workers", str(args.workers), "--destroy", str(args.destroy),
                       "--gap", str(args.gap), "--ip-budget", str(args.ip_budget),
                       "--iteration-ip-budget", str(args.iteration_ip_budget),
                       "--lp-gap", str(args.lp_gap),
                       "--exact-pricing-interval", str(args.exact_pricing_interval),
                       "--gurobi-threads", str(args.gurobi_threads)]
            if args.bootstrap_method is not None:
                command.extend(["--bootstrap-method", args.bootstrap_method])
            if args.demand_seconds is not None:
                command.extend(["--demand-seconds", str(args.demand_seconds)])
            if args.ip_reserve is not None:
                command.extend(["--ip-reserve", str(args.ip_reserve)])
            if args.capture_columns:
                command.append("--capture-columns")
            if args.outbound_only:
                command.append("--outbound-only")
            try:
                with (case_out / "process.log").open("w") as log:
                    subprocess.run(command, cwd=roots[version], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True,
                                   timeout=args.budget + (1800 if version in {"iteration_ip_astar_seed", "iteration_ip_astar_centered"} else 240))
                row = json.loads((case_out / "result.json").read_text())
            except (subprocess.SubprocessError, OSError) as exc:
                row = {"version": version, "scenario": scenario, "seed": seed,
                       "error": str(exc), "log": str(case_out / "process.log")}
                dump(case_out / "error.json", row)
                measurement_path = case_out / "measurement.json"
                measurement = json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
                measurement.update(status="interrupted_or_failed", error=str(exc),
                    rounds_persisted=len(list((case_out / "flight_records").glob("*.jsonl.gz"))))
                dump(measurement_path, measurement)
            rows.append(row)
            dump(out / "results.json", rows)
            if "error" in row:
                print("ERROR " + json.dumps(row), flush=True)
            else:
                print(f"DONE {version} wall={row['simulation_wall_s']:.3f}s "
                      f"delay={row['total_delay_s']:.3f}s cost={row['filed_cost']:.3f} "
                      f"accepted={row['accepted']}/{row['n_requests']} "
                      f"verified={row['verified']} iterations={row['iterations']} "
                      f"termination={row['termination']}", flush=True)
        pair = rows[-len(order):]
        if all("error" not in row for row in pair):
            assert len({row["demand_config_sha"] for row in pair}) == 1, "A/B inputs differ"
    fields = sorted({key for row in rows for key in row})
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    return int(any("error" in row for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--baseline", required=True)
    run.add_argument("--fixed", required=True)
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--scenarios", nargs="+", default=[
        "colgen_test", "density_faa_wing_zipline", "density_future_wing_zipline",
    ])
    run.add_argument("--density-flights", type=int, default=100)
    run.add_argument("--first-version", choices=["baseline", "fixed"], default="baseline")
    run.add_argument("--versions", nargs="+", default=["baseline", "fixed"],
                     choices=["baseline", "fixed", "no_ground_bootstrap", "iteration_ip",
                              "iteration_ip_eager", "iteration_ip_cheap", "iteration_ip_astar_seed", "iteration_ip_astar_centered"])
    child_parser = sub.add_parser("child")
    child_parser.add_argument("--root", required=True)
    child_parser.add_argument("--version", required=True,
                              choices=["baseline", "fixed", "no_ground_bootstrap", "iteration_ip",
                                       "iteration_ip_eager", "iteration_ip_cheap", "iteration_ip_astar_seed", "iteration_ip_astar_centered"])
    child_parser.add_argument("--scenario", required=True)
    child_parser.add_argument("--seed", type=int, required=True)
    child_parser.add_argument("--flights", type=int, required=True)
    for command in (run, child_parser):
        command.add_argument("--out", required=True)
        command.add_argument("--iterations", type=int, default=3)
        command.add_argument("--budget", type=float, default=600.0)
        command.add_argument("--ip-budget", type=float, default=60.0)
        command.add_argument("--iteration-ip-budget", type=float, default=30.0,
                             help="Per-sweep IP cap for the iteration_ip arm; final cap is separate")
        command.add_argument("--ip-reserve", type=float)
        command.add_argument("--capture-columns", action="store_true")
        command.add_argument("--overrun", type=int, default=6,
                             help="Extra airborne hops beyond the shortest lattice route (default: 6)")
        command.add_argument("--columns-per-flight", type=int, default=1)
        command.add_argument("--bootstrap-method", choices=["dp", "astar"])
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--gurobi-threads", type=int, default=4)
        command.add_argument("--lp-gap", type=float, default=0.001)
        command.add_argument("--exact-pricing-interval", type=int, default=5)
        command.add_argument("--destroy", type=int, default=100)
        command.add_argument("--gap", type=float, default=0.001)
        command.add_argument("--demand-seconds", type=float)
        command.add_argument("--outbound-only", action="store_true",
                             help="Drop paired return legs after generating unchanged demand")
    args = parser.parse_args()
    if args.overrun < 0 or args.columns_per_flight < 1:
        parser.error("--overrun must be nonnegative and --columns-per-flight positive")
    if args.gurobi_threads < 1:
        parser.error("--gurobi-threads must be positive")
    os.environ["COLGEN_GUROBI_THREADS"] = str(args.gurobi_threads)
    if not math.isfinite(args.iteration_ip_budget) or args.iteration_ip_budget <= 0:
        parser.error("--iteration-ip-budget must be finite and positive")
    if args.mode == "run" and len(set(args.versions)) != len(args.versions):
        parser.error("--versions must not contain duplicate arms")
    try:
        return parent(args) if args.mode == "run" else child(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
