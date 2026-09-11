"""Run the archived full benchmark while capturing selected pricing inputs.

This analysis-only wrapper serializes immutable replay inputs immediately before
selected ``solver.price_sweep`` calls. It writes the already serialized bytes
after each sweep and reports all capture overhead separately.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import ModuleType

from analysis.colgen.common import sha256_bytes, source_hashes, write_json


def _load_benchmark(path: Path | None, module_name: str) -> ModuleType:
    """Load the requested benchmark without depending on its filename."""
    if path is None:
        return importlib.import_module(module_name)
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location("colgen_capture_benchmark", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_trace_path(benchmark: Path) -> Path | None:
    """Find the tracer used by either the packaged or historical benchmark layout."""
    for name in ("trace.py", "colgen_trace.py"):
        candidate = benchmark.with_name(name)
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    """Capture requested production sweeps during one archived benchmark child run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    benchmark_source = parser.add_mutually_exclusive_group()
    benchmark_source.add_argument(
        "--benchmark",
        type=Path,
        help="Benchmark script path; defaults to the packaged benchmark module.",
    )
    benchmark_source.add_argument(
        "--benchmark-module",
        default="analysis.colgen.benchmark",
        help="Importable benchmark module (default: analysis.colgen.benchmark).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 6, 11])
    parser.add_argument("--expected-production-flights", type=int, required=True)
    parser.add_argument(
        "--allow-missing-rounds",
        action="store_true",
        help="Complete the full run when convergence precedes a requested capture round.",
    )
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    capture_dir = args.out / "pricing_captures"
    capture_dir.mkdir()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    from freespace_sim.planner.colgen import solver

    assert source_root in Path(solver.__file__).resolve().parents
    benchmark_module = _load_benchmark(args.benchmark, args.benchmark_module)
    benchmark = Path(benchmark_module.__file__).resolve()
    trace_path = _selected_trace_path(benchmark)

    goal_warmup = {"available": False, "wall_s": 0.0, "signatures": 0}
    try:
        bootstrap_kernel = importlib.import_module(
            "freespace_sim.planner.colgen.bootstrap_kernel"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "freespace_sim.planner.colgen.bootstrap_kernel":
            raise
    else:
        goal_warm_started = time.perf_counter()
        bootstrap_kernel.warm_kernel()
        goal_warmup = {
            "available": True,
            "wall_s": time.perf_counter() - goal_warm_started,
            "signatures": len(bootstrap_kernel._advance.signatures),
        }

    requested_rounds = frozenset(args.rounds)
    if not requested_rounds or min(requested_rounds) < 1:
        parser.error("--rounds must contain positive call numbers")
    source_sha256 = source_hashes(source_root)
    harness_bytes = Path(__file__).read_bytes()
    (args.out / "capture_harness_used.py").write_bytes(harness_bytes)
    benchmark_bytes = benchmark.read_bytes()
    capture_records: list[dict] = []
    call_number = 0
    all_call_number = 0
    original = solver.price_sweep

    def observed(
        pricing_order,
        requests,
        graphs,
        cfg,
        params,
        catalog,
        duals,
        dual_view,
        flight_duals,
        known_columns,
        **kwargs,
    ):
        """Snapshot replay inputs before the selected production sweep call."""
        nonlocal all_call_number, call_number
        all_call_number += 1
        # The archived benchmark runs a tiny compilation simulation first. The
        # exact seed-specific production count separates that call from the run.
        if len(requests) != args.expected_production_flights:
            return original(
                pricing_order,
                requests,
                graphs,
                cfg,
                params,
                catalog,
                duals,
                dual_view,
                flight_duals,
                known_columns,
                **kwargs,
            )
        call_number += 1
        payload_bytes = None
        serialization_s = 0.0
        if call_number in requested_rounds:
            serialization_started = time.perf_counter()
            payload = {
                "capture_schema": 1,
                "round": call_number,
                "pricing_order": list(pricing_order),
                "requests": list(requests),
                "cfg": cfg,
                "params": params,
                "static_terminals": catalog.entries,
                "duals": duals,
                "flight_duals": flight_duals,
                "known_columns": known_columns,
                "heuristic_only": bool(kwargs.get("heuristic_only", False)),
                "remaining_deadline_s": (
                    None
                    if kwargs.get("deadline") is None
                    else kwargs["deadline"] - time.monotonic()
                ),
                "graphs_policy": (
                    "Parallel price_sweep ignores parent graphs; replay workers rebuild "
                    "graphs from captured requests, cfg, params, and static terminals."
                ),
            }
            payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            serialization_s = time.perf_counter() - serialization_started

        sweep_started = time.perf_counter()
        result = original(
            pricing_order,
            requests,
            graphs,
            cfg,
            params,
            catalog,
            duals,
            dual_view,
            flight_duals,
            known_columns,
            **kwargs,
        )
        sweep_s = time.perf_counter() - sweep_started
        if payload_bytes is not None:
            write_started = time.perf_counter()
            capture_path = capture_dir / f"round_{call_number:04d}.pkl"
            capture_path.write_bytes(payload_bytes)
            write_s = time.perf_counter() - write_started
            record = {
                "round": call_number,
                "capture": str(capture_path.relative_to(args.out)),
                "capture_sha256": sha256_bytes(payload_bytes),
                "capture_bytes": len(payload_bytes),
                "serialization_s": serialization_s,
                "write_s": write_s,
                "capture_overhead_s": serialization_s + write_s,
                "underlying_sweep_s": sweep_s,
                "complete": result.complete,
                "flights": len(result.flight_ids),
                "raw_diagnostics": "Preserved by the benchmark flight_records output.",
            }
            capture_records.append(record)
            write_json(args.out / "capture_records.json", capture_records)
        return result

    solver.price_sweep = observed
    forwarded = list(args.benchmark_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    child_argv = [
        str(benchmark),
        "child",
        "--root",
        str(source_root),
        "--out",
        str(args.out),
        *forwarded,
    ]
    manifest = {
        "status": "running",
        "source_root": str(source_root),
        "source_sha256": source_sha256,
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256_bytes(benchmark_bytes),
        "harness_sha256": sha256_bytes(harness_bytes),
        "tool_dependency_sha256": {
            "common.py": sha256_bytes(Path(__file__).with_name("common.py").read_bytes()),
            **(
                {str(trace_path): sha256_bytes(trace_path.read_bytes())}
                if trace_path is not None
                else {}
            ),
        },
        "requested_rounds": sorted(requested_rounds),
        "allow_missing_rounds": args.allow_missing_rounds,
        "expected_production_flights": args.expected_production_flights,
        "compiled_goal_warmup": goal_warmup,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "timing_note": (
            "Raw benchmark diagnostics include capture serialization and post-sweep file "
            "writes. simulation_wall_excluding_capture_s subtracts both; use fresh replay "
            "runs for matched candidate speed comparisons."
        ),
    }
    write_json(args.out / "capture_manifest.json", manifest)
    original_argv = sys.argv
    try:
        sys.argv = child_argv
        return_code = benchmark_module.main()
    finally:
        sys.argv = original_argv
        solver.price_sweep = original
    return_code = 0 if return_code is None else return_code

    missing = sorted(requested_rounds - {record["round"] for record in capture_records})
    overhead_s = sum(record["capture_overhead_s"] for record in capture_records)
    result_path = args.out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["capture_overhead_s"] = overhead_s
        result["simulation_wall_excluding_capture_s"] = (
            result["simulation_wall_s"] - overhead_s
        )
        write_json(result_path, result)
    manifest.update(
        status=(
            "complete"
            if return_code == 0 and (not missing or args.allow_missing_rounds)
            else "failed"
        ),
        observed_sweep_calls=call_number,
        all_sweep_calls_including_warmup=all_call_number,
        captures=len(capture_records),
        missing_rounds=missing,
        capture_overhead_s=overhead_s,
    )
    write_json(args.out / "capture_manifest.json", manifest)
    if missing and not args.allow_missing_rounds:
        raise RuntimeError(f"Run ended before requested captures: {missing}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
