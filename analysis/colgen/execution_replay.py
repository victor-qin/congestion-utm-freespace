"""Compare cold, warm and changed GPU snapshots without invoking full CPU pricing.

Run this script against each source checkout through PYTHONPATH, under the shared
Metal benchmark lock. Captures must be trusted local pickles. Timings include
instrumentation and independent canonical validation, reported separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import fields
import cProfile
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from analysis.colgen.common import digest, sha256_bytes, write_json
from analysis.colgen.gpu_validation import serializable, validate_column
from analysis.colgen.platform_replay import _json_safe, load_capture, result_record
from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.gpu_pricing import _Batch
from freespace_sim.planner.colgen.gpu_runtime import ComputeRuntime
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing_platform import (
    PricingOptions, PricingPlatform, PricingProblem,
)


class _Observed:
    """Forward a Metal object while observing allocations or completed command waits."""

    def __init__(self, native: Any, measured: dict[str, float]) -> None:
        """Retain the native object and this phase's counters."""
        self.native, self.measured = native, measured

    def __getattr__(self, key: str) -> Any:
        """Forward unchanged operations to the actual Metal backend."""
        return getattr(self.native, key)

    def newBufferWithLength_options_(self, length: int, options: int) -> Any:
        """Count actual native allocations, including on the base without runtime counters."""
        self.measured["allocations"] += 1
        self.measured["allocated_bytes"] += length
        return self.native.newBufferWithLength_options_(length, options)

    def commandBuffer(self) -> _Observed:
        """Observe each real command without changing encoding or scheduling."""
        return _Observed(self.native.commandBuffer(), self.measured)

    def waitUntilCompleted(self) -> None:
        """Measure host waiting and Metal-reported GPU time independently."""
        started = time.perf_counter()
        self.native.waitUntilCompleted()
        self.measured["wait_s"] += time.perf_counter() - started
        self.measured["waits"] += 1
        self.measured["gpu_s"] += max(0., self.native.GPUEndTime() - self.native.GPUStartTime())


def _timed(method: Callable, name: str, measured: dict[str, float]) -> Callable:
    """Wrap one method without changing arguments, return values or exceptions."""
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Accumulate inclusive method time even when a deadline interrupts it."""
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            measured[name + "_s"] += time.perf_counter() - started
    return wrapped


@contextmanager
def _observe(runtime: ComputeRuntime) -> Iterator[dict[str, float]]:
    """Instrument both base and resident implementations with the same observation points.

    Parameters
    ------------
    - runtime (ComputeRuntime): actual Metal runtime, initialized before phase timing.

    Return
    --------
    - output (Iterator[dict[str, float]]): mutable counters reset between phases.
    """
    measured = defaultdict(float)
    original_run = runtime.run

    def run(name: str, count: int, arrays: Any, outputs: Any,
            static: dict | None = None, write_only: tuple = ()) -> Any:
        """Estimate base host-array transfers; current runtime counters cover owned buffers too."""
        if count:
            for index, array in enumerate(arrays):
                if not hasattr(array, "flags") or index in write_only:
                    continue
                cached = runtime._static.get((static or {}).get(index))
                if cached is None or cached[0] is not array:
                    measured["run_upload_bytes"] += array.nbytes
            measured["run_download_bytes"] += sum(arrays[i].nbytes for i in outputs)
        return original_run(name, count, arrays, outputs, static=static, write_only=write_only)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        """Fail if any original full CPU pricing entry point is invoked."""
        raise AssertionError("Original full CPU pricing is forbidden")

    with ExitStack() as stack:
        for name in ("device", "queue"):
            stack.enter_context(patch.object(runtime, name, _Observed(getattr(runtime, name), measured)))
        stack.enter_context(patch.object(runtime, "run", run))
        for name in ("prepare", "prepare_visit_prices", "retain", "allocate", "complete"):
            stack.enter_context(patch.object(_Batch, name, _timed(getattr(_Batch, name), name, measured)))
        try:
            from freespace_sim.planner.colgen.gpu_resident import ResidentSearch
        except ImportError:
            pass  # PR142 predates the resident path.
        else:
            stack.enter_context(patch.object(ResidentSearch, "ensure_prefix", _timed(
                ResidentSearch.ensure_prefix, "resident_prefix", measured)))
        for name in ("price_flight", "_best_column", "_best_column_compiled"):
            stack.enter_context(patch.object(pricing, name, forbid))
        yield measured


def main(argv: list[str] | None = None) -> int:
    """Replay identical graphs with immutable price snapshots and record matched certificates.

    Parameters
    ------------
    - argv (list[str] | None): command-line arguments, or the process arguments.

    Return
    --------
    - output (int): zero after writing results, including explicit incomplete outcomes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--flight-ids", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--budget", type=float, default=180.)
    parser.add_argument("--max-labels", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--profile", action="store_true", help="Profile only the final restored phase")
    args = parser.parse_args(argv)
    if len(args.capture) not in (1, 2) or args.batch_size < 1 or args.budget <= 0:
        parser.error("Choose one or two captures and positive batch size/budget")
    loaded = [load_capture(path, args.flight_ids) for path in args.capture]
    captures, hashes = zip(*loaded)
    first = captures[0]
    params = ColGenParams(**{f.name: getattr(first["params"], f.name)
                            for f in fields(ColGenParams) if hasattr(first["params"], f.name)})
    requests = {r.flight_id: r for r in first["requests"]}
    execution_options = {"n_pricing_workers", "pricing_device", "pricing_gpu_backend",
                         "pricing_cpu_recompute", "pricing_gpu_batch_size",
                         "pricing_gpu_max_labels", "pricing_gpu_chunk_size"}
    for captured in captures[1:]:
        other = {r.flight_id: r for r in captured["requests"]}
        if (captured["cfg"] != first["cfg"] or
            captured["static_terminals"] != first["static_terminals"] or
            any(digest(serializable(other[fid])) != digest(serializable(requests[fid]))
                for fid in args.flight_ids) or
            any(getattr(captured["params"], f.name, f.default) != getattr(params, f.name)
                for f in fields(ColGenParams) if f.name not in execution_options)):
            raise ValueError("This replay shares graphs: selected geometry/configuration must match")
    catalog = StaticTerminalCatalog(first["static_terminals"], first["cfg"])
    started = time.perf_counter()
    graphs = {fid: build_flight_graph(requests[fid], first["cfg"], catalog, params)
              for fid in args.flight_ids}
    graph_s = time.perf_counter() - started
    started = time.perf_counter()
    views = [pricing.DualView(c["duals"], c["cfg"]) for c in captures]
    snapshot_index_s = time.perf_counter() - started
    platform = PricingPlatform(PricingOptions(device="gpu", gpu_backend="metal",
                                              max_labels=args.max_labels, chunk_size=args.chunk_size))
    started = time.perf_counter()
    platform._runtime = runtime = ComputeRuntime("metal")
    init_s = time.perf_counter() - started
    report = dict(batch_size=args.batch_size, max_labels=args.max_labels, chunk_size=args.chunk_size,
                  graph_s=graph_s, snapshot_index_s=snapshot_index_s, init_s=init_s,
                  source=str(Path(pricing.__file__).resolve()),
                  source_sha256={p.name: sha256_bytes(p.read_bytes())
                                 for p in Path(pricing.__file__).parent.iterdir()
                                 if p.suffix in (".py", ".h")},
                  capture_sha256=hashes, cpu_full_pricing_guard=True, runs=[])
    phases = [("round1_cold", 0), ("round1_warm", 0)]
    if len(captures) == 2:
        phases.extend((("round11_changed", 1), ("round11_warm", 1), ("round1_restored", 0)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with _observe(runtime) as measured:
        for label, index in phases:
            measured.clear()
            captured = captures[index]
            problems = [PricingProblem(graphs[fid], first["cfg"], params, views[index],
                                       captured["flight_duals"].get(fid, 0.),
                                       captured["known_columns"].get(fid)) for fid in args.flight_ids]
            records = []
            before = dict(getattr(runtime, "counters", {}))
            profile = cProfile.Profile() if args.profile and label == "round1_restored" else None
            started = time.perf_counter()
            if profile:
                profile.enable()
            for start in range(0, len(problems), args.batch_size):
                batch = problems[start:start + args.batch_size]
                results = platform.price_batch(batch, deadline=time.monotonic() + args.budget)
                records.extend(result_record(r) for r in results)
                validation_started = time.perf_counter()
                for problem, result in zip(batch, results, strict=True):
                    validate_column(problem, result)
                measured["independent_validation_s"] += time.perf_counter() - validation_started
            if profile:
                profile.disable()
                profile.dump_stats(str(args.out) + ".prof")
            elapsed = time.perf_counter() - started
            counters = {key: value - before[key]
                        for key, value in getattr(runtime, "counters", {}).items()}
            report["runs"].append(dict(label=label, elapsed_s=elapsed, measured=dict(measured),
                                       counters=counters, results=records))
            nonfinite = []
            safe = _json_safe(report, nonfinite)
            safe["nonfinite_fields"] = nonfinite
            write_json(args.out, safe)
            print(label, round(elapsed, 4), "s", sum(r["certified"] for r in records),
                  "certified of", len(records), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
