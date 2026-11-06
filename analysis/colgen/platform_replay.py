"""Price explicitly selected flights from a trusted local capture on CPU or GPU."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import pickle
import time
from typing import TYPE_CHECKING, Any

from analysis.colgen.common import digest, encode, sha256_bytes, source_hashes

if TYPE_CHECKING:
    from freespace_sim.planner.colgen.pricing_platform import PricingResult


def load_capture(path: Path, flight_ids: Sequence[int] | None) -> tuple[dict[str, Any], str]:
    """
    Load and validate one trusted capture without constructing any flight graphs.

    Parameters
    ------------
    - path (Path): trusted local pickle produced by analysis.colgen.capture.
    - flight_ids (Sequence[int] | None): distinct requested IDs, or None for every request.

    Return
    --------
    - output (tuple[dict[str, Any], str]): captured inputs and their byte-level SHA-256.
    """
    if flight_ids is not None and (not flight_ids or len(set(flight_ids)) != len(flight_ids)):
        raise ValueError("Choose a nonempty list of distinct flight IDs")
    payload = path.read_bytes()
    captured = pickle.loads(payload)  # Only load artifacts from a trusted local source.
    if not isinstance(captured, dict) or type(captured.get("capture_schema")) is not int:
        raise ValueError("Expected a schema-1 pricing capture")
    if captured["capture_schema"] != 1:
        raise ValueError(f"Unsupported capture schema: {captured['capture_schema']!r}")
    required = {
        "requests",
        "cfg",
        "params",
        "static_terminals",
        "duals",
        "flight_duals",
        "known_columns",
    }
    missing = required.difference(captured)
    if missing:
        raise ValueError(f"Capture is missing required fields: {sorted(missing)}")
    request_ids = [request.flight_id for request in captured["requests"]]
    if not request_ids:
        raise ValueError("Capture contains no requests")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in request_ids):
        raise ValueError("Capture request IDs must be integers")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Capture contains duplicate flight IDs")
    unknown = set() if flight_ids is None else set(flight_ids).difference(request_ids)
    if unknown:
        raise ValueError(f"Flight IDs absent from capture: {sorted(unknown)}")
    return captured, sha256_bytes(payload)


def result_record(result: PricingResult) -> dict[str, Any]:
    """
    Record a pricing result and the canonical identity of its returned column.

    Parameters
    ------------
    - result (PricingResult): one backend's result, including its certificate status.

    Return
    --------
    - output (dict[str, Any]): numerical results, route identity, claims digest and work counts.
    """
    column = result.column
    identity = (
        None
        if column is None
        else {
            "flight_id": column.flight_id,
            "departure_step": column.departure_step,
            "level": column.level,
            "origin_lane_idx": column.origin_lane_idx,
            "dest_lane_idx": column.dest_lane_idx,
            "cell_path": column.cell_path,
        }
    )
    return {
        "flight_id": result.flight_id,
        "reduced_cost": result.reduced_cost,
        "upper_bound": result.upper_bound,
        "search_complete": result.search_complete,
        "certified": result.certified,
        "backend": result.backend,
        "cpu_recompute": result.cpu_recompute,
        "status": result.status,
        "stats": result.stats,
        "column": None
        if column is None
        else {
            "identity": identity,
            "identity_sha256": digest(identity),
            "delay_s": column.delay_s,
            "claim_count": len(column.claims),
            "claims_sha256": digest(column.claims),
        },
    }


def _json_safe(value: Any, nonfinite_fields: list[str], path: str = "$") -> Any:
    """
    Convert report values to JSON types, recording each nonfinite number replaced by null.

    Parameters
    ------------
    - value (Any): nested report data, including NumPy scalar or array values.
    - nonfinite_fields (list[str]): destination for paths of replaced numbers.
    - path (str): location of this value in the report.

    Return
    --------
    - output (Any): JSON-compatible value with no NaN or infinity literals.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            nonfinite_fields.append(path)
            return None
        return number
    if is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, nonfinite_fields, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, nonfinite_fields, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), nonfinite_fields, path)
    raise TypeError(f"Unsupported report value at {path}: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Replay selected flights once and save their results without running a master LP.

    Parameters
    ------------
    - argv (Sequence[str] | None): CLI arguments, or None to read the process arguments.

    Return
    --------
    - output (int): zero after writing a report; inspect per-flight status for completion.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="Trusted local pickle only.")
    parser.add_argument("--out", type=Path, required=True, help="Fresh JSON output file.")
    parser.add_argument("--flight-ids", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-backend", choices=("metal", "cuda"))
    parser.add_argument("--cpu-recompute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-labels", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--endpoint-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--endpoint-max-rows", type=int, default=64)
    parser.add_argument("--endpoint-cache-bytes", type=int, default=1_048_576)
    parser.add_argument("--endpoint-cache-work", type=int, default=1_000_000)
    parser.add_argument(
        "--budget", type=float, default=60.0, help="Whole replay budget in seconds."
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("--budget must be finite and positive")
    if args.max_labels <= 0 or args.chunk_size <= 0:
        parser.error("--max-labels and --chunk-size must be positive")
    if len(set(args.flight_ids)) != len(args.flight_ids):
        parser.error("--flight-ids must contain distinct IDs")
    if (args.device == "gpu") != (args.gpu_backend is not None):
        parser.error("Provide --gpu-backend exactly when --device gpu is selected")
    if args.out.exists():
        parser.error("Choose a fresh --out JSON file")

    started = time.monotonic()
    deadline = started + args.budget
    captured, capture_hash = load_capture(args.capture, args.flight_ids)
    from freespace_sim.planner.colgen import pricing_platform
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.planner.colgen.params import ColGenParams

    # Reconstruct legacy slot-based parameter pickles using today's dataclass defaults.
    old_params = captured["params"]
    params = ColGenParams(
        **{
            field.name: getattr(old_params, field.name)
            for field in fields(ColGenParams)
            if hasattr(old_params, field.name)
        }
    )
    options = pricing_platform.PricingOptions(
        device=args.device,
        gpu_backend=args.gpu_backend,
        cpu_recompute=args.cpu_recompute,
        max_labels=args.max_labels,
        chunk_size=args.chunk_size,
        endpoint_bounds=args.endpoint_bounds,
        endpoint_max_rows=args.endpoint_max_rows,
        endpoint_cache_bytes=args.endpoint_cache_bytes,
        endpoint_cache_work=args.endpoint_cache_work,
    )
    requests = {request.flight_id: request for request in captured["requests"]}
    catalog = StaticTerminalCatalog(captured["static_terminals"], captured["cfg"])
    graph_started = time.monotonic()
    problems = [
        pricing_platform.PricingProblem(
            graph=build_flight_graph(requests[flight_id], captured["cfg"], catalog, params),
            cfg=captured["cfg"],
            params=params,
            duals=captured["duals"],
            pi_f=captured["flight_duals"].get(flight_id, 0.0),
            known_column=captured["known_columns"].get(flight_id),
        )
        for flight_id in args.flight_ids
    ]
    graph_s = time.monotonic() - graph_started
    pricing_started = time.monotonic()
    results = pricing_platform.PricingPlatform(options).price_batch(problems, deadline=deadline)
    pricing_s = time.monotonic() - pricing_started
    elapsed_s = time.monotonic() - started
    source_root = Path(pricing_platform.__file__).resolve().parents[3]
    report = {
        "report_schema": 1,
        "capture": str(args.capture.resolve()),
        "capture_schema": captured["capture_schema"],
        "capture_sha256": capture_hash,
        "captured_round": captured.get("round"),
        "captured_heuristic_only": captured.get("heuristic_only", False),
        "flight_ids": args.flight_ids,
        "options": options,
        "params": encode(params),
        "simulation_config": encode(captured["cfg"]),
        "budget_s": args.budget,
        "timing": {"graph_s": graph_s, "pricing_s": pricing_s, "elapsed_s": elapsed_s},
        "source_root": str(source_root),
        "source_sha256": {
            **source_hashes(source_root),
            **{
                str(path.relative_to(source_root)): sha256_bytes(path.read_bytes())
                for path in sorted((source_root / "freespace_sim").rglob("*.h"))
            },
        },
        "harness_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "results": [result_record(result) for result in results],
    }
    nonfinite_fields: list[str] = []
    report = _json_safe(report, nonfinite_fields, "$")
    report["nonfinite_fields"] = nonfinite_fields
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
