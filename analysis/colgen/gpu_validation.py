"""Validate unchanged long flights from a trusted capture using Metal and Gurobi."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from analysis.colgen.common import digest, encode, sha256_bytes, source_hashes, write_json
from analysis.colgen.platform_replay import _json_safe, load_capture, result_record


def select_requests(requests: list[Any], count: int = 10, required_id: int = 1752) -> list[Any]:
    """
    Select the longest original flights while retaining the known failed flight.

    Parameters
    ------------
    - requests (list[Any]): original captured requests, with unchanged coordinates and clocks.
    - count (int): number of requests to retain.
    - required_id (int): regression flight that must be included.

    Return
    --------
    - output (list[Any]): selected original objects ordered by descending horizontal distance.
    """
    if count < 1 or count > len(requests):
        raise ValueError("Selection count is outside the captured request count")
    ordered = sorted(requests, key=lambda r: (-distance_m(r), r.flight_id))
    required = next((r for r in ordered if r.flight_id == required_id), None)
    if required is None:
        raise ValueError(f"Required flight {required_id} is absent")
    selected = ordered[:count]
    if required_id not in {r.flight_id for r in selected}:
        selected[-1] = required
    return sorted(selected, key=lambda r: (-distance_m(r), r.flight_id))


def distance_m(request: Any) -> float:
    """Return the original horizontal endpoint distance in metres."""
    return math.hypot(*(float(b - a) for a, b in zip(request.origin[:2], request.dest[:2])))


def serializable(value: Any) -> Any:
    """Preserve full array precision and enum values in input and filing artifacts."""
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {f.name: serializable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Write strict JSON while explicitly identifying any nonfinite fields."""
    nonfinite: list[str] = []
    safe = _json_safe(serializable(report), nonfinite)
    safe["nonfinite_fields"] = nonfinite
    write_json(path, safe)


def validate_column(problem: Any, result: Any) -> None:
    """
    Recompute geometry, claims, objective, and reduced cost for a returned column.

    Parameters
    ------------
    - problem (Any): graph, configuration, objective and fixed prices for this result.
    - result (Any): GPU result, possibly with no improving column.

    Return
    --------
    - output (None): raises if the returned route or its coefficients are inconsistent.
    """
    from freespace_sim.planner.colgen import pricing
    from freespace_sim.planner.colgen.network import column_claims
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.translate import column_to_intent

    column = result.column
    if column is None:
        return
    if result.flight_id != column.flight_id or column.flight_id != problem.graph.request.flight_id:
        raise AssertionError("Returned column has the wrong flight ID")
    if column.claims != column_claims(column, problem.graph, problem.cfg):
        raise AssertionError("Returned column claims differ from canonical claims")
    if column.claims & problem.forbidden_rows:
        raise AssertionError("Returned column uses a currently forbidden resource row")
    intent = column_to_intent(column, problem.graph.request, problem.cfg)
    if not intent.accepted:
        raise AssertionError("Returned column does not translate to an accepted intent")
    model = cost_model(problem.cfg, problem.params)
    actual_cost = model.intent_cost(intent, problem.cfg)
    view = (problem.duals if isinstance(problem.duals, pricing.DualView)
            else pricing.DualView(problem.duals, problem.cfg))
    actual_rc = model.reduced_cost(benefit=problem.params.M, cost=actual_cost,
                                  dual_cost=view.claim_cost(column.claims), pi_f=problem.pi_f)
    if not math.isclose(column.delay_s, actual_cost, rel_tol=0., abs_tol=1e-8):
        raise AssertionError("Returned column objective differs from canonical objective")
    if not math.isclose(result.reduced_cost, actual_rc, rel_tol=0., abs_tol=1e-8):
        raise AssertionError("Returned reduced cost differs from canonical reduced cost")


def failure_reasons(
    records: list[dict[str, Any]], selected_flight_ids: set[int], result: Any | None = None,
) -> list[str]:
    """
    Evaluate GPU execution and completion independently of restricted-master success.

    Parameters
    ------------
    - records (list[dict[str, Any]]): every observed GPU pricing result.
    - selected_flight_ids (set[int]): every flight required to have a certified pricing result.
    - result (Any | None): optional full simulation result, including independent filing checks.

    Return
    --------
    - output (list[str]): explicit reasons validation has not passed.
    """
    reasons = []
    observed = {record["flight_id"] for record in records}
    missing = selected_flight_ids - observed
    unexpected = observed - selected_flight_ids
    if not selected_flight_ids or missing:
        reasons.append(f"Selected flights without pricing results: {sorted(missing)}")
    if unexpected:
        reasons.append(f"Unexpected flights in pricing results: {sorted(unexpected)}")
    if not records or not any(r["stats"].get("gpu_dispatches", 0) > 0 for r in records):
        reasons.append("No GPU dispatch was observed")
    for record in records:
        if record["backend"] != "metal":
            reasons.append(f"Unexpected pricing backend: {record['backend']}")
        if not record["search_complete"] or not record["certified"]:
            reasons.append(f"Flight {record['flight_id']}: {record['status']}, incomplete or uncertified")
        bound, reduced_cost = record.get("upper_bound"), record.get("reduced_cost")
        if (bound is None or reduced_cost is None
                or not math.isfinite(bound) or not math.isfinite(reduced_cost)
                or bound < reduced_cost):
            reasons.append(f"Flight {record['flight_id']}: missing or invalid certified upper bound")
    if result is not None:
        stats = result.planner_stats
        if stats.get("pricing_search_complete") is not True:
            reasons.append("Final solver pricing search is incomplete")
        if stats.get("pricing_bound_certified") is not True:
            reasons.append("Final solver pricing bound is uncertified")
        if stats.get("pricing_failure_status") is not None:
            reasons.append(f"Final solver pricing failure: {stats['pricing_failure_status']}")
        if not result.verified:
            reasons.append("Independent simulation conflict verification failed")
        if result.denied:
            reasons.append(f"Simulation denied {len(result.denied)} selected flights")
        if result.planner_stats.get("kernel_fell_back", 0):
            reasons.append("Solver reported a CPU kernel fallback")
    return reasons


def main(argv: list[str] | None = None) -> int:
    """
    Prepare a reproducible selection or run fixed-price/full-solver GPU validation.

    Parameters
    ------------
    - argv (list[str] | None): CLI arguments; None reads the process command line.

    Return
    --------
    - output (int): zero only for a prepared manifest or a fully passing validation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "replay", "solve"))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--budget", type=float, default=1800.0)
    parser.add_argument("--max-labels", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--integer-bounds", action="store_true")
    parser.add_argument("--audit-integer", action="store_true")
    parser.add_argument("--approach-bounds", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-flights", action="store_true",
                        help="Include every captured request in prepare/replay mode.")
    selection.add_argument("--flight-ids", type=int, nargs="+",
                           help="Select captured IDs in the supplied order, including a fresh subset solve.")
    args = parser.parse_args(argv)
    if args.all_flights and args.mode == "solve":
        parser.error("--all-flights supports prepare/replay only")
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    if not math.isfinite(args.budget) or args.budget <= 65:
        parser.error("--budget must be finite and exceed the 65-second final IP reserve")
    if min(args.max_labels, args.chunk_size, args.batch_size) <= 0:
        parser.error("GPU resource limits must be positive")
    requested_ids = args.flight_ids if args.flight_ids is not None else (
        None if args.all_flights else [1752]
    )
    captured, capture_hash = load_capture(args.capture, requested_ids)
    if args.flight_ids is not None:
        by_id = {request.flight_id: request for request in captured["requests"]}
        requests = [by_id[flight_id] for flight_id in args.flight_ids]
        selection_description = "Explicit flight IDs in caller-supplied order"
    elif args.all_flights:
        requests = list(captured["requests"])
        selection_description = "All captured flights in captured request order"
    else:
        requests = select_requests(captured["requests"])
        selection_description = (
            "Longest nine plus required flight 1752; descending distance, ID tie break"
        )
    from freespace_sim.planner.colgen.params import ColGenParams

    params = ColGenParams(**{
        f.name: getattr(captured["params"], f.name)
        for f in fields(ColGenParams) if hasattr(captured["params"], f.name)
    })
    original_params = params
    defaulted_param_fields = [
        f.name for f in fields(ColGenParams) if not hasattr(captured["params"], f.name)
    ]
    if params.max_air_overrun_hops != 6:
        raise ValueError("Expected the original six-extra-hop route domain")
    params = replace(params, solver="gurobi", pricing_device="gpu", pricing_gpu_backend="metal",
                     pricing_cpu_recompute=True, pricing_gpu_max_labels=args.max_labels,
                     pricing_gpu_integer_bounds=args.integer_bounds,
                     pricing_gpu_integer_audit=args.audit_integer,
                     pricing_gpu_approach_bounds=args.approach_bounds,
                     pricing_gpu_chunk_size=args.chunk_size, pricing_gpu_batch_size=args.batch_size,
                     n_pricing_workers=0, time_limit_s=args.budget, ip_time_limit_s=60.0,
                     ip_reserve_s=65.0)
    args.out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "capture": str(args.capture.resolve()), "capture_sha256": capture_hash,
        "selection": selection_description,
        "flight_ids": [r.flight_id for r in requests],
        "flights": [{"flight_id": r.flight_id, "distance_m": distance_m(r),
                     "origin_type": "terminal" if r.origin_terminal else "freepoint",
                     "destination_type": "terminal" if r.dest_terminal else "freepoint",
                     "request_sha256": digest(serializable(r))} for r in requests],
        "requests_sha256": digest(serializable(requests)), "config_sha256": digest(serializable(captured["cfg"])),
        "static_terminals_sha256": digest(serializable(captured["static_terminals"])),
        "source_sha256": source_hashes(root),
        "harness_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "original_params": encode(original_params), "run_params": encode(params),
        "original_params_defaulted_fields": defaulted_param_fields,
        "simulation_config": encode(captured["cfg"]), "requests": serializable(requests),
        "mode": args.mode,
        "prices": "Original full-scenario duals" if args.mode == "replay" else "New subset LP duals",
    }
    save_report(args.out / "manifest.json", manifest)
    if args.mode == "prepare":
        return 0

    from freespace_sim.planner.colgen import pricing, pricing_platform, pricing_pool, solver
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.sim import run

    records: list[dict[str, Any]] = []
    batch_wall_s: list[float] = []
    original_price_batch = pricing_platform.PricingPlatform.price_batch

    def observe(self: Any, problems: Any, deadline: float | None = None) -> Any:
        """Persist every completed batch, including incomplete searches and repair calls."""
        batch_started = time.perf_counter()
        results = original_price_batch(self, problems, deadline=deadline)
        batch_wall_s.append(time.perf_counter() - batch_started)
        records.extend(result_record(result) for result in results)
        save_report(args.out / "pricing.json", {"results": records})
        for problem, result in zip(problems, results, strict=True):
            validate_column(problem, result)
        return results

    def forbid_cpu(*positional: Any, **kwargs: Any) -> Any:
        """Reject a CPU full-pricing call rather than accepting silent fallback."""
        raise AssertionError("CPU full pricing is forbidden in GPU validation")

    started = time.monotonic()
    result = None
    error = None
    with ExitStack() as stack:
        for module in (pricing, pricing_pool, solver):
            stack.enter_context(patch.object(module, "price_flight", forbid_cpu))
        for name in ("_best_column", "_best_column_compiled"):
            stack.enter_context(patch.object(pricing, name, forbid_cpu))
        if args.integer_bounds and not args.audit_integer:
            from freespace_sim.planner.colgen.gpu_pricing import _Batch

            stack.enter_context(patch.object(_Batch, "bound", forbid_cpu))
        stack.enter_context(patch.object(pricing_platform.PricingPlatform, "price_batch", observe))
        try:
            if args.mode == "solve":
                result = run(captured["cfg"], requests=requests, planner_name="colgen",
                             planner_params=params, static_terminals=captured["static_terminals"],
                             progress=False)
                save_report(args.out / "planner_stats.json", result.planner_stats)
                save_report(args.out / "filed.json", {
                    "verified": result.verified, "accepted": result.accepted, "denied": result.denied,
                })
            else:
                catalog = StaticTerminalCatalog(captured["static_terminals"], captured["cfg"])
                dual_view = pricing.DualView(captured["duals"], captured["cfg"])
                problems = [pricing_platform.PricingProblem(
                    graph=build_flight_graph(r, captured["cfg"], catalog, params),
                    cfg=captured["cfg"], params=params, duals=dual_view,
                    pi_f=captured["flight_duals"].get(r.flight_id, 0.0),
                    known_column=captured["known_columns"].get(r.flight_id),
                ) for r in requests]
                platform = pricing_platform.PricingPlatform(pricing_platform.PricingOptions(
                    device="gpu", gpu_backend="metal", max_labels=args.max_labels,
                    chunk_size=args.chunk_size,
                    gpu_integer_bounds=args.integer_bounds, gpu_integer_audit=args.audit_integer,
                    gpu_approach_bounds=args.approach_bounds,
                ))
                for start in range(0, len(problems), args.batch_size):
                    platform.price_batch(problems[start:start + args.batch_size],
                                         deadline=started + args.budget)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    failures = failure_reasons(records, {r.flight_id for r in requests}, result)
    if error:
        failures.append(error)
    report = {"passed": not failures, "failure_reasons": failures,
              "elapsed_s": time.monotonic() - started, "pricing_results": len(records),
              "flight_ids_priced": sorted({r["flight_id"] for r in records}),
              "accepted": None if result is None else len(result.accepted),
              "verified": None if result is None else result.verified,
              "cpu_full_pricing_guard": True,
              "cpu_numeric_verifier_guard": args.integer_bounds and not args.audit_integer,
              "pricing_wall_s": sum(batch_wall_s), "batch_wall_s": batch_wall_s}
    save_report(args.out / "result.json", report)
    print(report, flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
