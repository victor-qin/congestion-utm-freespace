"""Isolated A/B for the always-active ``column_clear`` shortcut.

The parent process runs patched and legacy modes in fresh child interpreters. Each child regenerates
the same deterministic demand, keeps the requested chronological prefix, retains the full demand
model so every permanent hub wall is registered, and hashes every deterministic result field. The
only intentionally excluded field is ``OperationalIntent.solve_time_s``.

Examples:

    uv run python analysis/ab_column_clear.py density_future_wing_zipline \
        --flights 3000 --require-speedup
    uv run python analysis/ab_column_clear.py density_future_wing_zipline \
        --flights 3000 --levels 30,70,110
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from time import perf_counter

import numpy as np

import freespace_sim.planner.terminal_capacity as tc
from freespace_sim.planner import get_planner
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return out


def _levels(value: str) -> tuple[float, ...]:
    try:
        out = tuple(float(z) for z in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated numbers, e.g. 30,70,110") from exc
    if not out:
        raise argparse.ArgumentTypeError("must contain at least one level")
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", default="density_faa_wing_zipline")
    parser.add_argument("--smoke", action="store_true",
                        help="use a 1,800 s horizon and 300 s demand window")
    parser.add_argument("--lam-scale", type=float, default=1.0,
                        help="scale global and per-USS offered load")
    parser.add_argument("--flights", type=_positive_int,
                        help="run exactly the first N chronological generated requests")
    parser.add_argument("--levels", type=_levels,
                        help="override the flight-level ladder, e.g. 30,70,110")
    parser.add_argument("--require-speedup", action="store_true",
                        help="fail if patched wall time is not lower than baseline")
    parser.add_argument("--_mode", choices=("baseline", "patched"), help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return parser


def _build(args):
    spec = get_scenario(args.scenario)
    if args.smoke:
        spec = with_overrides(spec, horizon_s=1800.0, demand_duration_s=300.0)
    if args.lam_scale != 1.0:
        demand_overrides = None
        if spec.demand.lam_per_uss:
            demand_overrides = {
                "lam_per_uss": {
                    key: round(value * args.lam_scale, 2)
                    for key, value in spec.demand.lam_per_uss.items()
                }
            }
        spec = with_overrides(
            spec,
            lam_per_hour=round(spec.lam_per_hour * args.lam_scale, 2),
            demand_overrides=demand_overrides,
        )
    if args.levels is not None:
        spec = with_overrides(spec, flight_levels_m=args.levels)

    cfg, demand = spec.config(), spec.demand_model()
    if not cfg.terminal_airspace_always_active:
        raise ValueError(
            f"{args.scenario} is not always-active, so the shortcut would be a no-op"
        )
    if demand is None:
        raise ValueError(f"{args.scenario} has no explicit demand model")

    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)),
        key=lambda request: request.sort_key(),
    )
    generated = len(requests)
    if args.flights is not None:
        if generated < args.flights:
            raise ValueError(
                f"{args.scenario} generated {generated} requests, fewer than --flights {args.flights}"
            )
        requests = requests[:args.flights]
    return cfg, demand, scenario_from_requests(requests), generated


def _value_sig(value):
    """Exact, pickle-stable representation of the deterministic simulation value graph."""
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, np.ndarray):
        return ("numpy.ndarray", value.dtype.str, tuple(value.shape), value.tobytes())
    if isinstance(value, np.generic):
        return (type(value).__module__, type(value).__qualname__, value.item())
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple((field.name, _value_sig(getattr(value, field.name)))
                  for field in dataclasses.fields(value)),
        )
    if isinstance(value, dict):
        items = [(_value_sig(key), _value_sig(item)) for key, item in value.items()]
        return ("dict", tuple(sorted(items, key=lambda pair: pickle.dumps(pair[0], protocol=5))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_value_sig(item) for item in value))
    return value


def _digest(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _intent_digest(intent) -> str:
    # Deliberately enumerate dataclass fields so a future deterministic field is included automatically.
    signature = (
        type(intent).__module__,
        type(intent).__qualname__,
        tuple(
            (field.name, _value_sig(getattr(intent, field.name)))
            for field in dataclasses.fields(intent)
            if field.name != "solve_time_s"
        ),
    )
    return _digest(signature)


def _committed_digest(ledger) -> str:
    digest = hashlib.sha256()
    for flight_id, volume in ledger.iter_committed():
        digest.update(pickle.dumps((flight_id, _value_sig(volume)), protocol=5))
    return digest.hexdigest()


def _static_digest(ledger) -> str:
    digest = hashlib.sha256()
    for volume in ledger.static_volumes():
        digest.update(pickle.dumps(_value_sig(volume), protocol=5))
    return digest.hexdigest()


def _run_child(args) -> int:
    if args._result is None:
        raise ValueError("internal child mode requires --_result")

    cfg, demand, scenario, generated = _build(args)

    # Compile/load the selected planner kernel before measurement in each fresh child. Scenario caches
    # remain cold and isolated for both modes.
    warm = get_planner(cfg.planner)
    del warm
    gc.collect()

    previous = tc.SKIP_FOREIGN_WHEN_WALLED
    tc.SKIP_FOREIGN_WHEN_WALLED = args._mode == "patched"
    t0 = perf_counter()
    try:
        result = run(
            cfg,
            scenario=scenario,
            demand=demand,
            progress=True,
            parallel=None,
        )
    finally:
        wall_s = perf_counter() - t0
        tc.SKIP_FOREIGN_WHEN_WALLED = previous

    static_walls = len(result.ledger.static_volumes())
    expected_static_walls = (
        len(demand.terminals(cfg))
        if hasattr(demand, "terminals")
        else None
    )
    if expected_static_walls is not None and static_walls != expected_static_walls:
        raise AssertionError(
            f"registered {static_walls} static terminal walls; expected {expected_static_walls}"
        )

    intent_digests = tuple(_intent_digest(intent) for intent in result.intents)
    flight_ids = tuple(intent.request.flight_id for intent in result.intents)
    config_digest = _digest(_value_sig(result.config))
    committed_digest = _committed_digest(result.ledger)
    static_digest = _static_digest(result.ledger)
    result_digest = _digest((
        config_digest,
        bool(result.verified),
        intent_digests,
        committed_digest,
        static_digest,
        result.telemetry is None,
    ))
    record = {
        "mode": args._mode,
        "scenario": args.scenario,
        "levels": tuple(cfg.flight_levels_m),
        "generated": generated,
        "flights": len(result.intents),
        "flight_ids": flight_ids,
        "accepted": len(result.accepted),
        "denied": len(result.denied),
        "verified": bool(result.verified),
        "static_walls": static_walls,
        "wall_s": wall_s,
        "intent_digests": intent_digests,
        "config_digest": config_digest,
        "committed_digest": committed_digest,
        "static_digest": static_digest,
        "result_digest": result_digest,
    }
    args._result.write_bytes(pickle.dumps(record, protocol=5))
    print(
        f"CHILD mode={args._mode} flights={record['flights']} accepted={record['accepted']} "
        f"denied={record['denied']} walls={static_walls} verified={record['verified']} "
        f"wall={wall_s:.3f}s "
        f"digest={result_digest}",
        flush=True,
    )
    return 0


def _child_command(args, mode: str, result_path: Path) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), args.scenario]
    if args.smoke:
        command.append("--smoke")
    if args.lam_scale != 1.0:
        command.extend(("--lam-scale", str(args.lam_scale)))
    if args.flights is not None:
        command.extend(("--flights", str(args.flights)))
    if args.levels is not None:
        command.extend(("--levels", ",".join(str(z) for z in args.levels)))
    command.extend(("--_mode", mode, "--_result", str(result_path)))
    return command


def _first_divergence(baseline, patched) -> str:
    for key in ("config_digest", "verified", "flights", "static_walls"):
        if baseline[key] != patched[key]:
            return f"{key} differs: {baseline[key]!r} != {patched[key]!r}"
    for index, (base, skip) in enumerate(
            zip(baseline["intent_digests"], patched["intent_digests"])):
        if base != skip:
            return (
                f"intent index {index} (flight_id={baseline['flight_ids'][index]}) differs: "
                f"{base} != {skip}"
            )
    for key in ("committed_digest", "static_digest"):
        if baseline[key] != patched[key]:
            return f"{key} differs: {baseline[key]} != {patched[key]}"
    return "composite result digest differs"


def _run_parent(args) -> int:
    print(
        f"scenario={args.scenario} smoke={args.smoke} lam_scale={args.lam_scale} "
        f"flights={args.flights or 'all'} levels={args.levels or 'scenario default'}",
        flush=True,
    )
    print(
        "Running patched first, then baseline, in fresh interpreters "
        "(any residual OS-cache advantage goes to the baseline).",
        flush=True,
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    records = {}
    with tempfile.TemporaryDirectory(prefix="column-clear-ab-") as tmp:
        tmp_path = Path(tmp)
        for mode in ("patched", "baseline"):
            result_path = tmp_path / f"{mode}.pickle"
            subprocess.run(
                _child_command(args, mode, result_path),
                check=True,
                env=env,
            )
            records[mode] = pickle.loads(result_path.read_bytes())

    baseline, patched = records["baseline"], records["patched"]
    if baseline["result_digest"] != patched["result_digest"]:
        raise AssertionError(f"NOT exact: {_first_divergence(baseline, patched)}")
    if not baseline["verified"] or not patched["verified"]:
        raise AssertionError(
            f"verification failed: baseline={baseline['verified']} patched={patched['verified']}"
        )

    n = baseline["flights"]
    base_s, patch_s = baseline["wall_s"], patched["wall_s"]
    speedup = base_s / patch_s
    faster = (1.0 - patch_s / base_s) * 100.0
    print(
        f"\nPARITY: EXACT ✓  {n} flights; all deterministic fields match "
        f"(solve_time_s excluded)\n"
        f"  digest={baseline['result_digest']}\n"
        f"  accepted={baseline['accepted']} denied={baseline['denied']} "
        f"walls={baseline['static_walls']} verified={baseline['verified']}\n",
        flush=True,
    )
    print(
        f"SPEED: baseline(scan)={base_s:.3f}s ({base_s / n * 1000.0:.3f} ms/flight)  "
        f"patched(skip)={patch_s:.3f}s ({patch_s / n * 1000.0:.3f} ms/flight)  "
        f"speedup ×{speedup:.3f} (delta={faster:+.2f}%)",
        flush=True,
    )
    if args.require_speedup and patch_s >= base_s:
        raise AssertionError(
            f"optimized mode did not improve wall time: {patch_s:.3f}s >= {base_s:.3f}s"
        )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args._mode is not None:
        return _run_child(args)
    if args._result is not None:
        raise ValueError("--_result is only valid with internal --_mode")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
