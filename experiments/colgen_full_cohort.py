"""Solve one fixed full-cohort input with DP or Zeta pricing, timed and audited the same way.

Both arms load the same JSON capture (config, requests, static terminals and ColGenParams by
field name), run the same untimed two-flight warm-up, apply the same parameter overrides and
are timed around the solve call alone. Only the pricing differs: ``--pricing dp`` runs
``ColGenSolver`` and ``--pricing zeta`` runs ``solve_zeta`` with the Zeta options below.

    python -m experiments.colgen_full_cohort --input experiments/worker_seed_full_input.json \
        --out results/full_cohort/zeta.json --pricing zeta --workers 44 --gurobi-threads 32 \
        [--options-file opts.json] [--param turn_below_90=True ...] \
        [--zeta-option forward_cruise=False ...] [--count N]

``--pricing dp,zeta`` runs each arm in turn, each in a fresh interpreter, writing
``<out stem>-<arm>.json``, so a single job measures both arms on the same node.

``--options-file`` takes JSON ``{"params": {...}, "zeta_options": {...}}`` keyed by
ColGenParams and ZetaOptions field names of the checkout that runs (so a branch's own fields,
such as ``price_snap``, pass through); lists become tuples for tuple fields. Precedence, low to
high: captured params, ``--workers``/gurobi, the file, ``--param``; ``ZETA_DEFAULTS``, the
file, ``--zeta-option``. Unknown names stop the run before any work. ``zeta_options`` apply to
the Zeta arm only; ``turn_below_90`` is a ``params`` field that the Zeta options follow.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

# The measured Zeta configuration: twelve directions, best-first, both history merges and the
# native index, with no work caps so no pricing call is cut off. Overridable per run.
ZETA_DEFAULTS: dict[str, Any] = dict(
    connection_mode="hex12", forward_cruise=False, traversal="best_first",
    lattice_distance_bound=True, max_expansions=None, max_labels=None,
    lazy_successors=True, expire_history=True, history_dominance=True,
    strong_history_dominance=True, history_group_skips=True,
    merge_equivalent_histories=True, merge_shorter_histories=True,
    backend="compiled", compiled_loop=True, compiled_history=True,
    route_coverage_cache=True, compiled_route_coverage=True, native_state_index=True,
    leg_cache=True, persistent_timing_cache=True,
    arrival_snap_s=1.0, speed_fraction=0.05, early_return=True,
)


def json_safe(value: Any) -> Any:
    """Convert evidence to strict JSON, keeping nonfinite floats as explicit strings."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parse_overrides(pairs: list[str], target: type, label: str) -> dict[str, Any]:
    """Parse repeated ``NAME=VALUE`` options into dataclass field overrides.

    Parameters
    ------------
    - pairs (list[str]): raw ``NAME=VALUE`` strings; values are Python literals, else strings
      (or, for tuple fields, comma-separated strings).
    - target (type): dataclass whose init fields the names must match.
    - label (str): option name used in error messages.

    Return
    --------
    - output (dict[str, Any]): field name to parsed value.
    """
    defaults = {field.name: field.default for field in fields(target) if field.init}
    parsed: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep or name not in defaults:
            raise SystemExit(f"{label} {pair!r}: expected NAME=VALUE with NAME a {target.__name__} field")
        try:
            parsed[name] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Tuple fields take bare comma lists (arrival,resource), which need no shell quoting.
            parsed[name] = (tuple(part for part in raw.split(",") if part)
                            if isinstance(defaults[name], tuple) else raw)
    return parsed


def file_overrides(raw: Any, target: type, label: str) -> dict[str, Any]:
    """Validate one options-file section against a dataclass's init fields.

    Parameters
    ------------
    - raw (Any): the section's JSON object (field name to JSON value).
    - target (type): dataclass whose init fields the names must match.
    - label (str): section name used in error messages.

    Return
    --------
    - output (dict[str, Any]): field name to value; JSON lists become tuples where the field's
      default is a tuple. Values are validated later by the dataclass itself.
    """
    if not isinstance(raw, dict):
        raise SystemExit(f"--options-file {label!r} must be a JSON object of field names")
    defaults = {field.name: field.default for field in fields(target) if field.init}
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise SystemExit(f"--options-file {label!r}: {unknown} are not {target.__name__} fields "
                         "in this checkout")
    return {name: (tuple(value) if isinstance(value, list) and isinstance(defaults[name], tuple)
                   else value) for name, value in raw.items()}


def resolve_overrides(args: argparse.Namespace) -> tuple[dict, dict, dict | None]:
    """Merge options-file and command-line overrides, rejecting unknown names before any work.

    Parameters
    ------------
    - args (argparse.Namespace): parsed ``--options-file``, ``--param`` and ``--zeta-option``.

    Return
    --------
    - output (tuple): ColGenParams overrides, ZetaOptions overrides, and the options-file
      record (path, sha256, parsed content) or None.
    """
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.zeta import ZetaOptions

    sections: dict[str, Any] = {}
    record = None
    if args.options_file is not None:
        text = args.options_file.read_text()
        try:
            sections = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--options-file {args.options_file}: invalid JSON ({exc})") from exc
        if not isinstance(sections, dict) or set(sections) - {"params", "zeta_options"}:
            raise SystemExit("--options-file must be a JSON object with only 'params' and "
                             "'zeta_options' keys")
        record = {"path": str(args.options_file), "content": sections,
                  "sha256": hashlib.sha256(text.encode()).hexdigest()}
    params = {**file_overrides(sections.get("params", {}), ColGenParams, "params"),
              **parse_overrides(args.param, ColGenParams, "--param")}
    zeta = {**file_overrides(sections.get("zeta_options", {}), ZetaOptions, "zeta_options"),
            **parse_overrides(args.zeta_option, ZetaOptions, "--zeta-option")}
    if "turn_below_90" in zeta:
        raise SystemExit("set the turn limit once, as the params field turn_below_90")
    return params, zeta, record


def load_input(path: Path, count: int | None) -> tuple[Any, list[Any], tuple[Any, ...], dict[str, Any]]:
    """Rebuild the captured config, FCFS-ordered requests and terminals by field name.

    Parameters
    ------------
    - path (Path): JSON capture with ``config``, ``requests``, ``static_terminals``, ``params``.
    - count (int | None): keep the first ``count`` requests in FCFS order; None keeps all.

    Return
    --------
    - output (tuple): SimConfig, requests, static terminals and the captured params dict.
    """
    import numpy as np
    from freespace_sim.config import SimConfig
    from freespace_sim.types import FlightRequest, Terminal, vec

    data = json.loads(path.read_text())
    config = dict(data["config"])
    for name in ("flight_levels_m", "region_center_latlon", "region_size_m"):
        config[name] = tuple(config[name])
    cfg = SimConfig(**config)
    known = {field.name for field in fields(FlightRequest)}
    requests = []
    for raw in data["requests"]:
        raw = dict(raw)
        # Older captures carry paired-leg metadata; this cohort is outbound-only.
        if raw.pop("paired_outbound_id", None) is not None:
            raise SystemExit("capture contains a paired return leg, which this harness does not rebuild")
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"capture request fields unknown to this checkout: {sorted(unknown)}")
        raw["origin"], raw["dest"] = vec(*raw["origin"]), vec(*raw["dest"])
        for name in ("origin_terminal", "dest_terminal"):
            if raw[name] is not None:
                raw[name] = Terminal(*raw[name])
        requests.append(FlightRequest(**raw))
    ordered = sorted(requests, key=lambda request: request.sort_key())
    terms = tuple((np.asarray(point, dtype=float), Terminal(*terminal))
                  for point, terminal in data["static_terminals"])
    return cfg, ordered[:count] if count else ordered, terms, data["params"]


def warm_case(cfg: Any, params: Any) -> tuple[Any, list[Any], Any]:
    """Two small crossing flights that absorb first-use compile and import cost untimed."""
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.types import FlightRequest, vec

    warm_cfg = replace(cfg, max_ground_delay_s=32.0, max_detour_factor=3.0)

    def point(q: int, r: int) -> Any:
        """Put one endpoint on the unchanged physical lattice."""
        return vec(*hg.hex_center(q, r, hg.circumradius(warm_cfg)), warm_cfg.ground_level_m)

    requests = [FlightRequest(0, point(-2, 0), point(2, 0), 0.0, 0.0),
                FlightRequest(1, point(0, -2), point(0, 2), 0.0, 0.0)]
    warm_params = replace(params, n_pricing_workers=0, max_air_overrun_hops=1,
                          time_limit_s=60.0, ip_time_limit_s=5.0, ip_reserve_s=6.0)
    return warm_cfg, requests, warm_params


def peak_rss_bytes() -> dict[str, int]:
    """Peak RSS of this process and of its largest finished child (Linux reports KiB)."""
    scale = 1 if sys.platform == "darwin" else 1024
    return {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
            "largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale}


def run_sequence(args: argparse.Namespace, arms: list[str]) -> None:
    """Run each arm in its own interpreter so no cache or allocator state carries over.

    Parameters
    ------------
    - args (argparse.Namespace): parsed options shared by every arm.
    - arms (list[str]): pricing kinds in run order.

    Return
    --------
    - output (None): each arm writes ``<out stem>-<arm>.json``; a failed arm stops the rest.
    """
    outputs = [args.out.with_name(f"{args.out.stem}-{arm}.json") for arm in arms]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing}")
    shared = ["--input", str(args.input), "--workers", str(args.workers),
              "--gurobi-threads", str(args.gurobi_threads)]
    if args.options_file is not None:
        shared += ["--options-file", str(args.options_file)]
    if args.count:
        shared += ["--count", str(args.count)]
    for pair in args.param:
        shared += ["--param", pair]
    if args.flight_records:
        shared.append("--flight-records")
    for arm, out in zip(arms, outputs):
        command = [sys.executable, "-m", "experiments.colgen_full_cohort", *shared,
                   "--pricing", arm, "--out", str(out)]
        if arm == "zeta":
            for pair in args.zeta_option:
                command += ["--zeta-option", pair]
        subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    """Run one timed, audited full-cohort solve and write its evidence as JSON."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pricing", required=True,
                        help="dp, zeta, or a comma-separated sequence such as dp,zeta")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gurobi-threads", type=int, default=4)
    parser.add_argument("--count", type=int, default=None,
                        help="solve only the first N requests in FCFS order")
    parser.add_argument("--param", action="append", default=[],
                        help="ColGenParams override NAME=VALUE (repeatable; applies to both arms)")
    parser.add_argument("--zeta-option", action="append", default=[],
                        help="ZetaOptions override NAME=VALUE (repeatable; zeta arm only)")
    parser.add_argument("--flight-records", action="store_true",
                        help="keep per-flight pricing records in each round's history")
    parser.add_argument("--options-file", type=Path, default=None,
                        help='JSON {"params": {...}, "zeta_options": {...}} field overrides')
    args = parser.parse_args(argv)
    arms = args.pricing.split(",")
    if not arms or any(arm not in ("dp", "zeta") for arm in arms):
        raise SystemExit("--pricing takes dp, zeta, or a comma-separated sequence of them")
    param_overrides, zeta_overrides, options_record = resolve_overrides(args)
    if len(arms) > 1:
        run_sequence(args, arms)
        return
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; measurements never overwrite a report")
    if args.zeta_option and args.pricing != "zeta":
        raise SystemExit("--zeta-option applies only to --pricing zeta")
    os.environ["COLGEN_GUROBI_THREADS"] = str(args.gurobi_threads)

    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.translate import column_to_intent
    from freespace_sim.planner.colgen.zeta import ZetaOptions
    from freespace_sim.verify import find_interflight_conflict

    started = time.monotonic()
    cfg, requests, terms, captured = load_input(args.input, args.count)
    names = {field.name for field in fields(ColGenParams) if field.init}
    unknown = sorted(set(captured) - names)
    if unknown:
        raise SystemExit(f"captured params unknown to this checkout: {unknown}")
    overrides = {"n_pricing_workers": args.workers, "solver": "gurobi", **param_overrides}
    params = ColGenParams(**{**captured, **overrides})
    # The Zeta solver requires both flags to agree; the ColGenParams one is the single knob.
    options = (ZetaOptions(**{**ZETA_DEFAULTS, **zeta_overrides,
                              "turn_below_90": params.turn_below_90})
               if args.pricing == "zeta" else None)
    input_s = time.monotonic() - started

    if args.pricing == "zeta":
        from freespace_sim.planner.colgen.zeta_solver import solve_zeta
        from freespace_sim.planner.zeta.native import prepare_compiled_loop

        prepare_compiled_loop()

        def solve(reqs: Any, config: Any, statics: Any, solve_params: Any) -> Any:
            """Run the Zeta pricing solve."""
            return solve_zeta(reqs, config, statics, solve_params, options=options)
    else:
        from freespace_sim.planner.colgen.solver import ColGenSolver

        def solve(reqs: Any, config: Any, statics: Any, solve_params: Any) -> Any:
            """Run the production DP pricing solve."""
            return ColGenSolver().solve(reqs, config, statics, solve_params)

    warm_cfg, warm_requests, warm_params = warm_case(cfg, params)
    warm_started = time.monotonic()
    solve(warm_requests, warm_cfg, (), warm_params)
    warmup_s = time.monotonic() - warm_started

    print(json.dumps({"event": "solve_start", "pricing": args.pricing,
                      "flights": len(requests)}), flush=True)
    solve_started = time.monotonic()
    result = solve(requests, cfg, terms, params)
    solve_s = time.monotonic() - solve_started

    audit_started = time.monotonic()
    by_id = {request.flight_id: request for request in requests}
    intents = (dict(result.intents) if args.pricing == "zeta" else
               {fid: column_to_intent(column, by_id[fid], cfg)
                for fid, column in result.columns.items()})
    conflict = find_interflight_conflict(list(intents.values()), cfg, terms)
    model = cost_model(cfg, params)
    physical = math.fsum(model.intent_cost(intent, cfg) for intent in intents.values())
    audit = {"served": len(intents), "denied": len(requests) - len(intents),
             "objective_cost_recomputed": physical + (len(requests) - len(intents)) * params.M,
             "collision_pair": conflict,
             "all_intents_accepted": all(intent.accepted for intent in intents.values()),
             "audit_s": time.monotonic() - audit_started}
    stats = dict(result.stats)
    history = stats.pop("history", [])
    report = {
        "pricing": args.pricing, "flights": len(requests),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "workers": args.workers, "gurobi_threads": args.gurobi_threads,
        "params": asdict(params), "param_overrides": overrides,
        "zeta_options": None if options is None else asdict(options),
        "zeta_option_overrides": zeta_overrides if options is not None else None,
        "options_file": options_record,
        "input_s": input_s, "warmup_s": warmup_s, "solve_s": solve_s,
        "termination_reason": stats.get("termination_reason"),
        "peak_rss_bytes": peak_rss_bytes(), "audit": audit,
        "history": [{key: value for key, value in row.items()
                     if args.flight_records or key not in ("search_statuses", "flight_pricing")}
                    for row in history if isinstance(row, dict)],
        "stats": {key: value for key, value in stats.items()
                  if not key.endswith("_flight_records")},
        "hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2) + "\n")
    print(json.dumps({"event": "done", "solve_s": solve_s, "served": audit["served"],
                      "termination": report["termination_reason"],
                      "collision": conflict is not None}), flush=True)
    if conflict is not None or not audit["all_intents_accepted"]:
        raise SystemExit("audit failed: selected schedule collides or a route was not accepted")


if __name__ == "__main__":
    main()
