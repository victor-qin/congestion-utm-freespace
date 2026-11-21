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
file, ``--zeta-option``. Unknown names and values of the wrong type (checked against each
field's annotation) stop the run before any work. ``zeta_options`` apply to the Zeta arm only;
``turn_below_90`` is a ``params`` field that the Zeta options follow.

Both arms apply ``lp_gap`` to their certified bounds. The Zeta arm also limits each search's
stored labels from its share of the job's memory. A capped search keeps its valid bound and
cannot claim exhaustive convergence. ``--memory-limit-gib`` can lower the detected allocation;
``--parent-reserve-gib`` overrides the parent reserve (default max(8 GiB, 20 MiB per flight)).
An explicit ``zeta_search_memory_bytes`` can tighten the automatically sized cap.

Input and options files are read once: a multi-arm run freezes copies next to its outputs and
every arm solves exactly the bytes whose sha256 it reports.
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
# native index, with no fixed work caps; the memory-derived label cap still applies.
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


def _field_types(target: type) -> dict[str, str]:
    """Map each init field of a dataclass to its annotation text (``int | None``, ...)."""
    return {field.name: (field.type if isinstance(field.type, str)
                         else getattr(field.type, "__name__", str(field.type)))
            for field in fields(target) if field.init}


def typed_value(name: str, value: Any, annotation: str, label: str) -> Any:
    """Check one override against its field annotation, converting only lossless cases.

    Parameters
    ------------
    - name (str): field name, for error messages.
    - value (Any): parsed CLI literal or JSON value.
    - annotation (str): the field's annotation text, e.g. ``bool``, ``int | None``.
    - label (str): option name used in error messages.

    Return
    --------
    - output (Any): the value, with ints widened for float fields and lists made tuples;
      a mismatch (``forward_cruise="false"``, ``max_labels=True``) stops the run.
    """
    kinds = [part.strip() for part in annotation.replace("Optional[", "").rstrip("]").split("|")]
    if value is None:
        if "None" in kinds or annotation.startswith("Optional["):
            return None
        raise SystemExit(f"{label} {name}: None is not allowed for {annotation}")
    kinds = [kind for kind in kinds if kind != "None"]
    for kind in kinds:
        base = kind.split("[", 1)[0]
        if base == "bool" and isinstance(value, bool):
            return value
        if base == "int" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if base == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if base == "str" and isinstance(value, str):
            return value
        if base == "tuple" and isinstance(value, (list, tuple)):
            return tuple(value)
    if any(kind.split("[", 1)[0] not in ("bool", "int", "float", "str", "tuple") for kind in kinds):
        return value  # An annotation this check does not model; the dataclass validates it.
    raise SystemExit(f"{label} {name}={value!r}: expected {annotation}")


def parse_overrides(pairs: list[str], target: type, label: str) -> dict[str, Any]:
    """Parse repeated ``NAME=VALUE`` options into type-checked dataclass field overrides.

    Parameters
    ------------
    - pairs (list[str]): raw ``NAME=VALUE`` strings; values are Python literals, ``true``/
      ``false``/``none``/``null`` in any case, bare strings, or (tuple fields) comma lists.
    - target (type): dataclass whose init fields the names must match.
    - label (str): option name used in error messages.

    Return
    --------
    - output (dict[str, Any]): field name to parsed value.
    """
    types = _field_types(target)
    words = {"true": True, "false": False, "none": None, "null": None}
    parsed: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep or name not in types:
            raise SystemExit(f"{label} {pair!r}: expected NAME=VALUE with NAME a {target.__name__} field")
        if raw.lower() in words:
            value = words[raw.lower()]
        else:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                # Tuple fields take bare comma lists (arrival,resource), which need no shell quoting.
                value = (tuple(part for part in raw.split(",") if part)
                         if "tuple" in types[name] else raw)
        parsed[name] = typed_value(name, value, types[name], label)
    return parsed


def file_overrides(raw: Any, target: type, label: str) -> dict[str, Any]:
    """Validate one options-file section against a dataclass's init fields and their types.

    Parameters
    ------------
    - raw (Any): the section's JSON object (field name to JSON value).
    - target (type): dataclass whose init fields the names must match.
    - label (str): section name used in error messages.

    Return
    --------
    - output (dict[str, Any]): field name to type-checked value; JSON lists become tuples.
    """
    if not isinstance(raw, dict):
        raise SystemExit(f"--options-file {label!r} must be a JSON object of field names")
    types = _field_types(target)
    unknown = sorted(set(raw) - set(types))
    if unknown:
        raise SystemExit(f"--options-file {label!r}: {unknown} are not {target.__name__} fields "
                         "in this checkout")
    return {name: typed_value(name, value, types[name], f"--options-file {label}")
            for name, value in raw.items()}


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


def load_input(raw_bytes: bytes, count: int | None) -> tuple[Any, list[Any], tuple[Any, ...], dict[str, Any]]:
    """Rebuild the captured config, FCFS-ordered requests and terminals by field name.

    Parameters
    ------------
    - raw_bytes (bytes): JSON capture with ``config``, ``requests``, ``static_terminals`` and
      ``params``, read once so the reported sha256 covers exactly what is solved.
    - count (int | None): keep the first ``count`` requests in FCFS order; None keeps all.

    Return
    --------
    - output (tuple): SimConfig, requests, static terminals and the captured params dict.
    """
    import numpy as np
    from freespace_sim.config import SimConfig
    from freespace_sim.types import FlightRequest, Terminal, vec

    data = json.loads(raw_bytes)
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


def memory_limit() -> tuple[int, str]:
    """Find the smallest memory allocation visible to this process.

    Return
    --------
    - output (tuple[int, str]): bytes and source among physical RAM, Slurm allocation and
      cgroup v1/v2 limits (including parent groups); unlimited groups are ignored.
    """
    found: list[tuple[int, str]] = []
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if physical > 0:
            found.append((physical, "physical"))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            root, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            root, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        # In a cgroup namespace the visible mount itself may be the constrained group.
        relative = Path(path.lstrip("/"))
        group = root if ".." in relative.parts else root / relative
        while True:
            limit_file = group / name
            try:
                raw = limit_file.read_text().strip()
            except OSError:
                raw = ""
            if raw.isdigit() and int(raw) < 2**60:
                found.append((int(raw), str(limit_file)))
            if group == root:
                break
            group = group.parent
    per_node = os.environ.get("SLURM_MEM_PER_NODE", "")
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE", "")
    if per_node.isdigit() and int(per_node) > 0:
        found.append((int(per_node) * 2**20, "SLURM_MEM_PER_NODE"))
    elif per_cpu.isdigit() and cpus.isdigit() and int(per_cpu) > 0 and int(cpus) > 0:
        found.append((int(per_cpu) * int(cpus) * 2**20, "SLURM_MEM_PER_CPU*SLURM_CPUS_ON_NODE"))
    if not found:
        raise ValueError("cannot determine a memory limit; pass --memory-limit-gib")
    return min(found)


def size_search_memory(
    params: Any, n_flights: int, memory_limit_gib: float | None,
    parent_reserve_gib: float | None,
) -> tuple[Any, dict[str, Any]]:
    """Apply the job-sized Zeta label budget, respecting any tighter caller budget.

    Parameters
    ------------
    - params (ColGenParams): resolved runtime settings.
    - n_flights (int): cohort size, which caps the useful worker count.
    - memory_limit_gib (float | None): optional lower memory limit for this run.
    - parent_reserve_gib (float | None): optional parent-process memory reserve.

    Return
    --------
    - output (tuple): updated parameters and the sizing evidence written to the report.
    """
    from freespace_sim.planner.colgen.zeta_pool import search_label_cap, search_memory_bytes

    for name, value in (("memory-limit-gib", memory_limit_gib),
                        ("parent-reserve-gib", parent_reserve_gib)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name} must be finite and positive")
    try:
        limit, source = memory_limit()
    except ValueError:
        if memory_limit_gib is None:
            raise
        limit, source = int(memory_limit_gib * 2**30), "--memory-limit-gib"
    if memory_limit_gib is not None and int(memory_limit_gib * 2**30) < limit:
        limit, source = int(memory_limit_gib * 2**30), "--memory-limit-gib"
    reserve = int(parent_reserve_gib * 2**30 if parent_reserve_gib is not None
                  else max(8 * 2**30, n_flights * 20 * 2**20))
    workers = max(1, min(params.n_pricing_workers, n_flights))
    budget = search_memory_bytes(limit, workers, n_flights, parent_reserve_bytes=reserve)
    requested = params.zeta_search_memory_bytes
    if requested is not None:
        budget = min(budget, requested)
    return replace(params, zeta_search_memory_bytes=budget), {
        "memory_limit_bytes": limit, "memory_limit_source": source,
        "parent_reserve_bytes": reserve, "effective_workers": workers,
        "requested_search_memory_bytes": requested,
        "search_memory_bytes": budget, "search_label_cap": search_label_cap(budget),
    }


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


def write_new(path: Path, data: bytes) -> None:
    """Write ``path`` atomically and only if it does not exist yet.

    The bytes go to a private temporary file that is then hard-linked into place, so a
    concurrent writer or an interruption can never leave a partial or overwritten file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise SystemExit(f"{path} exists; measurements never overwrite a report") from None
    finally:
        temporary.unlink()


def run_sequence(args: argparse.Namespace, arms: list[str]) -> None:
    """Run each arm in its own interpreter so no cache or allocator state carries over.

    Parameters
    ------------
    - args (argparse.Namespace): parsed options shared by every arm.
    - arms (list[str]): pricing kinds in run order.

    Return
    --------
    - output (None): each arm writes ``<out stem>-<arm>.json``; a failed arm stops the rest.
      The input and options files are frozen once beside the outputs, so every arm solves
      the same bytes even if the originals change mid-run.
    """
    outputs = [args.out.with_name(f"{args.out.stem}-{arm}.json") for arm in arms]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing}")
    frozen_input = args.out.with_name(f"{args.out.stem}-input.json")
    write_new(frozen_input, args.input.read_bytes())
    shared = ["--input", str(frozen_input), "--workers", str(args.workers)]
    if args.gurobi_threads is not None:
        shared += ["--gurobi-threads", str(args.gurobi_threads)]
    if args.options_file is not None:
        frozen_options = args.out.with_name(f"{args.out.stem}-options.json")
        write_new(frozen_options, args.options_file.read_bytes())
        shared += ["--options-file", str(frozen_options)]
    for flag, value in (("--memory-limit-gib", args.memory_limit_gib),
                        ("--parent-reserve-gib", args.parent_reserve_gib)):
        if value is not None:
            shared += [flag, str(value)]
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
    parser.add_argument("--gurobi-threads", type=int, default=None,
                        help="master LP/IP threads; default keeps COLGEN_GUROBI_THREADS or the solver default")
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
    parser.add_argument("--memory-limit-gib", type=float, default=None,
                        help="lower the detected memory allocation used to size the Zeta cap")
    parser.add_argument("--parent-reserve-gib", type=float, default=None,
                        help="parent peak set aside before splitting memory over workers")
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
    if args.gurobi_threads is not None:
        os.environ["COLGEN_GUROBI_THREADS"] = str(args.gurobi_threads)
    input_bytes = args.input.read_bytes()

    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.translate import column_to_intent
    from freespace_sim.planner.colgen.zeta import ZetaOptions
    from freespace_sim.verify import find_interflight_conflict

    started = time.monotonic()
    cfg, requests, terms, captured = load_input(input_bytes, args.count)
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
    sizing = None
    if args.pricing == "zeta":
        try:
            params, sizing = size_search_memory(
                params, len(requests), args.memory_limit_gib, args.parent_reserve_gib,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    input_s = time.monotonic() - started

    if args.pricing == "zeta":
        from freespace_sim.planner.colgen.zeta_solver import solve_zeta
        from freespace_sim.planner.zeta.native import prepare_compiled_loop

        # Only the compiled loop needs the native build (and a C compiler).
        if options.compiled_loop:
            prepare_compiled_loop()

        def solve(reqs: Any, config: Any, statics: Any, solve_params: Any) -> Any:
            """Run the Zeta pricing solve."""
            return solve_zeta(reqs, config, statics, solve_params, options=options)
    else:
        from freespace_sim.planner.colgen.solver import ColGenSolver

        dp_rounds: list[dict[str, Any]] = []
        # Model objects and per-column arrays stay out of the report; per-flight lists only
        # with --flight-records, matching the Zeta history.
        skipped = {"master", "lp_x", "lp_columns", "capacity_duals"}

        def record_round(state: dict[str, Any]) -> None:
            """Keep DP's per-round evidence, which it exposes only through this callback."""
            dp_rounds.append({key: value for key, value in state.items() if key not in skipped
                              and (args.flight_records or not isinstance(value, (list, tuple, dict)))})

        def solve(reqs: Any, config: Any, statics: Any, solve_params: Any) -> Any:
            """Run the production DP pricing solve, recording each round."""
            dp_rounds.clear()
            return ColGenSolver().solve(reqs, config, statics, solve_params, on_iteration=record_round)

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
    ground_s = math.fsum(intent.ground_delay_s for intent in intents.values())
    hold_s = math.fsum(intent.air_hold_s for intent in intents.values())
    detour_s = math.fsum(intent.air_detour_m / cfg.nominal_speed_mps for intent in intents.values())
    audit = {"served": len(intents), "denied": len(requests) - len(intents),
             "objective_cost_recomputed": physical + (len(requests) - len(intents)) * params.M,
             "cost_components": {
                 "ground_delay_s": ground_s, "air_hold_s": hold_s, "air_detour_s": detour_s,
                 "ground_cost": model.evaluate(ground_s=ground_s),
                 "hold_cost": model.evaluate(ground_s=0.0, air_hold_s=hold_s),
                 "detour_cost": model.evaluate(ground_s=0.0, air_detour_s=detour_s),
                 "denial_cost": (len(requests) - len(intents)) * params.M,
             },
             "collision_pair": conflict,
             "all_intents_accepted": all(intent.accepted for intent in intents.values()),
             "audit_s": time.monotonic() - audit_started}
    stats = dict(result.stats)
    history = stats.pop("history", []) if args.pricing == "zeta" else list(dp_rounds)
    # DP rounds always carry their LP gap; a Zeta solver applies lp_gap only if its rounds do.
    lp_gap_applied = any(isinstance(row, dict) and "lp_gap" in row for row in history)
    report = {
        "pricing": args.pricing, "flights": len(requests),
        "input": str(args.input), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "workers": args.workers, "gurobi_threads": os.environ.get("COLGEN_GUROBI_THREADS"),
        "lp_gap_applied": lp_gap_applied,
        "params": asdict(params), "param_overrides": overrides,
        "zeta_options": None if options is None else asdict(options),
        "zeta_option_overrides": zeta_overrides if options is not None else None,
        "options_file": options_record,
        "memory_sizing": sizing,
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
    write_new(args.out, (json.dumps(json_safe(report), indent=2) + "\n").encode())
    print(json.dumps({"event": "done", "solve_s": solve_s, "served": audit["served"],
                      "termination": report["termination_reason"],
                      "collision": conflict is not None}), flush=True)
    if conflict is not None or not audit["all_intents_accepted"]:
        raise SystemExit("audit failed: selected schedule collides or a route was not accepted")


if __name__ == "__main__":
    main()
