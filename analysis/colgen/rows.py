"""Benchmark public capacity separation with real captured incumbent columns."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import sys
import time

from analysis.colgen.common import sha256_bytes, source_hashes, write_json


def _encode(value):
    """Convert row data to a deterministic JSON representation."""
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return [[_encode(key), _encode(item)] for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_encode(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        return [_encode(item) for item in tuple(value)]
    except TypeError:
        return repr(value)


def _digest(value) -> str:
    """Hash a deterministic representation of row results."""
    return hashlib.sha256(json.dumps(_encode(value), sort_keys=True).encode()).hexdigest()


def _distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize a small repeat set without discarding its raw values."""
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total_s": math.fsum(ordered),
        "min_s": ordered[0],
        "median_s": statistics.median(ordered),
        "max_s": ordered[-1],
        "raw_s": values,
    }


class RecordingBackend:
    """Minimal injected backend that records rows without solving an LP or IP."""

    name = "recording"
    time_limit_s = math.inf
    ip_gap = 0.0

    def __init__(self, flight_ids) -> None:
        """Store the master's normalized flight IDs and future row calls."""
        self.flight_ids = tuple(flight_ids)
        self.columns = []
        self.rows = []

    def add_column(self, objective, flight_id, column_rows) -> None:
        """Record one column insertion requested by the public master API."""
        self.columns.append((objective, flight_id, tuple(column_rows)))

    def add_row(self, row, rhs, column_indices) -> None:
        """Record one row insertion and its immutable coefficient snapshot."""
        self.rows.append((row, rhs, tuple(column_indices)))

    def solve_lp(self):
        """Reject solves because this harness measures separation only."""
        raise AssertionError("row-scan harness does not solve an LP")

    def solve_ip(self, warm_start=None):
        """Reject solves because this harness measures separation only."""
        raise AssertionError("row-scan harness does not solve an IP")


def main() -> int:
    """Measure initial and no-new-row separation on one captured incumbent pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    from freespace_sim.planner.colgen.master import RestrictedMaster
    from freespace_sim.planner.colgen.network import RowIndex
    from freespace_sim.planner.colgen.params import ColGenParams

    with args.capture.open("rb") as handle:
        captured = pickle.load(handle)
    old_params = captured["params"]
    params = ColGenParams(
        **{
            field.name: getattr(old_params, field.name)
            for field in fields(ColGenParams)
            if hasattr(old_params, field.name)
        }
    )
    columns = [captured["known_columns"][flight_id] for flight_id in captured["pricing_order"]]
    columns = [column for column in columns if column is not None]
    flight_ids = tuple(column.flight_id for column in columns)
    terminals = {
        terminal.id: terminal.capacity for _center, terminal in captured["static_terminals"]
    }
    backend = RecordingBackend(flight_ids)
    master = RestrictedMaster(
        flight_ids,
        RowIndex(terminals),
        params,
        seed=captured["cfg"].seed,
        backend=backend,
    )
    setup_started = time.perf_counter()
    for column in columns:
        master.add_column(column)
    setup_s = time.perf_counter() - setup_started
    x = [1.0] * len(columns)
    load_started = time.perf_counter()
    loads = master.fractional_loads(x)
    fractional_loads_s = time.perf_counter() - load_started
    initial_started = time.perf_counter()
    added = master.add_violated_rows(x)
    initial_s = time.perf_counter() - initial_started
    repeat_s = []
    repeat_added = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        repeat_added.append(master.add_violated_rows(x))
        repeat_s.append(time.perf_counter() - started)
    result = {
        "columns": len(columns),
        "claims": sum(len(column.claims) for column in columns),
        "fractional_rows": len(loads),
        "fractional_loads_sha256": _digest(loads),
        "added_rows": added,
        "recorded_rows_sha256": _digest(backend.rows),
        "repeat_added_rows": repeat_added,
        "setup_s_excluded": setup_s,
        "fractional_loads_s": fractional_loads_s,
        "initial_add_violated_rows_s": initial_s,
        "repeat_no_new_rows": _distribution(repeat_s),
    }
    write_json(args.out / "result.json", result)
    manifest = {
        "source_root": str(source),
        "source_sha256": source_hashes(source),
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_bytes(args.capture.read_bytes()),
        "captured_round": captured.get("round"),
        "harness_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "tool_dependency_sha256": {
            "common.py": sha256_bytes(Path(__file__).with_name("common.py").read_bytes()),
        },
        "scope": (
            "Public RestrictedMaster.add_violated_rows over one real incumbent column per "
            "captured flight and x=1. Pool construction and fractional-load control timing "
            "are separate; this is not the full multi-column RMP."
        ),
    }
    write_json(args.out / "manifest.json", manifest)
    if args.reference is not None:
        expected = json.loads((args.reference / "result.json").read_text())
        checks = {
            key: result[key] == expected[key]
            for key in (
                "columns",
                "claims",
                "fractional_rows",
                "fractional_loads_sha256",
                "added_rows",
                "recorded_rows_sha256",
                "repeat_added_rows",
            )
        }
        parity = {"passed": all(checks.values()), "checks": checks}
        write_json(args.out / "parity.json", parity)
        if not parity["passed"]:
            raise AssertionError("row-scan result differs from the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
