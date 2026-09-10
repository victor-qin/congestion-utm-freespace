"""Replay selected captured sweeps serially through one explicit source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _dump(path: Path, value) -> None:
    """Write a JSON progress artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    """Run fresh-process sweep replays and preserve exact parity results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--references", type=Path, nargs="+")
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("benchmark_captured_sweep.py"),
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 6, 11])
    parser.add_argument("--budget", type=float, default=1800.0)
    args = parser.parse_args()
    if args.references is not None and len(args.references) != len(args.rounds):
        parser.error("--references must provide one directory for each --rounds value")
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    source = args.source_root.resolve()
    harness = args.harness.resolve()
    manifest = {
        "status": "running",
        "source_root": str(source),
        "capture_root": str(args.capture_root.resolve()),
        "reference_root": None if args.reference_root is None else str(args.reference_root.resolve()),
        "rounds": args.rounds,
        "budget_s_per_round": args.budget,
        "harness": str(harness),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "process_policy": "Rounds run serially in fresh processes; no measured overlap.",
    }
    _dump(args.out / "manifest.json", manifest)
    summaries = []
    for round_number in args.rounds:
        capture = args.capture_root / f"round_{round_number:04d}.pkl"
        round_out = args.out / f"round_{round_number:04d}"
        command = [
            sys.executable,
            str(harness),
            "--source-root",
            str(source),
            "--capture",
            str(capture),
            "--out",
            str(round_out),
            "--budget",
            str(args.budget),
        ]
        if args.references is not None:
            command.extend(["--reference", str(args.references[args.rounds.index(round_number)])])
        elif args.reference_root is not None:
            command.extend(["--reference", str(args.reference_root / f"round_{round_number:04d}")])
        log_path = args.out / f"round_{round_number:04d}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"round": round_number, "returncode": completed.returncode, "log": str(log_path)}
        sweep_path = round_out / "sweep.json"
        if sweep_path.exists():
            record.update(json.loads(sweep_path.read_text()))
        parity_path = round_out / "parity.json"
        if parity_path.exists():
            record["parity"] = json.loads(parity_path.read_text())
        summaries.append(record)
        _dump(args.out / "summary.json", summaries)
        if completed.returncode:
            manifest.update(status="failed", failed_round=round_number)
            _dump(args.out / "manifest.json", manifest)
            return completed.returncode
    manifest.update(status="complete", completed_rounds=args.rounds)
    _dump(args.out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
