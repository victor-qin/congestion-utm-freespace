"""Run the two-seed validation benchmark serially from frozen source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BASELINE = Path("/private/tmp/colgen-validation-baseline-20260910")
EXPECTED_FLIGHTS = {0: 1537, 1: 1545}


def _dump(path: Path, value) -> None:
    """Write one JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one experiment dependency."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    """Hash production Python sources in a frozen experiment tree."""
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "freespace_sim").rglob("*.py"))
    }


def _command(root: Path, out: Path, seed: int) -> list[str]:
    """Build one exact archived benchmark command."""
    return [
        sys.executable,
        str(HERE / "capture_full_run.py"),
        "--source-root",
        str(root),
        "--benchmark",
        str(HERE / "benchmark_colgen_lns.py"),
        "--out",
        str(out),
        "--rounds",
        "1",
        "6",
        "11",
        "--expected-production-flights",
        str(EXPECTED_FLIGHTS[seed]),
        "--allow-missing-rounds",
        "--",
        "--version",
        "iteration_ip_eager",
        "--scenario",
        "density_faa_wing_zipline",
        "--seed",
        str(seed),
        "--flights",
        "0",
        "--iterations",
        "30",
        "--budget",
        "14460",
        "--ip-budget",
        "600",
        "--iteration-ip-budget",
        "30",
        "--ip-reserve",
        "660",
        "--capture-columns",
        "--overrun",
        "6",
        "--columns-per-flight",
        "1",
        "--bootstrap-method",
        "astar",
        "--workers",
        "12",
        "--gurobi-threads",
        "4",
        "--lp-gap",
        "0.001",
        "--exact-pricing-interval",
        "5",
        "--destroy",
        "100",
        "--gap",
        "0",
        "--demand-seconds",
        "1200",
        "--outbound-only",
    ]


def main() -> int:
    """Plan or execute baseline0, combined0, baseline1, combined1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "full_matrix")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified completed cases and continue the fixed serial order.",
    )
    args = parser.parse_args()
    roots = {
        "baseline": args.baseline_root.resolve(),
        "combined": args.combined_root.resolve(),
    }
    for name, root in roots.items():
        if not (root / "freespace_sim").is_dir():
            parser.error(f"{name} source root is incomplete: {root}")
    order = [("baseline", 0), ("combined", 0), ("baseline", 1), ("combined", 1)]
    commands = [
        {
            "arm": arm,
            "seed": seed,
            "expected_flights": EXPECTED_FLIGHTS[seed],
            "out": str((args.out / f"{arm}_seed{seed}").resolve()),
            "command": _command(
                roots[arm], (args.out / f"{arm}_seed{seed}").resolve(), seed
            ),
        }
        for arm, seed in order
    ]
    plan = {
        "status": "ready" if not args.execute else "running",
        "order": [f"{arm}{seed}" for arm, seed in order],
        "sources": {name: str(root) for name, root in roots.items()},
        "source_sha256": {name: _source_hashes(root) for name, root in roots.items()},
        "harness_sha256": {
            name: _sha256(HERE / name)
            for name in (
                "run_full_matrix.py",
                "capture_full_run.py",
                "benchmark_colgen_lns.py",
                "colgen_trace.py",
            )
        },
        "process_policy": "Fresh processes run serially; no measured simulations overlap.",
        "timing": (
            "Warmup and demand generation are outside simulation_wall_s. Capture serialization "
            "and writes are reported and subtracted as a supporting adjusted wall; matched "
            "pricing replays provide the isolated timing comparison."
        ),
        "commands": commands,
    }
    plan_path = HERE / "full_matrix_plan.json"
    _dump(plan_path, plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.out.exists() and not args.resume:
        raise FileExistsError("Choose a fresh matrix output directory")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    env = dict(os.environ)
    env.update(
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        COLGEN_GUROBI_THREADS="4",
    )
    progress = []
    for item in commands:
        existing_result = Path(item["out"]) / "result.json"
        existing_manifest = Path(item["out"]) / "capture_manifest.json"
        if args.resume and existing_result.exists() and existing_manifest.exists():
            result = json.loads(existing_result.read_text())
            capture = json.loads(existing_manifest.read_text())
            if result.get("verified") and capture.get("status") == "complete":
                progress.append(
                    {
                        "arm": item["arm"],
                        "seed": item["seed"],
                        "returncode": 0,
                        "reused": True,
                        "result": result,
                    }
                )
                _dump(args.out / "progress.json", progress)
                if item["arm"] == "combined":
                    reference = next(
                        row["result"] for row in progress if row["seed"] == item["seed"]
                    )
                    for key in (
                        "requests_sha",
                        "generated_requests_sha",
                        "demand_config_sha",
                        "input_sha",
                    ):
                        if result[key] != reference[key]:
                            raise AssertionError(f"Seed {item['seed']} input mismatch: {key}")
                continue
            raise RuntimeError(f"Cannot resume incomplete case: {item['out']}")
        log_path = args.out / f"{item['arm']}_seed{item['seed']}_driver.log"
        with log_path.open("w") as log:
            completed = subprocess.run(
                item["command"],
                cwd=roots[item["arm"]],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result_path = Path(item["out"]) / "result.json"
        record = {
            "arm": item["arm"],
            "seed": item["seed"],
            "returncode": completed.returncode,
            "log": str(log_path.resolve()),
            "result": None if not result_path.exists() else json.loads(result_path.read_text()),
        }
        progress.append(record)
        _dump(args.out / "progress.json", progress)
        if completed.returncode:
            plan.update(status="failed", failed=record)
            _dump(plan_path, plan)
            return completed.returncode
        result = record["result"]
        if result is None or not result.get("verified"):
            plan.update(status="failed", failed=record)
            _dump(plan_path, plan)
            raise RuntimeError("Completed process has no verified result")
        if item["arm"] == "combined":
            reference = next(row["result"] for row in progress if row["seed"] == item["seed"])
            for key in ("requests_sha", "generated_requests_sha", "demand_config_sha", "input_sha"):
                if result[key] != reference[key]:
                    raise AssertionError(f"Seed {item['seed']} input mismatch: {key}")
    plan.update(status="complete", progress=str((args.out / "progress.json").resolve()))
    _dump(plan_path, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
