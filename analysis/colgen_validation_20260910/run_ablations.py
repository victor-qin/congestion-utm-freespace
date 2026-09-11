"""Run serial matched replays that isolate row scanning and parent validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def _write(path: Path, value) -> None:
    """Write one JSON progress artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    """Replay all available baseline captures through the requested frozen arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--row-only-root", type=Path, required=True)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--parent-only-root", type=Path)
    parser.add_argument("--full-matrix", type=Path, default=HERE / "full_matrix")
    parser.add_argument("--out", type=Path, default=HERE / "ablations")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    roots = {
        "baseline": args.baseline_root.resolve(),
        "row_only": args.row_only_root.resolve(),
        "combined": args.combined_root.resolve(),
    }
    if args.parent_only_root is not None:
        roots["parent_only"] = args.parent_only_root.resolve()
    for name, root in roots.items():
        if not (root / "freespace_sim").is_dir():
            parser.error(f"{name} source root is incomplete: {root}")
    pricing_captures = []
    row_captures = []
    for seed in (0, 1):
        capture_root = args.full_matrix / f"baseline_seed{seed}" / "pricing_captures"
        pricing_captures.append((seed, capture_root / "round_0006.pkl"))
        row_captures.extend(
            (seed, capture_root / name) for name in ("round_0001.pkl", "round_0011.pkl")
        )
    missing = [path for _, path in pricing_captures + row_captures if not path.is_file()]
    if missing and args.execute:
        parser.error(f"Missing completed baseline captures: {missing}")
    planned = {
        "status": "ready" if not args.execute else "running",
        "roots": {name: str(root) for name, root in roots.items()},
        "pricing_captures": [
            {"seed": seed, "path": str(path.resolve()), "available": path.is_file()}
            for seed, path in pricing_captures
        ],
        "row_captures": [
            {"seed": seed, "path": str(path.resolve()), "available": path.is_file()}
            for seed, path in row_captures
        ],
        "pricing_arms": ["baseline", "combined"],
        "row_arms": ["baseline", "row_only"],
        "row_repeats": 3,
        "scope": (
            "Pricing sweeps exercise worker pricing plus parent result handling but no master "
            "LP/IP. Row scans use one real captured incumbent column per flight; they do not "
            "reconstruct the full multi-column restricted master."
        ),
        "harness_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("benchmark_captured_sweep.py", "benchmark_row_scan.py")
        },
    }
    _write(HERE / "ablation_plan.json", planned)
    if not args.execute:
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0
    if args.out.exists() and not args.resume:
        raise FileExistsError("Choose a fresh ablation output directory")
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

    def run(command: list[str], label: str, cwd: Path) -> Path:
        """Run one fresh subprocess, record it, and stop the matrix on failure."""
        output = args.out / label
        log_path = args.out / f"{label}.log"
        status_path = output / "status.json"
        if args.resume and status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("state") == "complete":
                progress.append(
                    {"label": label, "returncode": 0, "reused": True, "output": str(output)}
                )
                _write(args.out / "progress.json", progress)
                return output
        if output.exists():
            raise RuntimeError(f"Cannot resume incomplete ablation: {label}")
        with log_path.open("w") as log:
            completed = subprocess.run(
                command + ["--out", str(output)],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        progress.append(
            {"label": label, "returncode": completed.returncode, "log": str(log_path.resolve())}
        )
        _write(args.out / "progress.json", progress)
        if completed.returncode:
            raise RuntimeError(f"Ablation failed: {label}")
        return output

    pricing_references = {}
    pricing_arms = ["baseline", "combined"]
    for seed, capture in pricing_captures:
        round_name = capture.stem
        for arm in pricing_arms:
            label = f"pricing_seed{seed}_{round_name}_{arm}"
            command = [
                sys.executable,
                str(HERE / "benchmark_captured_sweep.py"),
                "--source-root",
                str(roots[arm]),
                "--capture",
                str(capture),
                "--budget",
                "1800",
            ]
            key = (seed, round_name)
            if arm != "baseline":
                command.extend(["--reference", str(pricing_references[key])])
            output = run(command, label, roots[arm])
            if arm == "baseline":
                pricing_references[key] = output

    for seed, capture in row_captures:
        round_name = capture.stem
        reference = run(
            [
                sys.executable,
                str(HERE / "benchmark_row_scan.py"),
                "--source-root",
                str(roots["baseline"]),
                "--capture",
                str(capture),
                "--repeats",
                "3",
            ],
            f"rows_seed{seed}_{round_name}_baseline",
            roots["baseline"],
        )
        run(
            [
                sys.executable,
                str(HERE / "benchmark_row_scan.py"),
                "--source-root",
                str(roots["row_only"]),
                "--capture",
                str(capture),
                "--reference",
                str(reference),
                "--repeats",
                "3",
            ],
            f"rows_seed{seed}_{round_name}_row_only",
            roots["row_only"],
        )
    planned.update(status="complete", progress=str((args.out / "progress.json").resolve()))
    _write(HERE / "ablation_plan.json", planned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
