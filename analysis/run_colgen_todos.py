"""Prepare or run the seven isolated, sequential 1200-second outbound experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from benchmark_colgen_lns import digest, dump


ARMS = {
    "baseline": ("baseline", None, 3, 1),
    "corridor": ("corridor", None, 3, 1),
    "shifts": ("shifts", None, 3, 1),
    "bootstrap": ("bootstrap", "astar", 3, 1),
    "combined": ("combined", "astar", 3, 1),
    "wider6": ("combined", "astar", 6, 1),
    "multi3": ("combined", "astar", 3, 3),
}


def source_hashes(root):
    """Fingerprint executable Python and verify it against the frozen manifest."""
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "freespace_sim").rglob("*.py"))}
    if not hashes:
        raise ValueError(f"No source files in {root}")
    frozen = json.loads((root / "source-manifest.json").read_text())
    expected = {key for key in frozen if key.startswith("freespace_sim/") and key.endswith(".py")}
    if set(hashes) != expected:
        raise ValueError(f"Frozen source file set differs: {root}")
    for relative, sha in hashes.items():
        if frozen.get(relative) != sha:
            raise ValueError(f"Frozen source mismatch: {root / relative}")
    return hashes


def prepare(args):
    """Build reproducible commands and source identities without running simulations."""
    harness = Path(__file__).with_name("benchmark_colgen_lns.py").resolve()
    arms = []
    for name in args.arms:
        variant, bootstrap, overrun, columns = ARMS[name]
        root = (Path(args.snapshots_dir) / f"colgen-todos-{variant}-20260910").resolve()
        hashes = source_hashes(root)
        command = [sys.executable, str(harness), "run", "--baseline", str(root),
                   "--fixed", str(root), "--versions", "iteration_ip_eager",
                   "--scenarios", "density_faa_wing_zipline", "--seeds", "0",
                   "--density-flights", "0", "--demand-seconds", "1200",
                   "--outbound-only", "--workers", "12", "--gurobi-threads", "4",
                   "--iterations", "30", "--budget", "14460", "--ip-budget", "600",
                   "--iteration-ip-budget", "30", "--ip-reserve", "660",
                   "--lp-gap", "0.001", "--gap", "0", "--capture-columns", "--overrun", str(overrun),
                   "--columns-per-flight", str(columns)]
        if bootstrap is not None:
            command.extend(["--bootstrap-method", bootstrap])
        arms.append(dict(name=name, root=str(root), source_sha256=hashes,
                         source_digest=digest(hashes), command=command))
    return dict(schema_version=1, arms=arms, expected_requests=1537,
                demand="FAA paired stream, seed0,1200s, filtered outbound preserving IDs/clocks",
                timing="Sequential fresh processes; per-arm solver budget14460s; warmup excluded",
                harness_sha256=hashlib.sha256(harness.read_bytes()).hexdigest(),
                tracer_sha256=hashlib.sha256(harness.with_name("colgen_trace.py").read_bytes()).hexdigest(),
                orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    """Persist the plan, then optionally execute arms serially with resumable status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshots-dir", default="/private/tmp")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = parser.parse_args()
    if len(args.arms) != len(set(args.arms)):
        parser.error("duplicate arms")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = prepare(args)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != plan:
        raise ValueError("Existing manifest differs; use a new output directory")
    dump(manifest_path, plan)
    status_path = args.out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if args.mode == "prepare":
        print(f"Prepared {len(plan['arms'])} arms: {manifest_path}", flush=True)
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONHASHSEED="0", COLGEN_GUROBI_THREADS="4")
    demand_sha = None
    for arm in plan["arms"]:
        name = arm["name"]
        previous = status.get(name, {})
        if previous.get("state") == "complete":
            if demand_sha is not None and previous["demand_config_sha"] != demand_sha:
                raise ValueError("Previously completed arms used different demand/config")
            demand_sha = previous["demand_config_sha"]
            continue
        if source_hashes(Path(arm["root"])) != arm["source_sha256"]:
            raise ValueError(f"Source changed before execution: {name}")
        attempt = previous.get("attempt", 0) + 1
        arm_out = args.out / name / f"attempt_{attempt:02d}"
        arm_out.mkdir(parents=True)
        command = arm["command"] + ["--out", str(arm_out)]
        status[name] = dict(state="running", attempt=attempt, output=str(arm_out),
                            started=datetime.now(timezone.utc).isoformat(), command=command)
        dump(status_path, status)
        with (args.out / "progress.log").open("a") as progress:
            progress.write(f"{status[name]['started']} START {name} attempt{attempt}\n")
        print(f"START {name}: {arm_out}", flush=True)
        try:
            with (arm_out / "orchestrator.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    returncode = process.wait(timeout=15000)
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, command)
                except BaseException:
                    # Cancel descendants too: the harness has its own simulation child/workers.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            rows = json.loads((arm_out / "results.json").read_text())
            if len(rows) != 1 or rows[0].get("error"):
                raise ValueError(f"Arm failed: {rows}")
            row = rows[0]
            if row["n_requests"] != 1537 or not row["verified"]:
                raise ValueError("Demand count or conflict verification failed")
            if demand_sha is not None and row["demand_config_sha"] != demand_sha:
                raise ValueError("Arms used different demand/config")
            demand_sha = row["demand_config_sha"]
            status[name].update(state="complete", demand_config_sha=demand_sha, result=row)
        except BaseException as exc:
            status[name].update(state="interrupted_or_failed", error=repr(exc))
            raise
        finally:
            status[name]["updated"] = datetime.now(timezone.utc).isoformat()
            dump(status_path, status)
            with (args.out / "progress.log").open("a") as progress:
                progress.write(f"{status[name]['updated']} {status[name]['state']} {name}\n")
    print(f"Completed {len(plan['arms'])} arms: {status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
