"""Plan or run one V2 25-second iteration-IP policy comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from run_full_matrix import HERE, _command, _dump, _source_hashes


def _change(control: float, candidate: float) -> dict[str, float | None]:
    """Report one absolute and percentage policy difference."""
    return {
        "control": control,
        "candidate": candidate,
        "delta": candidate - control,
        "percent": None if control == 0 else 100.0 * (candidate / control - 1.0),
    }


def main() -> int:
    """Run one seed and compare it with its completed V2 30-second control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--matrix", type=Path, default=HERE / "full_matrix")
    parser.add_argument("--out-root", type=Path, default=HERE / "policy25")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = args.combined_root.resolve()
    control = args.matrix / f"combined_seed{args.seed}"
    output = args.out_root / f"combined_seed{args.seed}"
    command = _command(source, output.resolve(), args.seed)
    budget_index = command.index("--iteration-ip-budget") + 1
    command[budget_index] = "25"
    plan = {
        "status": "ready" if not args.execute else "running",
        "seed": args.seed,
        "source_root": str(source),
        "source_sha256": _source_hashes(source),
        "control": str(control.resolve()),
        "output": str(output.resolve()),
        "command": command,
        "policy_change": {"iteration_ip_time_limit_s": {"control": 30.0, "candidate": 25.0}},
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    _dump(args.out_root / f"seed{args.seed}_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise FileExistsError("Choose a fresh policy output directory")
    env = dict(os.environ)
    env.update(
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        COLGEN_GUROBI_THREADS="4",
    )
    log_path = args.out_root / f"combined_seed{args.seed}_driver.log"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        plan.update(status="failed", returncode=completed.returncode)
        _dump(args.out_root / f"seed{args.seed}_plan.json", plan)
        return completed.returncode

    control_result = json.loads((control / "result.json").read_text())
    candidate_result = json.loads((output / "result.json").read_text())
    control_planner = json.loads((control / "planner_stats.json").read_text())
    candidate_planner = json.loads((output / "planner_stats.json").read_text())
    control_iterations = json.loads((control / "iterations.json").read_text())
    candidate_iterations = json.loads((output / "iterations.json").read_text())
    control_inputs = json.loads((control / "inputs.json").read_text())
    candidate_inputs = json.loads((output / "inputs.json").read_text())
    control_params = dict(control_inputs["params"])
    candidate_params = dict(candidate_inputs["params"])
    control_budget = control_params.pop("iteration_ip_time_limit_s")
    candidate_budget = candidate_params.pop("iteration_ip_time_limit_s")
    input_keys = ("requests_sha", "generated_requests_sha", "demand_config_sha", "input_sha")
    comparison = {
        "input_hashes_match": {
            key: control_result.get(key) is not None
            and control_result.get(key) == candidate_result.get(key)
            for key in input_keys
        },
        "params_match_except_iteration_ip_time_limit_s": control_params == candidate_params,
        "iteration_ip_time_limit_s": {
            "control": control_budget,
            "candidate": candidate_budget,
        },
        "verified": candidate_result.get("verified") is True,
        "accepted_all": (
            isinstance(candidate_result.get("n_requests"), int)
            and candidate_result["n_requests"] > 0
            and candidate_result.get("accepted") == candidate_result["n_requests"]
        ),
        "quality": {
            key: {"control": control_result.get(key), "candidate": candidate_result.get(key)}
            for key in (
                "congestion_cost",
                "solver_objective",
                "global_cost_gap",
                "iterations",
                "columns",
                "simulation_wall_s",
                "iteration_ip_wall_s",
                "ip_wall_s",
            )
        },
        "deltas": {
            key: _change(float(control_result[key]), float(candidate_result[key]))
            for key in (
                "congestion_cost",
                "simulation_wall_s",
                "iteration_ip_wall_s",
                "ip_wall_s",
            )
        },
        "final_lp_gap_cost": {
            "control": control_iterations[-1].get("lp_gap_cost"),
            "candidate": candidate_iterations[-1].get("lp_gap_cost"),
        },
        "termination": {
            "control": control_result.get("termination"),
            "candidate": candidate_result.get("termination"),
        },
        "restricted_ip": {
            arm: {
                "status": planner.get("ip_status"),
                "optimal": planner.get("ip_optimal"),
                "gap": planner.get("restricted_ip_gap"),
            }
            for arm, planner in (
                ("control", control_planner),
                ("candidate", candidate_planner),
            )
        },
        "diagnostics": {
            arm: {
                key: planner.get(key)
                for key in (
                    "pricing_certified_columns",
                    "pricing_revalidated_columns",
                    "kernel_fell_back",
                    "kernel_label_restarts",
                    "pricing_worker_lost",
                    "repair_added",
                )
            }
            for arm, planner in (
                ("control", control_planner),
                ("candidate", candidate_planner),
            )
        },
        "canonical_iteration_ip_s": {
            arm: sum(
                float(row.get("stage_s", {}).get("canonical_iteration_ip", 0.0))
                for row in iterations
            )
            for arm, iterations in (
                ("control", control_iterations),
                ("candidate", candidate_iterations),
            )
        },
    }
    candidate_diagnostics = comparison["diagnostics"]["candidate"]
    comparison["candidate_safety"] = {
        "final_lp_gap_reached": (
            isinstance(comparison["final_lp_gap_cost"]["candidate"], (int, float))
            and comparison["final_lp_gap_cost"]["candidate"] <= 0.001
        ),
        "termination_lp_gap": candidate_result.get("termination") == "lp_gap",
        "restricted_ip_optimal": (
            candidate_planner.get("ip_status") == "optimal"
            and candidate_planner.get("ip_optimal") is True
        ),
        "all_priced_columns_certified": (
            isinstance(candidate_diagnostics["pricing_certified_columns"], int)
            and candidate_diagnostics["pricing_certified_columns"] > 0
            and candidate_diagnostics["pricing_revalidated_columns"] == 0
        ),
        "no_fallback_restart_workerloss_repair": all(
            candidate_diagnostics[key] == 0
            for key in (
                "kernel_fell_back",
                "kernel_label_restarts",
                "pricing_worker_lost",
                "repair_added",
            )
        ),
    }
    comparison["passed"] = (
        all(comparison["input_hashes_match"].values())
        and comparison["params_match_except_iteration_ip_time_limit_s"]
        and comparison["iteration_ip_time_limit_s"] == {"control": 30.0, "candidate": 25.0}
        and comparison["verified"]
        and comparison["accepted_all"]
        and all(comparison["candidate_safety"].values())
    )
    _dump(args.out_root / f"seed{args.seed}_comparison.json", comparison)
    plan.update(status="complete", comparison=str((args.out_root / f"seed{args.seed}_comparison.json").resolve()))
    _dump(args.out_root / f"seed{args.seed}_plan.json", plan)
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
