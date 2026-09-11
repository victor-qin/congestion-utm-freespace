"""Describe recorded iteration-IP incumbents at fixed runtime prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Convert recorded maximize objectives to exact penalized-cost prefixes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[10.0, 25.0, 30.0])
    parser.add_argument("--cost-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    inputs = json.loads((args.case / "inputs.json").read_text())
    calls = json.loads((args.case / "ip_calls.json").read_text())
    benefit = len(inputs["requests"]) * float(inputs["params"]["M"])
    rows = []
    for call in calls:
        if call["phase"] != "iteration":
            continue
        trajectory = call.get("incumbent_trajectory", ())
        prefixes = {}
        for cutoff in args.cutoffs:
            available = [entry for entry in trajectory if float(entry[0]) <= cutoff]
            best = max(available, key=lambda entry: float(entry[1])) if available else None
            if best is None:
                prefixes[str(cutoff)] = None
                continue
            cost = benefit - float(best[1])
            improvement = float(call["before_penalized_cost"]) - cost
            cost_above_end = cost - float(call["after_penalized_cost"])
            prefixes[str(cutoff)] = {
                "runtime_s": float(best[0]),
                "penalized_cost": cost,
                "improvement_from_before": improvement,
                "meaningful_improvement": improvement > args.cost_tolerance,
                "cost_above_recorded_end": cost_above_end,
                "matches_recorded_end": abs(cost_above_end) <= args.cost_tolerance,
            }
        rows.append(
            {
                "iteration": call["sweep"],
                "before_penalized_cost": call["before_penalized_cost"],
                "recorded_end_penalized_cost": call["after_penalized_cost"],
                "prefixes": prefixes,
            }
        )
    output = {
        "case": str(args.case.resolve()),
        "n_requests": len(inputs["requests"]),
        "M": float(inputs["params"]["M"]),
        "total_benefit": benefit,
        "cutoffs_s": args.cutoffs,
        "cost_tolerance": args.cost_tolerance,
        "rows": rows,
        "scope": (
            "Read-only descriptions of incumbents actually recorded by the 30-second solve. "
            "They are not counterfactual simulations: a shorter cap can change later LPs, "
            "pricing, warm starts, and Gurobi search."
        ),
    }
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
