"""Audit paired LNS simulations and plot wall time, heuristic time, and quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for directory in args.directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["baseline_source_sha256"] == manifest["fixed_source_sha256"]
        grouped = {}
        for row in json.loads((directory / "results.json").read_text()):
            assert "error" not in row, row
            assert row["verified"] is True, row
            assert abs(row["congestion_cost_minus_solver_objective"]) < 1e-5
            grouped.setdefault((row["scenario"], row["seed"], row["n_requests"]), {})[
                row["version"]] = row
        for (scenario, seed, flights), arms in grouped.items():
            base, lns = arms["baseline"], arms["fixed"]
            assert base["input_sha"] == lns["input_sha"]
            name = f"{'FAA' if 'faa' in scenario else 'Future'} {flights}, seed {seed}"
            trace = {}
            for arm in ("baseline", "fixed"):
                trace[arm] = json.loads((directory / f"{scenario}_seed{seed}_{arm}" /
                                        "iterations.json").read_text())
            reference = base["penalized_congestion_cost"]
            time_to_reference = {}
            certified_times = {}
            for tolerance in (.001, .01, .05):
                certified_times[tolerance] = {}
                for arm, row in arms.items():
                    times = [it["elapsed_s"] for it in trace[arm]
                             if it["heuristic_gap_cost"] <= tolerance]
                    if row["global_cost_gap"] <= tolerance:
                        times.append(row["simulation_wall_s"])
                    certified_times[tolerance][arm] = min(times) if times else None
            for tolerance in (.01, .05):
                time_to_reference[tolerance] = {}
                for arm, row in arms.items():
                    # An offline benchmark target, not an online optimality certificate.
                    times = [it["elapsed_s"] for it in trace[arm]
                             if it["heuristic_cost"] <= (1 + tolerance) * reference]
                    if row["penalized_congestion_cost"] <= (1 + tolerance) * reference:
                        times.append(row["simulation_wall_s"])
                    time_to_reference[tolerance][arm] = min(times) if times else None
            pairs.append({
                "name": name, "directory": str(directory.resolve()),
                "baseline": base, "lns": lns,
                "wall_speedup": base["simulation_wall_s"] / lns["simulation_wall_s"],
                "wall_reduction_pct": 100 * (1 - lns["simulation_wall_s"] /
                                             base["simulation_wall_s"]),
                "cost_change_pct": 100 * (lns["penalized_congestion_cost"] / reference - 1),
                "time_to_baseline_final_cost_tolerances": time_to_reference,
                "time_to_certified_gap_s": certified_times,
                "trace": trace,
            })
    (args.out / "paired_results.json").write_text(json.dumps(pairs, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.8, len(pairs) * .8 + 2)))
    y = np.arange(len(pairs))
    for ax, field, title in zip(axes, ["simulation_wall_s", "heuristic_stage_s",
                                     "penalized_congestion_cost"],
                               ["Complete simulation (s)", "Primal heuristic work (s)",
                                "Final penalized congestion cost"]):
        for offset, arm, label, color in [(-.17, "baseline", "Rounding", "#738297"),
                                           (.17, "lns", "LNS", "#087F8C")]:
            values = [pair[arm][field] for pair in pairs]
            bars = ax.barh(y + offset, values, height=.30, label=label, color=color)
            ax.bar_label(bars, labels=[f"{v:,.2f}" if field != "penalized_congestion_cost"
                                      else f"{v:,.1f}" for v in values],
                         padding=4, fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, pad=16)
        ax.set_yticks(y, [p["name"] for p in pairs] if ax == axes[0] else [])
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.27)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.13), frameon=False, ncol=2)
    fig.suptitle("Column generation: rounding versus LNS", x=.03, ha="left",
                 fontsize=19, weight="bold")
    fig.text(.03, .915, "Gurobi · 4 pricing workers · same requests and tolerances · warmed kernels",
             fontsize=11, color="#52677B")
    fig.tight_layout(rect=(0, .04, 1, .88), w_pad=3)
    fig.savefig(args.out / "benchmark.png", dpi=160, facecolor="white")
    largest = max(pairs, key=lambda p: p["baseline"]["n_requests"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm, trace_arm, label, color in [("baseline", "baseline", "Rounding", "#738297"),
                                        ("lns", "fixed", "LNS", "#087F8C")]:
        trace = largest["trace"][trace_arm]
        final = largest[arm]
        times = [r["elapsed_s"] for r in trace] + [final["simulation_wall_s"]]
        costs = [r["heuristic_cost"] for r in trace] + [final["penalized_congestion_cost"]]
        gaps = [r["heuristic_gap_cost"] * 100 for r in trace] + [final["global_cost_gap"] * 100]
        axes[0].step(np.array(times) / 60, costs, where="post", color=color, label=label, lw=2)
        axes[0].scatter(np.array(times) / 60, costs, color=color, s=22)
        axes[1].step(np.array(times) / 60, np.maximum(gaps, .001), where="post",
                     color=color, label=label, lw=2)
        axes[1].scatter(np.array(times) / 60, np.maximum(gaps, .001), color=color, s=22)
    axes[1].axhline(5, color="#BA6D16", linestyle="--", lw=1.2, label="5% certified gap")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Feasible schedule cost (lower is better)")
    axes[1].set_ylabel("Certified global cost gap (%)")
    for ax in axes:
        ax.set_xlabel("Elapsed minutes")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle(f"{largest['name']}: when a useful schedule becomes available",
                 fontsize=16, weight="bold")
    fig.text(.5, .025, "Dots are completed pricing iterations; last dot includes the final integer solve. "
             "Incumbent timestamps are conservative.", ha="center", fontsize=9, color="#52677B")
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(args.out / "anytime.png", dpi=160, facecolor="white")
    lines = ["# Paired column-generation LNS benchmark", "",
             "Actual `sim.run` simulations with final conflict verification. Demand generation "
             "and kernel warmup are excluded; measured runs never overlap. Both arms use "
             "Gurobi, four Gurobi threads, four pricing workers, the same ladder and objective, "
             "and identical inputs. Only `lns_destroy_flights` differs.", "",
             "| Case | Rounding wall | LNS wall | Wall reduction | Rounding/LNS cost | "
             "Rounding/LNS global gap | Iterations |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        base, lns = p["baseline"], p["lns"]
        lines.append(f"| {p['name']} | {base['simulation_wall_s']:.2f}s | "
                     f"{lns['simulation_wall_s']:.2f}s | {p['wall_reduction_pct']:+.2f}% | "
                     f"{base['penalized_congestion_cost']:.2f} / {lns['penalized_congestion_cost']:.2f} | "
                     f"{100*base['global_cost_gap']:.3f}% / {100*lns['global_cost_gap']:.3f}% | "
                     f"{base['iterations']} / {lns['iterations']} |")
    lines += ["", "A positive wall reduction means LNS was faster. Heuristic-stage savings "
              "are not whole-simulation speedups. Fixed iteration/time limits do not prove "
              "convergence; the table reports the resulting global gaps.", "",
              "| Case | Rounding heuristic | LNS heuristic | Stage speedup | "
              "Time to certified 5% gap, rounding/LNS |",
              "|---|---:|---:|---:|---:|"]
    for p in pairs:
        base, lns = p["baseline"], p["lns"]
        times = p["time_to_certified_gap_s"][.05]
        formatted = ["not reached" if times[arm] is None else f"{times[arm]:.2f}s"
                     for arm in ("baseline", "fixed")]
        lines.append(f"| {p['name']} | {base['heuristic_stage_s']:.3f}s | "
                     f"{lns['heuristic_stage_s']:.3f}s | "
                     f"{base['heuristic_stage_s']/lns['heuristic_stage_s']:.2f}× | "
                     f"{' / '.join(formatted)} |")
    lines += ["", "Heuristic time decreased in these pairs. The whole-run difference also "
              "includes pricing-time changes; the incumbent affects processing order and "
              "known-column hints, and timing also varies between runs. Do not interpret the "
              "largest observed wall reduction as a guaranteed acceleration of exact pricing.", "",
              "The JSON also reports the first recorded time within 1% and 5% of the baseline's final "
              "cost. This is an offline time-to-quality comparison, not a certified stopping "
              "rule. Iteration timestamps are conservative: the incumbent was available "
              "before that iteration's pricing sweep completed.", "",
              "`time_to_certified_gap_s` instead uses the actual global pricing certificate "
              "at 0.1%, 1%, and 5%. Null means that tolerance was not reached. These are "
              "observed availability times in the strict-tolerance run, not timings from "
              "separate runs configured with looser stopping rules.", "",
              "Each input directory retains the exact harness, source hashes, parameter/input "
              "fingerprints, iteration traces, full planner statistics, and compressed trajectories."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
