"""Audit the full 600-second demand experiment and bootstrap ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


VERSIONS = ("baseline", "fixed", "no_ground_bootstrap")
LABELS = ("Rounding", "LNS + ground seed", "LNS without ground seed")
COLORS = ("#6D7F93", "#087F8C", "#C4772C")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    rows = {r["version"]: r for r in json.loads((directory / "results.json").read_text())}
    assert set(rows) == set(VERSIONS)
    manifest = json.loads((directory / "manifest.json").read_text())
    source_hashes = [manifest[f"{version}_source_sha256"] for version in VERSIONS]
    assert all(hashes == source_hashes[0] for hashes in source_hashes)
    assert len({r["input_sha"] for r in rows.values()}) == 1
    traces, stats = {}, {}
    for version, row in rows.items():
        assert "error" not in row, row
        assert row["verified"] is True
        assert row["n_generated"] == row["n_requests"]
        assert row["demand_duration_s"] == 600
        assert abs(row["congestion_cost_minus_solver_objective"]) < 1e-5
        case = directory / f"{row['scenario']}_seed{row['seed']}_{version}"
        traces[version] = json.loads((case / "iterations.json").read_text())
        stats[version] = json.loads((case / "planner_stats.json").read_text())
    flights = rows["baseline"]["n_requests"]
    baseline, lns, removed = (rows[version] for version in VERSIONS)
    cost_reduction_pct = 100 * (1 - lns["penalized_congestion_cost"] /
                               baseline["penalized_congestion_cost"])
    wall_reduction_pct = 100 * (1 - lns["simulation_wall_s"] / baseline["simulation_wall_s"])
    audit = {"full_demand_flights": flights, "matching_inputs": True,
             "heuristic_timing_scope": "completed pricing sweeps only; the solver callback "
                                       "omits any final incomplete sweep",
             "matching_sources": True, "all_final_schedules_verified": True,
             "seeded_lns_cost_reduction_pct": cost_reduction_pct,
             "seeded_lns_observed_wall_reduction_pct": wall_reduction_pct,
             "seeded_lns_ground_delay_saved_s": baseline["ground_delay_s"] - lns["ground_delay_s"],
             "rows": rows, "traces": traces}
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    y = np.arange(3)
    for ax, field, title, scale in zip(
        axes,
        ("simulation_wall_s", "accepted", "penalized_congestion_cost"),
        ("Complete simulation (minutes)", f"Flights accepted (of {flights:,})",
         "Final penalized congestion cost"),
        (60, 1, 1),
    ):
        values = [rows[v][field] / scale for v in VERSIONS]
        bars = ax.barh(y, values, color=COLORS, height=.55)
        ax.bar_label(bars, labels=[f"{v:,.0f}" if field == "accepted" else f"{v:,.2f}"
                                  for v in values], padding=5, fontsize=10)
        ax.set_yticks(y, LABELS if ax == axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, pad=16)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(values) * 1.35)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"Full 600-second FAA demand: {flights:,} flight legs", x=.02,
                 ha="left", fontsize=18, weight="bold")
    fig.text(.02, .90, "Same flights and source · Gurobi · 4 pricing workers · "
             "15-minute solver budget · final conflict verification", color="#52677B")
    fig.text(.02, .015, "One demand seed; time-limited results do not establish convergence "
             "or a repeatable wall-time speedup.", fontsize=10, color="#52677B")
    fig.tight_layout(rect=(0, .06, 1, .86), w_pad=2)
    fig.savefig(directory / "benchmark.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for version, label, color in zip(VERSIONS, LABELS, COLORS):
        row, trace = rows[version], traces[version]
        times = [t["elapsed_s"] / 60 for t in trace] + [row["simulation_wall_s"] / 60]
        costs = [t["heuristic_cost"] for t in trace] + [row["penalized_congestion_cost"]]
        # Costs are nonnegative. Use the same zero floor on the global cost
        # lower bound as the final benchmark result, including early sweeps
        # whose unnormalized solver telemetry can report a gap above 100%.
        gaps = [max(0, (t["heuristic_cost"] - max(0, t["cost_lower_bound"])) /
                    max(1, abs(t["heuristic_cost"]))) * 100 for t in trace] + [
            row["global_cost_gap"] * 100]
        for ax, values in zip(axes, (costs, gaps)):
            ax.step(times, values, where="post", lw=2, color=color, label=label)
            ax.scatter(times, values, s=27, color=color)
    for ax, title in zip(axes, ("Feasible penalized cost", "Certified global cost gap (%)")):
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Elapsed minutes")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("LNS and initial ground-delay schedule: observed checkpoints", x=.02,
                 ha="left", fontsize=17, weight="bold")
    fig.text(.02, .015, "Points mark completed pricing sweeps and the final filed schedule; "
             "coverage losses are charged the solver's denial penalty.", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .90))
    fig.savefig(directory / "anytime.png", dpi=170, facecolor="white")
    plt.close(fig)

    lines = [
        "# Full 600-second FAA demand and initial ground-delay ablation",
        "",
        f"**Keep the initial ground-delay schedule.** Seeded LNS reduced cost by "
        f"{cost_reduction_pct:.2f}% while serving {lns['accepted']:,}/{flights:,} flights. "
        f"Disabling that initialization left {removed['denied']:,} flights unscheduled "
        "under the same budget, although the seeded runs demonstrate a feasible "
        "schedule for the whole batch. The ground-shift pass cost only "
        f"{baseline['ground_bootstrap']['elapsed_s']:.1f}–"
        f"{lns['ground_bootstrap']['elapsed_s']:.1f} seconds. "
        f"Seeded LNS's observed wall reduction was {wall_reduction_pct:.2f}%, "
        "which does not establish a meaningful runtime speedup in this capped, single-pair test.",
        "",
        f"Generated the entire 600-second demand window, seed 0: {flights:,} flight legs "
        f"({flights // 2:,} deliveries with paired returns), with all 182 static terminals. "
        "No first-N truncation. The simulation horizon remains 7,200 seconds so flights "
        "and ground waits can finish. The demand model shifts the request clock to avoid "
        "negative filing times, so absolute departures need not lie in [0, 600].",
        "",
        "Each arm used the same source and generated input fingerprint, a fresh process, "
        "Gurobi with four master threads, four pricing workers, a 900-second solver limit "
        "and at most four iterations. The final integer solve has a 60-second cap, "
        "but the solver reserves only five seconds when pricing exhausts the overall budget. "
        "Warmup and demand generation were excluded from measured simulation time; "
        "batch filing and final conflict verification were included. The pricing kernel "
        "was warmed; the separate feasibility kernel was not precompiled by that warmup. "
        "Measured simulations ran sequentially. This is one seed and one run per arm.",
        "",
        "The removal arm skips only `_initial_feasible_selection` in the benchmark harness. "
        "It retains nominal route seeds, all 20 departure-ladder shifts, the pricing "
        "bootstrap (`bootstrap_roots=1`), and the legal ground-delay decision variable. "
        "The solver's existing first-LP rounding fallback supplies its initial incumbent. "
        "Production initialization defaults were not changed.",
        "",
        "| Arm | Wall, s | Heuristic in completed sweeps, s | Accepted | Penalized cost | Global gap | "
        "Sweeps completed | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for version, label in zip(VERSIONS, LABELS):
        row = rows[version]
        lines.append(f"| {label} | {row['simulation_wall_s']:.2f} | "
                     f"{row['heuristic_stage_s']:.3f} | {row['accepted']}/{flights} | "
                     f"{row['penalized_congestion_cost']:,.2f} | "
                     f"{row['global_cost_gap'] * 100:.3f}% | "
                     f"{row['pricing_sweeps_completed']} | {row['termination']} |")
    lines.extend(["", "Heuristic timing covers only iterations whose pricing sweep completed. "
                  "The current solver callback omits an interrupted final sweep, even if its "
                  "heuristic already ran. These values must not be read as total heuristic "
                  "time in a capped run. Total simulation and pricing wall times include "
                  "the interrupted sweep.", "", "## Initialization and work distribution", ""])
    for version, label in zip(VERSIONS, LABELS):
        row = rows[version]
        init = row["ground_bootstrap"]
        lines.append(f"- **{label}:** ground-shift pass {init['elapsed_s']:.3f} s, "
                     f"initial coverage {init['covered']}/{flights}, initial selected-route "
                     f"cost {init['cost']:,.2f}. Pricing {row['pricing_wall_s']:.2f} s; "
                     f"final IP {row['ip_wall_s']:.2f} s. "
                     f"Pre-IP incumbent cost {row['heuristic_cost']:,.2f}. "
                     f"Seed-column construction {stats[version]['seed_elapsed_s']:.2f} s.")
    lines.extend([
        "", "| Arm / complete sweep | Incumbent cost | Global cost lower bound, floored at 0 | "
        "Maximum heuristic-trial coverage | Joint-repair swaps |",
        "|---|---:|---:|---:|---:|",
    ])
    for version, label in zip(VERSIONS, LABELS):
        for iteration in traces[version]:
            diagnostic = iteration["round_stats"]
            trial_coverage = max(diagnostic.get("try_covered", ()),
                                 default=diagnostic.get("covered_flights", 0))
            swaps = diagnostic.get("joint_repair", {}).get("n_swapped", 0)
            lines.append(f"| {label} / {iteration['iteration']} | "
                         f"{iteration['heuristic_cost']:,.2f} | "
                         f"{max(0, iteration['cost_lower_bound']):,.2f} | "
                         f"{trial_coverage}/{flights} | {swaps} |")
    lines.extend(["", "Trial coverage describes the heuristic attempts, not the retained "
                  "incumbent. The solver can reject these candidates in favor of its "
                  "complete ground-delay seed. The raw trace's `n_uncovered` describes "
                  "fractional LP coverage and is not used as integer schedule coverage."])
    lines.extend([
        "", "## Interpretation and limits", "",
        "Costs are the column-generation congestion objective: weighted ground delay, "
        "air hold, and lateral detour; mandatory climb/descent is excluded. Each denied "
        "flight carries the configured M penalty. The final trajectory-derived cost was "
        "checked against the reported solver objective in every arm, and all final "
        "schedules passed simulation conflict verification.",
        "",
        "LNS optimizes only flights already present in the incumbent. Its joint repair "
        "selects exactly one pooled column per released incumbent flight; it does not "
        "admit a missing flight. Removing initial coverage therefore delegates coverage "
        "recovery to the final integer solve and subsequent repair. Pool quality, pricing "
        "bounds and final integer-solve success can also change when shifted incumbent "
        "columns are absent, so this ablation is not merely a subtraction of startup time.",
        "",
        "The return clock uses the existing column-generation nominal-anchor convention; "
        "this experiment does not add realized-return coupling. A global gap above 0.1% "
        "does not meet the requested integer tolerance. The plots show observed "
        "checkpoints, not separate runs configured to stop at those quality levels.",
        "",
        "All three runs finished two full pricing sweeps and entered a third iteration. "
        "The seeded LNS run's pre-IP incumbent already equals its final cost, so its "
        "improvement comes from LNS rather than the final integer solve. That includes "
        "additional heuristic improvement in the last, incomplete sweep, absent from "
        "the completed-sweep timing table. The normalized global gap is 100% for all "
        "three runs: the valid cost lower bound is only zero, so none provides a "
        "useful optimality certificate. No compiled-pricing fallback occurred; the "
        "removal arm did restart its label pool twice on one difficult flight.",
        "", "## Reproduction", "", "```sh",
        f"PYTHONPATH=. {sys.executable} analysis/benchmark_colgen_lns.py run \\",
        "  --baseline . --fixed . --scenarios density_faa_wing_zipline --seeds 0 \\",
        "  --density-flights 0 --demand-seconds 600 \\",
        "  --versions baseline fixed no_ground_bootstrap --iterations 4 \\",
        "  --budget 900 --ip-budget 60 --out analysis/colgen_lns_600s_repeat",
        "```", "",
        "The manifest, archived harness and source snapshot identify the exact measured "
        "implementation. Each arm retains complete inputs, iteration telemetry, planner "
        "statistics and compressed trajectories.", "",
    ])
    (directory / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
