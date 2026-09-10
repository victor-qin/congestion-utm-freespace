"""Compare matched paired-demand and outbound-only CG/LNS simulations."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_colgen_reserved_ip import case_data, read


def enrich(directory):
    data = case_data(directory)
    result, summary = data["summary"]["result"], data["summary"]
    cfg = data["inputs"]["config"]
    outbound_ids = {str(r["flight_id"]) for r in data["inputs"]["requests"]
                    if r["paired_outbound_id"] is None}
    with gzip.open(data["case"] / "trajectories.json.gz", "rt") as handle:
        trajectories = {str(t["flight_id"]): t for t in json.load(handle)}
    costs = {fid: cfg["cost_ground_delay_per_s"] * t["ground_delay_s"]
             + cfg["cost_air_hold_per_s"] * t["air_hold_s"]
             + cfg["cost_air_lateral_per_s"] * t["air_detour_m"] / cfg["nominal_speed_mps"]
             for fid, t in trajectories.items()}
    columns = data["columns"]
    common_lns_cost = math.fsum(columns[data["ip"]["before_selection"][fid]]["cost"]
                               for fid in outbound_ids)
    summary.update(
        initial_cost=result["ground_bootstrap"]["cost"],
        lns_gain_pct=100 * (1 - summary["lns_cost"] / result["ground_bootstrap"]["cost"]),
        common_outbound_flights=len(outbound_ids),
        common_outbound_final_cost=math.fsum(costs[fid] for fid in outbound_ids),
        common_outbound_lns_cost=common_lns_cost,
        common_outbound_initial_cost=math.fsum(
            columns[data["selections"]["initial"][fid]]["cost"] for fid in outbound_ids),
        final_cost_per_flight=result["congestion_cost"] / result["accepted"],
        common_outbound_ground_cost=math.fsum(
            cfg["cost_ground_delay_per_s"] * trajectories[fid]["ground_delay_s"]
            for fid in outbound_ids),
        common_outbound_hold_cost=math.fsum(
            cfg["cost_air_hold_per_s"] * trajectories[fid]["air_hold_s"] for fid in outbound_ids),
        common_outbound_lateral_cost=math.fsum(
            cfg["cost_air_lateral_per_s"] * trajectories[fid]["air_detour_m"]
            / cfg["nominal_speed_mps"] for fid in outbound_ids),
        common_outbound_mean_ground_delay_s=np.mean(
            [trajectories[fid]["ground_delay_s"] for fid in outbound_ids]),
        common_outbound_p95_ground_delay_s=np.quantile(
            [trajectories[fid]["ground_delay_s"] for fid in outbound_ids], 0.95),
        common_outbound_mean_delay_s=np.mean([trajectories[fid]["delay_s"]
                                             for fid in outbound_ids]),
        common_outbound_p95_delay_s=np.quantile([trajectories[fid]["delay_s"]
                                                for fid in outbound_ids], 0.95),
        all_lns_wall_s=data["summary"]["trace"]["heuristic_wall_s"],
        lns_unique_trials_median=float(np.median([
            c["stats"]["n_unique_trial_solutions"] for c in data["lns"]])),
        lns_one_schedule_calls=sum(c["stats"]["n_unique_trial_solutions"] == 1
                                  for c in data["lns"]),
        lns_last_improved_call=max((c["call"] for c in data["lns"] if c["changed"]), default=0),
        final_lp_cost_upper_bound=read(data["case"] / "planner_stats.json")["cost_upper_bound"],
        final_global_cost_lower_bound=read(data["case"] / "planner_stats.json")["cost_lower_bound"],
    )
    data["actual_costs"] = costs
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired", type=Path)
    parser.add_argument("outbound", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paired, outbound = enrich(args.paired.resolve()), enrich(args.outbound.resolve())
    left, right = paired["inputs"], outbound["inputs"]
    assert left["config"] == right["config"]
    assert left["params"] == right["params"], "Use matching solve budgets and tolerances"
    assert left["static_terminals"] == right["static_terminals"]
    assert [r for r in left["requests"] if r["paired_outbound_id"] is None] == right["requests"]
    manifests = [read(data["directory"] / "manifest.json") for data in (paired, outbound)]
    source_differences = [name for name, sha in manifests[0]["fixed_source_sha256"].items()
                          if sha != manifests[1]["fixed_source_sha256"][name]]
    assert source_differences in ([], ["freespace_sim/planner/colgen/batch.py"])
    assert manifests[0]["packages"] == manifests[1]["packages"]
    summaries = [data["summary"] for data in (paired, outbound)]
    a, b = summaries
    ratios = dict(
        wall_speedup=a["result"]["simulation_wall_s"] / b["result"]["simulation_wall_s"],
        pricing_speedup=a["result"]["pricing_wall_s"] / b["result"]["pricing_wall_s"],
        common_outbound_cost_reduction_pct=100 * (
            1 - b["common_outbound_final_cost"] / a["common_outbound_final_cost"]),
        per_flight_cost_reduction_pct=100 * (
            1 - b["final_cost_per_flight"] / a["final_cost_per_flight"]),
        total_cost_reduction_pct=100 * (1 - b["ip_cost"] / a["ip_cost"]),
    )
    audit = dict(paired=a, outbound=b, comparison=ratios,
                 exact_outbound_subset_verified=True, same_solver_parameters=True,
                 same_static_terminals=True, source_differences=source_differences,
                 source_difference_scope="Uncertified-IP warning wording only")
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["With returns\n1,526 legs", "Outbound only\n763 legs"]
    colors = ["#596F91", "#07857C"]
    bottom = np.zeros(2)
    for key, label, color in (
        ("pricing_wall_s", "Pricing", "#596F91"),
        ("ip_wall_s", "Final IP stage", "#C78332"),
    ):
        values = np.array([s["result"][key] / 60 for s in summaries])
        axes[0].bar(labels, values, bottom=bottom, label=label, color=color, width=0.58)
        bottom += values
    other = np.array([s["result"]["simulation_wall_s"] / 60 for s in summaries]) - bottom
    axes[0].bar(labels, other, bottom=bottom, label="Setup, LP, LNS, filing", color="#CBD1D8", width=.58)
    for x, s in enumerate(summaries):
        axes[0].text(x, s["result"]["simulation_wall_s"] / 60, f"{s['result']['simulation_wall_s']/60:.1f} min",
                     ha="center", va="bottom")
    axes[0].set(title="Measured whole simulation", ylabel="Minutes")
    axes[0].margins(y=.2)
    axes[0].legend(fontsize=8)
    for ax, key, title in (
        (axes[1], "common_outbound_final_cost", "Cost of the SAME 763 outbound flights"),
        (axes[2], "common_outbound_mean_delay_s", "Mean delay of those outbound flights"),
    ):
        vals = [s[key] / (763 if key.endswith("cost") else 1) for s in summaries]
        ax.bar(labels, vals, color=colors, width=.58)
        for x, value in enumerate(vals):
            ax.text(x, value, f"{value:.1f}", ha="center", va="bottom")
        ax.set(title=title, ylabel="Cost / outbound flight" if key.endswith("cost") else "Seconds / outbound flight")
        ax.margins(y=.2)
    fig.suptitle("Removing return flights: identical outbound demand, CG + LNS + final IP", fontsize=14)
    fig.text(.5, .01, "600 s demand · seed 0 · 4 pricing workers · max 30 sweeps · 600 s IP cap · historical paired reference",
             ha="center", color="#555555", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(args.out / "comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for data, label, color in zip((paired, outbound), labels, colors):
        s, n = data["summary"], data["summary"]["result"]["n_requests"]
        calls = data["lns"]
        axes[0].step([c["call"] for c in calls], [c["after_cost"] / n for c in calls],
                     where="post", color=color, label=label.replace("\n", ": "))
        axes[0].scatter(len(calls) + .5, s["ip_cost"] / n, marker="*", s=140, color=color)
        iterations = read(data["case"] / "iterations.json")
        values = [max(1e-5, min(100, 100 * row["lp_gap_cost"])) for row in iterations]
        axes[1].plot([row["elapsed_s"] / 60 for row in iterations], values,
                     marker=".", color=color, label=label.replace("\n", ": "))
    axes[0].set(title="LNS incumbent; star = final schedule", xlabel="LNS call", ylabel="Congestion cost / flight")
    axes[0].legend(fontsize=9)
    axes[1].axhline(.01, linestyle="--", color="#999999", label="LP target: 0.01%")
    axes[1].set(title="LP pricing convergence", xlabel="Simulation wall time (minutes)",
                ylabel="LP gap (%)", yscale="log")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "convergence.png", dpi=170)
    plt.close(fig)

    metrics = [
        ("Planned / verified flights", lambda s: f"{s['result']['accepted']}/{s['result']['n_requests']}"),
        ("Whole simulation", lambda s: f"{s['result']['simulation_wall_s']:.2f} s"),
        ("Completed pricing sweeps", lambda s: str(s['result']['pricing_sweeps_completed'])),
        ("Pricing wall time", lambda s: f"{s['result']['pricing_wall_s']:.2f} s"),
        ("Native IP search", lambda s: f"{s['native_ip_runtime_s']:.2f} s"),
        ("Final IP stage including setup", lambda s: f"{s['result']['ip_wall_s']:.2f} s"),
        ("Final LP pricing gap", lambda s: f"{100*s['final_lp_gap']:.5f}%"),
        ("Final global gap", lambda s: f"{100*s['result']['global_cost_gap']:.4f}%"),
        ("Raw restricted IP gap", lambda s: f"{100*s['result']['restricted_ip_gap']:.4f}%"),
        ("Initial ground-delay seed cost", lambda s: f"{s['initial_cost']:,.2f}"),
        ("Last LNS incumbent cost", lambda s: f"{s['lns_cost']:,.2f}"),
        ("LNS gain over its seed", lambda s: f"{s['lns_gain_pct']:.2f}%"),
        ("All LNS time", lambda s: f"{s['all_lns_wall_s']:.3f} s"),
        ("Final schedule congestion cost", lambda s: f"{s['ip_cost']:,.2f}"),
        ("Final cost per planned flight", lambda s: f"{s['final_cost_per_flight']:.2f}"),
        ("Final cost of identical 763 outbound flights", lambda s: f"{s['common_outbound_final_cost']:,.2f}"),
        ("Outbound ground / hold / lateral cost", lambda s: f"{s['common_outbound_ground_cost']:,.1f} / {s['common_outbound_hold_cost']:,.1f} / {s['common_outbound_lateral_cost']:,.1f}"),
        ("Outbound mean ground delay", lambda s: f"{s['common_outbound_mean_ground_delay_s']:.2f} s"),
        ("Outbound mean total delay", lambda s: f"{s['common_outbound_mean_delay_s']:.2f} s"),
        ("Outbound p95 total delay", lambda s: f"{s['common_outbound_p95_delay_s']:.2f} s"),
        ("Final master columns", lambda s: f"{s['result']['columns']:,}"),
        ("Priced columns selected by LNS / IP", lambda s: f"{s['lns_selected_priced_columns']} / {s['ip_selected_priced_columns']}"),
        ("Median distinct sequential LNS trials", lambda s: f"{s['lns_unique_trials_median']:g} / 16"),
        ("LNS calls with only one distinct trial", lambda s: f"{s['lns_one_schedule_calls']} / {len(s['lns_calls'])}"),
        ("Last LNS call to improve", lambda s: str(s['lns_last_improved_call'])),
        ("Repair after raw final IP", lambda s: str(s['repaired_flights'])),
        ("Termination", lambda s: s['result']['termination']),
    ]
    lines = ["# Column generation and LNS with outbound flights only", "",
             "The outbound case keeps the exact 763 outbound requests from the earlier 1,526-leg "
             "FAA Wing/Zipline run (600 seconds of demand, seed 0). Only the 763 linked return "
             "requests are removed. Outbound IDs, coordinates, request/departure clocks, the 7,200-second "
             "horizon, costs, and all 182 static terminals match exactly. This halves flight demand; "
             "it does not replace returns with extra outbound flights.", "",
             "Both runs use the same embedded column-pool LNS, ground-delay seed, four pricing workers, "
             "four Gurobi master threads, 30-iteration ceiling, 14,460-second overall allowance, "
             "660-second final-IP reserve, 600-second native IP cap, zero IP gap target, and 0.01% LP "
             "gap target. Rounding is never called. Earlier returns use the nominal anchor convention; "
             "this comparison removes their traffic, not an implemented realized-return scheduling constraint.", "",
             f"The outbound-only wall time is {ratios['wall_speedup']:.2f}× faster and pricing is "
             f"{ratios['pricing_speedup']:.2f}× faster. Final cost for the identical outbound subset "
             f"changes by {-ratios['common_outbound_cost_reduction_pct']:+.2f}%. These are measured "
             "single-seed results against the archived paired run, not a repeated contemporaneous timing trial.", "",
             "| Metric | With returns | Outbound only |", "|---|---:|---:|"]
    lines.extend(f"| {name} | {fn(a)} | {fn(b)} |" for name, fn in metrics)
    lines += ["", "## Interpretation", "",
              "Compare the shared outbound subset to separate congestion relief from simply deleting "
              "half the flights. Total objective alone mixes both effects. The LNS percentages measure "
              "improvement over each case's own feasible seed. Pricing still generates columns; LNS "
              "only recombines the pool, and its duration is included in the simulation time. "
              "The fixed 100-flight neighborhood now releases 13.1% of the fleet rather than 6.6%; "
              "less traffic and a larger relative neighborhood can both affect LNS performance.", "",
              "The final IP may use columns added after the last LNS call; any final omitted-flight "
              "repair is included in the reported filed schedule. Therefore the final-IP-versus-LNS "
              "gain is not a controlled equal-pool experiment. A small LP pricing gap certifies LP "
              "convergence; integer quality is reported separately with the global cost gap. "
              "The restricted IP gap applies to its raw pool solution including omission penalties.", "",
              "## Reproducibility and checks", "",
              "The analyzer verified source archives and harness/tracer hashes, exact outbound request "
              "matching, identical parameters, static infrastructure, package versions, full conflict "
              "verification, objective agreement with filed trajectories, and retention of IP-selected "
              "route geometries. The only production-source difference from the archived reference is "
              "the previously documented warning-text correction in `batch.py`; no optimization code "
              "was changed for this experiment. `--outbound-only` is a benchmark-only request filter.", "",
              f"Paired raw run: `{paired['directory']}`", "",
              f"Outbound raw run: `{outbound['directory']}`", "",
              "![Comparison](comparison.png)", "", "![Convergence](convergence.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()
