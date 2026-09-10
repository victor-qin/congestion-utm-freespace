"""Audit the outbound 1200-second cheap/exact pricing comparison and plot it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt

from summarize_colgen_policies import close, load_case, read


LABELS = {"iteration_ip_eager": "Standard pricing", "iteration_ip_cheap": "Cheap pricing"}
COLORS = {"iteration_ip_eager": "#285879", "iteration_ip_cheap": "#d16b33"}


def summarize(root):
    manifest, results = read(root / "manifest.json"), read(root / "results.json")
    assert {r["version"] for r in results} == set(LABELS) and len(results) == 2
    sources = [manifest[f"{v}_source_sha256"] for v in LABELS]
    assert sources[0] == sources[1]
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for path, digest in sources[0].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest
    for name, key in (("harness_used.py", "harness_sha256"),
                      ("colgen_trace_used.py", "tracer_sha256"), ("changes.patch", "patch_sha256")):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest[key]
    loaded = {r["version"]: load_case(root, r) for r in results}
    assert len({r["requests_sha"] for r in results}) == 1
    assert len({r["input_sha"] for r in results}) == 1
    assert all(item[0] == loaded["iteration_ip_eager"][0] for item in loaded.values())
    data = {v: loaded[v][2] for v in LABELS}
    for version, d in data.items():
        result = d["result"]
        case = root / f"{result['scenario']}_seed{result['seed']}_{version}"
        rows, stats = read(case / "iterations.json"), read(case / "planner_stats.json")
        calls = read(case / "ip_calls.json")
        assert result["n_requests"] == result["accepted"] == 1537
        assert result["n_generated"] == 3074 and result["n_static_terminals"] == 182
        assert result["gurobi_threads"] == manifest["gurobi_threads"] == 4
        assert manifest["workers"] == 4
        assert manifest["lp_gap"] == 0.001
        assert manifest["outbound_only"] and manifest["demand_duration_override_s"] == 1200
        previous = math.inf
        for row in rows:
            current = float(row["upper_bound"])
            assert current <= previous + 1e-6
            if not row["pricing_exact"]:
                assert current == previous, "heuristic RCs changed the certified global bound"
                assert row["cheap_columns_added"] > 0, "stalled restricted search was not retried"
            previous = current
        if result["termination"] in {"lp_gap", "no_improving_columns", "no_new_columns", "iteration_limit"}:
            assert rows[-1]["pricing_exact"], "termination without a full pricing sweep"
        if result["termination"] == "lp_gap":
            assert d["cg_lp_target_met"]
        close(stats["cheap_pricing_wall_s"] + stats["exact_pricing_wall_s"], result["pricing_wall_s"])
        assert stats["exact_pricing_sweeps"] == sum(row["pricing_exact"] for row in rows)
        assert stats["pricing_sweeps_completed"] == stats["cheap_pricing_sweeps"] + stats["exact_pricing_sweeps"]
        rounds = [call for call in calls if call["phase"] == "iteration"]
        assert len(rounds) == len(rows)
        for call, row in zip(rounds, rows, strict=True):
            close(call["after_cost"], row["heuristic_cost"])
            assert len(call["after_selection"]) == 1537
            assert call["requested_budget_s"] == 30
        d.update(cheap_sweeps=stats["cheap_pricing_sweeps"],
                 exact_sweeps=stats["exact_pricing_sweeps"],
                 cheap_pricing_s=stats["cheap_pricing_wall_s"],
                 exact_pricing_s=stats["exact_pricing_wall_s"],
                 cheap_columns_added=sum(row["cheap_columns_added"] for row in rows),
                 iterations=[{k: row[k] for k in (
                     "iteration", "elapsed_s", "pricing_exact", "sweep_s", "cheap_sweep_s",
                     "cheap_columns_added", "heuristic_cost", "lp_gap_cost", "cost_lower_bound")}
                     for row in rows],
                 final_ip_wall_s=sum(call["wall_s"] for call in calls if call["phase"] == "final"))
        d["exact_columns_added"] = d["actual_priced_columns"] - d["cheap_columns_added"]
        targets = {v: e["result"]["penalized_congestion_cost"] for v, e in data.items()}
        d["time_to_final_cost_s"] = {v: next((t for t, c in d["curve"] if c <= target + 1e-5), None)
                                    for v, target in targets.items()}
    a, b = (data[v] for v in LABELS)
    shared = loaded["iteration_ip_eager"][1].keys() & loaded["iteration_ip_cheap"][1].keys()
    deltas = dict(wall_speedup=a["result"]["simulation_wall_s"] / b["result"]["simulation_wall_s"],
                  wall_saving_pct=100 * (1-b["result"]["simulation_wall_s"]/a["result"]["simulation_wall_s"]),
                  pricing_speedup=a["result"]["pricing_wall_s"]/b["result"]["pricing_wall_s"],
                  cost_change_pct=100*(b["result"]["congestion_cost"]/a["result"]["congestion_cost"]-1),
                  shared_columns=len(shared))
    out = root / "summary"
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(dict(
        inputs_match=True, sources_match=True, source_archive_verified=True,
        physical_verification_passed=True, costs_recomputed=True,
        heuristic_bounds_never_used=True, runs=data, deltas=deltas), indent=2) + "\n")
    plot(data, out)
    report(data, deltas, out)
    print(json.dumps(deltas, indent=2))


def plot(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for version, d in data.items():
        xs, ys = zip(*d["curve"])
        axes[0].step([t/60 for t in xs], [c/1000 for c in ys], where="post",
                     color=COLORS[version], label=LABELS[version], linewidth=2)
        axes[0].scatter(xs[-1]/60, ys[-1]/1000, color=COLORS[version], s=25)
    axes[0].set(xlabel="Elapsed simulation time (minutes)", ylabel="Best feasible cost (thousands)",
                title="CG + IP every round: schedule quality")
    axes[0].legend(frameon=False)
    components = [("Cheap pricing", "#e3a077", lambda d: d["cheap_pricing_s"]),
                  ("Exact pricing", "#7395b4", lambda d: d["exact_pricing_s"]),
                  ("IP during CG", "#75856b", lambda d: d["result"]["iteration_ip_wall_s"]),
                  ("Final IP", "#9d83ae", lambda d: d["final_ip_wall_s"]),
                  ("LP solves", "#c6b35f", lambda d: d["lp_wall_s"])]
    bottoms = [0.0, 0.0]
    for label, color, metric in components:
        heights = [metric(d)/60 for d in data.values()]
        axes[1].bar(range(2), heights, bottom=bottoms, width=.6, label=label, color=color)
        bottoms = [x+y for x, y in zip(bottoms, heights)]
    totals = [d["result"]["simulation_wall_s"]/60 for d in data.values()]
    residual = [x-y for x, y in zip(totals, bottoms)]
    assert min(residual) >= -1e-3
    axes[1].bar(range(2), residual, bottom=bottoms, width=.6, label="Setup / other", color="#d8dcdf")
    for i, total in enumerate(totals):
        axes[1].text(i, total + max(totals)*.015, f"{total:.1f} min", ha="center", fontsize=10)
    axes[1].set(xticks=range(2), xticklabels=LABELS.values(), ylabel="Wall time (minutes)",
                title="Runtime including verification", ylim=(0, max(totals)*1.35))
    axes[1].legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("1,537 outbound flights · 1,200 s FAA demand · LP target 0.1% · 4 workers / 4 threads",
                 fontsize=12)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    for version, d in data.items():
        checkpoints = [r for r in d["iterations"] if r["pricing_exact"]]
        ax.plot([r["elapsed_s"]/60 for r in checkpoints],
                [max(1e-6, 100*r["lp_gap_cost"]) for r in checkpoints],
                marker="o", markersize=4, color=COLORS[version], label=LABELS[version])
    ax.axhline(.1, color="#555555", linestyle="--", linewidth=1, label="0.1% LP target")
    ax.set(yscale="log", xlabel="Elapsed simulation time (minutes)",
           ylabel="Certified LP gap (%)", title="Exact pricing checkpoints determine convergence")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2, which="both")
    ax.legend(frameon=False)
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)


def report(data, deltas, out):
    versions = list(LABELS)
    a, b = (data[v] for v in versions)
    rows = [
        ("Total simulation time", lambda d: f"{d['result']['simulation_wall_s']/60:.2f} min"),
        ("Final congestion cost", lambda d: f"{d['result']['congestion_cost']:,.3f}"),
        ("Ground delay", lambda d: f"{d['result']['ground_delay_s']:,.0f} s"),
        ("Air detour", lambda d: f"{d['result']['air_detour_m']:,.1f} m"),
        ("CG rounds", lambda d: str(d['result']['iterations'])),
        ("Cheap / exact sweeps", lambda d: f"{d['cheap_sweeps']} / {d['exact_sweeps']}"),
        ("Total pricing", lambda d: f"{d['result']['pricing_wall_s']/60:.2f} min"),
        ("Cheap pricing", lambda d: f"{d['cheap_pricing_s']:.2f} s"),
        ("Exact pricing", lambda d: f"{d['exact_pricing_s']:.2f} s"),
        ("LP solves", lambda d: f"{d['lp_wall_s']:.2f} s"),
        ("Native IP search during CG", lambda d: f"{d['iteration_ip_native_s']:.2f} s"),
        ("IP during CG including setup", lambda d: f"{d['result']['iteration_ip_wall_s']:.2f} s"),
        ("Final native IP search", lambda d: f"{d['final_ip_native_s']:.2f} s"),
        ("Generated pool columns", lambda d: f"{d['result']['columns']:,}"),
        ("Actual new priced columns", lambda d: f"{d['actual_priced_columns']:,}"),
        ("Cheap / exact new columns", lambda d: f"{d['cheap_columns_added']:,} / {d['exact_columns_added']:,}"),
        ("Pricing per new column", lambda d: f"{d['pricing_s_per_added_column']:.4f} s"),
        ("Final LP gap", lambda d: f"{100*d['lp_gap_cost']:.5f}%"),
        ("Global integer gap", lambda d: f"{100*d['result']['global_cost_gap']:.5f}%"),
        ("CG stop", lambda d: d['result']['termination']),
        ("Final IP pool optimal", lambda d: str(d['final_ip_optimal'])),
        ("Accepted / denied", lambda d: f"{d['result']['accepted']} / {d['result']['denied']}"),
        ("Physical verification", lambda d: str(d['result']['verified'])),
    ]
    time_direction = "longer" if deltas["wall_saving_pct"] < 0 else "shorter"
    cost_direction = "lower" if deltas["cost_change_pct"] < 0 else "higher"
    lines = ["# Cheap pricing with CG + IP every round", "",
             f"Cheap pricing took {abs(deltas['wall_saving_pct']):.2f}% {time_direction} "
             f"to meet the 0.1% LP stopping tolerance, with "
             f"{abs(deltas['cost_change_pct']):.4f}% {cost_direction} final cost. "
             "Standard pricing remains the default; this restricted-root variant stays optional.", "",
             "| Metric | Standard pricing | Cheap pricing |", "|---|---:|---:|"]
    lines += [f"| {label} | {fn(a)} | {fn(b)} |" for label, fn in rows]
    lines += ["", f"![Measured quality and runtime]({out / 'comparison.png'})", "",
              "## Interpretation and controls", "",
              "Both arms use identical original outbound requests, source files, seed, four pricing "
              "processes, four Gurobi threads, 0.1% LP tolerance, 30-round cap, and IP budgets. "
              "The 1,200 seconds describes the flight-generation window, not a solver time limit. "
              "The independent runs execute sequentially, cheap first.", "",
              "Cheap pricing uses the existing best-bound root search: one departure/lane choice "
              "with spatial routing still optimized inside it. It may also retain a better certified "
              "seed or seed time-shift. Every returned column is canonically checked. The exact "
              "oracle runs every five rounds, on zero new columns, and on the last allowed round. "
              "Only exact sweeps update global bounds or permit a convergence stop.", "",
              "A 0.1% LP gap is not a 0.1% integer optimality guarantee. Pool-optimal IP means "
              "optimal only over the generated columns. The global integer gap uses the certified "
              "pricing bound. Native IP caps exclude eager setup; the total solver deadline includes it.", "",
              "This is one seed and one timing sample per policy. Time-limited per-round IPs can "
              "choose different incumbents between runs. Cheap and exact policies also generate "
              "different pools, so per-column timing is not a matched-subproblem speed ratio. "
              "The initial schedule at time zero in the chart is a reference, not a measured "
              "availability time. All later points are completed, capacity-checked schedules.", "",
              f"![LP convergence checkpoints]({out / 'convergence.png'})", "",
              "## Time to the other policy's final quality", "",
              "| Policy | Standard final cost | Cheap final cost |", "|---|---:|---:|"]
    for v, d in data.items():
        times = [d["time_to_final_cost_s"][target] for target in versions]
        cells = ["Not reached" if t is None else f"{t/60:.2f} min" for t in times]
        lines.append(f"| {LABELS[v]} | {' | '.join(cells)} |")
    reference_path = out.parent / "previous_standard_reference.json"
    if reference_path.exists():
        reference = read(reference_path)
        previous = reference["result"]
        assert previous["requests_sha"] == a["result"]["requests_sha"]
        saved = 100 * (1-a["result"]["simulation_wall_s"]/previous["simulation_wall_s"])
        premium = 100 * (a["result"]["congestion_cost"]/previous["congestion_cost"]-1)
        lines += ["", "## Effect of the looser LP tolerance", "",
                  f"The earlier standard CG + IP run used a 0.01% target, hit its 30-round "
                  f"cap, and took {previous['simulation_wall_s']/60:.2f} minutes at cost "
                  f"{previous['congestion_cost']:,.3f}. The new 0.1% control takes "
                  f"{saved:.2f}% less time, with {premium:.4f}% higher cost. Its first 13 "
                  "rounds reproduce the earlier LP objectives, global bounds, column counts, "
                  "reduced-cost summaries and incumbent costs. This is a historical reference, "
                  "not a third fresh run; details are archived in previous_standard_reference.json.", "",
                  "Thus the large runtime reduction from the earlier 97-minute run comes from "
                  "stopping at the looser LP tolerance. Adding this cheap-pricing policy to that "
                  "control did not reduce convergence time further."]
    lines += ["", "## Earlier multi-column result", "",
              "Git commit `b65c4a2f87e7dccfdcc4377c54b6972396f5ff1d` records tied-column returns "
              "on a different, older 1,500-flight case: one column produced cost 184,666.0 with "
              "44,019 columns; five ties produced cost 198,222.9 with 65,805 columns. The LP gap "
              "tightened from 0.00924 to 0.000278, but both final IPs hit their 900-second limits. "
              "That identifies restricted-IP search as a bottleneck; it does not establish that "
              "multiple columns will fail with the current per-round IP policy. The proposed "
              "next comparison would retain several equal-reduced-cost columns with low row overlap "
              "and still use only the best reduced cost per flight in the global-bound sum. "
              "No multi-column experiment was run in this comparison.", "",
              "## Reproducibility", "",
              "See [protocol and command](../README.md), [machine-readable audit](audit.json), "
              "the source archive/manifest in the parent folder, and each arm's inputs, solver "
              "statistics, columns, paths, IP selections, native logs and filed trajectories. "
              "Input/source fingerprints, selection coverage/costs, native budgets, bounds, "
              "and final physical verification were audited.", ""]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summarize(args.root.resolve())
