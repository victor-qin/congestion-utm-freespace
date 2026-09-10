"""Audit the matched outbound-only CG rounding versus CG LNS experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read(path):
    return json.loads(path.read_text())


def load(directory):
    manifest = read(directory / "manifest.json")
    with tarfile.open(directory / "source_snapshot.tar.gz") as archive:
        for name, expected in manifest["fixed_source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    for name, key in (("harness_used.py", "harness_sha256"),
                      ("colgen_trace_used.py", "tracer_sha256")):
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == manifest[key]
    result = read(directory / "results.json")[0]
    assert "error" not in result, result
    case = directory / f"{result['scenario']}_seed{result['seed']}_{result['version']}"
    inputs, stats = read(case / "inputs.json"), read(case / "planner_stats.json")
    columns = [json.loads(line) for line in (case / "columns.jsonl").read_text().splitlines()]
    paths = [json.loads(line) for line in (case / "paths.jsonl").read_text().splitlines()]
    ip = read(case / "ip_calls.json")
    assert len(ip) == 1
    ip = ip[0]
    assert result["verified"] and result["accepted"] == result["n_requests"] == 763
    assert result["n_return_requests"] == 0 and result["outbound_only"]
    assert len(ip["before_selection"]) == 763
    assert ip["native_calls"][0]["time_limit_s"] >= 600 - 1e-3
    with gzip.open(case / "trajectories.json.gz", "rt") as handle:
        trajectories = {str(t["flight_id"]): t for t in json.load(handle)}
    cfg = inputs["config"]
    costs = {fid: cfg["cost_ground_delay_per_s"] * t["ground_delay_s"]
             + cfg["cost_air_hold_per_s"] * t["air_hold_s"]
             + cfg["cost_air_lateral_per_s"] * t["air_detour_m"] / cfg["nominal_speed_mps"]
             for fid, t in trajectories.items()}
    assert abs(math.fsum(costs.values()) - result["congestion_cost"]) < 1e-5
    assert abs(result["congestion_cost_minus_solver_objective"]) < 1e-5
    spacing = cfg["nominal_speed_mps"] * cfg["dt_s"]
    for fid, index in ip["after_selection"].items():
        assert abs(costs[fid] - columns[index]["cost"]) < 1e-5
        actual = iter(tuple(round(v, 4) for v in point[:2])
                      for point, _time in trajectories[fid]["centerline"])
        previous = None
        for q, r in paths[columns[index]["path_id"]]["cells"]:
            point = (round(spacing * (q + r / 2), 4),
                     round(spacing * math.sqrt(3) / 2 * r, 4))
            if point != previous:
                assert any(p == point for p in actual), fid
            previous = point
    trace = read(case / "trace_summary.json")
    iterations = read(case / "iterations.json")
    trial_coverage = [n for row in iterations for n in row["round_stats"]["try_covered"]]
    summary = dict(
        result=result, initial_cost=result["ground_bootstrap"]["cost"],
        heuristic_cost_before_ip=ip["before_cost"],
        heuristic_gain_pct=100 * (1 - ip["before_cost"] / result["ground_bootstrap"]["cost"]),
        native_ip_runtime_s=sum(c["gurobi_runtime_s"] for c in ip["native_calls"]),
        native_ip_nodes=sum(c["nodes"] for c in ip["native_calls"]),
        native_ip_limit_s=ip["native_calls"][0]["time_limit_s"],
        native_ip_optimal=ip["optimal"],
        heuristic_stage_s=sum(row["stage_s"].get(stage, 0) for row in iterations
                              for stage in ("round_heuristic", "lns_heuristic")),
        heuristic_canonicalization_s=sum(row["stage_s"].get("canonical_heuristic", 0)
                                         for row in iterations),
        final_lp_gap=stats["lp_gap_cost"], repair_added=stats["repair_added"],
        final_ip_gain_pct=100 * (1 - result["congestion_cost"] / ip["before_cost"]),
        trials=len(trial_coverage), min_trial_coverage=min(trial_coverage),
        max_trial_coverage=max(trial_coverage), full_trials=sum(n == 763 for n in trial_coverage),
        heuristic_trial_coverage=[dict(iteration=row["iteration"],
                                       min_coverage=min(row["round_stats"]["try_covered"]),
                                       max_coverage=max(row["round_stats"]["try_covered"]),
                                       full_coverage_trials=sum(n == 763 for n in row["round_stats"]["try_covered"]))
                                  for row in iterations],
        trace=trace,
    )
    pool = {}
    for c in columns:
        p = paths[c["path_id"]]
        key = (c["flight_id"], c["departure_step"], p["level"], p["origin_lane_idx"],
               p["dest_lane_idx"], tuple(map(tuple, p["cells"])))
        assert key not in pool
        pool[key] = c["cost"]
    return dict(summary=summary, manifest=manifest, inputs=inputs, iterations=iterations, pool=pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounding", type=Path)
    parser.add_argument("lns", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rounding, lns = load(args.rounding.resolve()), load(args.lns.resolve())
    x, y = rounding["inputs"], lns["inputs"]
    assert {k: v for k, v in x.items() if k != "params"} == {
        k: v for k, v in y.items() if k != "params"}
    assert x["params"]["lns_destroy_flights"] == 0 and y["params"]["lns_destroy_flights"] == 100
    assert {k: v for k, v in x["params"].items() if k != "lns_destroy_flights"} == {
        k: v for k, v in y["params"].items() if k != "lns_destroy_flights"}
    for key in ("fixed_source_sha256", "packages", "harness_sha256", "tracer_sha256"):
        assert rounding["manifest"][key] == lns["manifest"][key], key
    a, b = rounding["summary"], lns["summary"]
    assert a["result"]["input_sha"] == b["result"]["input_sha"]
    assert a["trace"]["lns_calls"] == b["trace"]["rounding_calls"] == 0
    common = rounding["pool"].keys() & lns["pool"].keys()
    assert all(abs(rounding["pool"][k] - lns["pool"][k]) < 1e-5 for k in common)
    comparison = dict(
        identical_inputs_except_heuristic=True, identical_sources=True,
        identical_final_trajectories=a["result"]["trajectory_sha"] == b["result"]["trajectory_sha"],
        identical_lp_gap_history=[r["lp_gap_cost"] for r in rounding["iterations"]]
        == [r["lp_gap_cost"] for r in lns["iterations"]],
        common_columns=len(common), rounding_only_columns=len(rounding["pool"]) - len(common),
        lns_only_columns=len(lns["pool"]) - len(common),
        lns_wall_speedup=a["result"]["simulation_wall_s"] / b["result"]["simulation_wall_s"],
        lns_wall_reduction_pct=100 * (1 - b["result"]["simulation_wall_s"] / a["result"]["simulation_wall_s"]),
        lns_final_cost_reduction_pct=100 * (1 - b["result"]["congestion_cost"] / a["result"]["congestion_cost"]),
    )
    (args.out / "audit.json").write_text(json.dumps(
        dict(rounding=a, lns=b, comparison=comparison), indent=2, allow_nan=False) + "\n")
    metrics = [
        ("Whole simulation", lambda s: f"{s['result']['simulation_wall_s']:.2f} s"),
        ("Pricing wall", lambda s: f"{s['result']['pricing_wall_s']:.2f} s"),
        ("Complete pricing sweeps", lambda s: str(s['result']['pricing_sweeps_completed'])),
        ("Heuristic calls (rounding / LNS)", lambda s: f"{s['trace']['rounding_calls']} / {s['trace']['lns_calls']}"),
        ("Heuristic stage time", lambda s: f"{s['heuristic_stage_s']:.3f} s"),
        ("Heuristic canonicalization", lambda s: f"{s['heuristic_canonicalization_s']:.3f} s"),
        ("Feasible incumbent before IP", lambda s: f"{s['heuristic_cost_before_ip']:,.2f}"),
        ("Heuristic improvement over seed", lambda s: f"{s['heuristic_gain_pct']:.2f}%"),
        ("Native IP search", lambda s: f"{s['native_ip_runtime_s']:.3f} s"),
        ("IP status", lambda s: s['result']['ip_status']),
        ("Final congestion cost", lambda s: f"{s['result']['congestion_cost']:,.2f}"),
        ("Final coverage, verified", lambda s: f"{s['result']['accepted']}/{s['result']['n_requests']}"),
        ("LP pricing gap", lambda s: f"{100*s['final_lp_gap']:.5f}%"),
        ("Global cost gap", lambda s: f"{100*s['result']['global_cost_gap']:.4f}%"),
        ("Columns", lambda s: f"{s['result']['columns']:,}"),
        ("Mean delay", lambda s: f"{s['result']['mean_delay_s']:.3f} s"),
    ]
    lines = ["# Outbound-only CG: rounding versus LNS", "",
             "The matched run uses the identical 763 outbound requests, 182 static terminals, "
             "source code, package versions, demand seed 0 and 600-second demand window. "
             "The only solver parameter difference is `lns_destroy_flights`: 0 enables the existing "
             "16-try randomized rounding and greedy fill; 100 enables embedded pool LNS. Both retain "
             "the same feasible ground-delay seed and final IP. This compares CG's existing rounding "
             "heuristic with LNS, not CG with every primal heuristic removed.", "",
             "Both allow 30 pricing sweeps, target LP gap 0.01%, and reserve 660 seconds for a "
             "600-second native IP cap with zero IP gap target. Measurements are separate fresh "
             "processes on the same host, with warmed pricing kernels, four pricing workers and four "
             "Gurobi master threads. One seed and one timing sample per arm do not establish a small "
             "runtime difference statistically.", "", "| Metric | CG + rounding | CG + LNS |",
             "|---|---:|---:|"]
    lines += [f"| {name} | {fn(a)} | {fn(b)} |" for name, fn in metrics]
    lines += ["", f"Measured LNS wall reduction: {comparison['lns_wall_reduction_pct']:.2f}%. "
              f"Final cost reduction: {comparison['lns_final_cost_reduction_pct']:.6f}%.", "",
              f"The pools share {comparison['common_columns']:,} exact timed routes; "
              f"{comparison['rounding_only_columns']:,} occur only with rounding and "
              f"{comparison['lns_only_columns']:,} only with LNS. Matching columns have matching costs.", "",
              f"Identical final trajectory fingerprints: {comparison['identical_final_trajectories']}. "
              f"Identical LP-gap histories: {comparison['identical_lp_gap_history']}.", "",
              f"Rounding's {a['trials']} trials cover {a['min_trial_coverage']}–{a['max_trial_coverage']} "
              f"of the 763 flights; {a['full_trials']} yield full coverage. Its retained incumbent "
              f"improves the seed by {a['heuristic_gain_pct']:.2f}%. LNS preserves coverage and improves "
              f"its seed by {b['heuristic_gain_pct']:.2f}%. The heuristic and canonicalization "
              f"stage savings are {a['heuristic_stage_s'] + a['heuristic_canonicalization_s'] - b['heuristic_stage_s'] - b['heuristic_canonicalization_s']:.2f} "
              f"of the {a['result']['simulation_wall_s'] - b['result']['simulation_wall_s']:.2f} "
              f"seconds of measured whole-run difference. Pricing also differs by "
              f"{a['result']['pricing_wall_s'] - b['result']['pricing_wall_s']:.2f} seconds. "
              "A repeat is needed before treating the entire timing difference as a reliable LNS speedup.", "",
              "Source archives and harness/tracer hashes were verified. The audit also checked exact "
              "request/config/parameter equality apart from the heuristic, full final coverage and "
              "simulation conflict verification, trajectory cost/objective agreement, and preservation "
              "of the selected IP route geometry. A pool-optimal IP does not prove global integer "
              "optimality; the separately reported global gap uses the pricing bound.", "",
              "![Heuristic comparison](heuristic_comparison.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for data, label, color in ((rounding, "CG + rounding", "#596F91"), (lns, "CG + LNS", "#07857C")):
        rows, s = data["iterations"], data["summary"]
        axes[0].step([r["iteration"] for r in rows], [r["heuristic_cost"] for r in rows],
                     where="post", label=label, color=color)
        axes[0].scatter(len(rows) + .5, s["result"]["congestion_cost"], color=color, marker="*", s=130)
        axes[1].plot([r["elapsed_s"] / 60 for r in rows],
                     [100*r["lp_gap_cost"] for r in rows], color=color, label=label)
    axes[0].set(title="Feasible incumbent; star = final IP", xlabel="Pricing iteration", ylabel="Congestion cost")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    if comparison["identical_final_trajectories"]:
        axes[0].annotate("Same final schedule", xy=(len(lns["iterations"]) + .5, b["result"]["congestion_cost"]),
                         xytext=(len(lns["iterations"]) - 6, b["result"]["congestion_cost"] + 1600),
                         arrowprops=dict(arrowstyle="->", color="#777777"), fontsize=9)
    axes[1].set(title="Pricing convergence", xlabel="Simulation time (minutes)", ylabel="LP gap (%)", yscale="log")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("Same 763 outbound flights: CG with rounding versus CG with LNS")
    fig.tight_layout()
    fig.savefig(args.out / "heuristic_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
