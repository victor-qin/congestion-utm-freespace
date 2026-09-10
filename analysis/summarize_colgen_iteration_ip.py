"""Compare full-pool IP after every CG sweep with the archived outbound controls."""

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


def load(root):
    manifest = read(root / "manifest.json")
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for name, digest in manifest["fixed_source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    for name, key in (("harness_used.py", "harness_sha256"),
                      ("colgen_trace_used.py", "tracer_sha256")):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest[key]
    result = read(root / "results.json")[0]
    assert "error" not in result, result
    case = root / f"{result['scenario']}_seed{result['seed']}_{result['version']}"
    inputs = read(case / "inputs.json")
    stats = read(case / "planner_stats.json")
    iterations = read(case / "iterations.json")
    calls = read(case / "ip_calls.json")
    trace = read(case / "trace_summary.json")
    columns = [json.loads(line) for line in (case / "columns.jsonl").read_text().splitlines()]
    paths = [json.loads(line) for line in (case / "paths.jsonl").read_text().splitlines()]
    n = result["n_requests"]
    benefit = inputs["params"]["M"]
    assert result["verified"] and result["accepted"] == n == 763
    assert result["n_return_requests"] == 0 and result["outbound_only"]
    with gzip.open(case / "trajectories.json.gz", "rt") as handle:
        trajectories = json.load(handle)
    cfg = inputs["config"]
    cost = math.fsum(
        cfg["cost_ground_delay_per_s"] * t["ground_delay_s"]
        + cfg["cost_air_hold_per_s"] * t["air_hold_s"]
        + cfg["cost_air_lateral_per_s"] * t["air_detour_m"] / cfg["nominal_speed_mps"]
        for t in trajectories if t["accepted"]
    )
    assert abs(cost - result["congestion_cost"]) < 1e-5
    assert abs(result["congestion_cost_minus_solver_objective"]) < 1e-5
    for call in calls:
        selection = call["after_selection"]
        assert len(selection) == len({columns[i]["flight_id"] for i in selection.values()})
        assert abs(call["after_cost"] - math.fsum(columns[i]["cost"]
                                                 for i in selection.values())) < 1e-5
        call["penalized_cost"] = call["after_cost"] + benefit * (n - len(selection))
        call["native_s"] = sum(c["gurobi_runtime_s"] for c in call["native_calls"])
    round_calls = [c for c in calls if c.get("phase", "final") == "iteration"]
    final_calls = [c for c in calls if c.get("phase", "final") == "final"]
    if round_calls:
        assert len(round_calls) == stats["pricing_sweeps_completed"]
        assert trace["rounding_calls"] == trace["lns_calls"] == 0
        assert [c["sweep"] for c in round_calls] == list(range(1, len(round_calls) + 1))
        costs = [row["iteration_ip"]["cost"] for row in iterations]
        assert all(b <= a + 1e-6 for a, b in zip(costs, costs[1:]))
        assert all(abs(row["heuristic_cost"] - row["iteration_ip"]["cost"]) < 1e-6
                   for row in iterations)
        assert all(row["cost_lower_bound"] <= row["heuristic_cost"] + 1e-5
                   for row in iterations)
        for call in round_calls:
            assert call["requested_budget_s"] == inputs["params"]["iteration_ip_time_limit_s"]
            assert all(c["time_limit_s"] <= call["requested_budget_s"] + 1e-3
                       for c in call["native_calls"])
    if final_calls:
        assert final_calls[-1]["requested_budget_s"] == 600
    events = [(0.0, result["ground_bootstrap"]["cost"])]
    for call in read(case / "lns_calls.json"):
        events.append((call["simulation_elapsed_s"], call["after_cost"]
                       + benefit * (n - call["covered"])))
    for call in calls:
        events.append((call["start_elapsed_s"] + call["wall_s"], call["penalized_cost"]))
    events.append((result["simulation_wall_s"], result["penalized_congestion_cost"]))
    best = math.inf
    curve = []
    for elapsed, value in sorted(events):
        best = min(best, value)
        curve.append((elapsed, best))
    pool = {}
    first_round_pool = {}
    for c in columns:
        p = paths[c["path_id"]]
        key = (c["flight_id"], c["departure_step"], p["level"], p["origin_lane_idx"],
               p["dest_lane_idx"], tuple(tuple(cell) for cell in p["cells"]))
        pool[key] = c["cost"]
        if c["sweep"] <= 1:
            first_round_pool[key] = c["cost"]
    summary = {
        "result": result, "trace": trace, "curve": curve,
        "first_round_pool_sha256": hashlib.sha256(json.dumps(
            sorted(first_round_pool.items(), key=repr)).encode()).hexdigest(),
        "first_round_columns": len(first_round_pool),
        "iteration_ip_calls": len(round_calls),
        "iteration_ip_optimal_calls": sum(bool(c["optimal"]) for c in round_calls),
        "iteration_ip_native_s": sum(c["native_s"] for c in round_calls),
        "iteration_ip_wall_s": stats.get("iteration_ip_wall_s", 0),
        "final_ip_native_s": sum(c["native_s"] for c in final_calls),
        "all_ip_native_s": sum(c["native_s"] for c in calls),
        "ip_rows_added_during_cg": sum(r.get("iteration_ip", {}).get("rows_added", 0)
                                       for r in iterations),
        "heuristic_and_canonical_s": sum(
            r["stage_s"].get(k, 0) for r in iterations
            for k in ("round_heuristic", "lns_heuristic", "canonical_heuristic")),
        "lp_wall_s": sum(r["stage_s"].get("solve_lp", 0) for r in iterations),
        "final_lp_gap": stats["lp_gap_cost"], "repair_added": stats["repair_added"],
        "rounds": [dict(r["iteration_ip"], native_s=c["native_s"])
                   for r, c in zip(iterations, round_calls)],
    }
    return inputs, manifest, pool, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iteration_ip", type=Path)
    parser.add_argument("--lns", type=Path,
                        default=Path("analysis/colgen_outbound_only_20260908/long"))
    parser.add_argument("--rounding", type=Path,
                        default=Path("analysis/colgen_outbound_only_20260908/cg_rounding"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eager", type=Path,
                        help="Optional short follow-up with all bindable IP rows prepared up front")
    args = parser.parse_args()
    sources = {"CG + rounding": args.rounding, "CG + LNS": args.lns,
               "CG + IP each round": args.iteration_ip}
    eager_label = "CG + IP, eager (3 rounds)"
    if args.eager:
        sources[eager_label] = args.eager
    loaded = {label: load(path) for label, path in sources.items()}
    controls = []
    for inputs, _, _, _ in loaded.values():
        copy = dict(inputs)
        params = dict(copy["params"])
        params.pop("lns_destroy_flights", None)
        params.pop("iteration_ip_time_limit_s", None)
        if len(controls) == 3:
            assert params["max_iterations"] == 3
            params["max_iterations"] = controls[0]["params"]["max_iterations"]
        copy["params"] = params
        controls.append(copy)
    assert all(c == controls[0] for c in controls), "uncontrolled input difference"
    packages = [manifest["packages"] for _, manifest, _, _ in loaded.values()]
    assert all(p == packages[0] for p in packages)
    summary = {label: data[3] for label, data in loaded.items()}
    old = summary["CG + LNS"]
    new = summary["CG + IP each round"]
    target = old["result"]["congestion_cost"]
    new["time_to_lns_final_cost_s"] = next((t for t, c in new["curve"] if c <= target + 1e-5), None)
    new["time_to_within_0_1pct_lns_final_cost_s"] = next(
        (t for t, c in new["curve"] if c <= target * 1.001), None)
    new["whole_run_speedup"] = old["result"]["simulation_wall_s"] / new["result"]["simulation_wall_s"]
    new["cost_improvement_pct"] = 100 * (1 - new["result"]["congestion_cost"] / target)
    if args.eager:
        eager = summary[eager_label]
        assert eager["first_round_pool_sha256"] == new["first_round_pool_sha256"]
        eager["time_to_lns_final_cost_s"] = next(
            (t for t, c in eager["curve"] if c <= target + 1e-5), None)
        eager["cost_difference_pct"] = 100 * (eager["result"]["congestion_cost"] / target - 1)
    old_sources = loaded["CG + LNS"][1]["fixed_source_sha256"]
    new_sources = loaded["CG + IP each round"][1]["fixed_source_sha256"]
    if args.eager:
        assert loaded[eager_label][1]["fixed_source_sha256"] == new_sources
    changed = [k for k in set(old_sources) | set(new_sources) if old_sources.get(k) != new_sources.get(k)]
    assert set(changed) <= {f"freespace_sim/planner/colgen/{name}.py"
                            for name in ("solver", "master", "params")}, changed
    old_pool, new_pool = loaded["CG + LNS"][2], loaded["CG + IP each round"][2]
    shared = old_pool.keys() & new_pool.keys()
    assert all(abs(old_pool[key] - new_pool[key]) < 1e-6 for key in shared)
    audit = dict(runs=summary, source_changes=sorted(changed), inputs_match=True,
                 source_archives_verified=True, packages_match=True,
                 eager_pilot_iteration_cap=3 if args.eager else None,
                 first_round_lazy_eager_pools_match=True if args.eager else None,
                 shared_pool_columns=len(shared), old_only_columns=len(old_pool.keys() - shared),
                 new_only_columns=len(new_pool.keys() - shared))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    colors = ["#8b94a3", "#177e89", "#cf5b21", "#6645ac"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for (label, data), color in zip(summary.items(), colors):
        xs, ys = zip(*data["curve"])
        axes[0].step([x / 60 for x in xs], [y / 1000 for y in ys], where="post",
                     label=label, color=color, linewidth=2)
        axes[0].scatter(xs[-1] / 60, ys[-1] / 1000, color=color, s=30, zorder=3)
    axes[0].axhline(target / 1000, color="#555", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Elapsed simulation time (minutes)", ylabel="Best feasible cost (thousands)",
                title="Quality available while CG is running")
    axes[0].legend(frameon=False, fontsize=9)
    rounds = new["rounds"]
    axes[1].bar([r["iteration"] - (0.18 if args.eager else 0) for r in rounds],
                [r["wall_s"] for r in rounds], width=.35 if args.eager else .8,
                color=colors[2], label="IP with lazy row separation")
    if args.eager:
        axes[1].bar([r["iteration"] + .18 for r in eager["rounds"]],
                    [r["wall_s"] for r in eager["rounds"]], width=.35,
                    color=colors[3], label="IP with all rows prepared")
    axes[1].axhline(30, color="#555", linestyle=":", linewidth=1, label="30 s search cap")
    axes[1].set(xlabel="CG round", ylabel="Time (seconds)", title="Cost of solving the IP every round")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylim(0, max(r["wall_s"] for r in rounds +
                           (eager["rounds"] if args.eager else [])) * 1.27)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("763 outbound flights · identical seed-0 demand · 4 pricing workers", fontsize=13)
    fig.savefig(args.out / "comparison.png", dpi=180)
    plt.close(fig)

    lines = ["# CG with a full-pool IP after every pricing sweep", "",
             "Same 763 outbound flights, 600-second demand window, 182 static terminals, "
             "four pricing workers and four Gurobi threads. Per-round IP replaces LNS/rounding, "
             "uses a 30-second cap and lazy capacity separation, and passes its feasible incumbent "
             "to the next round. Final IP retains its 600-second cap and 660-second reserve. "
             "LP gap target is 0.01%; native IP gap target is zero.", "",
             "The controls are archived timing samples, not repeated contemporaneous runs. "
             "IP-added capacity rows and LP/MIP transitions can change later duals and pricing; "
             "this compares complete solver policies rather than identical column pools.", "",
             "| Metric | " + " | ".join(summary) + " |",
             "|---|" + "---:|" * len(summary)]
    metrics = [
        ("Whole simulation, s", lambda d: d["result"]["simulation_wall_s"]),
        ("Pricing, s", lambda d: d["result"]["pricing_wall_s"]),
        ("LP solving, s", lambda d: d["lp_wall_s"]),
        ("Pricing sweeps", lambda d: d["result"]["iterations"]),
        ("Per-round IP calls", lambda d: d["iteration_ip_calls"]),
        ("Per-round IP wall, s", lambda d: d["iteration_ip_wall_s"]),
        ("All native IP search, s", lambda d: d["all_ip_native_s"]),
        ("Final native IP search, s", lambda d: d["final_ip_native_s"]),
        ("Final verified congestion cost", lambda d: d["result"]["congestion_cost"]),
        ("Final coverage", lambda d: d["result"]["accepted"]),
        ("Total ground delay, s", lambda d: d["result"]["ground_delay_s"]),
        ("Final LP gap, %", lambda d: 100 * d["final_lp_gap"]),
        ("Final global cost gap, %", lambda d: 100 * d["result"]["global_cost_gap"]),
        ("Columns", lambda d: d["result"]["columns"]),
    ]
    for label, metric in metrics:
        values = [metric(d) for d in summary.values()]
        lines.append("| " + label + " | " + " | ".join(
            f"{v:,}" if isinstance(v, int) else f"{v:,.3f}" for v in values) + " |")
    lines += ["", f"Per-round IPs proven optimal: {new['iteration_ip_optimal_calls']} / "
              f"{new['iteration_ip_calls']}. IP separation added "
              f"{new['ip_rows_added_during_cg']:,} capacity rows during CG.", "",
              f"The full per-round-IP run stopped on `{new['result']['termination']}` "
              f"with LP gap {100 * new['final_lp_gap']:.5f}%, above the 0.01% target. "
              f"Its final restricted IP status was `{new['result']['ip_status']}`.", "",
              f"Time to match or beat the old LNS-plus-final-IP cost: "
              f"{new['time_to_lns_final_cost_s']} seconds. "
              f"Time to get within 0.1%: {new['time_to_within_0_1pct_lns_final_cost_s']} seconds.", "",
              f"Pools share {len(shared):,} exact timed columns; "
              f"{len(old_pool.keys() - shared):,} only in the LNS control and "
              f"{len(new_pool.keys() - shared):,} only in the per-round IP run.", "",
              "Source archives and harness/tracer hashes verified. Requests, configuration, "
              "packages and other solver parameters match. Filed costs were recomputed from "
              "trajectory metrics. Every final flight passed simulation conflict verification. "
              "A pool-optimal IP does not establish global integer optimality. "
              "The plot shows seed cost at time zero as a reference; improvements are plotted "
              "only after completed, capacity-checked solves.", "",
              "![Runtime and quality](comparison.png)", "",
              "| Round | Feasible cost | IP wall, s | Native IP, s | Status | Rows added |",
              "|---:|---:|---:|---:|---|---:|"]
    for r in rounds:
        lines.append(f"| {r['iteration']} | {r['cost']:,.3f} | {r['wall_s']:.3f} | "
                     f"{r['native_s']:.3f} | {r['status']} | {r['rows_added']} |")
    if args.eager:
        lines += ["", "## Complete-row follow-up (three rounds)", "",
                  "The follow-up changes the per-round IP to prepare all bindable capacity "
                  "rows before search, just as the final IP does. It retains the same "
                  "30-second search cap and 600-second final-IP cap, but deliberately stops "
                  "CG after at most three rounds. This tests early schedule quality; it is "
                  "not a comparison at equal CG convergence.", "",
                  f"The lazy and eager first-round pools match exactly: "
                  f"{eager['first_round_columns']:,} timed columns with identical costs. "
                  "That first IP comparison isolates row preparation on the same alternatives.", "",
                  f"Final cost difference versus the converged LNS control: "
                  f"{eager['cost_difference_pct']:.4f}%. "
                  + ("It did not reach the control's final cost."
                     if eager['time_to_lns_final_cost_s'] is None else
                     f"It first matched the control at {eager['time_to_lns_final_cost_s']:.1f} seconds."), "",
                  "| Round | Feasible cost | IP wall, s | Native IP, s | Status | Rows added |",
                  "|---:|---:|---:|---:|---|---:|"]
        for r in eager["rounds"]:
            lines.append(f"| {r['iteration']} | {r['cost']:,.3f} | {r['wall_s']:.3f} | "
                         f"{r['native_s']:.3f} | {r['status']} | {r['rows_added']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: new[k] for k in ("whole_run_speedup", "cost_improvement_pct",
                                         "time_to_lns_final_cost_s", "iteration_ip_native_s")}, indent=2))


if __name__ == "__main__":
    main()
