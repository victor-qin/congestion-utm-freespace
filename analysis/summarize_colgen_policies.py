"""Audit a matched multi-arm CG benchmark and plot measured quality and runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt


LABELS = {"baseline": "CG + rounding", "fixed": "CG + LNS",
          "iteration_ip_eager": "CG + IP each round"}


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-5), (a, b)


def load_case(root, result):
    assert "error" not in result, result
    case = root / f"{result['scenario']}_seed{result['seed']}_{result['version']}"
    inputs, stats, iterations, calls, trace, lns = [read(case / name) for name in (
        "inputs.json", "planner_stats.json", "iterations.json", "ip_calls.json",
        "trace_summary.json", "lns_calls.json")]
    columns, paths = [[json.loads(line) for line in (case / name).read_text().splitlines()]
                      for name in ("columns.jsonl", "paths.jsonl")]
    n, benefit = result["n_requests"], inputs["params"]["M"]
    assert result["verified"] and result["accepted"] + result["denied"] == n
    assert len(inputs["requests"]) == n
    assert result["n_return_requests"] == 0 and result["outbound_only"]
    with gzip.open(case / "trajectories.json.gz", "rt") as handle:
        trajectories = json.load(handle)
    assert len(trajectories) == n
    assert len({t["flight_id"] for t in trajectories}) == n
    accepted = [t for t in trajectories if t["accepted"]]
    assert len(accepted) == result["accepted"]
    cfg = inputs["config"]
    cost = math.fsum(cfg["cost_ground_delay_per_s"] * t["ground_delay_s"]
                    + cfg["cost_air_hold_per_s"] * t["air_hold_s"]
                    + cfg["cost_air_lateral_per_s"] * t["air_detour_m"]
                    / cfg["nominal_speed_mps"] for t in accepted)
    close(cost, result["congestion_cost"])
    close(cost, result["solver_objective"])
    close(cost + benefit * result["denied"], result["penalized_congestion_cost"])
    close(math.fsum(t["ground_delay_s"] for t in accepted), result["ground_delay_s"])
    assert len(columns) == trace["columns"] == result["columns"]
    assert all(c["index"] == i for i, c in enumerate(columns))
    for call in calls:
        selection = call["after_selection"]
        assert len(selection) == len({columns[i]["flight_id"] for i in selection.values()})
        assert all(int(fid) == columns[i]["flight_id"] for fid, i in selection.items())
        close(call["after_cost"], math.fsum(columns[i]["cost"] for i in selection.values()))
        call["penalized_cost"] = call["after_cost"] + benefit * (n - len(selection))
        call["native_s"] = math.fsum(c["gurobi_runtime_s"] for c in call["native_calls"])
        assert all(c["time_limit_s"] <= call["requested_budget_s"] + 1e-3
                   for c in call["native_calls"])
    round_calls = [c for c in calls if c["phase"] == "iteration"]
    final_calls = [c for c in calls if c["phase"] == "final"]
    assert len(round_calls) == stats["iteration_ip_calls"] == trace["iteration_ip_calls"]
    if round_calls:
        assert trace["rounding_calls"] == trace["lns_calls"] == 0
        assert len(round_calls) == stats["iterations"]
        assert all(c["requested_budget_s"] == inputs["params"]["iteration_ip_time_limit_s"]
                   for c in round_calls)
    assert all(c["requested_budget_s"] == inputs["params"]["ip_time_limit_s"] for c in final_calls)
    costs = [r["heuristic_cost"] for r in iterations]
    assert all(b <= a + 1e-5 for a, b in zip(costs, costs[1:]))
    assert all(float(r["cost_lower_bound"]) <= r["heuristic_cost"] + 1e-5 for r in iterations)
    # Only completed, capacity-checked outputs enter the quality curve. Native
    # callbacks can precede separation/canonicalization and are not plotted.
    # The seed at time zero is a reference, not a measured availability time.
    seed = result["ground_bootstrap"]
    events = [(0.0, seed["cost"] + benefit * (n - seed["covered"]))]
    events += [(r["elapsed_s"], r["heuristic_cost"]) for r in iterations]
    events += [(c["simulation_elapsed_s"], c["after_cost"] + benefit * (n - c["covered"]))
               for c in lns]
    events += [(c["start_elapsed_s"] + c["wall_s"], c["penalized_cost"]) for c in calls]
    events.append((result["simulation_wall_s"], result["penalized_congestion_cost"]))
    curve, best = [], math.inf
    for elapsed, value in sorted(events):
        best = min(best, value)
        curve.append((elapsed, best))
    pool = {}
    for c in columns:
        p = paths[c["path_id"]]
        key = (c["flight_id"], c["departure_step"], p["level"], p["origin_lane_idx"],
               p["dest_lane_idx"], tuple(tuple(cell) for cell in p["cells"]))
        assert key not in pool
        pool[key] = c["cost"]
    origins = Counter(c["phase"] for c in columns)
    priced = origins["pricing"]
    data = dict(result=result, trace=trace, curve=curve, lp_gap_cost=stats["lp_gap_cost"],
                cg_lp_target_met=stats["lp_gap_cost"] <= inputs["params"]["lp_gap"],
                kernel_label_restarts=stats.get("kernel_label_restarts", 0),
                kernel_budget_declined=stats.get("kernel_budget_declined", 0),
                repair_added=stats["repair_added"], initial_cost=events[0][1],
                actual_priced_columns=priced,
                pricing_s_per_added_column=result["pricing_wall_s"] / max(1, priced),
                iteration_ip_optimal_calls=sum(bool(c["optimal"]) for c in round_calls),
                iteration_ip_statuses=dict(Counter(c["status"] for c in round_calls)),
                iteration_ip_native_s=math.fsum(c["native_s"] for c in round_calls),
                final_ip_native_s=math.fsum(c["native_s"] for c in final_calls),
                final_ip_calls=len(final_calls),
                final_ip_optimal=stats["ip_optimal"],
                lp_wall_s=math.fsum(r["stage_s"].get("solve_lp", 0) for r in iterations),
                heuristic_and_canonical_s=math.fsum(r["stage_s"].get(k, 0) for r in iterations
                    for k in ("round_heuristic", "lns_heuristic", "canonical_heuristic")),
                per_round_ip=[dict(sweep=c["sweep"], cost=c["penalized_cost"],
                    available_s=c["start_elapsed_s"] + c["wall_s"],
                    native_s=c["native_s"], wall_s=c["wall_s"], setup_s=c["setup_s"],
                    status=c["status"], covered=len(c["after_selection"])) for c in round_calls])
    normalized = dict(inputs, params=dict(inputs["params"]))
    normalized["params"]["lns_destroy_flights"] = 0
    normalized["params"]["iteration_ip_time_limit_s"] = 0.0
    if "cheap_pricing" in normalized["params"]:
        normalized["params"].update(cheap_pricing=False, iteration_ip_eager=True)
    if "seed_nominal_routes" in normalized["params"]:
        normalized["params"].update(warm_start_planner=None, seed_nominal_routes=True,
                                    seed_ladder_steps=20, warm_start_max_shift_steps=8)
    if "provided_seed_ladder_steps" in normalized["params"]:
        normalized["params"]["provided_seed_ladder_steps"] = 0
    assert hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest() == result["input_sha"]
    return normalized, pool, data


def summarize(root, out):
    manifest, results = read(root / "manifest.json"), read(root / "results.json")
    assert len(results) == 3 and {r["version"] for r in results} == set(LABELS)
    assert len({(r["scenario"], r["seed"]) for r in results}) == 1
    sources = [manifest[f"{version}_source_sha256"] for version in LABELS]
    assert all(source == sources[0] for source in sources)
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for name, digest in sources[0].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
    for name, key in (("harness_used.py", "harness_sha256"), ("colgen_trace_used.py", "tracer_sha256"),
                      ("changes.patch", "patch_sha256")):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest[key]
    loaded = {r["version"]: load_case(root, r) for r in results}
    assert all(v[0] == loaded["baseline"][0] for v in loaded.values()), "uncontrolled input difference"
    data = {version: loaded[version][2] for version in LABELS}
    pairwise = []
    for a, b in itertools.combinations(LABELS, 2):
        pa, pb = loaded[a][1], loaded[b][1]
        common = pa.keys() & pb.keys()
        assert all(abs(pa[k] - pb[k]) < 1e-6 for k in common)
        pairwise.append(dict(a=a, b=b, shared_columns=len(common), a_only=len(pa)-len(common),
                             b_only=len(pb)-len(common), identical_pool=pa == pb))
    targets = {v: d["result"]["penalized_congestion_cost"] for v, d in data.items()}
    for d in data.values():
        d["time_to_final_cost_s"] = {v: next((t for t, c in d["curve"] if c <= target + 1e-5), None)
                                      for v, target in targets.items()}
    audit = dict(inputs_match=True, sources_match=True, archive_hashes_verified=True,
                 all_final_schedules_verified=True, costs_recomputed=True,
                 runs=data, pool_comparisons=pairwise)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(data, out)
    report(data, pairwise, manifest, out)
    print(json.dumps({LABELS[v]: dict(wall_s=d["result"]["simulation_wall_s"],
        cost=d["result"]["penalized_congestion_cost"], accepted=d["result"]["accepted"],
        rounds=d["result"]["iterations"], lp_gap=d["lp_gap_cost"],
        global_gap=d["result"]["global_cost_gap"]) for v, d in data.items()}, indent=2))


def plot(data, out):
    colors = ["#7b8794", "#177e89", "#bc5428"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for (version, d), color in zip(data.items(), colors):
        xs, ys = zip(*d["curve"])
        axes[0].step([t / 60 for t in xs], [c / 1000 for c in ys], where="post",
                     label=LABELS[version], color=color, linewidth=2)
        axes[0].scatter(xs[-1] / 60, ys[-1] / 1000, color=color, s=25)
    axes[0].set(xlabel="Elapsed simulation time (minutes)", ylabel="Best feasible cost (thousands)",
                title="Schedule quality available during the run")
    crossing = data["iteration_ip_eager"]["time_to_final_cost_s"]["fixed"]
    if crossing is not None:
        target = data["fixed"]["result"]["penalized_congestion_cost"] / 1000
        axes[0].scatter(crossing / 60, target, s=32, color=colors[2], zorder=4)
        axes[0].annotate(f"Matches controls' final cost\n{crossing/60:.1f} min",
                         xy=(crossing / 60, target), xytext=(.35, .15),
                         textcoords="axes fraction", fontsize=9, color=colors[2],
                         arrowprops=dict(arrowstyle="->", color=colors[2], lw=1))
    axes[0].legend(frameon=False, fontsize=9)
    components = [("Pricing", "#8fa6bd", lambda d: d["result"]["pricing_wall_s"]),
                  ("Rounding / LNS", "#389797", lambda d: d["heuristic_and_canonical_s"]),
                  ("IPs during CG", "#d47b4a", lambda d: d["result"]["iteration_ip_wall_s"]),
                  ("Final IP search", "#865999", lambda d: d["final_ip_native_s"])]
    bottoms = [0.0] * len(data)
    for label, color, value in components:
        heights = [value(d) / 60 for d in data.values()]
        axes[1].bar(range(3), heights, bottom=bottoms, label=label, color=color, width=.62)
        bottoms = [a + b for a, b in zip(bottoms, heights)]
    totals = [d["result"]["simulation_wall_s"] / 60 for d in data.values()]
    other = [a - b for a, b in zip(totals, bottoms)]
    assert min(other) > -1e-3, "runtime components overlap"
    axes[1].bar(range(3), other, bottom=bottoms, color="#d8dcdf", label="Setup / LP / other", width=.62)
    for i, total in enumerate(totals):
        axes[1].text(i, total + max(totals)*.015, f"{total:.1f} min", ha="center", fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=["Rounding", "LNS", "IP each round"],
                ylabel="Wall time (minutes)", title="Runtime by solver stage", ylim=(0, max(totals)*1.35))
    axes[1].legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    first = next(iter(data.values()))["result"]
    fig.suptitle(f"FAA density · {first['n_requests']:,} outbound flights · "
                 f"{first['demand_duration_s']:g} s demand window · seed {first['seed']}", fontsize=13)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


def report(data, pairwise, manifest, out):
    r = data["baseline"]["result"]
    lines = ["# FAA density: CG primal-policy comparison", "",
        f"Three fresh sequential runs on {r['n_requests']:,} outbound flights generated over "
        f"{r['demand_duration_s']:g} seconds, seed {r['seed']}, with {r['n_static_terminals']} "
        "static terminals. Every final schedule passed physical conflict verification.", "",
        "All three policies retain ground-delay initialization and a final IP with a 600-second "
        "native search allowance and a 660-second reserve. CG + IP also solves the full pool "
        "after each pricing sweep, preparing all bindable capacity rows before a 30-second "
        "search cap. CG + rounding and CG + LNS run their respective heuristics during CG. "
        "All use four pricing workers, four Gurobi threads, at most 30 CG rounds, a 0.01% LP "
        "cost-gap target and zero native IP gap target.", "",
        "| Metric | " + " | ".join(LABELS.values()) + " |", "|---|---:|---:|---:|"]
    metrics = [
        ("Total simulation, min", lambda d: d["result"]["simulation_wall_s"]/60),
        ("Final congestion cost", lambda d: d["result"]["congestion_cost"]),
        ("Final cost including denial penalties", lambda d: d["result"]["penalized_congestion_cost"]),
        ("Accepted flights", lambda d: d["result"]["accepted"]),
        ("Denied flights", lambda d: d["result"]["denied"]),
        ("Pre-final-IP cost including denial penalties", lambda d: d["result"]["heuristic_cost"]),
        ("Total ground delay, s", lambda d: d["result"]["ground_delay_s"]),
        ("CG rounds", lambda d: d["result"]["iterations"]),
        ("LP cost gap, %", lambda d: 100*d["lp_gap_cost"]),
        ("Global integer cost gap, %", lambda d: 100*d["result"]["global_cost_gap"]),
        ("Termination", lambda d: d["result"]["termination"]),
        ("Final IP status", lambda d: d["result"]["ip_status"]),
        ("Pricing, s", lambda d: d["result"]["pricing_wall_s"]),
        ("Rounding/LNS and canonicalization, s", lambda d: d["heuristic_and_canonical_s"]),
        ("Per-round IP wall, s", lambda d: d["result"]["iteration_ip_wall_s"]),
        ("Per-round native IP search, s", lambda d: d["iteration_ip_native_s"]),
        ("Final native IP search, s", lambda d: d["final_ip_native_s"]),
        ("Final pool columns", lambda d: d["result"]["columns"]),
        ("Unique pricing columns added", lambda d: d["actual_priced_columns"]),
        ("Pricing wall seconds per added column", lambda d: d["pricing_s_per_added_column"]),
        ("Pricing label-pool restarts", lambda d: d["kernel_label_restarts"]),
        ("Compiled-kernel fallbacks", lambda d: d["result"]["kernel_fell_back"]),
        ("Repair flights added after IP", lambda d: d["repair_added"]),
    ]
    def fmt(value):
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:,.4f}" if isinstance(value, float) else str(value)
    for label, value in metrics:
        lines.append("| " + label + " | " + " | ".join(fmt(value(d)) for d in data.values()) + " |")
    lines += ["", "## Comparisons", ""]
    for a, b in itertools.combinations(data, 2):
        ra, rb = data[a]["result"], data[b]["result"]
        lines.append(f"- {LABELS[b]} versus {LABELS[a]}: "
                     f"{rb['simulation_wall_s']/ra['simulation_wall_s']:.3f}× total time; "
                     f"{100*(rb['penalized_congestion_cost']/ra['penalized_congestion_cost']-1):+.4f}% "
                     "penalized cost.")
    lines += ["", "Wall-time ratios compare these completed runs; different final gaps or round-cap "
              "exits are not evidence of speedup at equal convergence. Each policy has one timing "
              "sample on one seed. IP-added rows and LP/MIP transitions can change duals and "
              "pricing columns, so these are complete policy comparisons.", "",
              "## Time to reach each final cost", "",
              "Only completed, capacity-checked outputs are counted; values in minutes. A dash "
              "means the run did not reach that target. The seed cost at time zero in the plot "
              "is a reference, not a measured availability time.", "",
              "| Running method / target | " + " | ".join(LABELS.values()) + " |", "|---|---:|---:|---:|"]
    for version, d in data.items():
        values = [d["time_to_final_cost_s"][v] for v in data]
        lines.append("| " + LABELS[version] + " | " + " | ".join("—" if v is None else f"{v/60:.2f}"
                                                                               for v in values) + " |")
    lines += ["", "## Audit", "",
              "Matched exact requests, configuration and all parameters except primal policy; "
              "verified identical source hashes, source archive, harness, tracer and patch. "
              "Recomputed cost from filed trajectories and checked against the solver objective. "
              "Audited IP selections, coverage, budgets, monotonic incumbent cost, and bounds. "
              "Congestion cost excludes mandatory climb/descent. Denied flights incur the "
              "configured cancellation penalty. Restricted-IP optimality does not establish "
              "global integer optimality.", "",
              f"Environment: {manifest['packages']}; {manifest['platform']}.", ""]
    for p in pairwise:
        lines.append(f"- {LABELS[p['a']]} / {LABELS[p['b']]} pools: {p['shared_columns']:,} "
                     f"shared timed columns; {p['a_only']:,} / {p['b_only']:,} unique to each.")
    lines += ["", "![Measured runtime and quality](comparison.png)", "",
              "## IP after every pricing round", "",
              "| Round | Available, min | Feasible penalized cost | Covered | Native search, s | Setup, s | Status |",
              "|---:|---:|---:|---:|---:|---:|---|"]
    for c in data["iteration_ip_eager"]["per_round_ip"]:
        lines.append(f"| {c['sweep']} | {c['available_s']/60:.2f} | {c['cost']:,.3f} | "
                     f"{c['covered']} | {c['native_s']:.3f} | "
                     f"{c['setup_s']:.3f} | {c['status']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.out or args.root / "summary")
