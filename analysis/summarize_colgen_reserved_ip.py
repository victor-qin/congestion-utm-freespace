"""Audit reserved-IP experiments and illustrate actual priced/selected columns."""

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
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def case_data(directory):
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
    inputs, trace = read(case / "inputs.json"), read(case / "trace_summary.json")
    columns, paths = jsonl(case / "columns.jsonl"), jsonl(case / "paths.jsonl")
    calls, ip_calls = read(case / "lns_calls.json"), read(case / "ip_calls.json")
    assert result["verified"] and result["accepted"] == result["n_requests"]
    assert result["n_generated"] == 1526
    expected_requests = 763 if result.get("outbound_only") else 1526
    assert result["n_requests"] == expected_requests
    if result.get("outbound_only"):
        assert all(r["paired_outbound_id"] is None for r in inputs["requests"])
    assert result["demand_duration_s"] == 600
    assert trace["rounding_calls"] == 0
    assert len(ip_calls) == 1
    assert len(columns) == trace["columns"]
    assert all(column["index"] == index for index, column in enumerate(columns))
    assert abs(result["congestion_cost_minus_solver_objective"]) < 1e-5
    selections = read(case / "selections.json")
    ip = ip_calls[0]
    before, after = ip["before_selection"], ip["after_selection"]
    assert len(before) == result["n_requests"]
    stats = read(case / "planner_stats.json")
    with gzip.open(case / "trajectories.json.gz", "rt") as handle:
        trajectories = {str(t["flight_id"]): t for t in json.load(handle)}
    cfg = inputs["config"]
    actual_costs = {
        fid: cfg["cost_ground_delay_per_s"] * t["ground_delay_s"]
        + cfg["cost_air_hold_per_s"] * t["air_hold_s"]
        + cfg["cost_air_lateral_per_s"] * t["air_detour_m"] / cfg["nominal_speed_mps"]
        for fid, t in trajectories.items()}
    assert abs(sum(actual_costs.values()) - result["congestion_cost"]) < 1e-5
    assert all(abs(actual_costs[fid] - columns[index]["cost"]) < 1e-5
               for fid, index in after.items()), "Final repair changed an IP-selected route"
    # Filing may interpolate mid-edge points. Every cruise lattice vertex must
    # nevertheless remain in order in the final polyline (ignore repeated holds).
    spacing = cfg["nominal_speed_mps"] * cfg["dt_s"]
    for fid, index in after.items():
        filed = iter(tuple(round(v, 4) for v in point[:2])
                     for point, _time in trajectories[fid]["centerline"])
        previous = None
        for q, r in paths[columns[index]["path_id"]]["cells"]:
            point = (round(spacing * (q + r / 2), 4),
                     round(spacing * math.sqrt(3) / 2 * r, 4))
            if point != previous:
                assert any(actual == point for actual in filed), fid
            previous = point
    repaired = sorted(set(trajectories) - set(after), key=int)
    assert len(repaired) == stats["repair_added"]
    assert abs(ip["after_cost"] + sum(actual_costs[fid] for fid in repaired)
               - result["congestion_cost"]) < 1e-5
    raw_ip_penalized_cost = ip["after_cost"] + len(repaired) * inputs["params"]["M"]
    last_lns_pool = calls[-1]["stats"]["n_columns"]
    native = ip["native_calls"]
    assert native and native[0]["time_limit_s"] >= inputs["params"]["ip_time_limit_s"] - 1e-3
    assert all(c["phase"] != "pricing" or c["reduced_cost"] is not None for c in columns)
    demand = {k: v for k, v in inputs.items() if k != "params"}
    demand_sha = hashlib.sha256(json.dumps(demand, sort_keys=True).encode()).hexdigest()
    completed = {row["iteration"] for row in read(case / "column_diversity.json")}
    diversity = []
    for sweep in sorted({c["sweep"] for c in columns if c["phase"] == "pricing"}):
        added = [c for c in columns if c["phase"] == "pricing" and c["sweep"] == sweep]
        diversity.append(dict(iteration=sweep, completed=sweep in completed,
                              priced_columns=len(added),
                              new_spatial_routes=sum(c["new_geometry"] for c in added),
                              timing_variants=sum(not c["new_geometry"] for c in added),
                              lns_selected=sum(columns[i]["sweep"] == sweep
                                               and columns[i]["phase"] == "pricing"
                                               for i in before.values()),
                              ip_selected=sum(columns[i]["sweep"] == sweep
                                              and columns[i]["phase"] == "pricing"
                                              for i in after.values())))
    summary = dict(
        name=directory.name, demand_sha=demand_sha, result=result,
        source_sha=hashlib.sha256(json.dumps(manifest["fixed_source_sha256"],
                                          sort_keys=True).encode()).hexdigest(),
        archived_sources_verified=True,
        raw_final_lp_gap=stats["lp_gap_cost"],
        final_lp_gap=min(1.0, max(0.0, stats["lp_gap_cost"])),
        lns_cost=ip["before_cost"], ip_cost=result["congestion_cost"],
        raw_ip_cost=ip["after_cost"], raw_ip_penalized_cost=raw_ip_penalized_cost,
        raw_ip_coverage=len(after), repaired_flights=repaired,
        lns_pool_columns=last_lns_pool,
        columns_added_after_last_lns=len(columns) - last_lns_pool,
        ip_selected_after_last_lns=sum(i >= last_lns_pool for i in after.values()),
        raw_ip_improvement_pct=100 * (1 - raw_ip_penalized_cost / ip["before_cost"]),
        ip_improvement_pct=100 * (1 - result["congestion_cost"] / ip["before_cost"]),
        native_ip_limit_s=native[0]["time_limit_s"],
        native_ip_runtime_s=sum(c.get("gurobi_runtime_s", c["wall_s"]) for c in native),
        ip_setup_s=ip["setup_s"], native_ip_status=ip["status"],
        lns_selected_priced_columns=sum(columns[i]["phase"] == "pricing" for i in before.values()),
        ip_selected_priced_columns=sum(columns[i]["phase"] == "pricing" for i in after.values()),
        lns_geometry_changes_from_initial=sum(
            columns[i]["geometry_id"] != columns[selections["initial"][fid]]["geometry_id"]
            for fid, i in before.items()),
        ip_geometry_changes_from_initial=sum(
            columns[i]["geometry_id"] != columns[selections["initial"][fid]]["geometry_id"]
            for fid, i in after.items()),
        trace=trace, diversity=diversity,
        lns_calls=[dict(call=c["call"], cost=c["after_cost"], changed_flights=len(c["changed"]),
                        geometry_changes=c["geometry_changes"],
                        unique_sequential_trials=c["stats"]["n_unique_trial_solutions"],
                        sequential_trials=len(c["stats"]["try_objectives"]),
                        joint_flights=c["stats"].get("joint_repair", {}).get("flights", 0),
                        joint_columns=c["stats"].get("joint_repair", {}).get("columns", 0),
                        joint_swaps=c["stats"].get("joint_repair", {}).get("n_swapped", 0))
                   for c in calls],
    )
    return dict(directory=directory, case=case, summary=summary, inputs=inputs,
                columns=columns, paths=paths, ip=ip, selections=selections, lns=calls)


def examples(data, out, *, lns_examples=False):
    columns, paths, ip = data["columns"], data["paths"], data["ip"]
    initial = data["selections"]["initial"]
    before, after = ip["before_selection"], ip["after_selection"]
    requests = {str(r["flight_id"]): r for r in data["inputs"]["requests"]}
    cfg = data["inputs"]["config"]
    dt, speed = cfg["dt_s"], cfg["nominal_speed_mps"]
    target = before if lns_examples else after
    comparison = initial if lns_examples else before
    candidates = [fid for fid, index in target.items()
                  if columns[index]["phase"] == "pricing" and fid in after]
    candidates.sort(key=lambda fid: (
        columns[target[fid]]["geometry_id"] != columns[initial[fid]]["geometry_id"],
        columns[comparison[fid]]["cost"] - columns[target[fid]]["cost"],
    ), reverse=True)
    chosen = candidates[:3]
    assert chosen, "No newly priced columns were selected"
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 3.5 * len(chosen)), squeeze=False)
    colors = ("#738297", "#087F8C", "#C4772C")
    labels = ("Ground seed", "LNS before IP", "Final schedule")
    records = []
    for row, fid in enumerate(chosen):
        indices = [initial[fid], before[fid], after[fid]]
        request = requests[fid]
        nominal_step = math.ceil(request["t_departure"] / dt)
        ground = [(columns[i]["departure_step"] - nominal_step) * dt for i in indices]
        costs = [columns[i]["cost"] for i in indices]
        ax, timing = axes[row]
        for index, color, label, style, width in zip(indices, colors, labels,
                                                    ("-", "--", "-"), (4, 2.5, 1.5)):
            cells = np.asarray(paths[columns[index]["path_id"]]["cells"])
            x = speed * dt * (cells[:, 0] + cells[:, 1] / 2) / 1000
            y = speed * dt * math.sqrt(3) / 2 * cells[:, 1] / 1000
            ax.plot(x, y, style, color=color, linewidth=width, alpha=.9, label=label)
            ax.scatter([x[0], x[-1]], [y[0], y[-1]], color=color, s=18)
        new = columns[target[fid]]
        selected_by = "LNS" if lns_examples else "IP"
        ax.set_title(f"Flight {fid}: {selected_by} column {new['index']} from sweep {new['sweep']}",
                     loc="left", fontsize=11, weight="bold")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        old_cells = {tuple(c) for c in paths[columns[initial[fid]]["path_id"]]["cells"]}
        changed_cells = [c for c in paths[new["path_id"]]["cells"] if tuple(c) not in old_cells]
        if changed_cells:
            q, r = changed_cells[len(changed_cells) // 2]
            zx, zy = speed * dt * (q + r / 2) / 1000, speed * dt * math.sqrt(3) / 2 * r / 1000
            zoom = ax.inset_axes([.035, .57, .29, .34])
            for index, color, style in zip(indices, colors, ("-", "--", "-")):
                cells = np.asarray(paths[columns[index]["path_id"]]["cells"])
                zoom.plot(speed * dt * (cells[:, 0] + cells[:, 1] / 2) / 1000,
                          speed * dt * math.sqrt(3) / 2 * cells[:, 1] / 1000,
                          style, color=color, linewidth=2)
            zoom.set(xlim=(zx - .4, zx + .4), ylim=(zy - .4, zy + .4), xticks=[], yticks=[])
            zoom.set_title("Route detail: 800 m square", fontsize=7)
            zoom.set_aspect("equal")
            ax.scatter([zx], [zy], s=80, facecolors="none", edgecolors="black", linewidth=.8)
        bars = timing.barh(np.arange(3), ground, color=colors, height=.55)
        timing.bar_label(bars, labels=[f"{delay:.0f} s; cost {cost:,.1f}"
                                      for delay, cost in zip(ground, costs)], padding=5, fontsize=10)
        timing.set_yticks(np.arange(3), labels)
        timing.invert_yaxis()
        timing.set_xlim(0, max(max(ground), 10) * 1.75)
        timing.set_xlabel("Ground delay (seconds)")
        rc = new["reduced_cost"]
        rc_label = f"{rc:.3g}" if abs(rc) < .01 else f"{rc:,.2f}"
        timing.set_title(f"Reduced cost when added: {rc_label}",
                         loc="left", fontsize=11)
        for panel in (ax, timing):
            panel.spines[["top", "right"]].set_visible(False)
            panel.grid(alpha=.15)
            panel.set_axisbelow(True)
        records.append(dict(flight_id=int(fid), seed=columns[indices[0]],
                            lns=columns[indices[1]], final_ip=columns[indices[2]],
                            ground_delay_s=ground, costs=costs))
    fig.suptitle(f"Actual columns generated by pricing and selected by {selected_by}",
                 x=.025, ha="left", fontsize=17, weight="bold")
    fig.text(.025, .945, f"{data['directory'].name} run · {len(requests):,} flights · cruise lattice paths shown; terminal "
             "connectors omitted · each choice was used in a complete feasible schedule", fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, .925), h_pad=3, w_pad=3)
    filename = "lns_column_examples" if lns_examples else "column_examples"
    fig.savefig(out / f"{filename}.png", dpi=170, facecolor="white")
    plt.close(fig)
    (out / f"{filename}.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [case_data(path.resolve()) for path in args.directories]
    summaries = [case["summary"] for case in cases]
    assert len({s["demand_sha"] for s in summaries}) == 1
    assert len({s["source_sha"] for s in summaries}) == 1
    (args.out / "audit.json").write_text(json.dumps(summaries, indent=2) + "\n")
    examples(cases[-1], args.out)
    examples(cases[-1], args.out, lns_examples=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for case_index, case in enumerate(cases):
        s, ip = case["summary"], case["ip"]
        color = ("#087F8C", "#C4772C")[case_index % 2]
        x = [c["simulation_elapsed_s"] / 60 for c in case["lns"]]
        y = [c["after_cost"] for c in case["lns"]]
        axes[0].step(x, y, where="post", marker="o", color=color, label=f"{s['name']}: LNS")
        axes[0].scatter([s["result"]["simulation_wall_s"] / 60], [s["ip_cost"]],
                        marker="*", s=160, color=color, label=f"{s['name']}: IP + repair")
        trajectory = ip["incumbent_trajectory"]
        points = [(float(t), len(case["inputs"]["requests"]) * case["inputs"]["params"]["M"]
                   - float(objective)) for t, objective, _ in trajectory
                  if math.isfinite(float(objective)) and abs(float(objective)) < 1e50]
        if points:
            points.append((s["native_ip_runtime_s"], s["raw_ip_penalized_cost"]))
            t, costs = zip(*points)
            axes[1].step(t, costs, where="post", marker=".", color=color, label=s["name"])
    axes[0].set_title("LNS incumbent and final repaired result", loc="left")
    axes[0].set_xlabel("Elapsed minutes")
    axes[1].set_title("Final IP incumbent progress", loc="left")
    axes[1].set_xlabel("Seconds since native IP search started")
    for ax in axes:
        ax.set_ylabel("Congestion cost (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=9)
    axes[1].set_ylabel("Cost + 10,000 per omitted flight")
    fig.suptitle("Column generation with an explicitly reserved 600-second IP allowance",
                 x=.025, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, .01, 1, .90))
    fig.savefig(args.out / "quality.png", dpi=170, facecolor="white")
    plt.close(fig)

    s = summaries[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rows = s["diversity"]
    sweeps = [r["iteration"] for r in rows]
    spatial = [r["new_spatial_routes"] for r in rows]
    timing = [r["timing_variants"] for r in rows]
    axes[0].bar(sweeps, spatial, color="#087F8C", label="New spatial route")
    axes[0].bar(sweeps, timing, bottom=spatial, color="#C4772C", label="Timing variant")
    axes[0].set(xlabel="Pricing sweep (last may be partial)", ylabel="Added columns",
                title="Pricing expands the route pool")
    rows = s["lns_calls"]
    calls = [r["call"] for r in rows]
    unique = [r["unique_sequential_trials"] for r in rows]
    duplicate = [r["sequential_trials"] - r["unique_sequential_trials"] for r in rows]
    axes[1].bar(calls, unique, color="#087F8C", label="Distinct schedule within call")
    axes[1].bar(calls, duplicate, bottom=unique, color="#B8C1CC", label="Repeated trial schedule")
    axes[1].set(xlabel="LNS call", ylabel="Sequential trials",
                title="How often LNS revisits the same schedule")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=.2)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"Observed route and solution diversity: {s['name']} run",
                 x=.025, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(args.out / "diversity.png", dpi=170, facecolor="white")
    plt.close(fig)

    lines = ["# Column generation with a reserved final-IP budget", "",
             "All cases use the full 600-second FAA Wing/Zipline demand, seed 0: 1,526 legs "
             "and 182 static terminals. Gurobi uses four master threads and four pricing "
             "workers. The complete ground-delay seed is retained; no rounding calls ran. "
             "LNS uses 100 released flights, up to 16 sequential trials, and a 0.5-second "
             "joint-repair allowance. Source snapshots, exact inputs and traces accompany each run.",
             "", "| Run | Full sweeps | LNS before IP | Raw IP + omission penalty | Final after repair | Improvement vs LNS | "
             "Native IP time | Global gap | Restricted IP gap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        r = s["result"]
        lines.append(f"| {s['name']} | {r['pricing_sweeps_completed']} | {s['lns_cost']:,.2f} | "
                     f"{s['raw_ip_penalized_cost']:,.2f} | {s['ip_cost']:,.2f} | {s['ip_improvement_pct']:.2f}% | "
                     f"{s['native_ip_runtime_s']:.2f} s | {r['global_cost_gap'] * 100:.3f}% | "
                     f"{float(r['restricted_ip_gap']) * 100:.3f}% |")
    lines.extend(["", "The native IP was checked to receive its full 600-second cap after "
                  "setup. It may terminate sooner on its tolerance/proof. Global pricing "
                  "bounds certify the modeled route universe; restricted IP bounds certify only "
                  "the generated pool. A solved restricted IP does not establish global "
                  "convergence. The LP/global gaps use the known nonnegative-cost lower "
                  "bound when a pricing bound is negative; raw solver gaps are preserved "
                  "in the audit. Each final schedule passed simulation conflict verification.",
                  "", "| Run | Whole simulation | All pricing | All LNS calls | IP setup | Maximum CG iterations | IP gap target | Final LP pricing gap |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for case in cases:
        s, p = case["summary"], case["inputs"]["params"]
        lines.append(f"| {s['name']} | {s['result']['simulation_wall_s']:.2f} s | "
                     f"{s['result']['pricing_wall_s']:.2f} s | "
                     f"{s['trace']['heuristic_wall_s']:.3f} s | {s['ip_setup_s']:.2f} s | "
                     f"{p['max_iterations']} | {p['ip_gap']} | {s['final_lp_gap'] * 100:.4f}% |")
    lines.extend(["", "## How LNS helps", "",
                  "LNS adds no columns. The trace asserts that the column count is unchanged "
                  "across every LNS call. It selects pooled alternatives while preserving "
                  "coverage, supplies a feasible incumbent to the final IP, and can close "
                  "a certified primal gap sooner. Pricing supplies every newly generated "
                  "route. The incumbent also affects pricing order and its known-column hint, "
                  "but a column already in the optimally solved restricted LP has nonpositive "
                  "reduced cost; that hint is not a source of new improving columns.",
                  "", "The separate planner in `freespace_sim/planner/lns/` calls A* or "
                  "SIPP to replan routes. The design note `context/solver_brainstorm_lacam_cbs.md` "
                  "section 4.3 proposes feeding such repaired routes into column generation. "
                  "That connection is not implemented in the restricted-master LNS measured "
                  "here. Canonicalizing those routes and checking their graph, claims and "
                  "active-objective cost would be required before importing them as columns. "
                  "The existing `warm_start.build` and `intent_to_column` helpers provide "
                  "conversion plumbing. Another possibility is reusing the solver's final "
                  "feasibility-pricing repair within a neighborhood so generated routes "
                  "already obey the column-generation graph. Neither integration ran here.",
                  "", "## Route and solution diversity", ""])
    for s in summaries:
        d = s["diversity"]
        lines.extend([f"### {s['name']}", "",
                      f"The final pool contains {s['trace']['columns']:,} columns and "
                      f"{s['trace']['spatial_routes']:,} distinct spatial route families. "
                      "A spatial family includes flight, altitude, terminal lanes and the "
                      "cell sequence with consecutive holds removed; departure shifts and "
                      "hold timing do not count as new spatial routes.", "",
                      f"LNS selected {s['lns_selected_priced_columns']} priced columns; "
                      f"the final IP selected {s['ip_selected_priced_columns']}. "
                      f"Compared with the original ground seed, LNS changed spatial routes "
                      f"for {s['lns_geometry_changes_from_initial']} flights; the IP did so "
                      f"for {s['ip_geometry_changes_from_initial']}. The raw IP covered "
                      f"{s['raw_ip_coverage']:,} flights. Final feasibility pricing restored "
                      f"{len(s['repaired_flights'])} omitted flights: {s['repaired_flights']}. "
                      "Repair routes are reported separately from master-pool diversity. "
                      "Retained IP columns were checked against the final per-flight costs. "
                      f"The raw IP improved penalized cost by {s['raw_ip_improvement_pct']:.2f}%; "
                      "the headline improvement includes the final repair. "
                      f"The last LNS call had {s['lns_pool_columns']:,} columns; "
                      f"{s['columns_added_after_last_lns']:,} were added afterward, and the IP "
                      f"selected {s['ip_selected_after_last_lns']} of those later columns. "
                      "This is a comparison of the actual pipeline stages, not a controlled "
                      "same-pool LNS-versus-IP ablation.", "",
                      "| Sweep | Completed | New columns | New spatial routes | Timing variants | LNS selected | IP selected |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in d:
            lines.append(f"| {row['iteration']} | {'yes' if row['completed'] else 'partial'} | "
                         f"{row['priced_columns']} | "
                         f"{row['new_spatial_routes']} | {row['timing_variants']} | "
                         f"{row['lns_selected']} | {row['ip_selected']} |")
        lines.extend(["", "| LNS call | Cost | Changed flights | Spatial changes | "
                      "Unique sequential trials / tried | Joint options / flights |",
                      "|---|---:|---:|---:|---:|---:|"])
        for row in s["lns_calls"]:
            lines.append(f"| {row['call']} | {row['cost']:,.2f} | {row['changed_flights']} | "
                         f"{row['geometry_changes']} | {row['unique_sequential_trials']}/"
                         f"{row['sequential_trials']} | {row['joint_columns']}/{row['joint_flights']} |")
    lines.extend(["", "Distinct trial schedules do not demonstrate sufficient exploration. "
                  "LNS retains one best incumbent, uses positive-LP-support alternatives "
                  "plus original columns, and fixes the rest of the fleet during repair. "
                  "The final IP can coordinate all flights and use zero-LP-weight columns. "
                  "The IP improvement shows headroom in the overall pipeline, but later "
                  "columns and final repair also contribute; it cannot all be attributed "
                  "to insufficient LNS search.", "",
                  "Two changes worth testing next are admitting a bounded set of low-cost "
                  "zero-LP-weight alternatives and releasing the flights that actually block "
                  "a promising alternative, rather than choosing only by LP disagreement. "
                  "Both expand the repair choices; neither is a demonstrated speedup in "
                  "this experiment. The measured solver was left unchanged for the comparison.", "",
                  "The long-run examples include priced columns with reduced costs around "
                  "1e-8 (numerically near zero) that remove hundreds of seconds of ground "
                  "delay when the final IP coordinates the fleet. LP reduced-cost magnitude "
                  "alone therefore did not identify their eventual integer-schedule value.", "",
                  "The plots show actual selected columns. Each route comes from a "
                  "complete claim-feasible schedule; individual alternatives cannot be "
                  "mixed arbitrarily without rechecking reservations. The experiment uses "
                  "the existing nominal return-anchor convention.", ""])
    lines.extend(["After both measurements, a logging-only correction changed the "
                  "uncertified-IP message to recommend the separate IP time cap and reserve, "
                  "and removed an outdated description of gap scaling. The archived "
                  "measured source is retained; optimization behavior was not changed.", ""])
    (args.out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
