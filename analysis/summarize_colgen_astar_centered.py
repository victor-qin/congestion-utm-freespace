"""Audit and compare FCFS A* centered seeds with normal CG initialization."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt

from summarize_colgen_policies import close, load_case, read


LABELS = {"iteration_ip_astar_centered": "A* + centered ladder",
          "iteration_ip_eager": "Nominal + later ladder + greedy"}
COLORS = {"iteration_ip_astar_centered": "#ad5b27", "iteration_ip_eager": "#285879"}


def summarize(root):
    manifest, results = read(root / "manifest.json"), read(root / "results.json")
    assert len(results) == 2 and {r["version"] for r in results} == set(LABELS)
    hashes = [manifest[f"{v}_source_sha256"] for v in LABELS]
    assert hashes[0] == hashes[1]
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for path, digest in hashes[0].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest
    for file, key in (("harness_used.py", "harness_sha256"),
                      ("colgen_trace_used.py", "tracer_sha256"), ("changes.patch", "patch_sha256")):
        assert hashlib.sha256((root / file).read_bytes()).hexdigest() == manifest[key]
    loaded = {r["version"]: load_case(root, r) for r in results}
    assert all(item[0] == loaded["iteration_ip_eager"][0] for item in loaded.values())
    data = {v: loaded[v][2] for v in LABELS}
    assert manifest["workers"] == 12 and manifest["gurobi_threads"] == 4
    assert manifest["lp_gap"] == .001 and manifest["outbound_only"]
    for version, d in data.items():
        r = d["result"]
        case = root / f"{r['scenario']}_seed{r['seed']}_{version}"
        inputs, stats, selections, rows, calls = [read(case / name) for name in (
            "inputs.json", "planner_stats.json", "selections.json", "iterations.json", "ip_calls.json")]
        columns = [json.loads(line) for line in (case / "columns.jsonl").read_text().splitlines()]
        assert r["n_requests"] == 1537 and r["n_generated"] == 3074
        assert r["gurobi_threads"] == 4 and r["verified"]
        initial = [c for c in columns if c["phase"] == "initialization"]
        first = {}
        for c in initial:
            first.setdefault(c["flight_id"], c)

        def category(c):
            if c["phase"] == "pricing":
                return "priced"
            origin = first[c["flight_id"]]
            shift = c["departure_step"] - origin["departure_step"]
            assert c["path_id"] == origin["path_id"]
            if version == "iteration_ip_astar_centered":
                return "A* center" if shift == 0 else "earlier A* variant" if shift < 0 else "later A* variant"
            return "nominal" if shift == 0 else "later ladder" if shift <= 20 else "greedy beyond ladder"

        def usage(selection):
            selected = [columns[i] for i in selection.values()]
            return dict(counts=dict(Counter(category(c) for c in selected)),
                        shifts_steps=dict(sorted(Counter(c["departure_step"] - first[c["flight_id"]]["departure_step"]
                            for c in selected if c["phase"] == "initialization").items())))

        n, benefit = r["n_requests"], inputs["params"]["M"]
        initial_cost = math.fsum(columns[i]["cost"] for i in selections["initial"].values())
        initial_penalized = initial_cost + benefit * (n - len(selections["initial"]))
        events = [(selections["initial_available_s"], initial_penalized)]
        events += [(row["simulation_elapsed_s"], row["heuristic_cost"]) for row in rows]
        events += [(c["start_elapsed_s"] + c["wall_s"], c["after_cost"] + benefit * (n - len(c["after_selection"]))) for c in calls]
        events.append((r["simulation_wall_s"], r["penalized_congestion_cost"]))
        full_service_events = [(c["start_elapsed_s"] + c["wall_s"], c["after_cost"])
                               for c in calls if len(c["after_selection"]) == n]
        if len(selections["initial"]) == n:
            full_service_events.append((selections["initial_available_s"], initial_cost))
        if r["accepted"] == n:
            full_service_events.append((r["simulation_wall_s"], r["congestion_cost"]))
        curve, best = [], math.inf
        for elapsed, value in sorted(events):
            best = min(best, value)
            curve.append((elapsed, best))
        round_calls = [c for c in calls if c["phase"] == "iteration"]
        assert len(round_calls) == len(rows)
        assert all(c["requested_budget_s"] == 30 for c in round_calls)
        assert all(row["pricing_exact"] for row in rows)
        rc_residuals = []
        for row in rows:
            inserted_rc = math.fsum(c["reduced_cost"] for c in columns
                                   if c["phase"] == "pricing" and c["sweep"] == row["iteration"])
            residual = row["rc_sum"] - inserted_rc
            assert abs(residual) < 1e-3, (row["iteration"], residual)
            rc_residuals.append(residual)
        assert d["trace"]["rounding_calls"] == d["trace"]["lns_calls"] == 0
        close(initial_cost, stats["initial_heuristic_delay_s"])
        assert len(selections["initial"]) == stats["initial_heuristic_flights"]
        if version == "iteration_ip_astar_centered":
            assert stats["initial_seed_columns"] == 0
            assert not r["ground_bootstrap"]["enabled"]
            assert not stats["initial_greedy_completed"]
            assert r["astar_seed"]["max_shift_steps"] == 0
            assert not r["astar_seed"]["require_joint_feasibility"]
            assert len(first) == r["astar_seed"]["converted_flights"]
            request = {q["flight_id"]: q for q in inputs["requests"]}
            dt = inputs["config"]["dt_s"]
            for fid, origin in first.items():
                actual = {c["departure_step"] for c in initial if c["flight_id"] == fid}
                base = math.ceil(request[fid]["t_departure"] / dt)
                latest = base + int(inputs["config"]["max_ground_delay_s"] / dt)
                center = origin["departure_step"]
                assert actual == set(range(max(base, center - 10), min(latest, center + 10) + 1))
        d.update(curve=curve, full_service_curve=sorted(full_service_events), initial_pool_columns=len(initial),
                 initial_covered=len(selections["initial"]), initial_cost=initial_penalized,
                 initial_available_s=selections["initial_available_s"],
                 positive_rc_not_in_added_columns=rc_residuals,
                 initial_column_usage=usage(selections["initial"]),
                 final_column_usage=usage(selections["final"]),
                 round_column_usage=[dict(round=c["sweep"], covered=len(c["after_selection"]),
                                         **usage(c["after_selection"])) for c in round_calls],
                 iterations=[{k: row[k] for k in ("iteration", "simulation_elapsed_s", "sweep_s", "columns_added", "lp_gap_cost", "heuristic_cost")} for row in rows],
                 final_ip_wall_s=sum(c["wall_s"] for c in calls if c["phase"] == "final"))
    treatment, control = (data[v] for v in LABELS)
    deltas = dict(wall_speedup=control["result"]["simulation_wall_s"] / treatment["result"]["simulation_wall_s"],
                  cost_change_pct=100*(treatment["result"]["penalized_congestion_cost"] / control["result"]["penalized_congestion_cost"] - 1))
    targets = {v: d["result"]["penalized_congestion_cost"] for v, d in data.items()}
    for d in data.values():
        d["time_to_final_cost_s"] = {v: next((t for t, c in d["curve"] if c <= cost + 1e-5), None)
                                     for v, cost in targets.items()}
        d["time_to_quality_s"] = {str(cost): next((t for t, c in d["full_service_curve"]
                                                  if c <= cost + 1e-5), None)
                                    for cost in (160000, 158000, 157000)}
    out = root / "summary"
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(dict(inputs_and_sources_match=True,
        source_archive_verified=True, physical_verification_passed=True, costs_recomputed=True,
        runs=data, deltas=deltas), indent=2) + "\n")
    plot(data, out)
    report(data, deltas, out)
    print(json.dumps(deltas, indent=2))


def plot(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    for version, d in data.items():
        xs, ys = zip(*d["full_service_curve"])
        axes[0].step([x/60 for x in xs], [y/1000 for y in ys], where="post", lw=2,
                     color=COLORS[version], label=LABELS[version])
        axes[0].scatter(xs[-1]/60, ys[-1]/1000, color=COLORS[version], s=25)
        source = d["result"]["astar_seed"]
        if source:
            axes[0].scatter(source["source_wall_s"]/60, source["source_cost"]/1000,
                            marker="D", s=45, color="#777777", label="Standalone A* reference", zorder=4)
    axes[0].set(xlabel="Elapsed simulation time (minutes)", ylabel="Best feasible cost (thousands)",
                title="CG pool schedules serving all 1,537 flights")
    axes[0].legend(frameon=False, fontsize=8)
    components = [("A* planning + conversion", "#d89d73", lambda d: sum(d["result"]["astar_seed"].get(k, 0) for k in ("source_wall_s", "conversion_wall_s"))),
                  ("Pricing", "#7395b4", lambda d: d["result"]["pricing_wall_s"]),
                  ("IP during CG", "#75856b", lambda d: d["result"]["iteration_ip_wall_s"]),
                  ("Final IP", "#9d83ae", lambda d: d["final_ip_wall_s"]),
                  ("LP solves", "#c6b35f", lambda d: d["lp_wall_s"])]
    bottom = [0., 0.]
    for label, color, metric in components:
        values = [metric(d)/60 for d in data.values()]
        axes[1].bar(range(2), values, bottom=bottom, width=.6, label=label, color=color)
        bottom = [a+b for a, b in zip(bottom, values)]
    totals = [d["result"]["simulation_wall_s"]/60 for d in data.values()]
    residual = [a-b for a, b in zip(totals, bottom)]
    assert min(residual) > -1e-3
    axes[1].bar(range(2), residual, bottom=bottom, width=.6, label="Setup / other", color="#d8dcdf")
    for i, total in enumerate(totals):
        axes[1].text(i, total + max(totals)*.02, f"{total:.1f} min", ha="center")
    axes[1].set(xticks=range(2), xticklabels=["A* centered", "Normal seeds"],
                ylabel="Wall time (minutes)", ylim=(0, max(totals)*1.4), title="Complete runtime")
    axes[1].legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("1,537 outbound flights · FAA 1,200 s demand · 12 workers / 4 threads · CG + IP each round")
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for version, d in data.items():
        ax.plot([row["simulation_elapsed_s"]/60 for row in d["iterations"]],
                [max(1e-7, row["lp_gap_cost"]*100) for row in d["iterations"]],
                marker="o", markersize=4, color=COLORS[version], label=LABELS[version])
    ax.axhline(.1, ls="--", color="#555555", label="0.1% LP target")
    ax.set(xlabel="Elapsed simulation time (minutes)", ylabel="Certified LP gap (%)", yscale="log")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2, which="both")
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)


def report(data, deltas, out):
    versions = list(LABELS)
    measures = [
        ("Total wall time (s)", lambda d: f"{d['result']['simulation_wall_s']:.2f}"),
        ("Final penalized cost", lambda d: f"{d['result']['penalized_congestion_cost']:.3f}"),
        ("Accepted / verified", lambda d: f"{d['result']['accepted']} / {d['result']['verified']}"),
        ("CG rounds / stop", lambda d: f"{d['result']['iterations']} / {d['result']['termination']}"),
        ("Initial pool columns", lambda d: str(d['initial_pool_columns'])),
        ("Final pool columns", lambda d: str(d['result']['columns'])),
        ("Pricing time (s)", lambda d: f"{d['result']['pricing_wall_s']:.2f}"),
        ("Per-round IP wall / native (s)", lambda d: f"{d['result']['iteration_ip_wall_s']:.2f} / {d['iteration_ip_native_s']:.2f}"),
        ("Final IP wall / native (s)", lambda d: f"{d['final_ip_wall_s']:.2f} / {d['final_ip_native_s']:.2f}"),
        ("Final restricted IP optimal", lambda d: str(d['final_ip_optimal'])),
        ("Certified LP gap (%)", lambda d: f"{d['lp_gap_cost']*100:.5f}"),
        ("Global integer gap (%)", lambda d: f"{d['result']['global_cost_gap']*100:.4f}"),
        ("Ground delay (s)", lambda d: f"{d['result']['ground_delay_s']:.1f}"),
    ]
    treatment, control = (data[v] for v in versions)
    ratio = treatment["result"]["simulation_wall_s"] / control["result"]["simulation_wall_s"]
    runtime_word = "longer" if ratio > 1 else "shorter"
    lines = ["# A* centered initialization benchmark", "",
             f"Replacing nominal seeds with FCFS A* routes and centered departure alternatives took {abs(ratio-1)*100:.1f}% {runtime_word} in this paired run. Final cost changed by {deltas['cost_change_pct']:+.4f}%.", "",
             f"A* centered met the 0.1% LP target: {treatment['cg_lp_target_met']}. Normal initialization met it: {control['cg_lp_target_met']}. Pool-IP optimality is reported separately below.", "",
             "| Metric | A* centered | Normal initialization |", "|---|---:|---:|"]
    for label, value in measures:
        lines.append("| " + label + " | " + " | ".join(value(data[v]) for v in versions) + " |")
    lines += ["", f"Normal/A* elapsed-time ratio: {deltas['wall_speedup']:.3f}x. A* final cost change: {deltas['cost_change_pct']:+.4f}%.", "",
              "These are one paired seed-0 run, not repeated timing estimates. Both arms use the same requests, source snapshot, 12 pricing workers and 4 Gurobi threads. A* planning and conversion are included in wall time; the CG budget begins after that prepass. The 0.1% target applies to the LP certificate, not integer optimality.", ""]
    lines += ["## Time to fully served solution quality", "",
              "| Cost at most | A* centered (min) | Normal (min) |", "|---:|---:|---:|"]
    for cost in ("160000", "158000", "157000"):
        values = [data[v]["time_to_quality_s"][cost] for v in versions]
        lines.append("| " + cost + " | " + " | ".join(f"{t/60:.2f}" if t is not None else "Not reached" for t in values) + " |")
    lines.append("")
    for v, d in data.items():
        lines += [f"## {LABELS[v]}", "", "Final selected-column provenance:", "",
                  "| Origin | Selected columns |", "|---|---:|"]
        lines += [f"| {origin} | {count} |" for origin, count in d["final_column_usage"]["counts"].items()]
        lines += ["", "Time to each arm's final cost, including initialization:", "",
                  "| Target | Minutes |", "|---|---:|"]
        lines += [f"| {LABELS[target]} final cost | {seconds/60:.2f} |" if seconds is not None
                  else f"| {LABELS[target]} final cost | Not reached |"
                  for target, seconds in d["time_to_final_cost_s"].items()]
        lines.append("")
        if d["result"]["astar_seed"]:
            a = d["result"]["astar_seed"]
            reasons = Counter()
            for reason, count in a["conversion_stats"].items():
                if reason.startswith("unconvertible:"):
                    key = "Outside the CG corridor" if "outside the flight graph corridor" in reason else reason.removeprefix("unconvertible: ")
                    reasons[key] += count
            lines += [f"FCFS A* itself accepted and verified {a['source_accepted']} flights in {a['source_wall_s']:.2f} seconds, at cost {a['source_cost']:.3f}. Conversion took {a['conversion_wall_s']:.2f} seconds and retained {a['converted_flights']} routes. Their centers overloaded {a['conversion_stats']['rows over cap']} CG rows, so they were pool alternatives rather than an initial incumbent.", "",
                      "| Import rejection | Flights |", "|---|---:|"]
            lines += [f"| {reason} | {count} |" for reason, count in reasons.items()]
            lines.append("")
    lines += ["The A* arm imports all individually valid centers without a greedy feasibility pass. It adds up to ten earlier and ten later four-second alternatives per center. Earlier alternatives stop at requested departure. Airborne holds and routes outside the CG domain cannot be imported; pricing may recover those flights. Original A* centers are used as an incumbent only if jointly feasible under CG rows.", "",
              "The standalone A* marker is an operationally verified reference, available after its prepass. It contains holds and routes outside the CG domain, so it is not counted as a feasible CG-pool incumbent or used in the CG time-to-target calculations.", "",
              "Input/source hashes, native IP budgets, column uniqueness, initial ladder bounds, objective recomputation and physical verification were audited. See audit.json for per-round column usage and solution availability.", "",
              f"![Runtime and quality]({out / 'comparison.png'})", "",
              f"![LP convergence]({out / 'convergence.png'})"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root.resolve())
