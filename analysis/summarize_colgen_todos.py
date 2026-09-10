"""Summarize completed and partial isolated column-generation experiments."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from benchmark_colgen_lns import dump


COMPARISONS = [("corridor", "baseline"), ("shifts", "corridor"),
               ("bootstrap", "corridor"), ("combined", "corridor"),
               ("combined", "baseline"), ("wider6", "combined"), ("multi3", "combined")]


def read(path, default=None):
    """Read available snapshots; incomplete concurrent writes remain unavailable."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def numeric(value):
    """Return finite observations only; missing values are never replaced by zero."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def total(values):
    """Sum observed finite values, leaving wholly missing measurements unavailable."""
    values = [v for v in values if numeric(v)]
    return math.fsum(values) if values else None


def collect_arm(arm, status):
    """Aggregate durable observations for one arm, including an unfinished attempt."""
    row = dict(name=arm["name"], state=status.get("state", "pending"),
               source_root=arm["root"], source_digest=arm["source_digest"],
               error=status.get("error"))
    if not status.get("output"):
        return row
    output = Path(status["output"])
    cases = sorted(output.glob("*_seed*_iteration_ip_eager"))
    if not cases:
        return row
    case = cases[-1]
    result = read(case / "result.json", {})
    stats = read(case / "planner_stats.json", {})
    iterations = read(case / "iterations.json", [])
    last = iterations[-1] if iterations else {}
    inputs = read(case / "inputs.json", {})
    params = inputs.get("params", {})
    measurement = read(case / "measurement.json", {})
    row.update(case=str(case), result_available=bool(result), rounds=len(iterations),
               result=result, planner_stats=stats, params=params,
               demand_config_sha=result.get("demand_config_sha", measurement.get("demand_config_sha")),
               algorithm_params_sha=result.get("algorithm_params_sha", measurement.get("algorithm_params_sha")),
               accepted=result.get("accepted"), verified=result.get("verified"),
               cost=result.get("penalized_congestion_cost", last.get("heuristic_cost")),
               cost_is_final=bool(result), runtime_s=result.get("simulation_wall_s"),
               observed_elapsed_s=last.get("simulation_elapsed_s"),
               pricing_s=result.get("pricing_wall_s", total(i.get("sweep_s") for i in iterations)),
               lp_gap=stats.get("lp_gap_cost", last.get("lp_gap_cost")),
               global_integer_gap=stats.get("global_integer_gap_cost", result.get("global_cost_gap")),
               restricted_gap=result.get("restricted_ip_gap"),
               termination=result.get("termination"),
               seed_s=stats.get("seed_elapsed_s"), graph_build_s=stats.get("graph_build_elapsed_s"),
               time_to_master_s=stats.get("time_to_master_s"),
               greedy_initialization_s=stats.get("initial_greedy_elapsed_s"),
               initialization=result.get("ground_bootstrap"),
               diagnostics_write_s=total(i.get("flight_diagnostics_write_s") for i in iterations))
    summaries = [read(p, {}) for p in sorted((case / "flight_records").glob("*.summary.json"))]
    metrics = {key for summary in summaries for key in summary.get("metrics", {})}
    row["flight_metric_totals"] = {
        key: total(s.get("metrics", {}).get(key, {}).get("total") for s in summaries)
        for key in sorted(metrics)}
    row["flight_records"] = sum(s.get("records", 0) for s in summaries) if summaries else None
    row["flight_round_summaries"] = summaries
    workers = {}
    for summary in summaries:
        for key, worker in summary.get("workers", {}).items():
            combined = workers.setdefault(key, dict(records=0, task_s=0.0))
            combined["records"] += worker["records"]
            combined["task_s"] += worker["task_s"]
    row["worker_load"] = workers
    calls = read(case / "ip_calls.json", [])
    row["ip_calls"] = calls
    for phase in ("iteration", "final"):
        group = [c for c in calls if c["phase"] == phase]
        row[f"{phase}_ip"] = dict(calls=len(group),
            wall_s=total(c.get("wall_s") for c in group),
            setup_s=total(c.get("setup_s") for c in group),
            native_wall_s=total(n.get("wall_s") for c in group for n in c.get("native_calls", [])),
            native_solver_s=total(n.get("gurobi_runtime_s") for c in group for n in c.get("native_calls", [])))
    row["trace_summary"] = read(case / "trace_summary.json", {})
    row["diversity_rounds"] = read(case / "column_diversity.json", [])
    columns = []
    if (case / "columns.jsonl").exists():
        with (case / "columns.jsonl").open() as handle:
            for line in handle:
                try:
                    columns.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    selection = read(case / "selections.json", {}).get("final")
    by_id = {c["index"]: c for c in columns}
    chosen = [by_id[i] for i in selection.values() if i in by_id] if selection is not None else None
    row["column_counts"] = dict(
        observed_columns=len(columns), unique_spatial_routes=len({c["geometry_id"] for c in columns}),
        unique_timed_paths=len({(c["flight_id"], c["departure_step"], c["path_id"]) for c in columns}),
        pricing_columns=sum(c["phase"] == "pricing" for c in columns),
        final_columns_by_origin=dict(Counter(c["phase"] for c in chosen)) if chosen is not None else None,
        final_spatial_routes=len({c["geometry_id"] for c in chosen}) if chosen is not None else None,
        unique_claim_sets=None,
        limitations="Trace omits capacity claims and does not distinguish winner from extra pricing columns; exact claim diversity and extra-only final usage are unavailable.")
    if not (case / "columns.jsonl").exists():
        row["column_counts"] = {}
    row["iteration_metrics"] = [dict(iteration=i.get("iteration"),
        elapsed_s=i.get("simulation_elapsed_s"), cost=i.get("heuristic_cost"),
        lp_gap=i.get("lp_gap_cost"), integer_gap=i.get("heuristic_gap_cost"),
        pricing_s=i.get("sweep_s"), ip=i.get("iteration_ip")) for i in iterations]
    return row


def comparisons(arms):
    """Compare only completed, matching-demand arms and qualify convergence differences."""
    rows = []
    by_name = {a["name"]: a for a in arms}
    for target, reference in COMPARISONS:
        a, b = by_name.get(target, {}), by_name.get(reference, {})
        row = dict(target=target, reference=reference, available=False)
        if a.get("state") == b.get("state") == "complete" and a.get("result") and b.get("result"):
            row["available"] = bool(a.get("demand_config_sha") and a["demand_config_sha"] == b.get("demand_config_sha"))
            if row["available"]:
                row.update(cost_change_pct=(100 * (a["cost"] / b["cost"] - 1) if b["cost"] else None),
                           runtime_ratio=(a["runtime_s"] / b["runtime_s"] if b["runtime_s"] else None),
                           target_gap=a.get("global_integer_gap"), reference_gap=b.get("global_integer_gap"),
                           target_rounds=a["rounds"], reference_rounds=b["rounds"],
                           note=("Domain correction changes feasible routes; this is not a pure speed comparison."
                                 if reference == "baseline" else
                                 "Compare cost, certification gap, rounds, and termination alongside runtime; no unqualified speed claim."))
        rows.append(row)
    return rows


def fmt(value):
    """Format a compact table cell while making unavailable measurements explicit."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:,.4g}"
    return str(value)


def report(summary):
    """Render the available evidence and its limits as a reviewable Markdown report."""
    arms = summary["arms"]
    lines = ["# Column-generation experiment results", "",
        f"**{'PARTIAL — experiments remain unfinished' if summary['partial'] else 'COMPLETE'}**. Updated {summary['updated']}.", "",
        "Intended protocol: FAA seed 0, original paired stream filtered to 1,537 outbound flights over 1,200 s; 12 pricing workers, 4 Gurobi threads; up to 30 rounds, LP gap 0.001, IP each round (30 s), final IP 600 s, reserve 660 s, total solver limit 14,460 s. Actual parameters and source identities are retained in summary.json.", "",
        "Arms: baseline retains the original corridor; corridor corrects terminal-lane pruning; shifts adds certificate reuse; bootstrap adds A* followed by restricted-DP refinement to corridor; combined includes both optimizations. Wider6 raises the extra-hop limit from 3 to 6, enlarging both spatial and time search domains. Multi3 returns up to three certified surviving sink columns per flight, not exhaustive k-best paths. Each arm has one measured run on one demand seed.", "",
        "Incomplete arms show the last durable callback where available. Missing measurements are marked —, never interpreted as zero. Worker bootstrap/main seconds are summed work across flights, not simulation wall time. Native IP means Gurobi Runtime; phase wall includes setup, separation, and caller processing. Wrapper-only times are also preserved in summary.json.", "",
        "| Arm | State | Accepted / verified | Cost | Wall s | Pricing s | Rounds | LP gap | Integer gap | Restricted gap | Termination |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for a in arms:
        lines.append("| " + " | ".join(fmt(x) for x in [a["name"], a["state"],
            f"{fmt(a.get('accepted'))} / {fmt(a.get('verified'))}", a.get("cost"),
            a.get("runtime_s"), a.get("pricing_s"), a.get("rounds"), a.get("lp_gap"),
            a.get("global_integer_gap"), a.get("restricted_gap"), a.get("termination")]) + " |")
    lines += ["", "| Arm | Seed s | Pre-seed master setup s | Bootstrap worker s | Main worker s | Non-bootstrap task s | Main labels | Bootstrap labels | Round IP native / phase wall s | Final IP native / phase wall s |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for a in arms:
        m = a.get("flight_metric_totals", {})
        ip = a.get("iteration_ip", {})
        final = a.get("final_ip", {})
        lines.append("| " + " | ".join(fmt(x) for x in [a["name"], a.get("seed_s"),
            a.get("time_to_master_s"), m.get("bootstrap_s"),
            total([m.get("compiled_s"), m.get("fallback_s")]),
            m.get("non_bootstrap_task_s"), m.get("n_labels"), m.get("bootstrap_labels"),
            f"{fmt(ip.get('native_solver_s'))} / {fmt(a.get('result', {}).get('iteration_ip_wall_s'))}",
            f"{fmt(final.get('native_solver_s'))} / {fmt(a.get('result', {}).get('ip_wall_s'))}"]) + " |")
    lines += ["", "Non-bootstrap task time includes main search and overhead. Arms without a separate main-search clock show — for that measurement. Per-round timing percentiles, worker loads, labels, initialization details, and IP calls are retained in summary.json.", "",
              "| Comparison | Cost change % | Runtime ratio | Target / reference integer gap | Interpretation |", "|---|---:|---:|---|---|"]
    for c in summary["comparisons"]:
        lines.append(f"| {c['target']} vs {c['reference']} | {fmt(c.get('cost_change_pct'))} | {fmt(c.get('runtime_ratio'))} | {fmt(c.get('target_gap'))} / {fmt(c.get('reference_gap'))} | {c.get('note', 'Waiting for completed, matching-demand results.')} |")
    lines += ["", "| Arm | Pool columns | Spatial routes | Timed paths | Pricing columns | Final columns by origin |", "|---|---:|---:|---:|---:|---|"]
    for a in arms:
        c = a.get("column_counts", {})
        lines.append("| " + " | ".join(fmt(x) for x in [a["name"], c.get("observed_columns"), c.get("unique_spatial_routes"), c.get("unique_timed_paths"), c.get("pricing_columns"), c.get("final_columns_by_origin")]) + " |")
    lines += ["", "The trace records spatial routes and timed paths, but omits capacity claim sets and does not tag extra columns separately from each flight's winning column. Unique claim counts and extra-only final usage cannot be recovered; pricing-origin final usage is reported instead.", "", "| Arm | Demand/config SHA | Source SHA |", "|---|---|---|"]
    lines += [f"| {a['name']} | {fmt(a.get('demand_config_sha'))} | {a['source_digest']} |" for a in arms]
    return "\n".join(lines) + "\n"


def main():
    """Refresh summary.json and report.md without modifying experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).with_name("colgen_todos_1200s_20260910"))
    parser.add_argument("--additional-directory", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifests = []
    selected = {}
    superseded = []
    for directory in [args.directory, *args.additional_directory]:
        manifest = read(directory / "manifest.json", {})
        manifests.append(dict(directory=str(directory.resolve()), manifest=manifest))
        status = read(directory / "status.json", {})
        for arm in manifest.get("arms", []):
            name = arm["name"]
            previous = selected.get(name)
            if previous and previous[1].get("state") == "complete":
                raise ValueError(f"Refusing to replace completed arm {name}")
            if previous and previous[1].get("output"):
                superseded.append(dict(arm=previous[0], status=previous[1]))
            selected[name] = (arm, status.get(name, {}))
    arms = [collect_arm(arm, status) for arm, status in selected.values()]
    summary = dict(updated=datetime.now(timezone.utc).isoformat(),
                   partial=not arms or any(a["state"] != "complete" for a in arms),
                   arms=arms, comparisons=comparisons(arms), manifests=manifests,
                   superseded_attempts=superseded)
    output = args.out or args.directory
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "summary.json", summary)
    rendered = report(summary)
    if superseded:
        rendered += "\nSuperseded attempts remain in summary.json with their source identity and logs. The original heuristic-only bootstrap was interrupted after flight 1640 exhausted 67 million labels and fell back to Python; it is not a completed performance result. The revised bootstrap always refines the heuristic incumbent with restricted DP.\n"
    (output / "report.md").write_text(rendered)
    print(output / "report.md")


if __name__ == "__main__":
    main()
