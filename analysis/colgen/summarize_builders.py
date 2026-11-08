"""Validate builder comparisons and emit reproducible tables/CSV for scientific figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

DATA = Path(__file__).resolve().parent / 'builder_data'


def main() -> None:
    """Check exact preparation parity and summarize only completed raw measurement files."""
    rows, comparisons = [], []
    baseline = json.loads((DATA / 'base-prep-1.json').read_text())
    for path in sorted(DATA.glob('*-prep-*.json')) + sorted(DATA.glob('*-controls.json')):
        if '.launcher.' in path.name or '.process.' in path.name:
            continue
        report = json.loads(path.read_text())
        if 'phases' not in report:
            continue
        control = 'controls' in path.name
        reference = json.loads((DATA / 'base-controls.json').read_text()) if control else baseline
        assert report['capture_sha256'] == reference['capture_sha256']
        assert report['flight_ids'] == reference['flight_ids']
        for phase in report['phases']:
            expected = next(p for p in reference['phases'] if p['label'] == phase['label'])
            assert len(expected['rows']) == len(phase['rows'])
            for old, new in zip(expected['rows'], phase['rows'], strict=True):
                assert old['flight_id'] == new['flight_id']
                assert old['status'] == new['status'] == 'complete'
                assert old['fingerprints'] == new['fingerprints'], (path, phase['label'], new['flight_id'])
                comparisons.append(dict(file=path.name, phase=phase['label'], flight_id=new['flight_id']))
            rows.append(dict(file=path.name, variant=path.name.split('-')[0], control=control,
                             phase=phase['label'], prepare_s=phase['prepare_s'],
                             rss_peak_mib=phase['rss_peak_bytes']/2**20))
    with (DATA / 'preparation.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for control in (False, True):
        for variant in ('base', 'corridor', 'roles', 'geometry', 'endpoint', 'all'):
            for phase in ('cold', 'warm', 'changed', 'changed_warm', 'restored'):
                selected = [r for r in rows if (r['variant'], r['control'], r['phase']) ==
                            (variant, control, phase)]
                if selected:
                    values = [r['prepare_s'] for r in selected]
                    groups.append(dict(variant=variant, control=control, phase=phase, n=len(values),
                                       mean_s=statistics.mean(values), min_s=min(values), max_s=max(values)))
    solves = []
    for path in sorted(DATA.glob('*-solve-w*-*')):
        if not path.is_dir() or not (path/'result.json').exists():
            continue
        result = json.loads((path/'result.json').read_text())
        stats = json.loads((path/'planner_stats.json').read_text())
        rounds = json.loads((path/'rounds.json').read_text())['rounds']
        assert result['passed'], path
        solves.append(dict(name=path.name, elapsed_s=result['elapsed_s'], stats=stats, rounds=rounds))
    if solves:
        matched_keys = ("objective", "final_lp_objective", "master_objective", "upper_bound",
                        "cost_lower_bound", "cost_upper_bound", "iterations", "pricing_flights_completed",
                        "pricing_sweeps_completed", "gpu_dispatches", "pricing_bound_certified",
                        "pricing_search_complete", "kernel_fell_back", "termination_reason")
        reference = solves[0]
        round_keys = ("columns", "columns_added", "lp_objective", "upper_bound", "gpu_dispatches",
                      "pricing_bound_certified", "pricing_search_complete", "rc_max", "rc_sum")
        solve_rows = []
        for run in solves:
            assert {k: run["stats"].get(k) for k in matched_keys} == {
                k: reference["stats"].get(k) for k in matched_keys}, run["name"]
            assert [{k: r.get(k) for k in round_keys} for r in run["rounds"]] == [
                {k: r.get(k) for k in round_keys} for r in reference["rounds"]], run["name"]
            solve_rows.append(dict(name=run["name"], elapsed_s=run["elapsed_s"],
                                   pricing_s=run["stats"]["pricing_wall_s"],
                                   objective=run["stats"]["objective"],
                                   rounds=run["stats"]["iterations"],
                                   completed=run["stats"]["pricing_flights_completed"],
                                   dispatches=run["stats"]["gpu_dispatches"]))
        with (DATA / "solver.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(solve_rows[0]))
            writer.writeheader()
            writer.writerows(solve_rows)
    summary = dict(base='b605aba3b5d9a29ad70cfe5b4e6ab6e4ed57182c',
                   preparation=groups, exact_component_comparisons=len(comparisons), solves=solves)
    (DATA/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    text = ['Two fresh processes per hard88 variant; values are mean [observed range] seconds.', '',
            '| Builder enabled alone | Cold preparation | Reduction from base |',
            '| --- | ---: | ---: |']
    base = next(g['mean_s'] for g in groups if g['variant']=='base' and not g['control'] and g['phase']=='cold')
    for g in groups:
        if g['control'] or g['phase']!='cold':
            continue
        text.append(f"| {g['variant']} (n={g['n']}) | {g['mean_s']:.3f} [{g['min_s']:.3f}, {g['max_s']:.3f}] | {(1-g['mean_s']/base)*100:.1f}% |")
    text += ['', 'Individual reductions are **not additive**. These are CPU preparation times.', '',
             '| Phase | Base | All builders |', '| --- | ---: | ---: |']
    for phase in ('cold','warm','changed','changed_warm','restored'):
        values=[]
        for variant in ('base','all'):
            g=next(g for g in groups if g['variant']==variant and g['phase']==phase and not g['control'])
            values.append(f"{g['mean_s']:.3f} [{g['min_s']:.3f}, {g['max_s']:.3f}]")
        text.append(f"| {phase} | {' | '.join(values)} |")
    text += ['', f"All {len(comparisons):,} recorded flight calls match the base's exact component fingerprints.",
             '', '![Isolated cold preparation and warm phases](../../context/figures/cpu_builder_preparation.png)',
             '', '![Direct production solver comparison](../../context/figures/cpu_builder_solver.png)']
    if solves:
        text += ["", "Direct production solver wall (same integer/approach search and result):", "",
                 "| Preparation workers | Base wall | All builders wall | Reduction |",
                 "| --- | ---: | ---: | ---: |"]
        for workers in (0, 12):
            values = [[r["elapsed_s"] for r in solves if r["name"].startswith(
                f"{variant}-solve-w{workers}-")] for variant in ("base", "all")]
            if all(values):
                means = [statistics.mean(v) for v in values]
                cells = [f"{m:.3f} [{min(v):.3f}, {max(v):.3f}] (n={len(v)})" for m,v in zip(means, values)]
                text.append(f"| {workers} | {' | '.join(cells)} | {(1-means[1]/means[0])*100:.1f}% |")
        stats=solves[0]["stats"]
        text += ["", f"All {len(solves)} completed solver runs match exact final cost "
                 f"{stats['objective']:.12f}, LP objective/bounds, per-round columns and reduced-cost "
                 f"statistics, {stats['iterations']} rounds, {stats['pricing_flights_completed']} completed "
                 f"pricing calls and {stats['gpu_dispatches']:,} dispatches. All are certified and "
                 "verified; no full CPU fallback or CPU numeric verifier is allowed."]
    audit_path = DATA / "all-audited"
    if (audit_path / "result.json").exists():
        result = json.loads((audit_path / "result.json").read_text())
        audited = json.loads((audit_path / "pricing.json").read_text())["results"]
        audit_stats = json.loads((audit_path / "planner_stats.json").read_text())
        assert result["passed"] and result["verified"]
        assert result["cpu_full_pricing_guard"] and result["cpu_numeric_verifier_guard"]
        assert all(r["certified"] and r["search_complete"] and r["status"] == "complete" for r in audited)
        for key in ("objective", "final_lp_objective", "upper_bound", "iterations", "gpu_dispatches",
                    "pricing_flights_completed"):
            assert audit_stats[key] == solves[0]["stats"][key]
        audit_groups = [r["stats"]["preparation_group"] for r in audited if r["stats"].get("preparation_group")]
        flights = [r["stats"]["preparation_flight"] for r in audited if r["stats"].get("preparation_flight")]
        audit = dict(results=len(audited), groups=len(audit_groups), elapsed_with_audits_s=result["elapsed_s"],
                     matched_direct_outcome=True,
                     integer_prepare_s=sum(r["stats"].get("integer_prepare_s", 0) for r in audited),
                     group_wall_s=sum(g.get("group_wall_s", 0) for g in audit_groups),
                     worker_cpu_prepare_sum_s=sum(f.get("cpu_prepare_s", 0) for f in flights),
                     cache_account_sum_s=sum(f.get("cache_account_s", 0) for f in flights),
                     canonical_checks=sum(r["stats"].get("canonical_validations", 0) for r in audited),
                     canonical_rejections=sum(r["stats"].get("canonical_rejections", 0) for r in audited),
                     gpu_s=sum(r["stats"].get("gpu_s", 0) for r in audited),
                     wait_s=sum(r["stats"].get("wait_s", 0) for r in audited),
                     dispatches=audit_stats["gpu_dispatches"],
                     timing_note="Nested timers overlap; worker sums are not wall time. Audited elapsed excluded from production comparisons.")
        (DATA / "audit_summary.json").write_text(json.dumps(audit, indent=2)+"\n")
    report=DATA.parent/'CPU_PREPARATION_BUILDERS.md'
    original=report.read_text()
    a=original.index('<!-- BUILDER_RESULTS_START -->')+len('<!-- BUILDER_RESULTS_START -->')
    b=original.index('<!-- BUILDER_RESULTS_END -->')
    report.write_text(original[:a]+'\n'+'\n'.join(text)+'\n'+original[b:])
    print(json.dumps(dict(comparisons=len(comparisons), groups=len(groups), solves=len(solves))))


if __name__ == '__main__':
    main()
