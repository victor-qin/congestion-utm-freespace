"""Summarize bounded native path-comparison replays without rerunning any pricing."""
from pathlib import Path
import json
import statistics


def read(path):
    """Load an existing JSON artifact, returning None while a run is incomplete."""
    return json.loads(path.read_text()) if path.exists() else None


def main():
    """Write a comparison report from completed standalone benchmark artifacts.

    Parameters
    ------------
    - None: Read the dated path-comparison directory beside this script.

    Return
    --------
    - None: Write report.md and summary.json with timings and exact parity results.
    """
    root = Path(__file__).resolve().parent/'colgen_pathcmp_20260910'
    base = read(root/'baseline_single/records.json') or []
    candidate = {r['flight_id']:r for r in read(root/'candidate_single/records.json') or []}
    pairs = []
    for a in base:
        if a['flight_id'] not in candidate:
            continue
        b = candidate[a['flight_id']]
        pairs.append(dict(flight_id=a['flight_id'], baseline_total_s=a['median_wall_s'],
            candidate_total_s=b['median_wall_s'], total_change_percent=100*(b['median_wall_s']/a['median_wall_s']-1),
            baseline_native_s=a['median_native_s'], candidate_native_s=b['median_native_s'],
            native_change_percent=100*(b['median_native_s']/a['median_native_s']-1),
            baseline_arena_bytes=a['trials'][0]['label_arena_bytes'],
            candidate_arena_bytes=b['trials'][0]['label_arena_bytes']))
    sweeps = []
    for p in sorted(root.glob('*_sweep*')):
        result = read(p/'sweep.json')
        if result:
            sweeps.append(dict(run=p.name, **result))
    sweep_order = ['baseline_sweep', 'candidate_sweep', 'candidate_sweep_repeat', 'baseline_sweep_repeat']
    sweeps.sort(key=lambda r: sweep_order.index(r['run']))
    parity = {p.parent.name:read(p) for p in sorted(root.glob('*/parity.json'))}
    counters = []
    old = {r['flight_id']:r for r in read(root/'baseline_counts/records.json') or []}
    for r in read(root/'candidate_counts/records.json') or []:
        if r['flight_id'] in old:
            a=old[r['flight_id']]['instrumentation']['counters']
            b=r['instrumentation']['counters']
            changed_path_keys = {'calls_fill_path', 'calls_path_cmp', 'calls_path_cmp_equal_depth',
                                 'path_nodes_materialized', 'path_elements_compared', 'path_parent_pairs_visited'}
            counters.append(dict(flight_id=r['flight_id'],
                other_counters_match=all(a[k] == b[k] for k in a.keys() & b.keys() - changed_path_keys),
                path_call_counts_match=a['calls_path_cmp'] == b['calls_path_cmp_equal_depth'],
                old_materialized_nodes=a['path_nodes_materialized'],
                old_full_path_calls=a['calls_path_cmp'], new_parent_pairs=b['path_parent_pairs_visited'],
                new_materialized_nodes=b['path_nodes_materialized'], new_equal_depth_calls=b['calls_path_cmp_equal_depth']))
    medians = {arm: statistics.median(s['wall_s'] for s in sweeps if s['run'].startswith(arm))
               for arm in ('baseline', 'candidate') if any(s['run'].startswith(arm) for s in sweeps)}
    summary=dict(single_flights=pairs, native_counters=counters, sweeps=sweeps, parity=parity,
                 sweep_median_change_percent=None if len(medians) < 2 else
                     100*(medians['candidate']/medians['baseline']-1))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Native path-comparison benchmark', '',
        'Both sources use the current six-extra-hop corridor and hybrid A* plus restricted-DP bootstrap. They differ only in `dp_kernel.py`. Inputs are the same frozen corrected first-LP capture. This is a kernel/pricing benchmark, not a new master/IP simulation.', '',
        'Each single-flight result is the median of three warm uninstrumented trials, preceded by one untimed solve on its own graph. Native timing wraps whole `_price_dag` calls. Instrumentation is a separate run and is excluded from these timings.', '',
        '| Flight | Baseline total s | Candidate total s | Total change | Baseline native s | Candidate native s | Native change |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['flight_id']} | {p['baseline_total_s']:.3f} | {p['candidate_total_s']:.3f} | {p['total_change_percent']:+.1f}% | {p['baseline_native_s']:.3f} | {p['candidate_native_s']:.3f} | {p['native_change_percent']:+.1f}% |")
    lines += ['', 'The relevant baseline is the current hybrid implementation (flight 934 native about 0.29 seconds), not the older roughly 12-second full-DP-bootstrap replay in the earlier native report.', '',
        '| Flight | Old materialized path nodes | New parent pairs visited | Candidate materialized nodes |',
        '|---|---:|---:|---:|']
    for c in counters:
        lines.append(f"| {c['flight_id']} | {c['old_materialized_nodes']:,} | {c['new_parent_pairs']:,} | {c['new_materialized_nodes']:,} |")
    if counters:
        lines.append('\nPath-comparison call counts and all non-path native work counters match for every compared instrumented flight: '+str(all(c['path_call_counts_match'] and c['other_counters_match'] for c in counters))+'.')
    lines += ['', 'A materialized node and a parent pair are different units of work; these counts explain the implementation change, not an elapsed-time attribution. The new comparison stops at a shared label ID and creates no full path arrays for these DP ties. The general unequal-depth path comparison remains available to the feasible-search frontier.', '',
        '| Full sweep run | Flights | Wall s | Worker task sum s | Pool setup s | Fallbacks |',
        '|---|---:|---:|---:|---:|---:|']
    for s in sweeps:
        lines.append(f"| {s['run']} | {s['flights']} | {s['wall_s']:.3f} | {s['task_total_s']:.3f} | {s['pool_setup_s']:.3f} | {s['kernel_counters'].get('fell_back',0)} |")
    for arm in ('baseline','candidate'):
        samples=[s['wall_s'] for s in sweeps if s['run'].startswith(arm)]
        if samples:
            lines.append(f"\n{arm.capitalize()} sweep wall range: {min(samples):.3f}–{max(samples):.3f} seconds across {len(samples)} run(s); median {statistics.median(samples):.3f} seconds.")
    if len(medians) == 2:
        lines.append(f"\nCandidate sweep median change: {summary['sweep_median_change_percent']:+.2f}%.")
    lines += ['', 'Sweep execution order is A–B–B–A (baseline, candidate, candidate repeat, baseline repeat). Regression tests ran between the first A and B, and separate instrumentation runs between the two B sweeps; none overlapped a measured run.', '', 'Sweep timings include 12-worker launch, preparation, and worker warmup; worker task sums are not wall time. Runs use the same capture and execute serially. Repeated samples expose observed variance but do not establish broad statistical confidence.', '',
        'Exact parity artifacts compare reduced-cost and full-column hashes, timed path identity, canonical claim sets, cost/RC bits, and per-flight label/stage work counts. No tolerance is used for these kernel-change comparisons. Individual records also preserve arena allocation bytes; unchanged arena size does not establish unchanged whole-process RSS.', '']
    for name,result in parity.items():
        lines.append(f"- {name}: {'PASS' if result['passed'] else 'FAIL'}, {result['compared']} flights compared.")
    if pairs:
        lines.append('\nLabel arena bytes match on all compared single flights: '+str(all(p['baseline_arena_bytes']==p['candidate_arena_bytes'] for p in pairs))+'.')
    lines += ['', 'Reproduce using `analysis/benchmark_colgen_pathcmp.py --source-root SOURCE --capture analysis/colgen_bootstrap_tail_20260910/subproblems.pkl --out NEW_DIRECTORY`. Use `--reference BASELINE_DIRECTORY`, `--mode sweep` for the 1,537-flight pricing sweep, and `--instrument --repeats 1` for a separate work-counter run. Every output directory preserves source/capture hashes and the harness used.', '']
    (root/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
