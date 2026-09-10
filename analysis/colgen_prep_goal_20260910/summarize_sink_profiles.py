"""Compare the two bounded six-flight sink-screen profiles without running pricing."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    """Write exact-parity, timing and nested sink attribution evidence.

    Parameters
    ------------
    - None: Read the fixed sibling sink_screen_profiles artifacts.

    Return
    --------
    - None: Write summary.json and report.md; fail on exact parity differences.
    """
    out = HERE / 'sink_screen_profiles'
    arms = {name: json.loads((out / name / 'profile.json').read_text())
            for name in ('old_combined', 'screened')}
    old, new = arms.values()
    assert old['capture_sha256'] == new['capture_sha256']
    assert old['params_sha256'] == new['params_sha256']
    rows = []
    totals = {name: Counter() for name in arms}
    for a, b in zip(old['records'], new['records'], strict=True):
        assert a['flight_id'] == b['flight_id']
        assert a['input_sha256'] == b['input_sha256']
        assert a['ordinary'][0]['signature'] == b['ordinary'][0]['signature']
        assert a['profiled']['phases']['goal'] == b['profiled']['phases']['goal']
        search = b['ordinary'][0]['search']
        row = dict(flight_id=a['flight_id'], exact_parity=True,
                   old_wall_s=a['median_wall_s'], new_wall_s=b['median_wall_s'],
                   old_goal_s=a['median_goal_s'], new_goal_s=b['median_goal_s'],
                   sinks_skipped=search['bootstrap_goal_sinks_skipped'],
                   sinks_asked=search['bootstrap_goal_sinks_asked'],
                   expanded=search['bootstrap_goal_labels'])
        rows.append(row)
        for name, item in zip(arms, (a, b), strict=True):
            totals[name]['ordinary_wall_medians_s'] += item['median_wall_s']
            totals[name]['ordinary_goal_medians_s'] += item['median_goal_s']
            totals[name].update(item['profiled']['phases']['inclusive_s'])
            totals[name]['sink_calls'] += item['profiled']['phases']['calls'].get('canonical_sink_in_goal', 0)
            totals[name]['canonical_calls'] += item['profiled']['phases']['calls'].get('canonical_candidate_in_goal', 0)
    wall_change = 100 * (totals['screened']['ordinary_wall_medians_s'] /
                         totals['old_combined']['ordinary_wall_medians_s'] - 1)
    goal_change = 100 * (totals['screened']['ordinary_goal_medians_s'] /
                         totals['old_combined']['ordinary_goal_medians_s'] - 1)
    summary = dict(exact_parity=True, rows=rows, totals=totals,
                   wall_change_pct=wall_change, goal_change_pct=goal_change,
                   capture_sha256=old['capture_sha256'], params_sha256=old['params_sha256'],
                   source_sha256={name: arm['source_sha256'] for name, arm in arms.items()})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Native goal sink screen: bounded warm comparison', '',
             'Same captured round-11 inputs, six tails selected by prior baseline goal time; '
             'one warmup, three ordinary warm trials, then a separate nested-instrumentation trial per flight. '
             'Arms ran serially in isolated processes; compilation excluded. Frozen old combined versus screened sources.', '',
             'Every final output, identity, claims, cost/reduced-cost and search-work signature matches exactly. '
             'Goal entry, result and 20,000-expansion count also match for every flight. '
             'Callback counts intentionally differ. This is a selected-tail result, not a full-solver speedup estimate.', '',
             '| Flight | Total old → screened (s) | Goal old → screened (s) | Skipped / asked labels |',
             '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['flight_id']} | {row['old_wall_s']:.4f} → {row['new_wall_s']:.4f} | "
                     f"{row['old_goal_s']:.4f} → {row['new_goal_s']:.4f} | "
                     f"{row['sinks_skipped']} / {row['sinks_asked']} |")
    lines.extend(['', f'Sum of per-flight ordinary medians: total {wall_change:.2f}%; goal {goal_change:.2f}%.', '',
                  'Separate instrumented inclusive goal components (seconds; nested rows must not be added twice):', '',
                  '| Component | Old combined | Screened |', '|---|---:|---:|'])
    for key in ('goal', 'canonical_sink_in_goal', 'path_claims_in_goal', 'path_delay_s_in_goal',
                'claim_cost_in_goal', 'canonical_candidate_in_goal', 'native_advance_in_goal', 'pack_duals_in_goal'):
        lines.append(f"| {key} | {totals['old_combined'][key]:.6f} | {totals['screened'][key]:.6f} |")
    lines.extend(['', f"Goal sink callbacks: {totals['old_combined']['sink_calls']} → {totals['screened']['sink_calls']}; "
                  f"canonical-candidate calls inside goal: {totals['old_combined']['canonical_calls']} → "
                  f"{totals['screened']['canonical_calls']}.", '',
                  'The conservative native bound screens before host claim construction and delay computation. '
                  'It does not compile canonical geometry, use queue priority as a pruning bound, or reduce expansions. '
                  'The positive paid-cost lower bound remains zero; near-cutoff uncertainty still asks Python.', '',
                  'Validation: 32 goal-kernel tests passed in 2.16 s, including near-cutoff, negative dual, '
                  'invalid sink/resume and strong-to-weak cached-envelope cases. Frozen source differs only in '
                  'bootstrap_kernel.py (screen) and pricing.py (diagnostics).', '',
                  'Instrumentation wall/ordinary-median ratios: ' + '; '.join(
                      f"{name} {min(r['instrumentation_wall_ratio'] for r in arm['records']):.3f}–"
                      f"{max(r['instrumentation_wall_ratio'] for r in arm['records']):.3f}"
                      for name, arm in arms.items()) + '. Ordinary timings are authoritative.', '',
                  'Reproduce each arm with profile_goal_phases.py --source-root <frozen-source> --out <fresh-output>; '
                  'default capture and six-flight selection paths match this report.'])
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    helper = HERE / 'profile_goal_phases.py'
    (out / 'profile_goal_phases_used.py').write_bytes(helper.read_bytes())
    (out / 'helper_sha256.txt').write_text(hashlib.sha256(helper.read_bytes()).hexdigest() + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
