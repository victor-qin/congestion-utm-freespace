"""Audit the completed full 1200-second native path-comparison experiment pair.

Run only after the timed processes have released the CPU. This script reads saved
artifacts and never imports the planner, reruns a solve, or infers unrecorded claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


ROUND_KEYS = ('lp_objective', 'lp_column_count', 'columns', 'columns_added', 'rc_sum',
              'rc_max', 'pricing_exact', 'upper_bound', 'cost_lower_bound',
              'cost_upper_bound', 'lp_gap_cost', 'n_uncovered')
RESULT_KEYS = ('verified', 'accepted', 'denied', 'n_requests', 'penalized_congestion_cost',
               'solver_objective', 'global_cost_gap', 'restricted_ip_gap', 'termination',
               'iterations', 'filed_cost', 'ground_delay_s', 'air_detour_m', 'p95_delay_s', 'total_delay_s', 'simulation_wall_s', 'solver_wall_s', 'pricing_wall_s',
               'iteration_ip_wall_s', 'ip_wall_s', 'ip_status', 'ip_skipped', 'kernel_fell_back', 'peak_process_rss_mib')


def read(path, default=None):
    """Read an available JSON artifact without substituting zero for missing data."""
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    """Read complete JSONL records; malformed or partial artifacts raise explicitly."""
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value):
    """Hash canonical JSON, preserving float round-trip values and list order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def total(values):
    """Sum available measurements, keeping an entirely missing quantity unavailable."""
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return math.fsum(values) if values else None


def compare_maps(a, b):
    """Compare every keyed record while retaining bounded mismatch examples."""
    common = a.keys() & b.keys()
    changed = [k for k in common if a[k] != b[k]]
    return dict(equal=a == b, baseline_count=len(a), candidate_count=len(b),
                baseline_only=len(a.keys()-b.keys()), candidate_only=len(b.keys()-a.keys()),
                changed=len(changed), changed_examples=sorted(changed, key=str)[:12],
                baseline_sha256=digest(a), candidate_sha256=digest(b))


def load_arm(directory):
    """Load final metrics and normalize complete route and pricing traces.

    Parameters
    ------------
    - directory (Path): One completed baseline or candidate harness output.

    Return
    --------
    - public, detail (dict): Compact metrics and normalized records for pairwise audit.
    """
    case = directory/'density_faa_wing_zipline_seed0_iteration_ip_eager'
    required = ('inputs.json', 'measurement.json', 'result.json', 'planner_stats.json',
                'iterations.json', 'ip_calls.json', 'paths.jsonl', 'columns.jsonl',
                'selections.json', 'trace_summary.json', 'trajectories.json.gz')
    missing = [name for name in required if not (case/name).exists()]
    inputs = read(case/'inputs.json', {})
    result = read(case/'result.json', {})
    stats = read(case/'planner_stats.json', {})
    iterations = read(case/'iterations.json', [])
    manifest = read(directory/'manifest.json', {})
    paths = {p['id']: {k:p.get(k) for k in
        ('flight_id', 'level', 'origin_lane_idx', 'dest_lane_idx', 'cells')}
        for p in lines(case/'paths.jsonl')}
    path_hashes = {key:digest(value) for key,value in paths.items()}
    columns = lines(case/'columns.jsonl')
    pool, by_index, descriptions, column_sequence = {}, {}, {}, {}
    unresolved_paths = []
    for col in columns:
        if col['path_id'] not in paths:
            unresolved_paths.append(col['index'])
            continue
        identity = (col['flight_id'], col['departure_step'], path_hashes[col['path_id']])
        key = digest(identity)
        value = dict(identity_sha256=key, cost=col['cost'])
        pool.setdefault(key, []).append(col['cost'])
        by_index[col['index']] = value
        column_sequence[str(col['index'])] = dict(value, phase=col.get('phase'), sweep=col.get('sweep'))
        descriptions[key] = dict(flight_id=col['flight_id'], departure_step=col['departure_step'],
                                  path_sha256=path_hashes[col['path_id']])
    pool = {key:sorted(values) for key,values in pool.items()}
    selections = read(case/'selections.json', {})
    missing_selection_indices = set()

    def selection(mapping):
        """Resolve arm-local column indices to stable timed identities and costs."""
        if mapping is None:
            return None
        resolved = {}
        for flight, index in mapping.items():
            if index not in by_index:
                missing_selection_indices.add(index)
            resolved[str(flight)] = by_index.get(index)
        return resolved

    initial, final = selection(selections.get('initial')), selection(selections.get('final'))
    calls = read(case/'ip_calls.json', [])
    ip_detail, ip_metrics = {}, {}
    for phase in ('iteration', 'final'):
        group = [c for c in calls if c['phase'] == phase]
        ip_metrics[phase] = dict(calls=len(group), wrapper_wall_s=total(c.get('wall_s') for c in group),
            native_wall_s=total(n.get('wall_s') for c in group for n in c.get('native_calls', [])),
            gurobi_runtime_s=total(n.get('gurobi_runtime_s') for c in group for n in c.get('native_calls', [])),
            setup_s=total(c.get('setup_s') for c in group))
    for i, call in enumerate(calls):
        key = f"{call['phase']}:{call.get('sweep')}:{i}"
        ip_detail[key] = dict(phase=call['phase'], sweep=call.get('sweep'), status=call.get('status'),
            optimal=call.get('optimal'), requested_budget_s=call.get('requested_budget_s'),
            before=selection(call.get('before_selection')), after=selection(call.get('after_selection')),
            route_cost_before=call.get('before_cost'), route_cost_after=call.get('after_cost'),
            native_outcomes=[{k:n.get(k) for k in ('status','objective','upper_bound','nodes','solution_count')}
                             for n in call.get('native_calls', [])])
    work, pricing_values, record_fields, duplicate_records = {}, {}, set(), []
    raw_round_counts = {}
    for p in sorted((case/'flight_records').glob('round_*.jsonl.gz')):
        round_id = int(p.name.split('_')[1].split('.')[0])
        records = lines(p)
        raw_round_counts[str(round_id)] = len(records)
        for record in records:
            key = f"{round_id}:{record['flight_id']}"
            if key in work:
                duplicate_records.append(key)
            filtered = {k:v for k,v in record.items() if not k.endswith('_s') and k not in ('pid','worker')}
            record_fields.update(filtered)
            work[key] = {k:v for k,v in filtered.items() if k not in ('entry_rc','final_rc')}
            pricing_values[key] = {k:v for k,v in filtered.items() if k in ('entry_rc','final_rc')}
    for iteration in iterations:
        rid = str(iteration['iteration'])
        expected = iteration.get('flight_diagnostics', {}).get('records')
        if rid not in raw_round_counts or (expected is not None and raw_round_counts[rid] != expected):
            missing.append('complete flight_records round '+rid)
    trajectories = {}
    if (case/'trajectories.json.gz').exists():
        with gzip.open(case/'trajectories.json.gz', 'rt') as handle:
            trajectories = {str(r['flight_id']):r for r in json.load(handle)}
    certificate_keys = ('lp_gap_cost','cost_lower_bound','cost_upper_bound','upper_bound',
                         'final_lp_objective','global_integer_gap_cost','restricted_ip_gap',
                         'ip_optimal','ip_status','termination_reason','n_columns',
                         'kernel_label_restarts','kernel_budget_declined','pricing_worker_lost')
    source = manifest.get('iteration_ip_eager_source_sha256', {})
    if not source:
        missing.append('actual iteration_ip_eager source manifest')
    if read(case/'measurement.json', {}).get('status') != 'complete':
        missing.append('completed measurement status')
    if final is None:
        missing.append('final selection mapping')
    integrity_issues = []
    if unresolved_paths or missing_selection_indices or duplicate_records:
        integrity_issues.append('unresolved references or duplicate pricing records')
    if len(columns) != len(by_index):
        integrity_issues.append('column indices are duplicated or unresolved')
    if result.get('n_requests') is not None and len(trajectories) != result['n_requests']:
        integrity_issues.append('trajectory coverage differs from request count')
    public = dict(directory=str(directory), case=str(case), missing=missing, integrity_issues=integrity_issues,
        result={k:result.get(k) for k in RESULT_KEYS},
        certificates={k:stats.get(k) for k in certificate_keys},
        source_files=source, published_hashes={k:result.get(k) for k in
            ('input_sha','demand_config_sha','algorithm_params_sha','requests_sha','trajectory_sha')},
        recomputed_hashes=dict(inputs=digest(inputs), config=digest(inputs.get('config')),
            requests=digest(inputs.get('requests')), static_terminals=digest(inputs.get('static_terminals')),
            full_algorithm_params=digest(inputs.get('params'))),
        params=inputs.get('params'), ip_metrics=ip_metrics, rounds=len(iterations),
        column_records=len(columns), unique_timed_identities=len(pool), unresolved_paths=unresolved_paths,
        duplicate_column_indices=len(columns)-len({c['index'] for c in columns}),
        initial_selection_count=None if initial is None else len(initial),
        final_selection_count=None if final is None else len(final),
        missing_selection_indices=sorted(missing_selection_indices),
        trajectories=len(trajectories), accepted_trajectories=sum(r.get('accepted',False) for r in trajectories.values()),
        raw_pricing_records=len(work), raw_round_counts=raw_round_counts,
        duplicate_pricing_records=duplicate_records, compared_work_fields=sorted(record_fields-{'entry_rc','final_rc'}),
        initialization_times={k:stats.get(k) for k in ('seed_elapsed_s','graph_build_elapsed_s',
            'time_to_master_s','initial_greedy_elapsed_s')},
        worker_task_sum_s=stats.get('pricing_task_total_s'),
        lp_backend_wall_s=total(r.get('stage_s',{}).get('solve_lp') for r in iterations))
    detail = dict(rounds={str(r['iteration']):{k:r.get(k) for k in ROUND_KEYS} for r in iterations},
        pool=pool, column_sequence=column_sequence, initial=initial, final=final, ip=ip_detail, work=work, pricing_values=pricing_values,
        trajectories=trajectories, descriptions=descriptions)
    return public, detail


def fmt(value):
    """Format missing values explicitly and keep cost differences visible."""
    return '—' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)


def main():
    """Compare the final pair and write compact summary, audit details and report.

    Parameters
    ------------
    - CLI --root (Path): Pair artifact directory, defaulting to the dated experiment.

    Return
    --------
    - None: Persist summary.json, parity_details.json and report.md without rerunning planning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'colgen_pathcmp_full_1200s_20260910')
    args=parser.parse_args()
    a,ad=load_arm(args.root/'baseline')
    b,bd=load_arm(args.root/'candidate')
    compared={k:compare_maps(ad[k],bd[k]) for k in ('rounds','pool','column_sequence','ip','work','pricing_values','trajectories')}
    for key in ('initial','final'):
        compared[key] = (dict(available=False) if ad[key] is None or bd[key] is None
                         else dict(available=True,**compare_maps(ad[key],bd[key])))
    for field in ('entry_rc','final_rc'):
        compared[field]=compare_maps({k:r[field] for k,r in ad['pricing_values'].items() if field in r},
                                     {k:r[field] for k,r in bd['pricing_values'].items() if field in r})
    work_fields=Counter()
    rc_fields=Counter()
    trajectory_fields=Counter()
    max_rc_delta={}
    for key in ad['work'].keys() & bd['work'].keys():
        x,y=ad['work'][key],bd['work'][key]
        work_fields.update(f for f in x.keys() | y.keys() if x.get(f)!=y.get(f))
        x,y=ad['pricing_values'][key],bd['pricing_values'][key]
        for field in x.keys() | y.keys():
            if x.get(field)!=y.get(field):
                rc_fields[field]+=1
                if isinstance(x.get(field),(int,float)) and isinstance(y.get(field),(int,float)):
                    max_rc_delta[field]=max(max_rc_delta.get(field,0),abs(x[field]-y[field]))
    for key in ad['trajectories'].keys() & bd['trajectories'].keys():
        x,y=ad['trajectories'][key],bd['trajectories'][key]
        trajectory_fields.update(f for f in x.keys() | y.keys() if x.get(f)!=y.get(f))
    changed_source=[p for p in a['source_files'].keys() | b['source_files'].keys()
                    if a['source_files'].get(p)!=b['source_files'].get(p)]
    hashes={k:a['recomputed_hashes'][k]==b['recomputed_hashes'][k] for k in a['recomputed_hashes']}
    round_audit=[]
    for rid in sorted(ad['rounds'].keys() | bd['rounds'].keys(),key=int):
        ar,br=ad['rounds'].get(rid,{}),bd['rounds'].get(rid,{})
        keys={k for k in ad['work'].keys() | bd['work'].keys() if k.startswith(rid+':')}
        round_audit.append(dict(round=int(rid),baseline=ar,candidate=br,
            changed_fields=[k for k in ROUND_KEYS if ar.get(k)!=br.get(k)],
            work_mismatches=sum(ad['work'].get(k)!=bd['work'].get(k) for k in keys),
            rc_mismatches=sum(ad['pricing_values'].get(k)!=bd['pricing_values'].get(k) for k in keys),
            entry_rc_mismatches=sum(ad['pricing_values'].get(k,{}).get('entry_rc')!=bd['pricing_values'].get(k,{}).get('entry_rc') for k in keys),
            final_rc_mismatches=sum(ad['pricing_values'].get(k,{}).get('final_rc')!=bd['pricing_values'].get(k,{}).get('final_rc') for k in keys)))
    ip_keys=sorted(ad['ip'].keys() | bd['ip'].keys(),
        key=lambda k:(k.split(':')[0] == 'final', int(k.split(':')[1]), int(k.split(':')[2])))
    first_ip=next((ad['ip'].get(k,bd['ip'].get(k))['sweep'] for k in ip_keys
                   if ad['ip'].get(k,{}).get('after')!=bd['ip'].get(k,{}).get('after')),None)
    durations={k:None if not a['result'].get(k) or b['result'].get(k) is None else
                   100*(b['result'][k]/a['result'][k]-1)
               for k in ('simulation_wall_s','solver_wall_s','pricing_wall_s','iteration_ip_wall_s','ip_wall_s')}
    shared=ad['pool'].keys() & bd['pool'].keys()
    cost_changes={k:dict(baseline=ad['pool'][k],candidate=bd['pool'][k],identity=ad['descriptions'][k])
                  for k in shared if ad['pool'][k]!=bd['pool'][k]}
    cost_delta = (None if a['result'].get('penalized_congestion_cost') is None or
        b['result'].get('penalized_congestion_cost') is None else
        b['result']['penalized_congestion_cost']-a['result']['penalized_congestion_cost'])
    label_totals={arm:{field:sum(r.get(field,0) for r in detail['work'].values())
                       for field in ('n_labels','bootstrap_labels','bootstrap_goal_labels','bootstrap_dp_labels')}
                  for arm,detail in (('baseline',ad),('candidate',bd))}
    summary=dict(updated=datetime.now(timezone.utc).isoformat(),baseline=a,candidate=b,
        complete=not a['missing'] and not b['missing'] and not a['integrity_issues'] and not b['integrity_issues'],input_hash_parity=hashes,
        source_changed_files=sorted(changed_source),comparisons=compared,rounds=round_audit,
        first_ip_selection_difference_round=first_ip,timing_change_percent=durations,cost_delta=cost_delta,
        work_changed_fields=dict(work_fields),rc_changed_fields=dict(rc_fields),max_rc_delta=max_rc_delta,
        trajectory_changed_fields=dict(trajectory_fields),label_totals=label_totals,claims_parity=None,limitation='Column trace omits capacity claims and per-round dual vectors; claims parity is unavailable.')
    details=dict(rounds=round_audit,shared_identity_cost_changes=cost_changes,
        baseline_only_timed_identities={k:ad['descriptions'][k] for k in ad['pool'].keys()-bd['pool'].keys()},
        candidate_only_timed_identities={k:bd['descriptions'][k] for k in bd['pool'].keys()-ad['pool'].keys()},
        ip_outcomes=dict(baseline=ad['ip'],candidate=bd['ip']))
    for name,value in (('summary.json',summary),('parity_details.json',details)):
        (args.root/name).write_text(json.dumps(value,indent=2)+'\n')
    report=['# Full 1200-second path-comparison experiment','',
        '**'+('COMPLETE' if summary['complete'] else 'INCOMPLETE — inspect missing artifacts')+'**. One serial baseline/candidate pair, one demand seed; observed differences are not statistical confidence intervals.','',
        'Both runs use 1,200 seconds of outbound demand, 12 pricing workers, 4 Gurobi threads, six extra hops, one column per flight, hybrid bootstrap, LP tolerance 0.001, and a 30-second IP search each round. Actual full parameter hashes are compared below.','',
        '| Metric | Baseline | Candidate |','|---|---:|---:|']
    for key in RESULT_KEYS:
        report.append(f"| {key} | {fmt(a['result'][key])} | {fmt(b['result'][key])} |")
    report += [f"| final LP cost gap | {fmt(a['certificates'].get('lp_gap_cost'))} | {fmt(b['certificates'].get('lp_gap_cost'))} |",
               f"| configured-domain cost lower bound | {fmt(a['certificates'].get('cost_lower_bound'))} | {fmt(b['certificates'].get('cost_lower_bound'))} |"]
    report += ['',f"Penalized cost change: {fmt(cost_delta)}. Simulation wall change: {fmt(durations['simulation_wall_s'])}%; pricing wall change: {fmt(durations['pricing_wall_s'])}%.",
        '', 'Simulation wall contains solver wall; solver wall contains pricing, initialization, LP, and IP phases. Do not add inclusive totals. IP wrapper wall, native wall, Gurobi Runtime and setup are nested views, not additional disjoint phases. Worker task sums are parallel work, not wall time.', '',
        '| IP phase | Baseline wrapper / native / Gurobi s | Candidate wrapper / native / Gurobi s |','|---|---:|---:|']
    for phase in ('iteration','final'):
        av,bv=a['ip_metrics'][phase],b['ip_metrics'][phase]
        report.append('| '+phase+' | '+' / '.join(fmt(av[k]) for k in ('wrapper_wall_s','native_wall_s','gurobi_runtime_s'))+' | '+' / '.join(fmt(bv[k]) for k in ('wrapper_wall_s','native_wall_s','gurobi_runtime_s'))+' |')
    report += ['',f"Input equality (all fields): {hashes}. Production source differences: {sorted(changed_source)}.", '',
        '| Evidence | Equal | Baseline / candidate records | Changed shared records |','|---|---|---:|---:|']
    for name,c in compared.items():
        report.append(f"| {name} | {c.get('equal','unavailable')} | {c.get('baseline_count','—')} / {c.get('candidate_count','—')} | {c.get('changed','—')} |")
    report += ['', '| Round | LP objective B / C | LP pool B / C | Added B / C | Round fields changed | Flight work / entry RC / final RC mismatches |', '|---|---:|---:|---:|---|---:|']
    for r in round_audit:
        ar,br=r['baseline'],r['candidate']
        report.append(f"| {r['round']} | {fmt(ar.get('lp_objective'))} / {fmt(br.get('lp_objective'))} | {ar.get('lp_column_count')} / {br.get('lp_column_count')} | {ar.get('columns_added')} / {br.get('columns_added')} | {', '.join(r['changed_fields']) or 'none'} | {r['work_mismatches']} / {r['entry_rc_mismatches']} / {r['final_rc_mismatches']} |")
    pricing_equal=all(compared[k]['equal'] for k in ('rounds','pool','column_sequence','final_rc'))
    report += ['',f"Work differences are confined to label-count fields: {dict(work_fields)}. RC-field differences: {dict(rc_fields)}; maximum absolute differences {max_rc_delta}. Recorded final RCs: {compared['final_rc']['baseline_count']} per arm, exact equality {compared['final_rc']['equal']}.",
        '',f"Summed label work: {label_totals}. Work-count equality is not claimed after the IP selection divergence.",
        '',f"Trajectory field differences: {dict(trajectory_fields)}. Equal aggregate cost does not mean identical routes or per-flight costs.",
        '', f"All recorded LP/certificate fields, timed column pools/order/costs, and final priced reduced costs match: {pricing_equal}.", '',f"First changed IP selection round: {first_ip}. IP outcomes are compared after resolving local column indices to timed identities and costs. Time-limited IP searches can choose different incumbents; later pricing may then receive different known columns. A subsequent work or pool difference is therefore not, by itself, a same-input comparator failure. Exact fixed-dual kernel parity is separately established in the bounded pathcmp replay artifacts.", '',
        'Final restricted-IP optimality applies to the generated pool. The global configured-domain integer gap and LP cost certificate remain separate; a pool-optimal result with a positive global gap is not a proof of global integer optimality.', '',
        'All timed column identities include absolute departure, flight, level, lanes and the full cell path; path IDs and geometry IDs are not compared across runs. Shared identity costs are compared exactly. Final selections and full saved trajectory records are compared independently. Capacity claims are absent from the trace, so no claims-equality claim is made. Per-round dual vectors are also absent.', '',
        f"Missing baseline artifacts: {a['missing']}; candidate: {b['missing']}. Missing selection indices: {a['missing_selection_indices']} / {b['missing_selection_indices']}. Unresolved path references: {a['unresolved_paths']} / {b['unresolved_paths']}. Integrity issues: {a['integrity_issues']} / {b['integrity_issues']}.", '',
        'Raw flight comparison excludes fields ending in `_s` plus worker/PID; work and RC fields are compared separately by round and flight. Details, input/source hashes, final certificates, and full IP selection mappings are retained in summary.json and parity_details.json.', '']
    (args.root/'report.md').write_text('\n'.join(report))


if __name__=='__main__':
    main()
