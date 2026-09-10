# Full 1200-second path-comparison experiment

**COMPLETE**. One serial baseline/candidate pair, one demand seed; observed differences are not statistical confidence intervals.

Both runs use 1,200 seconds of outbound demand, 12 pricing workers, 4 Gurobi threads, six extra hops, one column per flight, hybrid bootstrap, LP tolerance 0.001, and a 30-second IP search each round. Actual full parameter hashes are compared below.

| Metric | Baseline | Candidate |
|---|---:|---:|
| verified | True | True |
| accepted | 1537 | 1537 |
| denied | 0 | 0 |
| n_requests | 1537 | 1537 |
| penalized_congestion_cost | 156817.302553 | 156817.302553 |
| solver_objective | 156817.302553 | 156817.302553 |
| global_cost_gap | 0.011826 | 0.011826 |
| restricted_ip_gap | 0.000000 | 0.000000 |
| termination | lp_gap | lp_gap |
| iterations | 11 | 11 |
| filed_cost | 361750.635886 | 361750.635886 |
| ground_delay_s | 1876.000000 | 1876.000000 |
| air_detour_m | 1549413.025525 | 1549413.025525 |
| p95_delay_s | 73.222781 | 73.222781 |
| total_delay_s | 53523.100851 | 53523.100851 |
| simulation_wall_s | 1356.332580 | 1256.205863 |
| solver_wall_s | 1351.868614 | 1251.824059 |
| pricing_wall_s | 730.703414 | 641.643991 |
| iteration_ip_wall_s | 348.360555 | 345.446020 |
| ip_wall_s | 18.143535 | 18.277119 |
| ip_status | optimal | optimal |
| ip_skipped | False | False |
| kernel_fell_back | 0 | 0 |
| peak_process_rss_mib | 9816.859375 | 9549.531250 |
| final LP cost gap | 0.000852 | 0.000852 |
| configured-domain cost lower bound | 154962.781693 | 154962.781693 |

Penalized cost change: 0.000000. Simulation wall change: -7.382166%; pricing wall change: -12.188177%.

Simulation wall contains solver wall; solver wall contains pricing, initialization, LP, and IP phases. Do not add inclusive totals. IP wrapper wall, native wall, Gurobi Runtime and setup are nested views, not additional disjoint phases. Worker task sums are parallel work, not wall time.

| IP phase | Baseline wrapper / native / Gurobi s | Candidate wrapper / native / Gurobi s |
|---|---:|---:|
| iteration | 332.789223 / 321.234409 / 320.060281 | 330.500806 / 319.543003 / 318.391371 |
| final | 18.129824 / 17.776091 / 17.654810 | 18.263640 / 17.934207 / 17.822386 |

Input equality (all fields): {'inputs': True, 'config': True, 'requests': True, 'static_terminals': True, 'full_algorithm_params': True}. Production source differences: ['freespace_sim/planner/colgen/dp_kernel.py'].

| Evidence | Equal | Baseline / candidate records | Changed shared records |
|---|---|---:|---:|
| rounds | True | 11 / 11 | 0 |
| pool | True | 42879 / 42879 | 0 |
| column_sequence | True | 42879 / 42879 | 0 |
| ip | False | 12 / 12 | 8 |
| work | False | 16907 / 16907 | 96 |
| pricing_values | False | 16907 / 16907 | 1511 |
| trajectories | False | 1537 / 1537 | 568 |
| initial | True | 1537 / 1537 | 0 |
| final | False | 1537 / 1537 | 568 |
| entry_rc | False | 15580 / 15580 | 1511 |
| final_rc | True | 15580 / 15580 | 0 |

| Round | LP objective B / C | LP pool B / C | Added B / C | Round fields changed | Flight work / entry RC / final RC mismatches |
|---|---:|---:|---:|---|---:|
| 1 | 15203997.114633 / 15203997.114633 | 32422 / 32422 | 1226 / 1226 | none | 0 / 0 / 0 |
| 2 | 15210628.842622 / 15210628.842622 | 33648 / 33648 | 1389 / 1389 | none | 0 / 0 / 0 |
| 3 | 15213320.685614 / 15213320.685614 | 35037 / 35037 | 1135 / 1135 | none | 0 / 0 / 0 |
| 4 | 15214286.764366 / 15214286.764366 | 36172 / 36172 | 1320 / 1320 | none | 0 / 0 / 0 |
| 5 | 15214698.420228 / 15214698.420228 | 37492 / 37492 | 656 / 656 | none | 0 / 0 / 0 |
| 6 | 15214819.574238 / 15214819.574238 | 38148 / 38148 | 1299 / 1299 | none | 18 / 85 / 0 |
| 7 | 15214873.060425 / 15214873.060425 | 39447 / 39447 | 441 / 441 | none | 7 / 161 / 0 |
| 8 | 15214894.355827 / 15214894.355827 | 39888 / 39888 | 1179 / 1179 | none | 12 / 268 / 0 |
| 9 | 15214900.124188 / 15214900.124188 | 41067 / 41067 | 393 / 393 | none | 11 / 348 / 0 |
| 10 | 15214904.263131 / 15214904.263131 | 41460 / 41460 | 212 / 212 | none | 16 / 346 / 0 |
| 11 | 15214905.075535 / 15214905.075535 | 41672 / 41672 | 1207 / 1207 | none | 32 / 303 / 0 |

Work differences are confined to label-count fields: {'bootstrap_labels': 41, 'bootstrap_goal_labels': 40, 'n_labels': 65, 'bootstrap_dp_labels': 27}. RC-field differences: {'entry_rc': 1511}; maximum absolute differences {'entry_rc': 3.9366444875868183}. Recorded final RCs: 15580 per arm, exact equality True.

Summed label work: {'baseline': {'n_labels': 1251816978, 'bootstrap_labels': 438258149, 'bootstrap_goal_labels': 23135742, 'bootstrap_dp_labels': 415122407}, 'candidate': {'n_labels': 1251781902, 'bootstrap_labels': 438629106, 'bootstrap_goal_labels': 23193317, 'bootstrap_dp_labels': 415435789}}. Work-count equality is not claimed after the IP selection divergence.

Trajectory field differences: {'centerline': 568, 'cost': 113, 'delay_s': 113, 'air_detour_m': 110, 'ground_delay_s': 5}. Equal aggregate cost does not mean identical routes or per-flight costs.

All recorded LP/certificate fields, timed column pools/order/costs, and final priced reduced costs match: True.

First changed IP selection round: 5. IP outcomes are compared after resolving local column indices to timed identities and costs. Time-limited IP searches can choose different incumbents; later pricing may then receive different known columns. A subsequent work or pool difference is therefore not, by itself, a same-input comparator failure. Exact fixed-dual kernel parity is separately established in the bounded pathcmp replay artifacts.

Final restricted-IP optimality applies to the generated pool. The global configured-domain integer gap and LP cost certificate remain separate; a pool-optimal result with a positive global gap is not a proof of global integer optimality.

All timed column identities include absolute departure, flight, level, lanes and the full cell path; path IDs and geometry IDs are not compared across runs. Shared identity costs are compared exactly. Final selections and full saved trajectory records are compared independently. Capacity claims are absent from the trace, so no claims-equality claim is made. Per-round dual vectors are also absent.

Missing baseline artifacts: []; candidate: []. Missing selection indices: [] / []. Unresolved path references: [] / []. Integrity issues: [] / [].

Raw flight comparison excludes fields ending in `_s` plus worker/PID; work and RC fields are compared separately by round and flight. Details, input/source hashes, final certificates, and full IP selection mappings are retained in summary.json and parity_details.json.
