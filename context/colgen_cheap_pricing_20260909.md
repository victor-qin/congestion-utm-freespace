# Cheap pricing experiment (2026-09-09)

Status: both measured simulations complete, audited, and plotted. Final comparison below.
Root: `analysis/colgen_cheap_pricing_1200s_20260909`.
Protocol and exact command are in its README. Parent execution session: 81445.
Worktree: `/private/tmp/congestion-colgen-lns-20260905`, branch `codex/colgen-lns-main`.

User requested cheap pricing first, outbound-only FAA demand generated for 1,200 seconds,
LP tolerance 0.1%, and CG + IP every round as the new default. The benchmark runs the
cheap arm then the exact control sequentially on 1,537 flights, 4 pricing workers and
4 Gurobi threads, 30-round cap, 30 s per-round native IP, 600 s final native IP,
660 s final reserve, 14,460 s total cap, seed 0, native IP gap 0.

Implementation:
- ColGenParams defaults: iteration_ip_time_limit_s=30, iteration_ip_eager=True,
  lp_gap=0.001. Cheap pricing remains optional (False); exact_pricing_interval=5.
- price_flight(heuristic_only=True) returns the certified restricted bootstrap
  incumbent before unrestricted pricing. At least one best-bound departure/lane root
  is searched. Single-root graphs are searched too (the normal bootstrap skips them).
- The persistent worker protocol carries the per-sweep heuristic_only flag.
- Solver runs full pricing every fifth and final allowed round, or immediately after
  cheap pricing adds zero new columns. Only full sweeps supply new global bounds or
  permit convergence/stagnation termination. Cheap rounds retain the previous bound.
- Separate cheap/exact counters and times are reported on every solver exit.
- Per-round IP uses eager preparation, explicit max_eager_ip_rows, and preserved final
  reserve. CLI exposes per-round IP budget, LP gap, cheap mode, exact interval.
- Benchmark has iteration_ip_cheap arm, --lp-gap and --gurobi-threads. Older rounding/LNS
  arms explicitly disable per-round IP to preserve their intended policies.
- Tracer identifies per-round IP by explicit eager argument, not eager=False.
- Source/harness frozen once measurement began; exact source snapshot and hashes archived.

Validation before timed simulations: 321 distinct relevant tests passed, plus Ruff and
whitespace checks. This consists of 175 solver/pool/IP/CLI tests, 137 compiled/LNS/review
regressions, and 9 new cheap-pricing tests. Two failures from new telemetry schema and
runtime-key comparisons were fixed and rerun. Tests cover canonical cheap columns,
exact-score dominance, no unrestricted search in cheap mode, fallback on stagnation,
no false bounds, final exact sweep, defaults/validation, and persistent worker switching.
No independent measured simulations overlap.

Reporter prepared: `analysis/summarize_colgen_cheap_pricing.py`; run after both results
exist. It imports the generic load_case auditor from summarize_colgen_policies.py,
which was extended to normalize the new mode fields and accept infinite pre-certificate
bounds. It verifies inputs/source archive, budgets, coverage, costs, monotone bounds,
no heuristic-derived certificates, final physical verification, and generates figures.

Early cheap-arm results (not the final comparison):
- Round 1: 557.37 s total, 375.03 s pricing, 1,216 new spatial routes, IP cost189,365.36.
- Round 5: 1,256.73 s total, first full sweep138.43 s; LP gap1.8487%, cost165,811.65.
- Round 9: 169 new columns; round10 full search then found1,242 columns.
- Round10: 1,942.75 s total, LP gap0.18164%, cost157,219.06.
- Round14: first pool-optimal IP, cost156,750.04, elapsed2,462.03 s.
- Round15: 2,634.40 s elapsed; full pricing119.64 s; LP gap0.13784% still misses0.1%;
  pool-optimal cost156,663.81. Do not declare convergence here.
- Round16: 2,760.71 s elapsed, pool-optimal cost156,607.02.
All per-round schedules cover1,537 flights. One label-arena restart on flight1600,
same as previous benchmarks; final fallback/verification stats pending.

Earlier multi-column evidence found in Git commit
`b65c4a2f87e7dccfdcc4377c54b6972396f5ff1d`, archived as multi_column_history.txt.
On a different older1,500-flight experiment, pricing_tied_columns1->5 increased pool
44,019->65,805, tightened LP gap0.00924->0.000278, but final schedule cost worsened
184,666.0->198,222.9 with both IPs hitting900 s. Thus earlier result was limited by
integer search; it does not settle performance with the new per-round IP policy.
It returned exact RC ties with low row overlap, not arbitrary top-k. A future test
must continue to sum only one best reduced cost per flight for the global bound.
No multi-column implementation or benchmark was performed for this request.

Cheap arm COMPLETE; standard control now running in parent session81445:
- Wall3,367.68260499998 s (56.1280 min),20 rounds =16 cheap+4 exact.
- LP gap0.0009463753580620304 (0.0946375%), termination lp_gap.
- Final cost156,490.54718195138; ground delay1,484 s; air detour1,550,065.471819514 m.
- All1,537 physically verified,0 denials,0 repairs,0 kernel fallbacks.
- Columns51,076;18,655 actually priced beyond32,421 initial columns.
- Pricing2,169.494264374487 s: cheap1,689.7361568745691, exact479.75810749991797.
- Per-round IP wall608.5782605390996 s; native556.3744487762451 s.
- Final native IP19.585751056671143 s; wall~20.117 s; pool optimal, no cost change.
-7 pool-optimal per-round IP calls (rounds14-20); preceding13 hit30 s limits.
- LP solves244.66936387540773 s. Global integer gap0.009324727171735506 (0.93247%).
- Generic load_case audit of coverage, cost, input hash and IP budgets passed.
- Standard control version iteration_ip_eager is the second child, same exact protocol.
  Do not conclude the fresh comparison until it completes and run the prepared reporter.

Final comparison (supersedes the progress notes above):
- Both arms finished successfully and stopped on the requested 0.1% LP tolerance.
- Standard: 2,759.4237107499503 s (45.9904 min), 13 exact rounds, cost
  156,714.56744042016, ground delay 1,688 s, air detour 1,550,265.6744042013 m.
- Cheap: 3,367.68260499998 s (56.1280 min), 20 rounds (16 cheap / 4 exact),
  cost 156,490.54718195138, ground delay 1,484 s.
- Cheap took 22.043% longer, with 0.14295% lower final cost. It used 36.1582 min
  of pricing versus 33.0520 min for standard. Less time per returned column did
  not compensate for more rounds, LP solves, and per-round IP calls.
- Standard / cheap final LP gaps: 0.0384414% / 0.0946375%; global integer gaps:
  1.01853% / 0.93247%. These are different certificates.
- Both final IPs proved optimality over their generated pools: native search took
  16.78 s / 19.59 s of the available 600 s. Both physically verified all 1,537
  outbound flights, with zero denials, repairs, or reported kernel fallbacks.
- Cheap reached the standard final cost at 43.86 min, versus 45.56 min for
  standard, so the convergence-time result does not rule out an anytime benefit.
- Final pools: 45,879 / 51,076 columns, including 13,458 / 18,655 newly priced
  columns; 38,020 columns shared. Original inputs are identical apart from mode.
- The new standard run reproduces the first 13 rounds of the historical 0.01%
  target run in LP objective, bounds, column counts, reduced-cost summaries and
  incumbent costs. That old run took 97.34 min, hit 30 rounds, cost 156,405.160.
  The looser LP target reduces time by 52.75% with 0.1978% higher final cost.
  This is a historical comparison, not an additional fresh measured arm.
- Decision: CG + eager IP each round and LP gap 0.001 are defaults; standard
  exact pricing remains default. Restricted cheap pricing stays experimental.
  Multiple diverse tied columns with this IP policy is still an untested next step.

Final artifacts: summary/report.md, summary/audit.json, summary/comparison.png,
summary/convergence.png under the experiment root. Both figures visually inspected;
the audit verifies source and input fingerprints, costs, coverage, budgets, bounds,
termination and physical verification. Measured parent session 81445 exited 0.

Postmeasurement cleanup, performed only after BOTH measured runs completed:
- Corrected indentation of four empty-result telemetry entries (whitespace only).
- Clarified heuristic-only worker protocol documentation.
- Removed redundant eager-IP harness monkeypatch and budget replacement now handled
  by production parameters; added positive --gurobi-threads CLI validation.
- Refined report wording and added the historical tolerance comparison.
The exact measured source and harness remain in source_snapshot.tar.gz,
harness_used.py, colgen_trace_used.py, changes.patch, and manifest.json.
Ruff and git diff --check passed. A final 57-test targeted rerun passed in 5.79 s
after cleanup; these overlap the 321 distinct premeasurement tests, not 378 distinct
tests. The report was regenerated after cleanup and its audit passed.

Next benchmark preference (user instruction after reviewing these results): use
12 pricing workers instead of 4. Keep Gurobi at 4 threads to isolate the worker
change unless the user specifies otherwise. This is for the next run; no new
simulation was requested or started, and the measured results above used 4 workers.
