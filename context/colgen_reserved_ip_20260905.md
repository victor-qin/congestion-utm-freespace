# Reserved IP search and column provenance

The follow-up requests an actual 600-second final IP search allowance, no rounding
when the existing LNS path can avoid it, examples of priced columns, LNS diversity
measurements, and a long column-generation comparison approaching 30 iterations.

Worktree: `/private/tmp/congestion-colgen-lns-20260905`, branch
`codex/colgen-lns-main`. Main and the source LNS worktree remain untouched.

## Budget and reporting changes

`ColGenParams.ip_reserve_s` explicitly allocates a tail of the existing whole-solve
deadline to final IP setup and search. None preserves the previous automatic
reserve. It must be finite, nonnegative, and less than the whole budget.
`--colgen-ip-reserve` exposes it through the experiment CLI. It does not silently
extend the whole solve. Every solver exit reports the effective reserve.

The benchmark uses 660 seconds of reserve with a 600-second native IP budget:
the extra 60 seconds protects setup and cleanup. LNS with the complete ground
seed already bypasses randomized rounding; the observer counts those calls.

The Gurobi MIPSOL trace previously read MIPSOL_OBJBST, which can describe the
previous incumbent or a sentinel on the first callback. It now records the best
MIPSOL_OBJ seen, so warm-start and improvement timing are meaningful. This
changes reporting only. LNS also reports the exact number of distinct sequential
trial schedules using stable column indices, independently of objective ties.

## Measurement

`analysis/benchmark_colgen_lns.py` supports `--ip-reserve` and `--capture-columns`.
`analysis/colgen_trace.py` observes one actual simulation batch after warmup:

- Each timed path is stored once; columns refer to stable path/master indices.
- Every priced column records its originating sweep and actual reduced cost.
- Spatial families include the flight, altitude, terminal lanes and cell path
  with consecutive holds removed. Timing shifts are counted separately.
- All LNS calls are recorded, including work before an incomplete last sweep.
- The observer asserts LNS never changes the number of columns.
- Final-IP setup, actual native time limit/runtime, incumbent history and the
  before/after selections are recorded, with a native Gurobi progress log.
- Exact inputs, source hashes, source archive and measured harness/tracer are
  retained. The analyzer audits their hashes and compares identical demand.

The short run has 1,560 seconds overall: 900 for construction/LP/pricing and a
660-second reserved tail. It uses the entire FAA Wing/Zipline seed-0 sample
generated over 600 seconds: 1,526 legs, 182 terminals, horizon 7,200 seconds,
nominal return anchors, four pricing workers and four Gurobi master threads.

Output: `analysis/colgen_reserved_ip_20260905/short/`.
The completed short run used 1,559.767 seconds of simulation wall time. Native
Gurobi received a 600-second cap after 4.146 seconds of setup and ran for 600.021
seconds. It hit the time limit after one root node; it did not prove optimal.

| Stage | Coverage | Congestion cost | Penalized cost |
|---|---:|---:|---:|
| Ground seed | 1,526 | 260,047.879 | 260,047.879 |
| Last LNS | 1,526 | 257,182.118 | 257,182.118 |
| Raw IP | 1,525 | 241,509.445 | 251,509.445 |
| Final IP + repair | 1,526 | 241,742.597 | 241,742.597 |

Raw IP improved penalized cost by 2.206% over LNS. Final repair restored flight
1432, yielding a verified full schedule 6.003% cheaper than LNS. Its ground delay
was 36 seconds; no pooled column matched its departure step and spatial path.
The restricted IP gap is 21.934% for the raw IP. The final normalized global gap
is 100%; this is not converged column generation.

One complete pricing sweep and part of the second supplied 2,331 columns:
2,015 new spatial families and 316 timing variants. The final pool had 34,856
columns and 3,541 spatial families. All LNS calls totaled 1.969 seconds; rounding
calls were zero. LNS selected 18 priced columns; IP selected 229. **This is not a
controlled same-pool comparison:** 969 columns arrived after the last LNS call,
and IP selected 68 of them. Do not attribute all quality improvement to stronger
IP search or inadequate LNS exploration; later columns and final repair matter.

`analysis/summarize_colgen_reserved_ip.py` audits source archives, actual native
budgets, final per-flight costs and retained route geometry, including filing's
mid-edge interpolation. It produces a report, audit JSON, incumbent PNG and
two route galleries with zooms and ground delays. The completed short report is
under `analysis/colgen_reserved_ip_20260905/short_summary/`.

The long comparison is under `analysis/colgen_reserved_ip_20260905/long/`:
same full demand and worker counts, max_iterations=30, whole budget=14,460 seconds,
IP reserve=660 seconds, native IP budget=600 seconds, ip_gap=0. The first sweep
finished at 572.491 seconds with the same 1,362 columns as the short run.
Pricing subsequently met the LP tolerance after 25 full sweeps (0.0082% gap,
below the 0.01% target), so it stopped before the 30-iteration maximum. The pool
contains 48,945 columns: 16,420 additions comprising 15,120 new spatial families
and 1,300 timing variants. Native Gurobi confirmed the full 600-second time limit
and zero MIP gap target. The long run is complete; the audited comparison and
PNGs are in `analysis/colgen_reserved_ip_20260905/summary/`.

Long-run LNS took 23.130 seconds across 25 calls. The median call generated two
distinct sequential trial schedules out of 16; the range was 1–10, and 11 calls
repeated a single schedule throughout. Its last cost improvement was call 16;
the pre-IP cost is 247,015.739 with all 1,526 flights covered.

| Long-run stage | Coverage | Congestion cost | Penalized cost |
|---|---:|---:|---:|
| Last LNS | 1,526 | 247,015.739 | 247,015.739 |
| Raw IP | 1,525 | 183,117.124 | 193,117.124 |
| Final IP + repair | 1,526 | 183,520.457 | 183,520.457 |

Native Gurobi ran 600.074 seconds, after 6.199 seconds of setup, and timed out
without proving the restricted IP optimal. Final repair restored flight 1518
at cost 403.333. The full schedule passed conflict verification. Whole simulation
wall time was 8,511.115 seconds (2 h 21 m 51 s), of which 5,890.152 seconds were
pricing. The LP converged after 7,871.652 seconds and 25 full sweeps, below its
0.01% cost-gap tolerance, before the maximum of 30 iterations.

The final cost is 25.705% below the LNS incumbent and 24.084% below the completed
short run. Raw IP penalized cost improved 21.820% over LNS; the headline includes
final repair. The global lower bound is 179,908.267, giving a 1.968% gap measured
as (returned cost - lower bound) / returned cost. This is not an optimality proof.
The restricted IP gap, 5.242%, refers to the raw IP with its omitted-flight
penalty and a different (pool-only) bound.

LNS selected 105 priced columns; IP selected 961. Only 182 columns were added
after the last LNS call, and IP selected 20 of these, so the stage comparison is
still not a controlled identical-pool experiment. IP used 166 columns from
sweeps 16 onward; LNS stopped improving at call 16 and selected none of those
later columns. The final joint LNS repair had 126 options for 100 flights: at
least 74 released flights had no usable alternative. Its tiny model was solved
to optimality, so more repair time alone would not enlarge its choices.

Actual examples include flight 565 (column 40000, sweep 7, ground delay 628 to
0 seconds), flight 843 (column 39143, sweep 6, 600 to 20 seconds), and flight 443
(column 46861, sweep 17, 580 to 40 seconds). Their reduced costs when generated
were numerically near zero. They became valuable for the integer schedule when
the full IP coordinated the fleet; LP reduced-cost magnitude alone did not
indicate that eventual benefit.

After measurement, a logging-only fix in `batch.py` corrected advice to increase
the separate IP search cap and reserve, and removed stale gap-scaling wording.
It was the only production-source difference from the measured source archive;
`analysis/colgen_reserved_ip_20260905/post_benchmark_logging.patch` records it.
The existing uncertified-IP warning test passed, as did Ruff and diff checks.

## Interpretation

The restricted-master LNS reuses existing columns. It does not generate routes.
It provides a feasible primal incumbent and IP warm start. Existing master
columns normally have nonpositive reduced cost under optimal restricted-LP
duals, so passing an incumbent column into pricing does not supply a new positive
cutoff. Pricing's bootstrap search is a separate mechanism.

The repository's standalone `freespace_sim/planner/lns/` does replan with A* or
SIPP. `context/solver_brainstorm_lacam_cbs.md`, section 4.3, proposes using those
routes as columns. That integration is not present in the measured master LNS.

In the short run's first two LNS calls, 16 sequential trials produced only five
distinct schedules each. The first joint repair had 118 columns for 100 flights;
the second had 114. Original columns are retained, so at least 82 and 86 flights,
respectively, had no alternative in their joint repair. LP-support and frozen-
capacity filters sharply limit the usable neighborhood. The final IP can use
zero-LP-weight columns and coordinate every flight.

## Validation so far

- 73 LNS/experiment-CLI tests passed.
- The 86 nonslow solver tests initially exposed one missing stats key on empty
  exits. After fixing all exits, all eight relevant exit/reserve cases passed;
  the other 78 had passed in the full solver-file run.
- A real 98-flight smoke exercised native Gurobi IP, full tracing and final
  conflict verification: cost 3,996.742 to 3,439.076, restricted IP optimal.
- Ruff passed for changed production code, tests, harness and observer.

The small smoke is not the full 1,526-flight benchmark and must not be reported
as its result. Final schedules must pass simulation verification. A restricted
IP solve, even optimal, does not prove convergence over ungenerated routes.
