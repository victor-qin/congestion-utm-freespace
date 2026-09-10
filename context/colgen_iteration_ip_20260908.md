# CG with a full-pool IP every round

User request: "try CG + IP solve on every round" after discussing LNS as a
restricted IP warm-start heuristic. Work remains in
`/private/tmp/congestion-colgen-lns-20260905`, branch `codex/colgen-lns-main`.

## Implementation

`ColGenParams.iteration_ip_time_limit_s` (default 0) replaces the in-loop
rounding/LNS heuristic with a full-pool IP after each completed pricing sweep
and column insertion. Each IP is bounded by the pricing deadline, preserving
the final reserve, and its best capacity-checked schedule is canonicalized and
passed forward as the incumbent. It may retain the previous incumbent on a
timeout. The master still permits omitted flights at penalty M; this is not a
new full-coverage equality constraint. The measured final schedules cover all
763 flights and pass simulation conflict verification.

The initial implementation uses lazy IP row separation (`eager=False`) to avoid
eagerly carrying every bindable row through subsequent LPs. Per-round telemetry
records statuses, timings, columns, coverage, costs, separation rounds and rows
added. The tracer distinguishes iteration IPs from the final IP and writes
separate Gurobi logs. Reentrant IP calls clear old trajectory telemetry; skipped
final IPs no longer report the previous round's search as a final solve.

Production defaults are unchanged. The benchmark adds `iteration_ip` and
`iteration_ip_eager` arms, selected with `--versions`. The eager arm is a
benchmark-only wrapper that prepares all bindable rows before each IP, using
the same code path as the final IP. It preserves the separate native search
budget and captures row setup time. It is an intentional preparation-policy
ablation, not a new route generator.

## Completed full run (lazy capacity separation)

`analysis/colgen_iteration_ip_20260908/long` reproduces the exact outbound-only
763-flight subset of seed-0, 600-second demand, with 182 static terminals,
four pricing workers and four Gurobi threads. Maximum 30 CG rounds, LP cost-gap
target 0.01%, zero native IP gap, 30-second per-round IP cap, 600-second final
IP cap, 660-second final reserve, overall budget 14,460 seconds. The controls
are archived `analysis/colgen_outbound_only_20260908/long` (LNS) and
`analysis/colgen_outbound_only_20260908/cg_rounding` (rounding).

| Metric | CG + LNS | CG + per-round IP |
|---|---:|---:|
| Whole simulation | 1,260.196 s | 2,648.573 s |
| Pricing | 1,040.121 s | 1,553.592 s |
| CG rounds | 19 | 30 |
| Per-round IP wall | 0 | 830.069 s |
| Native final IP | 2.741 s | 4.078 s |
| Final congestion cost | 77,927.875 | 77,423.117 |
| Final verified coverage | 763 | 763 |
| Total ground delay | 980 s | 472 s |
| Global cost gap | 1.1783% | 0.5571% |
| Columns | 19,748 | 22,703 |

The IP-every-round run takes 2.10x as long for a 0.6477% lower final congestion
cost. Ground delay falls by 508 seconds (51.84%), offset by a 3.241-unit increase
in weighted air-detour cost. It hits the 30-round iteration cap without meeting the LP target;
the final IP is optimal only over its generated column pool.

The first 13 per-round calls retain the ground-delay seed (cost 94,127.940).
They repeatedly solve provisional IPs and discover missing capacity rows, or
time out without a better globally claim-feasible candidate. Round 14 first
returns a better feasible schedule at cost 77,598.294, available after
1,477.996 seconds. Later improvements reach 77,423.117 at round 30. Eight of
the 30 per-round IPs prove their current pool optimal. IP separation adds
63,634 rows during CG; later LP/MIP states and duals differ from the controls.
Pools share 17,478 exact timed columns, with 2,270 only in LNS and 5,225 only
in the per-round-IP run. This is a solver-policy comparison, not an
identical-pool warm-start comparison.

The final native IP takes 4.078 seconds after 2.652 seconds of eager row setup,
with no additional objective improvement over round 30. The long-run evidence
does not justify claiming that per-round IPs are inherently ineffective:
loading constraints lazily consumed most early per-round budgets.

## Follow-up

A separate three-round experiment at
`analysis/colgen_iteration_ip_20260908/eager_three_rounds` prepares the IP rows
up front, to test early schedule quality without repeated provisional solves.
It uses the identical demand and 30-second native search cap, retaining the
600-second final-IP cap, and deliberately does not target full CG convergence.
It starts only after the long run and a successful 30-flight eager smoke test
have finished.

The follow-up completed in 693.587 seconds (11.56 minutes), with all 763 flights
verified and zero denials. Final cost is 79,477.312, 1.9883% above the converged
LNS control. It intentionally stops after three CG rounds; LP gap is 4.960%,
and the final global cost gap is 7.092%, so it is not a converged comparison.
The final restricted IP is optimal. Pricing takes 383.473 seconds, per-round
IPs take 94.752 seconds including setup/canonicalization, and the final native
IP takes 134.279 seconds.

| Eager round | Feasible cost | IP wall incl. canonicalization | Native passes | Status |
|---|---:|---:|---:|---|
| 1 | 81,861.607 | 32.359 s | 1 | time limit |
| 2 | 80,386.677 | 31.177 s | 1 | time limit |
| 3 | 79,513.830 | 31.216 s | 1 | time limit |

The first-round pool matches the lazy run exactly: 16,638 timed columns with
identical costs. Its eager preparation takes 1.024 seconds and materializes
93,400 additional rows. Instead of the lazy call's eight provisional solves
and unchanged seed, one native solve returns a feasible improvement. It is
available after about 4.9 minutes and is already better than the LNS control's
best pre-final-IP cost (85,075.37). Later eager row preparation takes about
0.16 seconds per round. The full third-round schedule is available at about
9.25 minutes; the final IP spends a further 134 seconds searching/proving for
only a 36.518-unit cost improvement (0.046%).

Interpretation: complete-row IPs are useful for early primal quality; the
full 30-round lazy experiment is slower and does not establish a general
end-to-end speedup. The three-round eager pilot is an early-stopping quality
tradeoff, not evidence of equal-convergence speedup. A longer eager or periodic
IP experiment would be needed before making per-round IP the default policy.

## Audit and validation

`analysis/summarize_colgen_iteration_ip.py` verifies source archives and captured
harness/tracer hashes, package versions, exact inputs and solver parameters
(apart from the intended heuristic policy and the short follow-up's iteration
cap), per-round incumbent cost monotonicity, bounds and reported native budgets.
It recomputes filed congestion costs from trajectory metrics and verifies the
recorded full coverage and simulation conflict checks. Results and figures are
in `analysis/colgen_iteration_ip_20260908/summary`.

Six focused regression cases cover per-round IP/LP transitions on both backends,
incumbent feasibility and feedback, final reserve protection, budget validation
and stale IP telemetry. The existing solver/LNS suite has 118 cases; it caught
missing statistics keys on empty/timeout exits. That regression passed after
the correction, alongside all six new cases. Ruff and diff checks passed.
Both 30-flight smoke runs verified all flights and solved their one IP round
to optimality. The eager smoke also exercised the skipped-final-IP path.
