# GPU pruning allowance experiment

This follow-up keeps the original certified search as the default. An experimental
absolute allowance can remove numeric pruning checks from the CPU queue. An
unaudited allowance **never produces a certificate**, including when a search
exhausts its states. A margin that works on a capture is not a proof for new prices,
objectives, geometry, or search lengths.

## Modes

`PricingOptions.gpu_prune_epsilon=None` preserves original FP64 verification of
every proposed numeric deletion. With an epsilon, the GPU can delete a state when
its computed upper bound is at most a downward-converted \(b-\epsilon\), where
\(b\) is the current incumbent. Nonfinite experimental bounds cannot authorize
deletion. This uses the same resident expansion, roots, topology, restrictions,
and absence of approximate dominance as the certified path.

- Epsilon alone: GPU numeric pruning without CPU deletion verification. Completed
  routes still receive canonical checks. Keeping states inside the allowance can
  increase work and cause caps even when the certified verifier would close them.
- `gpu_prune_verify_border=True`: use the GPU rule for states below that cutoff;
  retain original CPU verification for near-cutoff proposals. This remains
  uncertified because the GPU rule has not been proved globally.
- `gpu_prune_audit=True`: original FP64 verification also checks every deletion
  satisfying the epsilon rule and restores rejected proposals. Completed searches
  can certify. Audit timings are separate from the modes that actually skip work.

Original FP64 preparation, signed-price summaries, complete claim memberships,
exact `fsum` repricing, and future-table preparation remain. None of these modes
calls the original full CPU pricer. Legacy `cpu_recompute=False` also changes
dominance and search execution; it is intentionally not the timing control here.

The optional analysis-only completion ablation skips `_canonical_candidate` for
GPU-generated routes. It still constructs and scores preliminary paths on the CPU
using `_path_claims`, `_path_delay_s`, and the original objective. Initial canonical
seed checks and independent post-search/final-solver checks remain. Thus this tests
removing **in-search deletion and completed-route verification**, not removing all
CPU arithmetic or implementing GPU canonical geometry.

## Numerical limits

The audit records actual GPU proposal bounds and the corresponding original FP64
completion bounds. Their difference is not exclusively floating-point roundoff:
the GPU affine geometry and accumulated prices differ from CPU canonical-order
geometry and exact claim unions. In particular, origin/visit overlap cannot be
resolved by choosing a hardware epsilon alone. `epsilon_rejected` counts rejected
deletions satisfying the chosen margin; `rejected_prunes` also includes any border
proposals rejected by the original verifier.

Scaled cancellation fixtures on actual Metal demonstrate that an absolute epsilon
does not generalize with objective magnitude. With benefit \(10^4\), \(10^8\),
and \(10^{12}\), observed zero-rejection thresholds were approximately
\(10^{-5}\), \(0.1\), and \(1000\). These are boundaries for these fixtures,
not recommended production constants. The original benefit-\(10^4\) case loses
an improvement of approximately \(10^{-5}\) when unchecked, and recovers it at
epsilon \(2\times10^{-5}\).

Integer IDs, clocks, topology and predecessor records already live on the GPU.
Scaled integers could also represent prices, but converting continuous LP prices
introduces quantization error and requires a sum/overflow bound. For instance,
nearest-grid spacing \(q\) permits error up to \(nq/2\) across \(n\) distinct
claims before other arithmetic. An outward interval representation could carry
lower and upper totals instead; its scales, signed arithmetic, full claim unions,
and overflow handling must be proved before it can certify deletions. Geometry,
folding, and canonical resource membership are separate porting work.

## Reproduction

Run sustained Metal work under the shared advisory lock. Use this checkout on
`PYTHONPATH`; the root environment supplies Metal and licensed Gurobi. Trusted
capture inputs and the original label cap, chunk size and completion requirements
must match between modes.

```sh
python -m analysis.colgen.execution_replay --capture CAPTURE --flight-ids IDS \
  --batch-size 2 --warm-repeats 2 --out baseline.json
python -m analysis.colgen.execution_replay --capture CAPTURE --flight-ids IDS \
  --batch-size 2 --warm-repeats 2 --epsilon 0.001 --verify-border --out mixed.json
python -m analysis.colgen.execution_replay --capture CAPTURE --flight-ids IDS \
  --batch-size 2 --epsilon 0.001 --verify-border --audit-epsilon --out audit.json
python -m analysis.colgen.gpu_epsilon_experiment replay --epsilon 0 \
  --skip-completion-validation --capture CAPTURE --flight-ids IDS --out unchecked.json
python -m analysis.colgen.gpu_epsilon_experiment solve --epsilon 0 \
  --skip-completion-validation --capture CAPTURE --budget 120 --out FRESH_DIRECTORY
```

The solver validation harness intentionally reports missing certificates as
failures in the unchecked experiments. Accepted-flight counts, independently
verified feasibility and final objective must be compared separately; an
uncertified LP gap is not an optimality guarantee.

## Integration boundaries

Resident controls add a seventh block counter for device deletions and an aligned
GPU-bound array for audits. Expansion controls append a numeric mode, and flight
cutoffs have two FP32 entries (CPU proposal and experimental deletion threshold).
Flag 16 denotes an unaudited deletion; flag 32 identifies audited epsilon
proposals. Endpoint flag 8 and opaque record extension words remain available.
Both shared-source Metal/CUDA adapters use the same interface; CUDA hardware has
not been tested.

The endpoint follow-up remains separate. Its latest mandatory-arrival union fixes
missing costs, not arithmetic noise, and its stronger incumbent search materially
improves capped routes here. On the identical early capture, its saved certified
flight-2 reduced cost is 14.0000000073 versus this branch's capped incumbent
0.5921792820. Equal incumbents across the epsilon modes therefore do not prove
optimal quality on capped flights. Endpoint integration is still needed before
extrapolating these execution-only measurements to the combined design.

## Measurements on 2026-11-07

Machine: Apple M1 Ultra; genuine Metal kernels. Early capture SHA-256
`2d8738324f510ec9f3be38aa67536158a1bcd01c7341cd58bb3816796084a2a0`.
Eight flights: 0, 2, 32, 34, 276, 278, 320, 322. Batch size 2, 250,000 labels per
batch/pass, chunk size 2048. Each mode has one cold call and two warm calls in a
fresh process. Timings exclude graph/runtime creation and include independent
returned-route auditing, which is also reported separately. These short repeated
runs are not a statistical end-to-end speedup estimate.

| Mode | Cold seconds | Warm mean seconds | Searches complete | Certified |
| --- | ---: | ---: | ---: | ---: |
| Original CPU verification | 30.894 | 26.857 | 4/8 | 4/8 |
| GPU only, epsilon 0 | 15.953 | 11.975 | 0/8 | 0/8 |
| GPU only, epsilon 0.001 | 16.138 | 12.184 | 0/8 | 0/8 |
| GPU + CPU border, epsilon 0.00001 | 15.699 | 11.750 | 4/8 | 0/8 |
| GPU + CPU border, epsilon 0.001 | 15.661 | 11.680 | 4/8 | 0/8 |
| GPU + CPU border, epsilon 0.01 | 15.662 | 11.738 | 4/8 | 0/8 |
| Epsilon 0.001 with every deletion audited | 31.516 | 27.527 | 4/8 | 4/8 |
| GPU only, epsilon 0; completion validation skipped | 14.882 | 10.848 | 0/8 | 0/8 |

Every returned reduced cost matched this branch's baseline exactly; independent
canonical checks found no invalid returned routes. All eight hit label caps in
the purely GPU-pruned modes, including the additional margins \(10^{-6}\),
\(10^{-5}\), \(10^{-4}\), and \(0.01\). Faster time to a cap is not an
equivalent completed search. With the CPU border, flights 276, 278, 320 and 322
complete, matching the baseline; the paired batches containing 2 and 32 cap.
The older PR144 batch-size-1 report completed six flights. A batch cap marks both
flights incomplete, explaining that difference without a default-mode regression.

For the first warm call, the 0.001 mixed mode reduced numeric CPU checks from
2,183,100 to 14,866 (99.32%). It performed 2,279,147 experimental device deletions.
The separate mixed audit verified 2,168,203 epsilon-qualified proposals and
rejected zero; no canonical completion was rejected in the captured baseline or
audit. Resident atomic ordering and shared caps can change work counts, so this
is a separate matching-configuration audit, not verification of every record in
the faster run. The pure zero-epsilon audit also rejected zero proposals; its
largest FP64-minus-GPU bound difference was 0.000513159621. The captures therefore
do not establish a required positive epsilon or a universal safe boundary.

The first warm baseline versus 0.001 mixed run separates the main costs:

| Measurement | Baseline | Mixed |
| --- | ---: | ---: |
| Initial CPU preparation, seconds | 1.162 | 1.153 |
| Future preparation, seconds | 6.796 | 6.790 |
| CPU deletion verification, seconds | 8.065 | 0.048 |
| CPU resident predecessor recovery, seconds | 4.610 | 0 |
| Completed-route processing, seconds | 1.336 | 1.343 |
| Metal GPU time, seconds | 0.277 | 0.258 |
| Observed host waits, seconds | 4.619 | 4.465 |
| Native allocations | 2,005 | 2,005 |
| Uploaded megabytes | 562.225 | 562.225 |
| Downloaded megabytes | 208.133 | 140.188 |
| Warm shared-price uploads | 0 | 0 |
| GPU candidate expansions | 3,978,798 | 4,000,135 |

Method timers overlap, including future preparation and waits; do not sum them
as a wall-time partition. The approximately 2.30x warm timing difference is an
experimental pruning result with matched incumbent values/completion counts,
not a certified speedup. Removing canonical checks saved another roughly 1.1s
in the capped no-numeric-verification run: 537 preliminary completions were
processed, while post-search checks accepted all returned routes.

### Full solver and solution quality

A fresh ten-flight subset LP and licensed Gurobi solve accepted all ten flights
and independently verified feasibility in every mode. All four runs returned
the identical total-cost objective **1895.591897429123**, with master objective
**98104.40810257087**.

| Mode | Wall seconds | Numeric CPU checks | Final stopping reason |
| --- | ---: | ---: | --- |
| Certified baseline | 10.465 | 136,371 | `lp_gap` |
| No numeric CPU verification | 10.418 | 0 | `approximate_no_new_columns` |
| Mixed epsilon 0.001 | 10.368 | 20 | `approximate_no_new_columns` |
| No in-search deletion/completion verification | 10.338 | 0 | `approximate_no_new_columns` |

These solver differences are too small and too lightly repeated to establish a
speedup. All four subset solves closed at roots (five bound dispatches, zero
expanded candidates), so the completion-validation ablation was not exercised
inside that solver workload; the fixed-capture experiment above did exercise it.
Only the baseline proves its final LP pricing bound. The unchecked solver's
same observed feasible objective does not provide its own optimality guarantee.
No full-1537-flight solve or NVIDIA hardware run was performed.

### Validation and durable evidence

Focused checks: 311 tests passed in 40.65s, followed by six scaled cancellation
cases in 1.65s (315 distinct cases across the two runs). One expected warning is
from the existing intentional lost-worker test. Tests cover preserved canonical
routes, signed/overlapping prices, unsafe zero-margin pruning and audit restoration,
large objective scales, border verification, explicit uncertified results,
device-side deletion without queue use, exceptional memory/queue/deadline exits,
runtime reuse, and original-CPU-pricer guards. Actual Metal and Gurobi/HiGHS ran.
Shared CUDA source syntax and runtime adapter tests are compatibility evidence,
not CUDA hardware validation.

[`gpu_epsilon_measurements.json`](gpu_epsilon_measurements.json) retains source
fingerprints, capture hashes, all cold/warm work and timing counters, route identity
hashes, statuses, boundary sweeps, and solver outcomes. The earlier pure-epsilon
sweep predates the optional border mode; its separate source fingerprint is
retained. The final-source zero-epsilon and baseline controls were repeated.
Raw local artifacts are under `.agent/epsilon-runs/`.
