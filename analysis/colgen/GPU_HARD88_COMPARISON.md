# Endpoint integration and the 88-flight comparison

The measurements below predate the final PR143 departure-screening and early-retry
changes. PR144 has since been rebased onto the PR143 merge `0cbd284a`. To compare
the **current source** using the same 88 requests, run from this checkout:

```sh
GPU_REF=/Users/victor/Documents/Research/congestion-utm-freespace
PYTHONPATH="$PWD" "$GPU_REF/.venv/bin/python" -m analysis.colgen.compare_gpu_checks \
  --capture "$GPU_REF/.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl" \
  --metal-lock "$GPU_REF/.agent/gpu-followups-20261106/metal-benchmark.lock" \
  --out ".agent/hard88-checks-$(date +%Y%m%d-%H%M%S)" \
  --budget 900 --max-labels 250000 --epsilon 0.001 --repeats 1
```

This runs three fresh LP/IP solves: full checks, epsilon with near-cutoff checks,
and no per-proposal pruning or completion checks. CPU preparation/screening,
canonical seeds, route construction/scoring, post-pricing audits and the final
independent interflight check remain active. The latter two search modes remain
uncertified. This is a solver comparison, not a fixed-price warm-cache replay.

`comparison.csv` and `comparison.json` report process and solver wall time,
final total cost, accepted flights, feasibility, completion, certification and
stopping reason. Per-run logs and detailed manifests/results are retained. Source
and input hashes must match across modes. Label caps and uncertified results are
reported without skipping later modes; execution or feasibility failures make the
comparison command fail. A retained certified bound does not imply the final
pricing sweep finished. Use `--repeats 3` to rotate the mode order across repeats.

Measured on Apple M1 Ultra, 2026-11-07 UTC. The endpoint algorithms improve the
final feasible schedule; the execution changes alone do not show a wall-time
win on this workload. Experimental epsilon pruning with near-cutoff CPU checks
reduces warm pricing time by 54.0% and the matched, capped solver run by 19.3%,
with the same observed results. Removing every numerical check terminates the
solver earlier and produces a worse schedule.

Production sources: original execution PR144 `23af7119`, endpoint PR143
`e1257b51` (production `d6ac84dc`), and integrated PR144 `623ab797`.
The endpoint source is pinned before its subsequent experiments. Full source
hashes, per-flight results, bounds, counters and input hashes are in
[gpu_hard88_measurements.json](gpu_hard88_measurements.json).

## Final solution quality and solver runtime

Every run accepted **88/88 flights** and passed the independent conflict check.
These are fresh subset LP/IP solves, with original coordinates/clocks and new
prices for the 88 selected requests. Total cost is the actual filed objective,
not a sum of independent pricing-route costs. Lower cost is better.

| Mode | Wall seconds | Final total cost | Pricing results | Termination |
| --- | ---: | ---: | ---: | --- |
| Original execution, original checks | 22.362 | 8860.244488722932 | 9 | Round 1 cap, flight 898 |
| Endpoint only, original checks | 80.896 | 8716.244488722932 | 103 | Round 2 cap, flight 2224 |
| Endpoint + execution, original checks | 82.116 | 8716.244488722932 | 103 | Round 2 cap, flight 2224 |
| Endpoint + execution, epsilon 0.001 + near-cutoff checks | 66.286 | 8716.244488722932 | 103 | Round 2 cap, flight 2224; uncertified mode |
| Endpoint + execution, no numerical checks, epsilon 0 | 5.389 | 8864.244488722932 | 1 | Round 1 cap, flight 1400; uncertified mode |
| Same, also no completion checks | 5.358 | 8864.244488722932 | 1 | Round 1 cap, flight 1400; uncertified mode |

The endpoint improvement reduces final cost by **144.0 (1.625%)** relative to
the original runtime. Both checked endpoint versions complete a full first
pricing sweep, retaining a valid cost lower bound of **8604.350169676123**.
Their final solution has a **1.2837% global cost gap** to that bound. The new
second-round cap prevents convergence: none of these full solves proves
optimality. `pricing_bound_certified=True` refers to the retained earlier
bound; it does not mean the final incomplete pricing sweep certified.

The checked endpoint-only and combined solves both expand **1,972,412**
candidates. Their 80.90/82.12-second timings show no execution-only speedup.
The epsilon/border solve performs the same candidate work and reaches the same
cap and cost in 66.29 seconds. CPU numerical checks fall from **2,355,282 to
7,291**, while preparation remains **51.28/51.45 seconds**. The experimental
run produces no pricing bound of its own. Its identical feasible cost can be
compared with the checked run's bound, but its search remains uncertified.

The 5-second runs are not equal-completion speedups. Flight 1400 finds the same
reduced cost, approximately 20, as the checked run, but fails to exhaust its
alternatives. The solver stops before pricing the other flights. Their final
cost is **148.0 worse** than the endpoint/border result. The master/IP and final
conflict checks remain active in every experiment. Completion-check removal
still uses preliminary CPU claim/objective calculations, canonical initial seeds
and independent post-search/final audits; it is not an all-GPU geometry solver.

## Fixed-price hard-case replay

The cohort is exactly `hard88.after.rows` from PR143's committed
`arrival_union_results.json`, in its original order. The early capture SHA256 is
`2d8738324f510ec9f3be38aa67536158a1bcd01c7341cd58bb3816796084a2a0`.
These prices come from the original 1537-flight scenario. They remain fixed
across cold/warm calls, unlike the fresh subset solve above.

| Mode | Cold seconds | Warm seconds | Complete / 88, each pass | Certified / 88 | Reference reduced costs matched / 88 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original execution | 529.469 | 492.887 | 26 | 26 | 63 |
| Endpoint only | 137.458 | 99.623 | 88 | 88 | 88 |
| Endpoint + execution | 143.944 | 104.580 | 88 | 88 | 88 |
| Epsilon 0.001 + near-cutoff checks | 88.283 | 48.062 | 88 | 0 | 88 |
| No numerical checks, epsilon 0 | 137.565 | 97.973 | 45 | 0 | 88 |
| Also no completion checks | 116.082 | 76.054 | 45 | 0 | 88 |
| Epsilon 0.001, every proposal audited | 147.339 | 107.347 | 88 | 88 | 88 |

The original runtime's 25 reduced-cost mismatches all occur on capped flights;
its largest shortfall is **16.0**, on flight 218. All its certified results agree
with the endpoint reference. The 26 flights certified by every checked version
have summed batch times, original/endpoint-only/combined, of
**118.759/34.434/34.466 seconds cold** and **105.910/20.902/21.659 seconds warm**.
These sums exclude the outer independent result audit. The whole-cohort
529-to-144-second comparison changes certificate coverage and must not be
presented as a matched 88-certificate speedup.

All endpoint modes match the reference's returned reduced costs within
\(10^{-6}\); the completion ablation's largest difference is
\(1.82\times10^{-12}\). Matching an incumbent does not mean search completion.
Without numerical checking, 43 flights still hit caps, and warm candidate work
rises from **5.90 million** to **73.26 million**. Near-cutoff CPU checks let the
search close tied alternatives that FP32 alone leaves unresolved. This is why
keeping those checks is faster than removing all numerical verification here.

## Preparation, transfers and verification

Warm values below are instrumented observations; timing regions overlap.
Metal bytes count explicit shared-buffer copies, not PCIe traffic. Endpoint-only
copy totals are observed through its host-array adapter; the combined runtime's
native counters also include owned buffers. Allocation bytes are cumulative,
not peak residency.

| Warm metric | Endpoint only | Combined, original checks | Combined, epsilon/border |
| --- | ---: | ---: | ---: |
| Initial preparation seconds | 20.726 | 19.264 | 19.029 |
| CPU numerical verification seconds | 43.905 | 45.017 | 0.661 |
| Resident predecessor recovery seconds | N/A | 7.266 | 0.188 |
| Completion handling seconds | 4.258 | 4.331 | 4.244 |
| GPU command time seconds | 0.640 | 0.675 | 0.646 |
| Observed host wait seconds | 7.134 | 7.071 | 5.673 |
| Allocations | 13873 | 17530 | 17530 |
| Cumulative allocated MB | 484.078 | 2438.605 | 2438.605 |
| Uploaded MB | 6125.184 | 867.196 | 867.196 |
| Downloaded MB | 453.177 | 273.286 | 189.968 |
| Command waits | 16401 | 16168 | 16168 |

The combined runtime uploads one 263964-byte price store on the cold call,
then records **zero price uploads and 88 cache hits** on each warm call.
Fewer copies do not translate into an execution-only wall-time win here.
The epsilon/border mode reduces warm CPU checks from **4,705,161 to 61,570**
(98.69% fewer), with **4,722,954 GPU deletions**. Canonical completion validation
still runs 1674 times. There is no asynchronous CPU/GPU pipeline or persistent
CPU preparation pool in this branch.

Across the cold/warm epsilon audit, **9,272,661 epsilon-qualified proposals**
reach FP64 verification, with **zero rejections**. Each pass also retains
**1555 ordinary near-cutoff proposals** rejected by the verifier. A rejected
proposal is not proof that deleting it would lose the optimum: the FP64 bound
can itself be conservative. Actual canonical completion checks record zero
rejections in the combined checked and border runs.

The largest observed FP64-minus-GPU bound deficit is **0.001170244643**, across
all audited proposals, not just epsilon-qualified ones. It includes affine and
claim-union evaluation differences, not solely rounding. Zero observed
qualified rejections does not prove epsilon 0.001 universally safe. The scaled
cancellation regressions from [GPU_EPSILON_EXPERIMENT.md](GPU_EPSILON_EXPERIMENT.md)
still pass and demonstrate failure of a universal absolute margin.

The audit is a separate run, not the exact fast-run trace. Atomic ordering and
aborted/retried passes can change counts. GPU direct-deletion counters can
include a final block that hits a cap before CPU verification would process its
queued records. Resident label counts also include partial allocations before
an overflow; the host implementation can stop before allocating that layer.

## Implementation and validation

Commit `623ab797` ports the committed endpoint algorithms and tests from PR143:
mandatory destination/final-visit claim unions, paid masks, endpoint-aware
preliminary ranking/diversity, and bounded remaining-hop future tables. The
[endpoint proof](https://github.com/victor-qin/congestion-utm-freespace/blob/e1257b51b6271ab2b1f1965dbb1923e8a7d5f022/analysis/colgen/GPU_SEARCH_BOUNDS.md)
applies unchanged to their FP64 bounds.

Resident records now have nine fixed words, including paid-mask words 7/8,
followed by recent cells. The shared-price arguments retain indices 12–16;
endpoint visits/metadata/entries/trees occupy 17–20. CPU prefix recovery
reconstructs paid masks from original membership and predecessors, independently
of device bits. Separate per-layer mask buffers preserve two-layer block
ownership. Copy, compaction and restoration preserve all record words.

Resident searches reuse mask trees discovered by roots and preliminary search.
Unseen masks safely fall back to the weaker bound. GPU future proposals use the
unrestricted suffix table; preliminary search and FP64 verification retain
exact remaining-hop queries. These are conservative interface limitations,
not stronger heuristics or unverified deletion rules. Mask/tree generations
remain fixed within a resident pass. PR143's later experiments are not imported;
the overlapping `gpu_pricing.py` / `pricing_gpu.h` interfaces need reconciliation
when integrating further endpoint changes.

**385 focused tests passed in 44.80 seconds**, including actual Metal, licensed
Gurobi/HiGHS small integrations, signed/new prices, changed roots/restrictions,
mask words, overlap, budgets, deadlines, rejected completions and restored
proposals. A new resident test corrupts device paid bits and forces proposals;
FP64 reconstruction still recovers the exhaustive oracle or reports a deadline.
The explicit-ID solver harness test verifies unchanged request selection and
fresh prices. All full CPU pricing entry points are guarded in the current
harness; none is used as fallback or certificate producer.

Generated CUDA source passes a C++ syntax/interface check with CUDA annotations
and atomics stubbed; **no NVIDIA hardware was used**. Ruff and whitespace checks
pass. The unrelated missing `analysis.profile_native_dp` fixture on the base is
why the full suite was not run. Full 88-flight solver reports intentionally fail
the harness's *complete pricing* requirement because of their explicit caps;
this is separate from their successful final feasibility checks.

## Reproduction and limits

Use the `cohort` list in the measurement JSON as `--flight-ids`, in order, with
the original early capture. Existing harnesses are:

- `execution_replay.py --batch-size 1 --max-labels 250000 --chunk-size 2048 --budget 1800 --warm-repeats 1` for cold/warm pricing. The budget is per flight.
- Add `--epsilon 0.001 --verify-border` for mixed checking, and `--audit-epsilon` for full auditing.
- `gpu_validation.py solve --flight-ids ... --batch-size 1 --budget 900` for a fresh subset solve, with 65 seconds reserved for the final IP and a 60-second IP limit.
- `gpu_epsilon_experiment.py solve --epsilon 0.001 --verify-border ...` for the diagnostic solver; use `--epsilon 0` and optionally `--skip-completion-validation` for the ablations.

Always take the shared `fcntl` lock at the reference root's
`.agent/gpu-followups-20261106/metal-benchmark.lock`. Every measured run was
serialized. Timings exclude queueing for that lock. There is one cold/warm pair
and one fresh solve per mode; these are not statistical estimates. Replay
excludes graph construction, snapshot indexing and runtime initialization,
which are recorded separately. Full-solver elapsed time includes initialization.

Original controls run from immutable archives of the named commits. The same
current selection/observation harness is used for the endpoint-only control,
with unsupported, unset epsilon constructor options omitted. Production hashes
are checked against their pinned source; every integrated mode uses identical
production files. Raw artifacts are local `.agent/hard88-runs`; the committed
JSON retains the evidence needed to inspect the comparisons.
