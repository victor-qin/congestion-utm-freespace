# Column-generation benchmark tools

These modules preserve the benchmark and validation workflow used for PR 133. Run them
from the repository root with the project environment. The benchmark, capture, replay,
and rows commands accept an explicit source tree and record hashes of that tree and its
inputs. Output directories must be fresh.

## Full benchmark

`benchmark` runs each arm in a fresh process and keeps simulations serial. For a small
two-tree comparison:

```sh
uv run python -m analysis.colgen.benchmark run \
  --baseline /path/to/baseline --fixed /path/to/candidate \
  --scenarios colgen_test --seeds 0 --iterations 2 \
  --workers 2 --out /private/tmp/colgen-pair
```

The validated 1,200-second outbound policy, with 12 pricing workers and a 30-second IP
solve after every round, can be captured from one selected source tree with:

```sh
uv run python -m analysis.colgen.capture \
  --source-root /path/to/source --out /private/tmp/colgen-capture \
  --rounds 1 6 11 --expected-production-flights 1537 --allow-missing-rounds -- \
  --version iteration_ip_eager --scenario density_faa_wing_zipline \
  --seed 0 --flights 0 --iterations 30 --budget 14460 \
  --ip-budget 600 --iteration-ip-budget 30 --ip-reserve 660 \
  --capture-columns --overrun 6 --columns-per-flight 1 --bootstrap-method astar \
  --workers 12 --gurobi-threads 4 --lp-gap 0.001 \
  --exact-pricing-interval 5 --destroy 100 --gap 0 \
  --demand-seconds 1200 --outbound-only
```

Use `--benchmark /path/to/benchmark.py` to capture with an archived benchmark file, or
`--benchmark-module package.module` to select another importable implementation. The
packaged `analysis.colgen.benchmark` module is the default.

## GPU pricing in the full solver

The full column-generation loop accepts `ColGenParams(pricing_device="gpu",
pricing_gpu_backend="metal")`. The benchmark and capture commands expose the same
selection as `--pricing-device gpu --pricing-gpu-backend metal`. CPU pricing remains
the default, with its existing worker pool. `--workers` controls only CPU pricing;
GPU batches execute in the solver process and reuse its device runtime across rounds.
The master LP and restricted IP continue to use Gurobi on CPU.

For a short end-to-end Metal run from this checkout:

```sh
PYTHONHASHSEED=0 .venv/bin/python -m analysis.colgen.benchmark child \
  --root "$PWD" --out ".agent/runs/metal-smoke-$(date +%Y%m%d-%H%M%S)" \
  --version iteration_ip_eager --scenario colgen_test --seed 0 --flights 1 \
  --iterations 3 --budget 120 --ip-budget 15 --ip-reserve 20 \
  --iteration-ip-budget 5 --overrun 0 --workers 0 --gurobi-threads 4 \
  --pricing-device gpu --pricing-gpu-backend metal \
  --pricing-gpu-batch-size 1 --pricing-gpu-max-labels 250000 \
  --pricing-gpu-chunk-size 2048
```

Install the optional `metal` and `gurobi` extras if the environment lacks either package.
The completed CPU capture in `.agent/runs/colgen-capture-20261105-211249` is the existing
1,537-flight baseline; GPU integration does not require rerunning it. To exercise that
full workload on Metal, add the pricing options above to its original capture command
and choose a fresh output directory. The certified prototype can exhaust its label cap
on larger flights; such a stop is an incomplete pricing search, not a convergence proof.

`--no-pricing-cpu-recompute` selects approximate FP32 pricing throughout the loop.
Returned routes still pass CPU trajectory validation, but those sweeps cannot establish
global pricing bounds or prove that no improving route exists. A final Gurobi IP can
prove optimality only over the generated route pool. `result.json` records the selected
backend, `gpu_dispatches`, `pricing_search_complete`, `pricing_bound_certified`, and
`last_pricing_status` so an interrupted or approximate search is visible.
Incomplete sweeps also emit their iteration callback and flight diagnostics before the
final IP begins. A certified pricing upper bound can slightly exceed the returned route's
reduced cost; the master uses that upper bound for its gap. If no column is added and a
positive bound remains, `pricing_bound_gap` reports that unresolved numerical gap.

GPU pricing currently supports one returned column per flight and full pricing sweeps;
`--columns-per-flight` values above one and the `iteration_ip_cheap` arm fail explicitly.
CUDA uses `--pricing-gpu-backend cuda` with an appropriate CuPy installation; NVIDIA
execution remains untested. Neither backend silently falls back to CPU pricing.

Full-solver checks exercise actual Metal pricing with both Gurobi and HiGHS, certified
and FP32 modes, label exhaustion, and final repair. They disable CPU pricing entry points
to catch accidental fallbacks. Valid routes returned by an incomplete search remain usable,
including a repair route returned at the deadline; they do not certify that search.

```sh
.venv/bin/pytest -q tests/test_colgen_gpu_solver.py tests/test_colgen_gpu_options.py
```

## Captured sweep replay

Replay one captured pricing round in a fresh process. Sweep timing includes worker startup
and worker graph setup. Parent acceptance and independent cold claim, cost, and reduced-cost
validation are timed separately.

```sh
uv run python -m analysis.colgen.replay \
  --source-root /path/to/source \
  --capture /private/tmp/colgen-capture/pricing_captures/round_0001.pkl \
  --out /private/tmp/colgen-replay --budget 1800 \
  --expected-workers 12 --gurobi-threads 4 \
  --require-independent-clean --require-all-parent-certified
```

Pass `--reference /path/to/reference-replay` to require output, identity, claim, cost,
reduced-cost, and search-work parity with an earlier replay.

## CPU/Metal pricing prototype

`platform_replay` uses the current source tree and rebuilds only the explicitly selected
flights from a trusted local schema-1 capture. It runs pricing without a master LP or a
Gurobi license. Metal uses Apple's runtime shader compiler through the optional PyObjC
bridge; full Xcode is unnecessary on the tested M1 Ultra. Use a fresh JSON output filename
for each invocation:

```sh
uv run --extra metal python -m analysis.colgen.platform_replay \
  --capture /private/tmp/colgen-capture/pricing_captures/round_0001.pkl \
  --flight-ids 900 --device gpu --gpu-backend metal \
  --budget 60 --max-labels 250000 --chunk-size 2048 \
  --out /private/tmp/pricing-metal.json
```

For a separate CPU comparison, replace `--device gpu --gpu-backend metal` with
`--device cpu` and change `--out`. There is no automatic CPU baseline run. The CPU adapter
calls the existing compiled pricing method. GPU expansion and grouping use the same
`freespace_sim/planner/colgen/pricing_gpu.h` source through small Metal/CUDA adapters.
The CUDA adapter requires an installed CuPy distribution matching the NVIDIA environment;
it has not been tested on NVIDIA hardware. An unavailable backend fails explicitly.

CPU numerical recomputation defaults to enabled. `--no-cpu-recompute` selects GPU
FP32 arithmetic without a pricing certificate; CPU trajectory validation still runs.
The report records each flight's returned route, reduced cost, upper bound, completion
and certificate status, work counts, and source/input hashes. Nonfinite numbers become
JSON `null`, with their field paths recorded in `nonfinite_fields`.
Successful report creation does not imply that search completed: inspect each result's
`status`, `search_complete`, and `certified`. The shared `--budget` includes input and
graph preparation and device setup; a synchronous operation may exceed the deadline.

The public Python interface is `PricingPlatform(PricingOptions(...)).price_batch(problems)`
in `freespace_sim/planner/colgen/pricing_platform.py`. It accepts independent fixed-price
`PricingProblem` records and preserves their order. Compatible flights share GPU time
layers; different occupancy-window configurations run in separate batches. The full solver
uses this interface when GPU pricing is selected, while CPU runs retain the existing
compiled pricing path and worker pool.

The certified prototype follows this algorithm:

```text
prepare topology, all legal departure/lane roots and price tables on CPU
obtain canonically valid shortest-path / supplied incumbent routes
prepare conservative flight-distance and mandatory-origin price bounds
GPU: evaluate completion bounds for every departure/lane root
GPU: bounded beam search over promising roots to find validated incumbent routes
discard heuristic predecessors; restart the certificate pass with every legal root
for each time layer, including future departures after an empty frontier:
    initialize roots; CPU-verify any proposed numeric root deletion
    for each bounded chunk of the immutable current frontier:
        GPU: expand six neighbor choices per label
        GPU: check integer geometry roles, time/hop/revisit limits, forbidden visits
        GPU: compute FP32 scores and propose completion-bound deletions
        CPU: verify each proposed numeric deletion using original prices
        CPU: canonically validate every remaining destination/lane completion
        retain every remaining continuation and its immutable predecessor
    advance only after all chunks and completions have been handled
if predecessor storage fills on this first pass:
    GPU: build a backward relaxed air-cost + positive-visit-price table
    CPU: independently recompute its numerical bounds in FP64
    discard all old predecessors; preserve only validated incumbent routes
    repeat the beam and full searches from every root using the stronger bound
return the best validated route and an upper bound covering every omitted alternative
```

The CPU recomputation does not rerun the CPU pricing DP. A safe completion bound includes
ground cost and minimum remaining flight-distance cost, the deduplicated positive prices
of mandatory origin and already visited lattice rows, and a conservative credit for **all**
negative input prices. Remaining distance uses the prepared reverse graph, including walls.
The distance calculation bounds coordinate, interpolation, folding and length-summation
error; uncertain terminal folds retain the ground-only bound. Outward-rounded FP64
arithmetic protects the final bound. Canonical claims include the discrete visit rows even
when endpoint geometry folds or retimes a route. Only canonically validated routes tighten
the incumbent. A failed completion never prevents other completions from being examined.
The bounded GPU search keeps up to 64 labels per flight and explores up to 32 promising
roots; it supplies only validated incumbents and cannot certify the full search.

Positive prefix prices are accumulated incrementally from original prices. The revisit
rule makes successive visit windows disjoint; exact origin claims are excluded from
later increments to avoid charging an overlapping origin row twice.

If those bounds are too weak, the backward table minimizes the sum of air cost and
positive visit prices from each cell/time to any destination. It separates legal first-hop arcs from later cruise/finish arcs, while ignoring hop,
revisit and endpoint constraints. Every real continuation therefore remains available
in this relaxed problem. Its suffix cost combines with a separate visit-only
prefix sum and an affine air-cost bound. At a destination, it also charges positive
terminal rows and destination cell rows strictly later than every visit window. These
rows cannot overlap the prefix or suffix visits; other endpoint additions are omitted
to avoid double-counting. Directed FP64 arithmetic and a rounding allowance protect the certified
bound. The CPU recomputes this relaxed numerical table, not the original constrained
pricing search. With recomputation disabled, only the GPU FP32 table is used and the
result remains uncertified. The table is built only after the initial label cap, and
retains one value per cell/time; its memory is additional to predecessor storage.

The full pass can close a branch whose proved upper bound is within \(10^{-6}\) of the
incumbent. Its bound is retained in the result's `upper_bound`, rather than replaced by
the incumbent's `reduced_cost`. This preserves a valid global bound and avoids enumerating
exponentially many numerical ties. It does not prove absence of improvement when the
remaining upper bound is positive. All choices dropped by the preliminary beam search
are reconsidered by the full pass or covered by one of these proved bounds.

In this first version, recomputation checks every proposed numeric deletion, rather than
only overlapping error intervals. Certified mode retains distinct histories: a proof that
partial-score dominance preserves canonical geometry costs is still outstanding. The
FP32 switch both removes numeric deletion verification and enables GPU hash grouping by
the full state (refined by root identity). It can miss improvements; its completion flag
means the approximate search finished, **not** that optimality was proved. Final trajectory
validation remains on CPU in both modes. Neither mode is a GPU speedup claim.

`max_labels` limits retained predecessor records per batch and search stage, not total process/GPU memory.
Reaching that cap again after the stronger-bound retry, or reaching the cooperative
deadline, returns an incomplete result with no pricing certificate. Allocation or device
failures raise errors. The current GPU domain is
single-level; unsupported prepared graphs return `status="unsupported"`. Larger flights
can still exhaust the cap; certified state merging remains future work. `labels` counts
all attempts, with `bootstrap_labels`, `initial_full_search_labels` (when retried), and
`full_search_labels` also reported separately. `future_s` and `future_gpu_dispatches`
identify backward-table work.

Focused checks cover exhaustive tiny-route optima, negative prices, endpoint unions,
terminal lanes/walls, forbidden arrivals, batched/chunked execution, near-zero reduced
costs, the FP32 certificate restriction and incomplete searches:

```sh
uv run --extra metal --extra dev pytest -q tests/test_colgen_pricing_platform.py
uv run --extra metal --extra dev pytest -q tests/test_colgen_gpu_bounds.py tests/test_colgen_gpu_future.py
uv run --extra metal --extra dev pytest -q tests/test_colgen_performance.py \
  -k endpoint_union_bound_matches_exhaustive_pricing_oracle
```

Metal cases skip when macOS/Metal is absent; compiler failures on a supported device fail
the tests. The near-zero test disables all CPU pricing entry points and demonstrates an
improvement that plain FP32 loses and CPU-verified GPU search retains.

## Capacity row scan

The row harness measures public capacity separation over the one-column-per-flight
incumbent in a capture. Pool construction and the fractional-load control are reported
outside the measured row scan.

```sh
uv run python -m analysis.colgen.rows \
  --source-root /path/to/source \
  --capture /private/tmp/colgen-capture/pricing_captures/round_0001.pkl \
  --out /private/tmp/colgen-rows --repeats 9
```

`analysis.colgen.trace.ColumnTrace` is an instrumentation helper used by
`benchmark --capture-columns`; it is not a standalone command. `capture` stores trusted local pickle
artifacts, so only replay captures produced by a source you trust.

## GPU validation on original long flights

`gpu_validation` selects the ten longest horizontal-distance requests from a trusted
capture, always retaining regression flight 1752. It preserves request IDs,
coordinates, clocks, simulation configuration, all static terminals, and the
six-extra-hop route domain. `prepare` only writes the selection and input/source
hashes; it does not build flight graphs or initialize the GPU.

```bash
python -m analysis.colgen.gpu_validation prepare \
  --capture .agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl \
  --out .agent/gpu-longflight-selection
```

Use `replay` with a fresh output directory to retain the original full-scenario
LP prices, or `solve` to run the complete simulation using Metal pricing and
Gurobi with new prices for the ten-flight subset. Both modes prohibit CPU full
pricing with runtime guards, persist every pricing result, require actual GPU
dispatches, and return nonzero if any search is incomplete or uncertified.
`solve` additionally requires all selected flights to be filed and the simulation's
independent conflict verification to pass. An optimal restricted-master IP alone
does not pass these checks. Resource controls are `--budget`, `--max-labels`,
`--chunk-size`, and `--batch-size`; they do not shorten any route or remove terminals.

For the ten-flight full solver check, run from the repository root:

```bash
COLGEN_GUROBI_THREADS=4 PYTHONHASHSEED=0 .venv/bin/python -m analysis.colgen.gpu_validation solve \
  --capture .agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl \
  --out ".agent/gpu-longflight/solve-ten-$(date +%Y%m%d-%H%M%S)" \
  --budget 600 --batch-size 2
```

Replace `solve` with `replay` and choose another fresh output directory to exercise
the original scenario's capacity prices. The small subset can legitimately finish in
one pricing round; its new LP prices need not create the congestion present in the
original full scenario. Use both checks when validating route expansion and solver integration.

GPU visit-price sums are cached once per shared `DualView` (one LP price snapshot).
All flights and batches using that snapshot gather their cell/time entries from the
same sparse table. Forbidden rows and future bounds stay flight-specific; a new LP
snapshot creates a fresh cache. CPU numerical verification remains enabled by default.
Callers of `PricingPlatform.price_batch` should reuse one `DualView` across batches;
raw mutable mappings are normalized anew on each call. Both the full solver and this
replay harness share the view automatically.

With persistent flight graphs, GPU geometry bounds are also reused across rounds.
The cache is keyed by topology, configuration and objective; each call still updates
root prices, exclusions and ground-dependent bounds. Root initialization and CPU
bound verification share the same origin occupancy union within that call. Metal
reuses dynamic allocations within each batch and copies directly from NumPy buffers;
every input is refreshed and returned outputs remain independent copies. Buffers
are released at the batch boundary, and certification rules are unchanged.

To time the production GPU implementation on the saved ten-flight workload:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m analysis.colgen.gpu_validation replay \
  --capture .agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl \
  --out ".agent/gpu-longflight/shared-cache-batch2-$(date +%Y%m%d-%H%M%S)" \
  --budget 600 --batch-size 2
```

The final report includes `pricing_wall_s` (sum of backend calls, including first
GPU initialization) and `batch_wall_s`, separately from graph construction and
independent validation in `elapsed_s`. A timing is usable only when `passed` is true.
This command uses the integrated shared cache and cutoff updates without profiling
replacements. Initial topology/row packing is inside pricing time; it is reused on
the same graph in later solver rounds. Saved-case replays construct fresh graphs.

To price every captured flight once at the saved LP prices, add `--all-flights`:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m analysis.colgen.gpu_validation replay \
  --capture .agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl \
  --all-flights --batch-size 2 --budget 3600 \
  --out ".agent/gpu-all-round1-$(date +%Y%m%d-%H%M%S)"
```

This is one cold pricing sweep in captured request order, with no master LP or IP
solve. `--all-flights` also supports `prepare` for inspecting the selection without
starting pricing. It does not change the ten-flight default. Each completed batch
is saved in `pricing.json`; `result.json` reports success only if all flights have
complete certified pricing results and returned routes pass validation.

To replay specific failed flights without a case-specific diagnostic script, supply
their IDs explicitly. The harness preserves this order, the original clocks and
route domain, and all canonical validation and CPU full-pricing guards:

```bash
.venv/bin/python -m analysis.colgen.gpu_validation replay \
  --capture /path/to/pricing_captures/round_0001.pkl \
  --flight-ids 2 32 278 322 --batch-size 1 --budget 600 \
  --out /private/tmp/gpu-selected-replay
```

Replace `replay` with `prepare` to inspect the selection without running pricing.
Duplicate or absent IDs are rejected. `--flight-ids` and `--all-flights` are mutually
exclusive and available only for `prepare` and `replay`; omitting both retains the
ten-flight default. Use `--batch-size 1` to measure each flight's completion status
without sharing its label cap with another flight.

The `.agent/` paths above refer to ignored local captures and diagnostic evidence,
not files supplied by a checkout. Keep those artifacts locally and pass your own
trusted capture path and a fresh output directory. Reusable replay commands live
in `analysis/colgen/`; archived profiling scripts and reports remain local evidence.

See [GPU development scope and endpoint-flag plan](GPU_DEVELOPMENT.md) for the
implemented reachability bound, the proposed cheaper endpoint accounting,
merge boundaries, and the separate CPU preparation work.
