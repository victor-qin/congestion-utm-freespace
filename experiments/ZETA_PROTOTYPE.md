# Simulator-backed any-angle column-generation prototype

Implementation now lives in the shared [Zeta package](ZETA_PACKAGE.md), usable
by column generation and standalone FCFS planning.

This is an explicit experimental entry point. The production planner and its DP remain the default. `solve_zeta` runs a live column-generation loop using the existing `RestrictedMaster`, with physical any-angle route pricing in place of the neighbouring-hop DP. It follows Zetahex's finite-waypoint, any-angle and paid-row-history approach; it is a Python reference implementation adapted to this simulator, not a call to Zetahex's native integer-clock kernel.

For the search model and the proposed reusable terminal-to-cell route caches, see
[How Zeta* searches and what we can precompute](ZETA_SEARCH_AND_CACHING.md).

## Route and reservation contract

- Candidate waypoints are the existing flight corridor's hex centres. A connection joins any two distinct candidates within `ZetaOptions.connection_radius_m`, initially 480 m. Motion can therefore cross hex boundaries at any angle. Bend locations remain finite.
- Departure steps, reported ground delay, the existing single cruise level, terminal lane choices, climb/lane clocks, endpoint hover volumes, buffers and objective weights retain their existing meanings.
- Horizontal geometry and subdivision come from the simulator builder. Nominal timing remains the default. With `arrival_snap_s=1.0` and `speed_fraction=0.05`, each candidate leg chooses the nearest one-second arrival reachable at 95–105% of nominal speed (later arrival wins a tie), or preserves nominal timing when none is reachable. Sub-box timing is proportional to its nominal duration. Search and final reservations use exactly the same clock. This defines one timing choice per leg; it does not optimize over all speeds in the band.
- Physical lateral length is bounded by the old hop allowance multiplied by the lattice pitch; the simulator's folded-distance detour limit is also enforced. Walks may revisit vertices or pass their eventual destination before terminating.
- New long connections cannot cut through an own-terminal exit-radius disk. Adjacent connections remain available under their existing geometry rules. This is an explicit restriction of the permitted connection set, preventing a long shortcut through a hub interior.
- All endpoint terminals are registered as permanent walls. Search checks the actual corridor boxes against those walls. Tagged terminal cylinders reserve pad rows; ordinary endpoint cylinders and every corridor box reserve every intersected hex/level/time row. Same-hub corridor boxes still contend.
- A column contains the union of all covered rows, including rows with zero current price. Each coefficient is one. Search tracks nonzero-price rows already paid under the frozen dual vector, so overlapping segments never pay twice. Optional expiry retains their exact accumulated charge after future overlap becomes impossible. Zero-price rows still participate in forbidden-row checks and canonical column construction.

The complete-volume cover is more conservative than the production DP's visit-window row representation. The benchmark reports that difference explicitly. Both arms use the actual simulator reservation builder and collision predicate; the adjacent-only experimental arm isolates the change in horizontal connections while retaining the new claim representation and nominal timing.

## Search and certificates

Two execution backends share this contract. `ZetaOptions.backend="reference"`
retains the Python reference. `backend="compiled"` uses numeric parent-pointer
labels and a compiled sparse lookup of priced and forbidden reservations. It
prepares geometry only for edges that search examines. Complete reservation sets
are still rebuilt and validated for returned columns. The compiled backend needs
the project's `compiled` dependency group; its first use includes compilation or
cache loading. By default its orchestration and priority-queue calls run in Python.

The compiled backend also compiles successor feasibility/bound calculations and
stable ordering, expiration partitions, and numeric label writes. Set
`compiled_search=False` (`--no-compiled-search`) to compare these operations with
their Python paths. Floating-point ordering and exact signed history charges are
preserved; arbitrary-size integer dominance comparisons remain in Python.

`compiled_loop=True` (`--compiled-loop`, with `--backend compiled`) also compiles
the shared orchestration source using Cython. Bounds and label buffers use native
numeric types; heap ordering, precise price sums, and arbitrary-size signed history
charges retain their existing semantics. The interpreted compact loop remains
available with this option off. Install the `zeta-native` extra and a C compiler
to enable it. Builds are cached by source, Python ABI, Cython/compiler versions,
platform, and compiler flags, with a lock and atomic binary installation for
parallel workers. First-use build/loading time is included in pricing elapsed
time. Preparation shares the pricing deadline, and the parent prepares the binary
before starting workers so terminating a worker cannot abandon a compiler process.
An expired preparation returns an incomplete deadline result; other build failures
raise rather than silently selecting the interpreted loop. `ZETA_LOOP_CACHE_DIR`
can select the binary cache directory. The compiled search explicitly releases its
helper closures on every exit so completed calls do not retain their search arenas.

The 2026-11-16 actual-demand comparison used the same 100 flights, 12 workers,
50,000-expansion/250,000-label caps, and early return. Each reported arm is one
complete run after a small warmup; the cached build was already available.

| Execution | Pricing wall time | Solve and audit wall time |
| --- | ---: | ---: |
| Previous compact loop | 135.66 s | 153.13 s |
| Refactored buffers, Python loop | 127.94 s | 145.06 s |
| Refactored buffers, Cython loop | 123.64 s | 141.48 s |

The combined change reduced observed pricing time by 8.9% and total time by 7.6%.
Compilation added a 3.4% pricing reduction beyond the buffer refactor in this
single-run comparison, so most of the gain did not require Cython. First-time
compilation separately took 4.76 s. Peak process-tree RSS was 13.26 GB versus
12.95 GB for the previous loop. These timings are preliminary, not a scaling claim.

All 100 selected routes and all 1,000 non-timing pricing results matched exactly,
including costs, bounds, search counts and statuses. Collision and capacity
audits passed; 347 targeted tests covered exact pricing, signed charges larger
than machine integers, cache controls, worker parity, deadlines and cleanup.
The search still expanded 42.93 million labels, with 847 capped calls and the
same 95.57% global integer gap. Compilation speeds execution without resolving
the certification bottleneck. Raw runs and comparison checks are retained in
the workspace's `.agent/zeta-compiled-loop-20261116/` directory.

An exact timed-segment cache stores reservation row IDs for an edge, endpoint
roles and its unrounded start timestamp. Each prefix still unions these rows
against its own paid history. The cache is local to one frozen-price call, so
changed prices and forbidden rows rebuild it. Its default accounting budget is
16 MiB per call, including an estimated Python entry overhead; this is not a
process-memory ceiling. FIFO eviction and oversized-entry bypass only discard
cached work. Set `timed_cache_bytes=0` (`--timed-cache-bytes 0`) to disable it.
Pricing results report hits, misses and peak accounted bytes. Both optimizations
apply only to the experimental compiled backend and change no search choices.

`lazy_successors=True` puts deferred segment choices on the same queue as route
prefixes, ordered by conservative completion bounds. Unexamined choices remain
in the frontier used for interruption bounds. This changes evaluation order, not
the permitted routes. Both eager and lazy modes use the configured physical timestamps and support
the same optional history expiry and dominance rules.

The reference search uses a priority queue and a conservative completion-cost bound. It preserves negative row prices. With the endpoint-fold preconditions established, the cost bound includes physical distance already flown, straight-line remaining distance and endpoint attachment lengths. Otherwise it falls back to the ground-cost bound.

State comparisons require identical waypoint, departure/lane context, exact arrival timestamp, both physical-length accumulation orders, terminal role, and the sequence of physical length summands. These conditions preserve future feasibility and canonical floating-point costs. Without stronger dominance, merging also requires identical live paid rows and retired charge.

`expire_history=True` discards a paid row only when its time interval ends at or before the earliest possible future reservation start, \(t-\max(0,b)\), where \(b\) is the time buffer. Its charge is retained exactly. Expiry occurs after an entire connection has been union-priced. Final columns always retain every reserved row.

`history_dominance=True` compares histories only within the physical-state key above. For live tails \(T_A,T_B\), retired charges \(R_A,R_B\) and signed row prices \(\mu_r\), A can replace B when

\[
R_A-R_B + \sum_{r\in T_A\setminus T_B}\max(\mu_r,0)
+ \sum_{r\in T_B\setminus T_A}\max(-\mu_r,0) \le 0.
\]

This bounds their cost difference for every common future reservation set, including negative dual prices. Comparisons use exact integer-scaled binary prices, avoiding tolerance-based pruning. Previously queued work for replaced histories is skipped; its replacement remains represented before any cap/deadline check. Tiny exhaustive physical-walk tests cover these modes and their interrupted bounds.

The opt-in `merge_shorter_histories` option additionally compares both unchanged length
accumulators when exact clock, endpoint context and live/retired price state match.
It retains a prefix only when neither accumulator is greater, under the existing
whole-route guard and nonnegative air cost. It requires `strong_history_dominance`
and works independently of `merge_equivalent_histories`. Both options expose their
numerical allowance in the pricing certificate. See the
[shorter-history contract](ZETA_SEARCH_AND_CACHING.md#guarded-shorter-history-comparison)
for feasibility conditions, counters and replacement-chain reasoning.

Every returned route is reconstructed through the simulator builder, and its complete claims and weighted cost are recomputed. A capped search returns an achievable route and a valid upper bound on its reduced cost; it never declares convergence. Useful capped columns may be added and repriced under the next LP duals. A later complete sweep is required for convergence.

`early_return=True` (`--early-return`) stops pricing when a newly discovered best
candidate is absent from the canonicalized known-column pool and passes the
driver's positive reduced-cost threshold. It reports `improving_column` with
`complete=False` and a valid frontier bound. Complete proof takes priority when
already available. An early seed return uses the conservative global floor
because departure roots have not yet been built. All ordinary search exits
occur between complete expansions, retaining unopened successor choices.

With `early_return=True`, the call also stops once its conservative reduced-cost
upper bound is at most `params.epsilon`. This returns `reason="no_improvement"`
and `complete=True`: pricing has proved that no permitted route can usefully
improve the master, although the returned route need not be the exact pricing
optimum. It preserves the actual incumbent and upper bound. This check runs
only after departure roots are represented, includes deferred successors, and
takes precedence over count caps and the deadline. An available exact-optimum
proof still returns `optimal` first.

If neither useful-column nor no-improvement proof is available, search continues
until proof, a count cap or the deadline. Thus no restricted waypoint search is
used, and the driver still requires a completed no-improvement sweep before
reporting convergence. Useful incomplete columns update the master and the next
round's dual prices. Statistics count useful-column returns as
`early_return_searches` and tolerance proofs as `no_improvement_searches`,
separately from `capped_searches`. With the option off, pricing retains its
exact-best-route stopping rule. Both modes work with reference, compact and
Cython execution and the existing persistent per-flight workers.

For flight bounds \(u_f\), the master uses the valid maximize-objective bound

\[
U = z_{\mathrm{RMP}} + \sum_f \max(0,u_f).
\]

The corresponding total-cost lower bound is \(nM-U\). The final restricted-IP bound is reported separately. A restricted IP solving optimally does not establish global integer optimality.

## Running the prototype

Run from this checkout with the project's Python environment:

```sh
python -m experiments.zeta_prototype --arm all --case crossing \
  --flights 2 --distance-cells 3 --ground-delay 32 --overrun 1 \
  --repeats 2 --seconds 45 --ip-seconds 5 \
  --max-expansions 5000 --max-labels 50000 \
  --out results/zeta-prototype-crossing2.json
```

`--arm all` runs production DP, adjacent-only physical search, and any-angle physical search. Repeats reverse the arm order. `--case hub` uses shared-terminal departures. `--arm zeta_seeded` first obtains a feasible adjacent-only schedule and pool, then retains them while adding any-angle columns; its reported total and pricing times include both stages. This diagnostic tests route improvement without discarding an existing feasible schedule.

Add `--backend compiled --lazy-successors` to measure the accelerated variant.
Omit those flags for the original reference, or select each independently for
comparison. The adjacent initial solve in `zeta_seeded` uses the same selected
backend and ordering. Record first-use time separately from warm repeats.

Add `--expire-history --history-dominance --arrival-snap 1 --speed-fraction 0.05`
to try pruning with the Zetahex-style speed allowance. Each flag is independent,
so compare pruning on/off at the same timing settings before attributing speed gains.
Snapping changes the permitted schedules; it is not an equivalent-work optimization.
Search counters report discarded histories, expired row entries and comparisons.

Add `--future-price-bound` to use an optional backward lower bound on remaining
travel and congestion cost. It relaxes the candidate graph into waypoint/time
intervals and assigns positive cell prices once per reservation period. This
avoids adding the same buffered reservation charge on consecutive edges. Only
already-paid cell rows whose sampling time is still ahead are discounted; sunk
charges remain in the prefix cost. Positive prices and
edge costs are rounded downward to integer units; the actual route objective
and reservations retain their original values.

For customer endpoints, the table couples travel cost with sampled prices and
includes conservative distances connecting the actual endpoints to the waypoint
graph. The independent minimum remaining graph distance remains a lower bound
after paid-row correction, so an unpriced detour still has a cost. Terminal endpoints and a zero reference
distance use only the price component. Unsupported geometry/arithmetic, excessive
table size or an interrupted build declines the optional bound and leaves the
original search bound in force. Prepared bounds propagate through deferred
successors, preserving interrupted-search certificates. The flag changes no
route, departure, timing, buffer or cost choice. Reports include preparation
time, table states, lookups and prefixes rejected at the additional bound check;
wall time includes preparation and any first-use compilation.

The bound assigns a reservation period to an edge only when that period's sample
lies inside every edge timing represented by its source and destination bins.
Successive edges therefore cannot charge the same period twice. Within each
assigned period, the bound unions every subpart guaranteed to overlap the period
after applying the time buffer, then adds the cheapest possible subpart at the
sample. Shared cells are charged once. This includes the old point-sample charge
and can capture additional required reservations elsewhere in the period. The
timing intervals include the exact snapping candidates and
sequential nominal fallback; they do not exclude any permitted timing.

The union prices are precomputed for subpart subsets. More than eight subparts
per edge, or more than eight million subset/period entries, uses the old
point-sample charge instead; these preparation limits do not remove routes.
`FuturePriceBound.build(..., buffered_periods=False)` also selects the old charge
for diagnostic comparisons. The numerical bounds and paid-row correction remain
the same in both modes.

Let \(D(v)\) be the quantized minimum remaining graph-distance cost and
\(H(v,b)\) the backward table's joint distance-and-price bound. If \(Q\) is the
quantized cost of already-paid sampled rows whose samples are still ahead, the
remaining bound is \(R=\max\{D(v),H(v,b)-Q\}\). Both arguments bound future travel
plus newly paid prices, so their maximum is safe. For customer endpoints, before
the separately computed rounding allowances, the completion floor is
\(g+P+N+\max\{0,a(\ell+A-r)+qR\}\), where \(g\) is ground cost, \(P\) is accumulated
positive price, \(N\) is all possible negative credit, \(a\) is the air-cost weight
divided by speed, \(\ell\) is prefix length, \(A\) lower-bounds endpoint attachments,
\(r\) is reference distance and \(q\) is the integer quantum. The implementation
rounds downward and subtracts allowances derived from the finite route-length
budget, canonical geometry arithmetic and price summation. The search retains
the maximum of this floor, its original geometric floor and the parent floor.

Additional completion relaxations can be selected independently with
`--completion-bounds arrival resource selected crossing`, or with the corresponding
`ZetaOptions.completion_bounds` tuple. They leave the finite route universe unchanged.
The search takes the maximum of their complete cost floors, the existing geometric
floor, and any inherited parent floor; their reservation charges are never blindly added.

- `arrival` minimizes mandatory destination reservation prices jointly with a safe
  arrival-time and distance envelope. It discounts previously paid rows, includes
  nominal timing fallback, and declines distance coupling for unsupported terminal folds.
- `resource` builds two distance-penalized sampled-price tables. Subtracting each
  penalty times an outward upper bound on remaining graph length gives a valid future
  price bound. This excludes the benefit of arbitrarily long cheap relaxed detours.
- `selected` remembers paid/unpaid membership for up to six expensive positive rows in
  a relaxed waypoint/time table. Guaranteed row masks are unioned across connections;
  other positive rows are free. Selection shrinks when its table would be excessive.
- `crossing` minimizes the largest guaranteed single-connection charge over relaxed
  completions. Every real completion must incur at least that charge somewhere;
  overlapping connections are never summed, and live paid rows are discounted.

Each component includes conservative arithmetic allowances and preserves the original
all-negative-price credit. Unsupported, oversized, or interrupted preparation contributes
no bound. Empty collections disable the lookup entirely. Static geometry may persist on
the flight graph, while prices and value tables are rebuilt for the current LP duals.
These four components use a separate relaxed geometry cache: every neighbor connection
is retained, wall restrictions are omitted, and lengths and subdivision clocks match
the physical graph exactly. Arrival preparation needs no corridor footprints. Other
components rasterize only near positively priced spatial cells, retain one support-keyed
variant, and keep at most 2,048 full cell footprints per flight. Omitting positive charges
weakens a bound safely. Selected-row and crossing tables use larger time intervals only
when the timing enclosure still includes every accepted physical continuation.

Preparation limits are separate from the 50,000-expansion search limit. A table that
exceeds its state or transition budget is discarded, and pricing continues with its other
valid bounds. Such a call has not demonstrated the discarded bound's pruning benefit.
`bound_build_s`, `bound_table_states`, `bound_lookups`, and `bound_pruned` include these
components; pricing wall time includes their preparation. Stronger bounds can change
search order and which useful column an early-return call discovers first.

The CLI defaults to 5,000 expansions and 50,000 generated labels per pricing call. `ZetaOptions` itself defaults to 50,000 and 250,000. Both limits are explicit compute caps, not restrictions of the certified route universe. Raise them to investigate convergence, while keeping the overall wall limit. `--no-search-caps` disables both counts (`max_expansions=None, max_labels=None` in the API), so pricing continues until completion or the wall deadline. This does not add an internal memory limit; use an external process-tree memory guard for diagnostic runs. Disabling counts does not change the route universe or the validity of completed/interrupted bounds.

Use `--workers N` to run up to N persistent processes; zero (the default) constructs seeds and prices in the parent process. Workers construct canonical initial columns, including supplied routes and departure ladders, then retain the same graphs and geometry caches for pricing. Each round freezes the row and flight duals, prices flights independently, and inserts results in flight-ID order. Ownership is static for this prototype. A failed seed sweep preserves valid returned columns for final selection and prevents a pricing certificate. A failed or interrupted pricing sweep cannot tighten the global bound from a partial set of flights. Workers close on normal completion and exceptions. Both seed geometry and final canonical checks query the existing wall spatial index before the unchanged exact collision predicate.

Parallel planning is across flights; a single flight's Zeta search is sequential.
For example, `--backend compiled --lazy-successors --early-return --workers 6`
uses up to six simultaneous any-angle pricing searches. This experimental driver
replaces DP pricing with Zeta while retaining the existing restricted master and
hex/time capacity coefficients. It is a separate entry point; the production
planner registry still selects the existing DP implementation.

All benchmark arms use the same configured worker count, objective/final-IP settings, and no intermediate IP search. Gurobi threads default to one. Reports include source hashes, selected routes, costs, global and restricted bounds, search caps, claim matrix size, process peak RSS, pricing time and inclusive solve/audit time. The first process call may include JIT/cache startup; warm repeats are separate observations.

## Validation and limits

Focused functional checks:

```sh
python -m pytest -q tests/test_colgen_network.py tests/test_colgen_objective.py \
  tests/test_colgen_e2e.py tests/test_colgen_pricing_pool.py -k 'anyangle or zeta'
```

These cover full-box/cylinder coverage and tangency, timing and departure/lane rules, union charges, exact pricing against exhaustive physical walks, negative prices, saturated airspace/pad rows, safe capped bounds, later convergence after useful capped proposals, and actual DSS ledger commits of selected crossing/shared-hub schedules.

The pricing oracle and end-to-end checks exercise reference/compiled and
eager/lazy variants. Interrupted runs include unexamined successors; clock and
cache-reuse cases change prices and forbidden rows on the same graph, including
nonbinary period boundaries and signed prices. Performance comparisons must also
check final served count, cost and global gap, because different equally good
priced routes can change the final integer schedule. Spawned-worker tests compare completed serial/parallel schedules and bounds, refresh signed duals and forbidden rows between sweeps, and exercise capped calls, worker death and deadline cleanup.

Per-round `flight_pricing` records report each flight's task time. Seed diagnostics include completion, failure, returned-flight count, and per-flight tasks. `seed_wall_s` includes seed startup, transport, and parent insertion; `seed_task_s` sums worker time. `seed_pool_setup_s` and aggregate `pricing_pool_setup_s` report the same startup paid during seeding, so do not add either again to the inclusive stage wall times. Pool close time remains separate. For an ideal one-worker-per-flight estimate, sum the maximum task time in each round and add work outside those tasks. Include both stages for `zeta_seeded`; dividing total pricing time by the worker count alone misses imbalance and startup.

The experimental pool is separate from the production DP pool. There is no planner registry switch, full-demand benchmark or claim of production competitiveness. Fixed commitments supplied through `fixed_claims` must use the same complete-volume converter and permanent-terminal protection; legacy DP row proxies cannot be imported as fixed reservations. Seed columns are physically recanonicalized before admission.

The earlier November 13 point-bound screen replayed one hard flight with frozen dual prices and
both count caps disabled. That bound reduced processed prefixes from 692,533
to 642,174 and accepted labels from 701,485 to 652,154, preserving the exact route,
claims, cost and certified pricing optimum. First-call time increased from 52.91
to 57.64 seconds; a second call reusing geometry increased from 34.32 to 36.59
seconds. Preparation took 1.82 and 0.058 seconds respectively; the remaining
overhead includes additional per-prefix checks. This is a pruning improvement,
not a measured speed improvement or a full-solver benchmark, so the option
remains disabled by default. Evidence is in `.agent/zeta-bounds-20261113/`.

The subsequent buffered-coverage screen reduced processed prefixes further, from
642,174 to 599,233, with the same route, full claims, cost and certified optimum.
Inclusive caller wall time (including return cleanup) improved from 57.75/37.65
seconds for first/reused-geometry calls to 56.52/35.90 seconds. The original
geometric bound still took only 53.00/34.29 seconds on the matching calls.
Thus buffered coverage improved on point sampling by 2.1%/4.6%, while remaining
6.6%/4.7% slower than the original bound. It remains opt-in. A separate exact-table
comparison found 1,443 of 6,640 states strictly strengthened and none weakened;
the refactored point-only table matched the archived implementation exactly.
All 256 focused tests and a two-worker collision-free solve passed. Evidence is
in `.agent/zeta-buffer-bound-20261113/`; these are single-flight pricing measurements,
not full-solver performance measurements.

The first integrated measurements show that native Zetahex speed does not directly transfer to exact simulator timing and box claims. Caching immutable claims eliminates repeated row construction, but difficult crossings still create many distinct histories. Preserve feasible incumbents and evaluate final schedule quality alongside search completion before expanding the prototype.

The additional compiled-loop and exact timed-segment-cache screen preserved the
hard replay's route, claims, certified pricing optimum and all search counts
(692,533 processed prefixes). Inclusive first/reused-geometry time changed from
52.99/34.03 seconds to 42.50/23.67 seconds. Compiled-loop-only and cache-only
variants took 32.03 and 25.87 seconds with reused geometry; both variants also
include shared canonical-length hoisting and bounded sequence memoization.
The cache reused 1,807,821 timed footprints against 19,434 misses, with 11.70 MiB
of peak accounted entries under its 16 MiB budget. Each call rebuilds this cache.

On the six-flight crossing case with six workers, two opposite-order repetitions
reduced mean pricing time from 55.38 to 42.39 seconds and inclusive solve/audit
time from 57.24 to 44.25 seconds (22.7% less total wall time). Every selected route,
round, pricing score, bound and search count matched. All six flights were served
without collisions or capacity-row violations, at cost 130.89676837. Both
versions stopped at the same search limit with the same 81.31% global integer
gap; this is faster equivalent work, not certified convergence of the full solve.
The 244 pricing and 72 integration tests passed, including changed prices,
forbidden rows, exact clocks and cache bypass/eviction. Evidence is in
`.agent/zeta-loop-cache-20261113/comparison.json`. Those comparisons used full
pricing; the separately controlled early-return option was added afterward.

The early-return screen used the same six-flight case with six workers, 120 seconds,
100 allowed rounds and 50,000/250,000 pricing caps. Two opposite-order repetitions
reduced mean total solve/audit time from 44.26 to 24.18 seconds (45.4%) and pricing
from 42.37 to 22.06 seconds (47.9%). Both modes served all six flights safely at
cost 130.89676837, but their selected routes and departures differ. The main stage
took 18 rounds with early return versus 15 without. Its valid cost lower bound
weakened from 24.4632 to zero; both modes ended at the search cap. The speed gain
therefore does not establish faster certified convergence, and early return
remains opt-in.

On the previously frozen hard pricing input, early return found the same route
after 746 expansions and 0.086 seconds with reused geometry, versus 692,533 and
23.51 seconds for the completed search. The early result's reduced-cost upper
bound was 20.1013 while its achieved reduced cost was 1.30806; completed search
proved that latter value optimal. This compares finding a useful route with
proving its optimality, not equivalent certificates. All 332 focused tests passed,
including canonical duplicate suppression, the driver's exact improvement
threshold, suboptimal early-route bounds and persistent serial/parallel parity.
Evidence is in `.agent/zeta-early-return-20261114/comparison.json`.

A subsequent worker-count comparison held that implementation and early-return
mode fixed. Two reversed-order repetitions on six crossing flights gave mean
total solve/audit times of 73.48, 48.85, 35.20 and 24.27 seconds for serial, two,
four and six workers. Six workers were 3.03 times faster overall and 3.24 times
faster in pricing. Every selected route, cost, bound, round and all 156 pricing
results per run matched exactly; the main stage still ended at its search cap.
The small shared-hub case instead took 0.80 seconds serially versus 1.42 with
six workers, including fresh worker startup. All eight hub runs matched and
converged. These 16 audited runs used the same 180-second budget and 50k/250k
search caps. Parallelism helps the expensive crossing workload but adds overhead
to short searches. Evidence is in
`.agent/zeta-worker-scaling-20261114/comparison.json`; solver code was unchanged.

## Four completion bounds on actual demand, 2026-11-16

All four optional components were implemented, independently reviewed and tested against
exhaustively enumerated physical routes. The 653 focused objective, worker-pool and
end-to-end tests passed (88 unrelated cases deselected). Coverage includes signed duals,
already-paid and expired rows, terminal attachments, nominal and snapped/fallback clocks,
interruption bounds, canonical reservations, and reference/compact/native parity. Real
edges in the relaxed geometry have identical lengths and durations; retained footprints
are subsets of the simulator's actual spatial coverage.

The experiment uses the same isolated 100-request actual-demand cohort, all static
terminals, 12 pricing workers, 50,000 expansions, 250,000 generated labels, early return,
native search loop and 300-second solver budget. Pricing reserves 35 seconds for final
selection, so a pricing deadline can produce a total near 266 seconds. Each variant ran
once under the shared optimizer lock, with a 32 GiB process-tree guard. Preparation is
included in pricing time. The fresh baseline matches the previous baseline's 100 selected
routes and all shared non-wall fields in the 1,000 recorded pricing results exactly.

| Variant | Pricing wall (s) | Total solve/audit (s) | Cost | Global lower bound | Fully returned sweeps + partial sweep |
| --- | ---: | ---: | ---: | ---: | --- |
| Current search | 110.66 | 128.60 | 1024.79 | 45.41 | 10 |
| Arrival reservations | 174.87 | 193.58 | 1024.79 | 84.70 | 10 |
| Remaining distance | 251.73 | 265.79 | 1026.10 | 31.12 | 1 + 98/100 |
| Selected paid rows | 251.53 | 265.78 | 1026.10 | 40.95 | 2 + 33/100 |
| Unavoidable crossing | 251.45 | 265.77 | 1026.10 | 40.95 | 2 + 45/100 |
| All four | 251.43 | 265.82 | 1026.10 | 70.19 | 1 + 64/100 |

Every variant served all 100 requests with clean physical-collision and capacity audits.
None completed pricing over the full defined route universe. Arrival preserved exactly
the same selected routes and the same 83 unresolved final-sweep flight IDs. Its stronger
global lower bound still leaves a 91.73% gap, versus 95.57% for the baseline; total wall
time increased 50.53%. The other three individual variants and the combination exhausted
the pricing budget before reaching the baseline's ten sweeps, with cost 0.127% higher.

Usable-table counts among returned pricing calls were arrival 74/1000, remaining distance
4/198, selected rows 86/233, crossing 32/245, and all four 66/164 (at least one component).
The three graph-table variants individually recorded zero bound-based label prunes.
Arrival recorded 1,229 and the combination 167. Summed preparation time across returned
workers was 21.9 seconds for arrival, versus 1,800–2,000 seconds for the others; these are
worker sums, not elapsed wall time, and exclude tasks that did not return before cutoff.

The fixed final-round replays explain the scale problem. Flight 1986 has 1,321 relaxed
waypoints, 69,736 edges and 213,920 segment parts; flight 3098 has 2,316 waypoints, 125,174
edges and 388,056 parts. Remaining-distance tables require 94.1 million and 194.0 million
part/time-bin entries, beyond the 24-million preparation limit. Crossing requires 28.0
million and 57.8 million. Selected-row memory fits flight 1986 (six rows, 845,440 states)
but prunes nothing; flight 3098 still requires 35.7 million part/time entries after reducing
the paid mask to three rows. These are preparation limits, separate from search expansions.
Rejecting an oversized table leaves a safe weaker bound; its pruning strength on that
flight was not evaluated.

Arrival is active for flight 3098 and reduces its final improvement upper bound from
52.562 to 48.777, but reused-call wall time rises from 0.735 to 5.784 seconds. Neither
illustrated flight becomes certified under any variant; every returned feasible column
matches its fixed-input control. Initial full-geometry preparation was even slower and
was replaced by the sparse relaxed geometry described above before the actual100 runs.

Keep these prototypes disabled by default. A follow-up should reject oversized tables
before rasterizing footprints, reuse spatial memberships more broadly, and make the
arrival lookup cheaper. Larger tables alone would increase preparation work without
addressing the observed lack of pruning on active cases. These measurements do not rule
out better abstractions or tighter combinations of the four ideas.

Evidence: root-workspace `.agent/zeta-completion-bounds-20261116/` contains immutable
runtime hashes, raw guarded runs, `comparison.json`, `decline-diagnostics.json`, the
exhaustive-test logs, and verified chart output. The PNG is
`context/figures/zeta_completion_bounds.png`; its generator is in the existing figure script.

## Shared leg and persistent timing caches, 2026-11-16

Two independent options reuse calculations across flight graphs and pricing rounds:

- `--leg-cache` shares immutable horizontal subparts and their complete hex/level
  footprints within each worker process. Keys include exact absolute coordinates and
  the geometry settings used to construct each value. Flight-specific terminal tags,
  terminal exemptions and static-wall checks still run on the receiving graph.
- `--persistent-timing-cache` shares exact subpart arrival times and buffered reservation
  period windows. Its key includes the exact start time, ordered nominal durations,
  arrival snapping, speed allowance, reservation period and time buffer. It does not
  round a departure time to increase reuse. The current pricing call resolves those
  windows against its current row table and forbidden rows.

Neither cache stores dual prices, paid-row histories, blocked-edge decisions or whole
routes. Both preserve the existing waypoint choices, connection and distance limits,
terminal rules and timing arithmetic. Cached arrays cannot be made writable. The
existing per-flight graph cache and per-call timed-row cache remain in use; these new
stores fill the gaps between graphs and calls. Independent worker processes have
separate stores, avoiding transport of cached geometry between workers.

The geometry store has a 64 MiB estimated-entry budget and the timing store 32 MiB per
process. Least-recently-used entries are evicted; entries larger than the budget are
computed but not retained. These are cache ownership estimates, not process-memory
limits: a graph may still reference an evicted immutable object. A cold geometry miss
constructs all subparts before flight-specific wall rejection, whereas the uncached
builder can stop constructing at the first blocked subpart. Measure seed startup and
pricing separately when assessing this tradeoff. Both options remain off by default.

The existing objective and worker-pool tests were extended for exhaustive route/claim
parity, signed and changing duals, newly priced and blocked rows, fresh flight graphs,
configuration changes, exact nonbinary timing boundaries, immutable arrays, eviction,
oversized-entry bypass and persistent timing with the per-call cache disabled. There
were 756 passing targeted cases across the suite run and one corrected-fixture rerun;
the initial failing new fixture placed a foreign terminal wall over the origin and
was corrected before passing. Independent source review found no correctness or
certificate blocker. Only wall-time and the six cache-performance counters may differ
in cache-on/cache-off parity checks.

The actual100 comparison used the same 100 requests, 12 persistent pricing workers,
50k/250k search limits, native loop, early return, unchanged completion bounds (off),
300-second solve allowance and 32 GiB process-tree guard. Runs were serialized under
the shared optimizer lock. All six returned identical 100 selected route payloads,
cost 1024.7926827393449, global lower bound 45.40537640254479 and the same 83 unresolved
final-sweep flights. All 1,000 pricing calls per run matched the control on every one
of the 14 recorded non-wall/non-cache fields. Collision and row-capacity audits passed;
the recorded source, input and harness hashes remained unchanged.

| Run, in execution order | Pricing wall (s) | Seed wall (s) | Total solve/audit (s) |
| --- | ---: | ---: | ---: |
| Control, opening | 112.047 | 10.639 | 130.389 |
| Shared geometry | 111.035 | 11.602 | 130.209 |
| Persistent timing | 106.592 | 10.615 | 124.650 |
| Both caches | 108.315 | 11.608 | 127.491 |
| Control, closing | 108.273 | 10.383 | 126.209 |
| Persistent timing, repeat | 109.265 | 10.358 | 127.146 |

The two timing runs averaged 107.929 seconds of pricing and 125.898 seconds total,
versus control averages of 110.160 and 128.299 seconds: reductions of 2.03% and 1.87%.
The timing repeat overlaps the control range, so a reliable speedup is not established.
Geometry alone did not improve total time; it also added about one second to seeding
in this sample. Sampled process-tree peaks ranged from 11.58 to 11.75 GiB. Keep both
options opt-in while evaluating other cohorts; neither reduces the number of histories.

During pricing, timing-enabled runs reused 505,761 windows and computed 28,193 (94.72%
hits); final round reuse was 53,399 hits and three misses. Shared geometry recorded only
69 leg hits against 16,779 misses, while full-footprint reuse recorded 30,286 hits and
66,646 misses. These are new shared-cache accesses after existing graph/per-call caches,
not all leg evaluations, and exclude seed-stage cache activity. Four fixed-input probe
variants also matched exactly through cold, reused-graph and fresh-graph calls. First-call
loading/compilation makes their initial timing unsuitable for a cache speedup claim.

Evidence, source snapshots, validation logs, six guarded measurements, reusable analysis,
incremental patches and the wall-time chart are in `.agent/zeta-leg-cache-20261116/`.
The PNG is `context/figures/zeta_leg_cache.png`.

## Deep cap study: final-round flight 1986, 2026-11-16

One of the 83 unresolved final-sweep searches was replayed with its original prices,
known columns and permitted route choices. The uncached native baseline reached a
no-useful-improvement certificate after 555,365 expanded histories and 556,069 generated
histories: 505,365 more expansions than the original 50,000. Successful calls took
8.33 and 8.72 seconds, with about 1.44 GB sampled peak process-tree memory. All eight
cap configurations returned the same route geometry, claims and achieved reduced cost.
The first 50k call included initial Numba loading/compilation; do not use its time as a
warm baseline or infer an 11-fold wall-time increase from the expansion ratio.

The remaining possible-improvement upper bound was 0.435758 at 50k, 0.313836 at 100k,
0.146965 at 250k, and \(1.0054\times10^{-7}\) at proof, below the \(10^{-6}\)
pricing tolerance. This is a tolerance-based no-improvement certificate, not an exact
argmin claim. An expansion cap of exactly 555,365 still interrupts 50 pending successor
checks. Cap 555,366 with generated-history cap 556,070 was confirmed sufficient.
Keeping the original generated-history cap of 250,000 stops at 238,406 expansions even
when the expansion allowance is one million. Both limits need attention.

The archived 50k spatial trace expanded only 92 distinct waypoints despite generating
57,878 histories. One waypoint appeared in 4,213 expansions; a subgroup of 45 histories
matched every recorded state-comparison field except the exact segment-length sequence.
That sequence remains part of the conservative comparison key to protect canonical
floating-point behavior. The successful proof recorded only 470 history prunes.
Its 619,526 expired paid-row entries were removals from individual histories with charges
retained, not deletion of that many histories. Within-call timing reuse was already
99.737%: 578,731 hits and 1,524 misses. These counts indicate repeated history processing
and conservative comparisons are worth investigating; they do not attribute wall time.

The original feasible route was available before any expansion. Search continued because
optimistic pending continuations still appeared capable of improving it: the active bound
omitted future positive reservation charges. Raising both caps successfully resolved this
one frozen pricing problem. It does not establish the expansions, memory or full-solver
runtime needed for the other 82 flights, or under different dual prices. No default caps
were changed. Raw runs, boundary checks, hashes and the detailed explanation are in
`.agent/zeta-deep-flight-20261116/`; the plot is `context/figures/zeta_deep_flight_1986.png`.
