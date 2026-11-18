# How Zeta* searches and what we can precompute

We can precompute a large part of the routing work. A useful design is a shared map of permitted connections, reusable terminal-to-cell path trees, and a collection of alternative routes. The remaining online search chooses among routes using the current departure time, available reservations, and congestion prices. A route collection can make that search start much closer to a good answer while preserving the full search for certification.

This explanation describes the simulator's Zeta* prototype. Here, a **reservation row** means one hex cell, altitude band, and time period, or one terminal's capacity in a time period. The master allocates these rows. A larger sector could organize the geometry cache without changing those allocation units.

## What happens in one solve

Each flight has a permitted corridor containing a finite set of candidate waypoints. Currently these are hex centres. Zeta* can join candidates up to 480 metres apart with straight segments, including segments that cross the grid at an angle. It therefore changes horizontal routing while keeping a finite set of bend locations. Terminal exit lanes, the cruise level, departure choices, buffers, and the flight's distance allowance still constrain the route.

Before column generation begins, static shortest-path searches create initial routes for the permitted terminal-lane combinations. These searches use physical distance and permanent obstacles. They do not yet have congestion prices. Valid departure shifts provide additional starting columns.

The master then solves its current collection of routes and produces a price for each reservation row. An expensive row represents a scarce resource in that LP solution. Prices can change substantially when new routes enter the collection.

For flight \(f\), Zeta* seeks a route \(P\) with low weighted physical cost plus its reservation prices:

\[
c(P)+\sum_{r\in R(P)}\mu_r.
\]

Here \(c(P)\) includes the configured delay and flight costs; \(R(P)\) is the **set** of all rows the route reserves; and \(\mu_r\) is the current row price. In the implementation's maximizing convention, an improving route has positive reduced cost:

\[
M-\pi_f-c(P)-\sum_{r\in R(P)}\mu_r>\epsilon,
\]

where \(M\) is the benefit of serving a flight, \(\pi_f\) is its master constraint's dual value, and \(\epsilon\) is the numerical acceptance threshold.

The search extends partial routes from a priority queue. A lower bound on the eventual cost lets it skip partial routes that cannot improve the answer. A search record, often called a **label**, contains the waypoint, arrival time, departure and lane choices, distance information, and the reservation rows already paid for.

Paid history matters because reservation prices are charged once per row. If two consecutive segments both cover the same cell during the same period and that row costs 3, their combined charge is 3. Two routes reaching the same waypoint at the same time can therefore have different continuation costs: one may already have paid for rows the next segment needs. A single cheapest label per waypoint is insufficient. The prototype merges or discards histories only under conditions that preserve every permitted continuation.

Every proposed complete route is rebuilt through the simulator's reservation builder. Its actual cost and complete union of claims are checked before insertion into the master. Zero-price rows remain part of this final reservation set. The next master solution supplies new prices, and the cycle continues. Early-return pricing can return a useful new route before completing its proof; certification requires subsequent pricing that rules out further improvement over the declared search choices. The final integer schedule can still have a gap relative to the global LP bound.

## What is already reused

The current graph and its geometry caches persist across pricing rounds in each worker. They are not rebuilt from scratch on every LP iteration. There are also bounded spatial-footprint caches and a timed-segment cache within a pricing call. Search histories and price-dependent work are rebuilt as prices change.

Within one pricing call, complete routes with identical reservation sets reuse their summed dual price. Their physical costs and geometry checks remain separate. This price cache is discarded before the next call, so a new dual vector cannot reuse stale prices. Departure setup also stops once ground delay, even after granting every negative row price, cannot beat the best known route; all later departures have at least that ground cost.

The seed fix makes the same worker construct a flight's initial routes and later price that flight, preserving the geometry it has already computed. Previously, the parent constructed seeds and pricing workers started with cold geometry caches. The fix also queries the existing terminal-wall spatial index before performing the exact collision test.

In the [original 100-request measurement](/Users/victor/Documents/Research/congestion-utm-freespace/.agent/zeta-real100-20261114/comparison.json), constructing the graph took about 1.5 seconds, whereas initial-route construction consumed 263.5 seconds before the deadline. A separate ten-second seed profile attributed about 95% of its instrumented time to scanning terminal walls. This distinction matters when deciding what to cache: eliminating repeated obstacle checks and route searches has more potential here than saving the initial graph object alone.

After the seed fix, two runs of the same 100-request cohort averaged 10.0 seconds for initial routes, 1.5 seconds for graph construction, and 137.6 seconds for pricing. Both served all 100 flights and returned identical routes and bounds. The remaining time is concentrated in price-dependent search, so further caching should help that search reuse geometry and find useful candidates. [Measured results](/Users/victor/Documents/Research/congestion-utm-freespace/.agent/zeta-seeds-20261114/comparison.json).

A subsequent test of the claim-price cache and departure cutoff measured 134.8 seconds of pricing and 152.2 seconds total. Three baseline runs, including a fresh control, averaged 139.2 and 156.7 seconds respectively. This is a provisional gain of about 3%, based on one changed-code run. All 100 selected routes and 1,000 pricing results were identical, with clean collision and capacity audits; 244 targeted tests passed. The search still expanded 42.9 million prefixes and reached the cap in 847 calls, so the larger opportunity remains reducing search work or compiling more of its Python bookkeeping. [Pricing acceleration measurements](/Users/victor/Documents/Research/congestion-utm-freespace/.agent/zeta-pricing-assessment-20261115/comparison.json).

## A reusable terminal-to-cell cache

The user's proposed “from this terminal, how do I reach every sector?” table can be represented compactly by a **shortest-path tree**. For each terminal exit lane, store a distance and predecessor for each reachable waypoint. Following predecessors reconstructs a route to any destination. Common prefixes are stored once instead of duplicating entire route lists. With an int32 predecessor and float64 distance, the basic arrays require 12 bytes per waypoint per source, before metadata and any additional variants.

One tree gives one geometric shortest route to each destination. Several spatially distinct alternatives could form a useful starting route collection. They would be especially valuable when many flights share a terminal but have different destinations, or when repeated solves use the same map. A path template is the geometric waypoint sequence; it still needs departure timing and reservations before it becomes a master column.

| Reusable object | Work it avoids | Work still needed for a flight or pricing round |
| --- | --- | --- |
| Directed connections, lengths, corridor sub-boxes, obstacle checks, and spatial hex coverage | Repeated geometry construction and intersection tests | Check request corridor and endpoint-role eligibility |
| Terminal/exit-lane predecessor trees | Repeated static route searches to many destinations | Reconstruct and validate the chosen path against this flight's bounds |
| A small collection of distinct path templates | Rediscovering useful initial alternatives | Apply legal departure choices, exact timing, and complete reservation claims |
| Valid columns at the same departure and timing | Rebuilding already known routes | Recompute their price against current duals and check newly unavailable rows |

A shared map can contain more connections than one flight is allowed to use. A cached path must still respect that flight's corridor and length limit. A shortest distance on the larger map may be a valid lower bound even when the corresponding path is not feasible for that flight; these are different uses of the same cache.

The shared cache should separate ordinary interior connections from endpoint attachments. The same segment can have different legality when it is the first or last segment or belongs to a particular terminal's exit lane. The simulator's terminal exemptions and own-terminal shortcut restrictions must survive caching.

## Why storing all routes is a different problem

The static graph compactly represents many possible routes. Explicitly listing every route generally loses that compactness. As an illustration, 30 independent choices with two alternatives each already represent \(2^{30}\), or more than a billion route combinations, before departure choices and reservation histories are included. That is an illustration of growth, not a count of this workload's routes. Our length limit makes the declared route universe finite, but does not make enumerating it practical.

Nor is the shortest geometric route always the cheapest priced route. Suppose a direct path costs 10 in physical cost and a detour costs 12. With no congestion charge, the direct path wins. If the direct path's reservations acquire a price of 5 while the detour's remain unpriced, the detour wins at 12 versus 15. A change in departure can change which time rows are priced. A library containing only the shortest geometric alternatives can miss the route the current LP needs.

There are two sound ways to use a route library:

1. **Keep the current search choices.** Reprice cached routes first and use them as initial columns and search incumbents. When that collection supplies no new improving route, run full Zeta* pricing. A completed search retains the current certificate scope.
2. **Define the route library as the permitted route set.** Optimize exactly over that library. The resulting bound certifies the restricted library model, rather than all paths through the current waypoint graph. This would be an explicit modeling decision.

The first approach matches the present requirement to retain the existing finite-waypoint search and its certification. Geometry caching itself can be exact; limiting the available routes is the part that changes the optimization problem.

## What Zetahex already does

Zetahex provides two useful patterns. Its [directed-leg geometry cache](/Users/victor/Documents/Research/zetahex/zh/geometry_cache.py:1) stores intersected hexes and entry/exit distances along a segment. It derives time-period rows from the actual leg timing when needed. This cleanly separates geometry from changing clocks and prices.

Its [sector path cache](/Users/victor/Documents/Research/zetahex/zh/sector_path_cache.py:1) goes further for the separate sector/gate allocator. It stores gate-to-gate distances and shared predecessor trees on disk, with separate bundles for terminal and customer attachments. Memory-mapped arrays allow workers to share the stored geometry. New customer locations can add endpoint bundles without rebuilding the map's gate trees.

That sector allocator chooses a static within-sector path for each arc. Our master prices every covered hex/time row, so different paths between the same sector boundaries can have different costs. Its cache organization is a useful model for ours, but adopting one path per sector crossing would also restrict our permitted alternatives.

A cache fingerprint must include every input that affects geometry or eligibility: the terminal map and obstacles, waypoint grid, connection radius, reservation dimensions, altitude, endpoint rules, and the builder version. In this simulator the waypoint pitch is `nominal_speed_mps * dt_s`, so changing either value can change the geometry. Timed claims additionally depend on the actual start time, buffers, period length, and the speed/snapping rule. Stored prices must never outlive their dual snapshot.

The next cache prototype should therefore share immutable connection geometry across flights, add terminal-lane trees and a bounded set of alternative templates, and retain full pricing as the certification fallback. Its evaluation should report offline construction, cold loading, warm solves, memory, cache misses, and final solution quality separately. The seed fix in this change preserves caches within a solve; persistent map-wide and terminal-to-cell caches remain proposed work.

Implementation references: [graph and static seeds](../freespace_sim/planner/colgen/zeta.py), [priced search](../freespace_sim/planner/colgen/zeta_compact.py), [reservation construction](../freespace_sim/planner/colgen/zeta_geometry.py), [persistent workers](../freespace_sim/planner/colgen/zeta_pool.py), and [column-generation driver](../freespace_sim/planner/colgen/zeta_solver.py).

## Why one waypoint accumulated 4,213 histories

An observation-only replay captured all 57,878 accepted prefixes and all 50,000 expansions
of flight 1986 under its frozen final-round prices. Every functional result field,
including the complete returned column, matched the unobserved control and archived
native result. This trace covers the original 50k run, not the later 555k proof run.

Waypoint `(222, 87)` had 4,213 expanded histories and seven additional accepted histories
that had not yet expanded. The expanded histories followed 4,213 distinct waypoint chains
through just 48 waypoints and 158 directed legs. Earlier choices branch and rejoin; without
effective comparisons at the joins, their combinations remain separate histories.
Removing forward-collinear intermediate points leaves 1,982 ideal geometric tracks.
That geometric simplification is for explanation only: different leg subdivision and
arrival snapping can still produce different times and reservation claims.

| Exact arrival | Expanded histories | Physical comparison groups without length sequence |
| --- | ---: | ---: |
| 940 s | 9 | 2 |
| 941 s | 457 | 20 |
| 942 s | 1,723 | 24 |
| 943 s | 2,024 | 22 |
| Total | 4,213 | 68 |

Every history at this waypoint has the same departure/lane context and endpoint roles,
and an empty live set of already-paid priced rows. Past reservation charges remain paid;
they have expired only from the set needed to assess future overlap. Earlier arrival alone
is not sufficient to discard another history: arbitrary waiting is not available, and
changing time changes snapping and time-indexed capacity prices.

The figure highlights histories 4784 and 57862. Both arrive at 942 seconds, have stored
distance 3590.977233 m to six decimals, and have no future payment advantage from their
past reservations. History 4784 paid 0.26457161745922614 in reservation prices; 57862 paid
0.7643445953721305. The former has no greater value in either stored length accumulator.
At the time of this diagnostic, the algorithm did not compare them because their
segment-length sequences differed.

More generally, 1,988 histories have another history at the same exact time with neither
stored length larger and exact paid charge at least \(10^{-6}\) lower. Of these, 1,883
even match both stored length totals exactly. These are substantial opportunities for
stronger comparisons, **not certified deletions or measured expansion savings**.

The 943-second group is primarily a near-tie problem: its 2,024 histories span only
\(3.638\times10^{-12}\) m in sequentially accumulated distance and
\(2.650\times10^{-10}\) in paid charge. An offline comparison of both stored distances
and exact paid charge leaves 15 distinct nondominated vectors across all four times,
with 27 histories representing those vectors including ties. That optimistic count does
not include the remaining numerical proof obligations.

The existing rule already discards provably worse reservation histories after requiring
the same physical-state key. That key includes the full length sequence because final
cost is recomputed from the folded, resampled physical centerline with a whole-array NumPy
distance reduction. Equal stored totals do not establish equal final floating-point cost
or identical behavior at the final detour-feasibility boundary for every continuation.
Future timing itself depends on the current exact state and next leg, not that old sequence.

A stronger rule should group equal waypoint, exact time, departure/lane context and
endpoint roles. History A may replace B only when every continuation feasible from B
remains feasible from A and A's final physical cost plus union-priced reservations cannot
be greater. The existing signed union-price comparison supplies the reservation component.
For this empty-live-set case, the comparison needs a certified bound on physical-cost and
feasibility differences, together with the exact irrevocably paid charges. Strict price
advantages beyond that numerical allowance are the first target; near ties need a sufficient
compact cost representation or explicit error bounds. Blindly rounding stored lengths or
removing the sequence from the key would not establish the requested certificate.

Comparisons must be rebuilt under each new dual vector. Retaining fewer tied reservation
patterns may also affect final integer-schedule quality even when pricing remains correct;
functional tests should check exhaustive pricing bounds, timing/budget boundaries, signed
and changing prices, and final route diversity and schedule cost. No pruning rule changed
in this diagnostic.

Full capture, exact group analysis and the interactive inspector are in the root workspace's
`.agent/zeta-waypoint-histories-20261116/`. The standalone PNG is
`context/figures/zeta_waypoint_histories.png`, generated by `fig_zeta_waypoint_histories`
in `context/figures/make_figures.py`.

## Stronger history comparison prototype

`--history-dominance --strong-history-dominance` enables a guarded comparison between
different segment-length sequences. It is implemented in the reference, compact and
compiled loops and passes through the existing parallel pricing pool. It remains opt-in.
Departure choices, timing, speed allowance, terminal rules, buffers, costs and the finite
waypoint domain are unchanged.

The first version requires identical waypoint, exact arrival time, departure/lane context,
endpoint roles **and both stored length totals**. It does not yet compare arbitrary shorter
prefixes or round near-equal distances into the same group. The original exact-sequence
comparison remains available in every case.

For a shared continuation, let \(D\) be the worst possible additional union price of
history A relative to B, including exact expired charges and any possible overlap with
live reservations. The new comparison requires \(D<-\delta\), where \(\delta\) covers
the simulator's whole-route distance reduction, endpoint arithmetic, weighted cost and
signed-price summation. It uses exact integer representations of the binary row prices.
An uncertain terminal fold or a potentially binding final detour-factor gate disables the
broader comparison for that endpoint context. No physical feasibility check is relaxed.

Explicitly, \(R_A\) is A's exact price for expired reservation rows that a future segment
cannot overlap; \(T_A\) is its set of already-paid rows still capable of future overlap.
Define B analogously and let \(p_r\) be the signed dual price of row \(r\). For any common
future reservation set \(F\), their total reservation prices are

\[
P_A(F)=R_A+\sum_{r\in T_A\cup F}p_r,
\qquad
P_B(F)=R_B+\sum_{r\in T_B\cup F}p_r.
\]

The worst possible difference, allowing every future overlap pattern, is bounded by

\[
D_{AB}=R_A-R_B+
\sum_{r\in T_A\setminus T_B}\max(p_r,0)+
\sum_{r\in T_B\setminus T_A}\max(-p_r,0).
\]

Discard B if \(D_{AB}+\delta<0\). Every continuation of B then has a cheaper feasible
counterpart through A, even after the numerical allowance. This is a sufficient test;
some overlap patterns in the bound may themselves be unreachable. When both live sets
are empty, the comparison reduces to \(R_A+\delta<R_B\). For example, eligible prefixes
that paid 0.26 and 0.76 can discard the latter when \(\delta=10^{-5}\); near-equal charges
within that allowance remain distinct. Arrival-time or distance differences still fail
the grouping condition even when one route looks better in a plot.

Histories with the same live reservations share a comparison index. The lowest exact
expired charge is the strongest rejection witness; the highest is the easiest to replace.
All retained label IDs remain separately queued until a strict comparison actually
eliminates them. This avoids repeatedly comparing close retired charges while preserving
the earlier broad comparison's decisions. The `strong_history_pruned` diagnostic counts
all removals made by that comparison, including some the old rule could also make; use net
generated/expanded counts to measure additional search savings.

The derivation, immutable source variants, guarded benchmark inputs/results and analysis
are in the root workspace's `.agent/zeta-strong-history-20261116/`. `GUARD.md` records the
numerical envelope and `comparison.json` records the measurements. Tests cover exhaustive
physical-route optima and capped bounds, signed prices, strict-margin ties, partial removal
within a reservation group, terminal/detour fallback, known columns and serial/parallel parity.

Measured with the captured final-round duals, flight 1986 reaches the same no-improvement
certificate after **186,665 expansions / 188,714 generated histories**, down from
555,365 / 556,069. The final indexed implementation took **3.557 s versus 8.158 s** in
fresh guarded calls: 66.4% fewer expansions and 56.4% less wall time. This diagnostic
raised the expansion cap to one million and removed the generated-history cap; both
searches finished before their 45-second deadline. The selected route and final upper
bound \(1.0054\times10^{-7}\) were identical, below the \(10^{-6}\) improvement tolerance.

At the original 50k/250k caps, the two-flight replays still hit the expansion cap. Flight
1986's bound improves from 0.43576 to 0.27913 and its generated histories fall from 57,878
to 53,771. Flight 3098's bound improves slightly, from 52.56197 to 52.55023; its search
finds a different near-tied route, improving reduced cost by only \(1.72\times10^{-11}\),
well below the improvement tolerance. This is not a useful new improving column.

The actual100 workload, with 12 workers and unchanged original caps, **does not become
faster**. Two control runs took 106.424 / 107.320 s pricing and 124.940 / 125.232 s total;
the final indexed rule took 120.941 s pricing and 138.887 s total. All five measured
control/intermediate/final runs selected exactly the same 100 routes, objective cost
1024.7926827393449, and passed capacity/collision audits. All still ended with 17 completed
final searches and the same 83 unresolved flights. The valid global cost lower bound
improved from 45.40538 to 49.34803, but full convergence remains uncertified.

Across 1,000 pricing calls, the final rule reduces generated histories by 1.46% and
expansions by only 0.11%. It performs 122.5 million comparisons versus 9.6 million in the
control (the initial broad implementation performed 215.0 million). The index preserves
the broad rule's route, bound and net-work decisions in all fixed replays; its full-workload
runtime advantage over the simpler broad version is not established by these measurements.
Many equal or nearly equal histories remain below the conservative margin: flight 1986's
guard requires about \(1.02\times10^{-5}\) of strict price advantage, whereas the busy
943-second group's entire paid-price spread is only \(2.65\times10^{-10}\).

Validation: 899 relevant tests passed before the comparison-index optimization, followed
by 103 affected exhaustive/signed-history/parallel cases passing on the final version.
The feature remains off by default; it is useful evidence for reducing the work needed
to finish a difficult search, not yet an end-to-end acceleration at the current caps.

## Compiling the history comparison

`--compiled-history` compiles the shared `PriceHistory` implementation into the existing
Cython search binary; it requires `--compiled-loop` and the compiled backend. Worker
processes reuse the parent's prepared binary. The helper source is included in its cache
fingerprint, and the Python implementation remains available when the option is off.
No comparison rule changes. Price weights, accumulated charges and strict thresholds
retain arbitrary-precision integer arithmetic; they are not narrowed to 64-bit values.

This first compilation removes Python interpretation from method bodies and loops, but
retains Python dictionaries, sets, classes and integer objects. It **did not demonstrate
an end-to-end speedup** in the matched actual100 pair: Python helper 118.169 s pricing /
136.045 s total, compiled helper 120.705 s / 138.744 s. The roughly 2% difference is one
pair, not a precise estimate of a regression. Cold compilation is excluded from those
timings and cached for reuse. All 1,000 pricing calls match exactly in every recorded
non-wall field, including 122,529,379 comparisons; selected routes, cost, bound and the
83 unresolved flights are identical. The fixed flight-1986 proof changed from 3.660 to
3.556 s, a small single-pair difference.

Validation: 945 relevant cases passed, including native signed-price union oracles,
charges wider than 64 bits, margin ties, interrupted bounds and parallel-worker parity.
An expired compiler/load deadline initially rejected the missing helper instead of
returning the ordinary interrupted-setup bound; the final code preserves that exit and
does not run an uncompiled search after the deadline. The new option remains off by default.
Evidence and the four-file runtime patch are in `.agent/zeta-history-compile-20261117/`.

Further candidates are distinct from compiling these same object operations:

- Before scanning a broad group, use its minimum/maximum retired charges. Since both
  set-difference terms in \(D\) are nonnegative, an existing history can dominate a new
  one only if \(R_{\mathrm{new}}-\min R>\delta\); the new history can replace an existing
  one only if \(\max R-R_{\mathrm{new}}>\delta\). If these necessary conditions fail,
  skip the corresponding scan. This filter is now implemented as described below.
- Store frequently accessed history fields in native structures and use direct compiled
  calls, reducing Python attribute lookup and call overhead.
- For the integer row IDs used by the compact backend, test set differences without
  allocating temporary sets; preserve exact signed-price arithmetic and benchmark the
  representation against the existing sparse sets.

## Repricing cached complete routes and skipping comparison groups

`--route-coverage-cache` adds a persistent index to the complete coverage already kept
in canonical columns. For every discovered route, that coverage includes the full union
of hex/altitude/time reservations and terminal-capacity rows, including currently
zero-price rows. Overlapping legs reserve and pay a row only once.

If \(A_{ij}=1\) when route \(j\) covers reservation row \(i\), its current dual charge is
\[d_j=\sum_i A_{ij}\mu_i.\]
The index stores each row's covered route IDs. Each pricing call applies the current
dual vector to that fixed membership, visiting the smaller of the cached row set and
the current price map. It sums only routes actually requested by the search, preserving
their original `math.fsum` order. New routes are indexed on demand. The index keeps at
most 256 complete reservation unions per graph, uses least-recently-used eviction,
and survives pricing rounds in the existing persistent workers. Prices and forbidden
decisions are never stored in the shared index. Complete claims still govern forbidden
checks and the master coefficients.

This is an in-memory cache of discovered routes, not an enumeration of all permitted
routes. The finite waypoint graph can still have a combinatorial number of complete
walks, departures and lane combinations. Repricing a route library cannot prove that
an undiscovered route is unprofitable. Exact search and its interrupted-search bounds
therefore remain in place. The cache preserves incumbent evaluation order and does
not expand or restrict the route universe.

`--history-group-skips` requires `--strong-history-dominance`. For each eligible
comparison group, let \(R_{\min}\) and \(R_{\max}\) be its extreme accumulated charges
for expired reservations and let \(\delta\) be the existing rounding guard. Since the
remaining live-set difference penalty is nonnegative:

- If \(R_{\mathrm{new}}-R_{\min}\leq\delta\), skip the entire rejection scan.
- If \(R_{\max}-R_{\mathrm{new}}\leq\delta\), skip the entire replacement scan.

Otherwise the existing signed-union comparison runs unchanged. Exact integer thresholds
preserve the strict dominance requirement; extrema update after insertion and removal.
This avoids comparisons without changing which histories are retained. It works with
the reference, compact and native loops, including the compiled history helper.

On actual100, comparisons fall from 122.5 million to 11.5 million (90.6%) while all
1,000 pricing decisions remain identical. The final combined run takes 115.714 s
pricing / 133.484 s total, compared with two controls at 124.607 / 142.585 s and
119.382 / 137.459 s. That is a preliminary 5.15% pricing / 4.67% total reduction
against the control mean, with only one final-index measurement. Group skipping alone
takes 122.501 / 140.638 s and does not establish a separate wall-time gain.

The permanent coverage index has 31,772 hits and 5,058 registrations. Repricing the
captured 37-route and 36-route batches while changing both price values and priced rows
is about 2.6 times faster after indexing, but saves less than a millisecond per batch;
building the index costs roughly 3–4 ms. All six complete runs select the same 100
routes, pass collision/capacity audits, and leave the same 83 final searches unresolved.
The 992 initial targeted tests and 9 affected final-cache tests pass, as do 36 fixed
replays. Both options remain off by default. Full evidence, limitations and the checked
patch are in [the measurement report](/Users/victor/Documents/Research/congestion-utm-freespace/.agent/zeta-group-cache-20261117/README.md).

## Small live pricing demonstration and compiled coverage

`--compiled-route-coverage` now compiles the shared complete-coverage index into the
same native binary as the search; it requires `--compiled-loop --route-coverage-cache`.
Its source participates in the artifact fingerprint, and workers retain the compiled
index between calls. Exact union prices, summation order, eviction and expired-load
deadline handling are unchanged. Python containers remain; the option is off by default.

A small live experiment with two simultaneous 480 m crossing flights confirms actual
price-driven routing. In round 1 the straight route has zero physical cost but 31.72846
of capacity-one hex/time charges. Zeta chooses a detour costing 5.56922 physically and
zero in row charges. Replaying the same input with only row duals zeroed restores the
straight route. One flight alone has no material row prices. The two- and three-flight
cases both certify LP pricing convergence after five CG rounds without hitting a cap.
They still have integer gaps: LP lower bounds 8.78461 / 21.56922 versus feasible filed
costs 16.0 / 45.56922. Converged LP pricing does not certify the global integer optimum.

With compiled history and coverage, warm five-repeat median pricing/total times are
26.1/42.9 ms for two flights and 47.5/76.5 ms for three. Compiler and first-use kernel
setup are separate. Additional compilation shows a small, inconsistent total-time
effect, not a demonstrated large-workload speedup. Validation passes 1,017 targeted
cases, all 26 paired trace calls and 36 repeated live solves. Full equations, actual
iteration duals, counterfactuals and timing evidence are in
[the small pricing experiment](/Users/victor/Documents/Research/congestion-utm-freespace/.agent/zeta-small-prices-20261117/README.md).

## Restricted connections and cheap-first full pricing

`ZetaOptions.connection_mode` selects the existing `full` Euclidean-radius graph,
`hex12` (axial squared norms 1 and 3), or `hex12_long` (norms 1, 3 and 7).
At the actual100 cohort's 120 m pitch and 480 m radius, these allow 60, 12 and
24 destinations before corridor, terminal-wall and distance-budget exclusions.
A 240 m Euclidean radius permits 18 destinations. This is different from a
hex-hop limit; the twelve added long links have length sqrt(7) pitches, about
317.5 m, and hex-hop distance three. Every connection still files and pays its
complete buffered simulator reservation union. Arrival snapping remains 1 s in
the experiment, while reservation periods and time buffers remain 4 s.

`cheap_connection_mode="hex12_long"` on a full graph enables a separate restricted
discovery stage. Its default 5,000 expansion / 25,000 label allowances are deducted
from the full call's configured totals. As in the existing search, label caps are
checked between expansions, so the final atomic successor batch can cross the
label threshold. All stages share the same absolute deadline. History pruning,
signed union prices and both reduced-cost arithmetic checks remain authoritative.

A new improving cheap column returns with `complete=False` and an infinite
full-domain reduced-cost upper bound. A duplicate, a completed restricted search
or a failed cheap search triggers full pricing with the remaining allowance.
Only that full search can supply a finite full-domain bound or convergence proof.
Both graphs retain their caches across rounds. Restricted-only solves certify
their declared connection graph; they do not establish convergence on the full
480 m graph. Canonical validation rejects seeds outside each graph's domain unless
`seed_validation_radius_m` explicitly allows supplied seeds to be validated by a
full graph of that radius. Those accepted seeds form a fixed additional pool; pricing
does not acquire any other routes from the wider graph. The certificate then covers
the restricted graph union that fixed pool, whose existing columns satisfy the LP's
dual constraints. The authoritative timing, cost, volume claims and forbidden-row
checks still apply to every supplied seed.

The actual100 comparison uses one full-480 m initial pool shared by all Zeta
arms, with its preparation time charged to each arm. Current production DP has
its native seed policy and visit-window row model. The task report and source
snapshots live in `.agent/zeta-connectivity-20261117/` in the main workspace.
# Equivalent-history merging and its certificate allowance

`merge_equivalent_histories` is opt-in and requires the existing guarded strong-history
comparison. It only merges prefixes with the same waypoint, exact time, departure/lane
context, endpoint roles, both accumulated lengths, live paid-row set, and exact retired
charge. The segment-length sequence may differ. A missing `HistoryGuard.margin` retains
the original sequence-sensitive comparison.

The merge preserves the exact physical reservation model and permitted route graph.
The equal state determines the same future actions and timing. Appending a common
continuation preserves both length accumulators and exact union charges. Guarded
contexts also preserve canonical feasibility; the whole-route guard bounds the possible
final numerical cost difference from the different segment-length orders.

The returned reduced-cost upper bound includes the largest whole-route margin used by
any performed merge. This is not an unchecked equality assertion. A completed search
with positive allowance can report `no_improvement` only when the corrected bound is
within the configured improvement tolerance. Otherwise it reports incomplete
`merge_uncertainty`; exact-argmin mode also remains incomplete when the allowance is
positive. Capped and interrupted calls retain the corrected finite-domain bound.

The allowance is a whole-route envelope, not an error added once per edge or merge.
Every replacement preserves the geometric state; common extensions preserve that
equality. Exact signed-price dominance can only decrease the final exact union charge,
and equal-price merges leave it unchanged. Consequently the original and final
representative route have ordered exact charges and equal final length accumulators;
one whole-route envelope bounds their possible canonical numerical cost difference.
If a future dominance rule drops an additional state field or permits a price increase,
this argument must be revisited before sharing the same allowance.

Hybrid counters include both stages, but its finite full-domain certificate and
`merge_error_bound` come from the full stage only. No forward-direction restriction is
introduced by this option; avoiding cycles is a separate change to the route domain.

# Guarded shorter-history comparison

`merge_shorter_histories=True` (CLI `--merge-shorter-histories`) is a separate opt-in
extension requiring `strong_history_dominance`. It compares prefixes with identical
waypoint, exact clock, departure/lane context, endpoint roles, live paid rows and exact
retired charge. The retained prefix must be no longer in **both** stored accumulators:
the sequential subsegment sum used by search and the per-edge sum used by the canonical
graph budget. Neither value is rounded or rewritten. Pairs ordered in opposite
directions remain separate. The new index does not broaden the older strict-price
comparison groups or replace their charge-range summaries.

This rule also requires a valid `HistoryGuard` context and finite nonnegative air cost.
Under a common continuation, binary64 additions preserve both length inequalities.
The existing guard ensures stable endpoint folds and sufficient simulator detour slack.
The exact union charge is identical for every common suffix, including signed prices.
Thus the exact-arithmetic comparison cost cannot increase through any combination of
shorter, equal-state and strict-price replacements. Comparing the original and final
canonical routes needs the same one whole-route allowance, not an allowance per merge.

The retained label keeps its own path, lengths and queue floor. An accepted replacement
is queued before checking interruption limits; previously generated descendants stay
valid. A missing guard retains the original comparison. Search caps, deadlines and
`merge_uncertainty` keep their existing conservative semantics. Fewer stored histories
do not by themselves guarantee completion or full-CG convergence.

`shorter_histories_merged` counts discards made by this comparison and is a subset of
the total `equivalent_histories_merged`. Equal-length ties also count when handled by
the shorter rule. The legacy `history_comparisons` counter measures signed-price
comparisons, not the new length-pair checks. Implementation tests and matched actual100
measurements are recorded in `.agent/zeta-shorter-history-20261117/` in the main workspace.

# Forward-only cruise domain

`forward_cruise=True` (CLI `--forward-cruise`) is a separate opt-in route restriction.
Each cruise waypoint displacement must have projected progress greater than `1e-8 m`
along the normalized fixed request origin-to-destination XY direction. The small
positive threshold excludes numerically perpendicular links as well as reversals.
Zero horizontal origin-to-destination distance permits no cruise links. The same
filter applies to canonical column admission and is retained when supplied-seed
validation widens the connection radius. Existing terminal connections, departure
choices, altitude, timing, volume claims and buffer rules remain unchanged.

The waypoint graph is then acyclic: every edge strictly increases its projected
position, so a route cannot return to a previous waypoint. This does not justify
keeping a single price history at each waypoint/time; two different forward paths
can still have different paid-row unions. The existing exact-price history rules
and optional guarded equivalent-history merge remain necessary.

Any certificate in this mode covers only the forward-only domain plus explicitly
admitted supplied columns. It is not a proof for the unrestricted graph. Known convex
obstacles motivate this modeling choice but do not, by themselves, prove that every
unrestricted shortest path is monotone along the fixed OD direction. The actual100
forward experiment uses the same filtered initial pool in both comparison arms:
five of the original 3,500 seeds have backward links, so both start from 3,495.

# Restricted-direction distance bound

`lattice_distance_bound=True` (CLI `--lattice-distance-bound`) strengthens the
remaining-distance estimate without changing permitted routes. For axial displacement
`(q, r)`, sort `abs(q), abs(r), abs(q+r)` as `a <= b <= c`. The ideal shortest
12-direction distance is `pitch * (b + (sqrt(3)-1) * a)`. Adding the axial double
steps present in the 240 m graph does not shorten this distance. Norm-square-seven
links do: the formula is therefore disabled for `hex12_long` and full480.

The implementation in `zeta_lattice_bound.py` rounds the ideal expression downward
and subtracts a conservative whole-route allowance for physical geometry and cost
arithmetic. It requires stable endpoint folds, a nonbinding final detour gate and
supported coordinate/spacing scales; unsafe contexts retain the prior Euclidean
bound. Off-center customer attachments remain part of the existing physical ruler.
The maximum of the guarded lattice estimate and prior Euclidean estimate is used
at the existing remaining-cost and length-budget checks in all search backends.

Only price-independent geometric distances persist in the graph cache. Each price
vector gets a fresh scalar allowance, including signed prices. Exact union charging,
history expiration/dominance, optional merging, duplicate handling and interrupted
search certificates are unchanged. `lattice_bound_calls` counts active pricing calls;
`bound_build_s` sums worker-local preparation durations, not parallel elapsed time.

This option preserves the certificate's existing domain. In the actual100 comparison
that is each restricted graph plus the same 3,500 initial full480 routes, with forward
filtering disabled. The proof and fixed-dual evidence are filed under the root
`.agent/zeta-lattice-bound-20261117/`; matched full solves are in its `actual100/`.

