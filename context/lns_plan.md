# MAPF-LNS over the FCFS A\* schedule — design record

Companion to `context/solver_brainstorm_lacam_cbs.md` (§4 "idea 3"), which holds the literature
analysis and the measured conflict-set / swap-feasibility numbers. This note records what was
**built**, the world adaptations, and the seams it stands on. Papers (checked in here):

- `context/mapf-lns-anytime-ijcai2021.pdf` — Li, Chen, Harabor, Stuckey, Koenig, *Anytime
  Multi-Agent Path Finding via Large Neighborhood Search* (IJCAI-21). The algorithm implemented.
- `context/drop-lns-parallel-2024.pdf` — Chan et al., *Anytime MAPF using Operation Parallelism in
  LNS* (DROP-LNS). The future parallel effort; today's constraint from it: destroy/repair are pure
  transactions against one solution object with a tiny mutation surface, so a parallel port can
  hand workers private copies later.

Decisions fixed by Victor (2026-08-24): all three destroy heuristics; **A\*** (the sim's own
planner + compiled kernel) as the single-agent planner, not SIPP; **PP with random priority
orderings** as the repair; **utilitarian acceptance** (a system operator may move any USS's
intents — but keep a restriction hook); offline first, then online receding-horizon.

## 1. What was built

```
freespace_sim/planner/lns/
  neighborhood.py   destroy operators (paper Algs. 1-2 + random) + ALNS selector
  state.py          LNSState: incumbent, claim index, destroy->repair->accept/revert transaction
  solver.py         LNSConfig / LNSResult / run_lns / run_lns_on_result (anytime loop)
freespace_sim/ledger.py   + release_many() (tombstone removal), + _append/_compact
analysis/run_lns.py       offline runner CLI (FCFS baseline -> LNS -> trajectory json)
tests/test_lns.py         ledger semantics, operators, solver invariants
```

The loop (paper Section 4, adapted): seed = the completed FCFS A\* schedule (already feasible, so
LNS2's collision-repair phase is unnecessary — it survives only as a future escape hatch). Each
iteration: pick a destroy heuristic by ALNS roulette (`w_i <- gamma*max(improvement,0) +
(1-gamma)*w_i`, gamma=0.01), collect a neighborhood of `N` flights (default 8), release them from
the ledger, replan them one-by-one with A\* in a seeded random permutation (PP), and accept iff
**every victim is placed** and the neighborhood's summed weighted cost (`intent.cost`, the
1 ground : 3 air-lateral : 3 hover : 4 climb per-second currency via `cost.trajectory_cost`)
**strictly drops**; otherwise restore the old volumes verbatim. The incumbent is therefore a
complete, ledger-feasible schedule at every instant with monotone non-increasing cost — stop
whenever, file the incumbent. Budgets are **iterations**, never wall clock (reproducibility;
`time_limit_s` exists for cluster jobs only). Per-iteration RNG is
`default_rng(SeedSequence([seed, iteration]))`, so trajectories are machine-independent.

## 2. World adaptations (paper -> hex-lattice reservation world)

| Paper concept | Here | Why |
|---|---|---|
| `delay(p_i) = l(p_i) - d(s_i,g_i)` | weighted premium `intent.cost - unimpeded cost`, unimpeded = one A\* plan per flight against a static-walls-only ledger (computed once at init) | the paper's currency is hops; ours is the 1:3:3:4 cost, and the unimpeded plan already includes the lattice tax so the premium is pure congestion |
| walk start `t ∈ [0, l(p))` incl. pre-wait steps | start sampled from `[unimpeded launch step, arrival)`; starts during the ground hold sit at the launch cell | ground holding is our dominant delay mode (57-95% held) but is not part of the airborne path; without this the walk of a held-but-geodesic flight collects nobody |
| collision at `(v, t+1)` | owners of committed **claim rows** covering `(cell, t+1)` — rows are already W=4-expanded by `rasterize_ranges`, so no extra k-robust window | claim semantics, not visit semantics; swaps/edge conflicts are subsumed by the W=4 same-cell separation |
| map-based: intersection vertices (degree >= 3) | **contention cells** (>= 2 distinct claimants), BFS capped at `map_max_cells` | hex degree is uniformly 6 — topology carries no signal; measured contention concentrates at hub lane mouths, exactly the paper's "ordering through a junction" intent |
| repair failure | any victim denied -> iteration fails, restore | same as paper; see §5 on saturation |
| — (no analogue) | **paired-return anchor guard**: an outbound repair must keep `realized_release + turnaround <= return.t_departure` when the baseline ran `return_anchor="realized"` | we never re-time returns in v0, so the stored anchor must stay honest |
| — (no analogue) | `frozen_flight_ids` / `movable_uss_ids` on `LNSConfig` | the USS-restriction toggle: default None = system operator moves anything (Victor's ruling); flipping it restricts repairs without touching the algorithm |

Documented deviations from the paper: after a tabu reset the most-delayed flight is re-selected
(the paper would walk a delay-0 flight for a guaranteed no-op iteration); map-based BFS is capped
(`4096` cells — the paper explores the whole map, ours is 144k cells/level); neighborhoods may
overshoot `N` by one collection batch (paper behavior, kept).

## 3. The transaction — why it is exact (the three seams)

1. **`ledger.release_many(fids)`** tombstones volumes in place (owner -> `TOMBSTONE_FID = -2`,
   AABB -> empty box). Every read path is AABB-pruned (`conflicts` / `any_conflict` / the
   terminal-capacity column scan reuses `ledger._aabb`), so dead entries are invisible without any
   reader knowing about tombstones; `iter_committed` skips them by owner id and keeps each
   flight's volumes contiguous (`_absorb` groups by adjacent runs). Crucially it does **not**
   re-feed observers — unlike legacy `release()`, which re-commits every kept volume one at a
   time through every subscriber (test-pinned behavior, left untouched). Arrays compact once dead
   entries outnumber live ones.
2. **The shrink tripwire heals the planners.** The A\* occupancy services are incremental and
   add-only; `plan()` compares `ledger.n_volumes` (now a live count) against `n_added` and does
   `reset()` + re-absorb on shrink — for the reference service, the compiled pools, and
   `TerminalCapacity` (dwell counts included). Every LNS iteration starts with a `release_many`,
   so the first repair plan of each iteration rebuilds an exact "everyone but the victims"
   occupancy. A rejected iteration leaves the services stale-with-ghosts by design: nothing reads
   them before the next iteration's first plan, whose tripwire rebuilds again.
   `tests/test_lns.py::test_release_many_heals_planner_occupancy` pins this end-to-end
   (byte-identical intent vs a fresh planner on a never-contained-the-flight ledger).
3. **`evict_floor = 0.0` makes random repair orders exact.** Eviction watermarks are monotone in
   `req.t_request`; replanning a later victim before an earlier one would evict the earlier one's
   obstacles (spurious denials — the exact `any_conflict` backstop catches them, but quality
   dies). The Track A out-of-order-dispatch knob caps the watermark; at 0.0 eviction is a no-op
   and any PP permutation sees the full-horizon occupancy. Costs kernel time on late flights,
   buys the paper's random-priority repair.

Revert restores the old `intent.volumes` verbatim (one `commit` per flight); acceptance updates
the incumbent map, the claim index, the delay cache, and the running total incrementally. The
solver replays the final schedule through `verify.find_interflight_conflict` (an independent
fresh-ledger replay) and reports `verified`.

**The per-iteration rebuild, measured and then removed.** Profiled on the FAA 120 s cut
(290 legs): 3.99 s/iteration median, of which **3.74 s (94%) was the first repair plan's shrink
rebuild** — the actual A\* replans cost 10–100 ms each. That motivated the follow-up, built the
same day as opt-in **incremental release** (`LNSConfig.incremental_release`, default on):

* `ledger.subscribe_release(cb)` — the removal analogue of the commit publish hook;
  `release_many` fires `cb(fid, volumes)` per removed flight. Legacy `release()` refuses to run
  while release subscribers exist (it would desync the per-owner rows).
* Each service, constructed with `track_removal=True` by `AStarPlanner(incremental_release=True)`,
  records its applied rows per owner at commit time and reverses exactly those rows on release —
  no recomputation, so no symmetry bugs:
  - `HexOccupancyService`: step buckets become per-cell **refcounts** (two flights' inflated
    rasters can share a (step, cell), so removal must decrement, not discard); membership queries
    are unchanged. Rows clip at the eviction watermark exactly as inserts do.
  - `CompiledHexOccupancy`: per-cell claim lists `(s0, s1, fid)` beside the interval pools;
    removal resets just the touched cells' linked lists (`_Pool.reset_cell`) and re-applies the
    surviving claims. Abandoned overflow slots are tolerated by the walk (dead-slot semantics the
    pool already has); `col_owners` stays a documented conservative superset.
  - `TerminalCapacity`: dwell rows removed by value (counting semantics); the lazy foreign-transit
    index is invalidated wholesale (its append-only tail assumption dies once the ledger shrinks).
* Every service decrements `n_added` in lockstep, so the shrink tripwire — kept intact — becomes a
  pure safety net: if anything desyncs, the next plan still rebuilds from scratch and warns.
* With removal live, the reject path no longer leaves ghosts at all: release+re-commit keep the
  services continuously exact.

Byte-parity is pinned two ways: flag off, no new code path executes (the service structures are
literally the old ones — the full A\*/occupancy/ledger/terminal suites pass unchanged); flag on,
`tests/test_lns.py::test_lns_incremental_release_matches_rebuild` proves the trajectory and final
schedule are identical to the rebuild path, and
`test_release_many_incremental_heals_planner_without_rebuild` proves plans are byte-identical to a
fresh planner with the shrink warning escalated to an error (the tripwire never fires).

Measured A/B (FAA 120 s cut, 290 legs, 60 iterations, seed 0 — `analysis/run_lns.py
[--no-incremental]`): identical outputs (both −1,285 → 76,898, 17/60 accepted, verified), wall
excluding init **244 s → 38 s (4.07 → 0.63 s/iteration, 6.4×)**. At 300 iterations the effect
compounds with scale because the rebuild grew with schedule size and the removal does not:
120 s cut **1,229 s → 154 s** (4.06 → 0.48 s/iter, 8×) and 300 s cut (754 legs) **3,051 s →
158 s** (10.09 → 0.44 s/iter, **23×**), identical −2.66% / −1.44% schedules. Per-iteration cost
is now flat in schedule size — only the neighborhood's plans — which is what full-FAA and
future-scale runs need.

## 4. On Victor's PP-randomization worry (locked geodesics) and PIBT

The worry is real but has a sharper form here than in the paper: our A\* is **deterministic**, so
replanning a flight in an unchanged environment reproduces its incumbent path exactly, and FCFS
is already *sequentially per-flight optimal* — a single-victim repair can literally never improve
(pinned by the debug traces: `cost_new == cost_old` no-ops). All improvement comes from
**intra-neighborhood reordering** (the B-before-A swap where B gains more than A loses) — which
is what the destroy heuristics are for: they select flights that measurably interact.

**Measured (agent-probe, FAA 120 s cut, 150 agent-based neighborhoods against the frozen FCFS
schedule, each evaluated under the drawn random order AND a forced seed-first order —
`agent_probe_rows.json`):** the lock-in is real and pervasive. Of the 87 multi-flight
neighborhoods: 47% of random-order repairs are *exact no-ops* (every victim reproduces its old
cost — the deterministic re-lock signature); the random order improves 32.2% vs seed-first
48.3% (reverse wins 1.1%); within the random draws themselves the improvement rate doubles when
the seed lands first (47.1% vs 22.6%). Seed-first captures **2,262 vs 1,547 weighted-seconds of
per-neighborhood improvement (+46%) from ordering alone**. Decomposition of seed-first outcomes:
the seed gains in 81.6% (mean 66 weighted-s — the walk does find the right blockers), and the
displaced partners re-plan **free** in 17.2% (the pure "equally-good alternative existed"
mechanism), pay-but-worth-it in 28.7%, and eat the whole gain in 35.6%; in 18.4% the seed
re-locks even with its blockers displaced (capacity/structural delay). Notably the free-swap and
ordering-loss sets overlap by only 3/15 — random ordering already catches most free swaps; what
it forfeits are mostly *paid trades*, which need the seed to claim its slot first. Also: 63/150
walks collect nobody (singletons, 98% no-ops by construction).

**Ordering experiment (same 150 neighborhoods, four PP orders — `agent_probe2_rows.json`):**

| PP order | improve rate | improvement mass (weighted-s) | exact no-ops |
|---|---|---|---|
| random (drawn) | 32.2% | 1,547 | 47.1% |
| seed-first | 48.3% | 2,262 | 12.6% |
| **premium-descending** | **49.4%** | **2,341 (+51% vs random)** | **11.5%** |
| premium-ascending (control) | 5.7% | 280 | 87.4% |

Premium-descending (most-delayed plans first) is the best order tested: it subsumes seed-first
(identical at k=2, which dominates at mean k=2.86; agrees on 94.3% of neighborhoods and edges it
on the rest), needs no seed identity so it works for map/random neighborhoods too, and collapses
the exact-no-op rate 47%→11.5%. The ascending control proves causality in the other direction:
planning the undelayed blockers first re-locks their geodesics and kills 87% of repairs. Random
still finds trades premium order misses in 2.3% of neighborhoods, so the recommended
implementation is premium-descending with RANDOM tie-breaking among equal premiums (partner
premiums are frequently tied at ~0), keeping exploration where the signal is silent.

Mitigations, in the order they should be spent (the probes re-rank them):
0. **Premium-descending repair order — SHIPPED as the default** (`LNSConfig.repair_order`,
   `"random"` kept for A/Bs). End-to-end at 300 iterations, seed 0, vs the random-order runs:
   120 s cut −2.66% → **−3.39%** (49 → 56 accepted, wall 154 → 93 s), 300 s −1.44% → **−1.84%**
   (63 → 81, 158 → 155 s), 600 s −1.49% → **−1.92%** (95 → 116, 255 → 252 s) — ~+28% more
   improvement at the same budget on every cut, all verified, and each run reaches the old
   final quality inside 39–74% of the budget then keeps going. Also still open: give singleton
   walks a fallback partner source (ledger conflict partners of the seed's unimpeded plan —
   the conflict-set machinery); 42% of walks collect nobody today.
1. **Already in**: per-iteration seeded permutations (`SeedSequence([seed, iter])`) — orderings
   never repeat across iterations by construction of the stream, and neighborhoods themselves are
   stochastic (walk starts, roulette).
2. **Next lever (flag-gated, unbuilt)**: seeded random tie-breaking among equal-`f` expansions in
   the A\* kernel, keyed by `(lns_seed, iteration, flight)`. This attacks the *exact* failure mode
   (many equal-cost geodesics, kernel always picks the same one): a repaired victim can vacate a
   contested geodesic at zero cost, which the acceptance test then monetizes through its
   partners. Must default OFF — FCFS byte-parity is test-pinned — and be A/B-measured before
   adoption.
3. **Cheap orderings beyond uniform** (knob, unbuilt): occasionally order by delay descending
   (most-delayed victim picks first) or regret-based insertion. The paper found uniform random +
   more iterations beats cleverer-but-slower repairs (their Exp 2: PP wins 74% on AUC).
4. **PIBT: not now, likely never for repair.** PIBT is cost-blind (it would launch immediately
   and pay 3x air holds), has no notion of our W=4 multi-cell footprints, release times, or pad
   capacity (the brainstorm's §2 verdict), and in the MAPF-LNS ecosystem PIBT/LaCAM serve as
   *initial-solution* generators — a role our FCFS A\* already fills with a far stronger seed.
   If repair throughput ever binds at future density, the levers are the compiled kernel and
   DROP-LNS parallelism, not a weaker planner.

## 5. Measured reality check (tiny-world debug, 2026-08-24)

At hard saturation (900 m box, admission-capped by `max_ground_delay_s=120`), 57/60 repairs are
**denied**: with the ground-hold escape capped, the first greedily-replanned victim squeezes the
rest out — our instances are not "well-formed" in the MAPF sense once the delay cap binds. LNS's
operating regime is the delay-dominated one (the density scenarios), where every flight fits but
FCFS charges a premium (measured: 11%→19%→68% of cost at 974/FAA/future scale). The improvement
e2e test therefore runs in that regime, and offline results on density cuts must always report
the deny-rate alongside cost.

## 6. Efficiency pass (2026-08-25) — what the loop actually spends

Measured on `density_faa_wing_zipline --demand-duration 120 --horizon 1500` (290 legs, 26,216
committed volumes), 300 iterations, seed 0. Every step below is answer-neutral and was A/B'd with a
sha over the whole trajectory (per-iteration operator, victims, accept/reject, costs) — it stayed
`d72adb89b9315204` throughout, and the result stayed 78,183 -> 75,404 (-3.554%, 62 accepted,
verified). Harness: `.context/perf/ab_lns.py`.

Headline, both arms run back to back so ambient machine load hits each equally (`b9d42c3` in a
pristine worktree vs this tree):

| | main `b9d42c3` | this branch | |
|---|---|---|---|
| total | 92.94 s | 78.32 s | **-15.7%** |
| state build (init) | 10.58 s | 4.23 s | **2.50x** |
| iteration loop | 82.36 s | 74.08 s | -10.1% |
| peak RSS | 1,068 MB | 906 MB | -162 MB |
| index + journals | 192.2 MB | 80.3 MB | **2.40x** |
| trajectory sha | `d72adb89b9315204` | `d72adb89b9315204` | identical |

The same pairing on the **600 s cut** (1,526 legs, 137,770 committed volumes — 5.3x the schedule),
which is what the asymptotic arguments above are actually about:

| | main `b9d42c3` | this branch | |
|---|---|---|---|
| total | 249.91 s | 184.11 s | **-26.3%** |
| state build (init) | 56.14 s | 15.23 s | **3.69x** |
| iteration loop | 193.77 s | 168.88 s | -12.8% |
| peak RSS | 2,923 MB | 2,285 MB | -639 MB |
| index + journals | 947.0 MB | 348.6 MB | **2.72x** |
| trajectory sha | `da73a15e18323cdc` | `da73a15e18323cdc` | identical |

Both arms improve 421,348 -> 413,289 (-1.913%, verified). The gains GROW with schedule size
(-15.7% -> -26.3% total, -10.1% -> -12.8% loop, 2.40x -> 2.72x memory), which is the point: what
was removed were terms linear in `n_volumes` sitting inside a per-iteration path whose real work is
linear in the neighborhood.

and the incremental attribution (single arm, same machine, quiet):

| step | total | init | loop | peak RSS | index+journal |
|---|---|---|---|---|---|
| before | 94.3 s | 11.3 s | 83.0 s | 1,029 MB | 192 MB |
| `release_many` slot runs | 95.9 s | 11.8 s | 84.1 s | 1,026 MB | 192 MB |
| packed journals + inlined refcounts | 86.2 s | 11.1 s | 75.0 s | 907 MB | 80 MB |
| parallel unimpeded ruler (8 workers) | 81.8 s | **4.5 s** | 77.3 s | 901 MB | 80 MB |
| rasterisation memo sized to a neighborhood | 78.7 s | 4.6 s | 74.1 s | 896 MB | 80 MB |

**1. `release_many` scanned every committed volume** (`for i, f in enumerate(self._fids)`) to
tombstone the ~380 that a neighborhood owns — a 69x over-scan, and the last per-iteration term
still growing with schedule size (incremental release exists precisely to make that cost flat).
`ReservationLedger._runs` now records each flight's `[start, stop)` slot runs at `_append` (the one
insertion point, coalescing contiguous appends), so the destroy touches only its own slots.
`_compact` renumbers the existing buckets instead of re-deriving every survivor's geometry.

*Honest caveat:* at 290 legs this is **inside timing noise** — the scan profiles at ~0.3% of loop
wall, not the 5.4% it was reported as, and the step's own A/B came back +1.7% (noise). The case
for it is asymptotic: the scan is linear in `n_volumes` while the work it does is linear in the
neighborhood, so at the 600 s cut it scans 137,770 slots per destroy to touch ~2,000. The loop win
does grow with schedule size (-10.1% at 290 legs vs -12.8% at 1,526), but that measures this step
together with the journal packing — do not attribute the whole of it here.

**2. The unimpeded delay ruler was sequential.** It is one A* plan per movable flight against a
ledger holding nothing but the static walls, and it is essentially the whole state build. Because
that ledger is never committed to, plan *i* cannot observe plan *j*: the pass is a pure function
per flight, so `lns/unimpeded.py` shards it across processes with no validation machinery at all
(contrast `freespace_sim.parallel`, whose read-envelopes exist because its plans DO see each
other's commits). A probe prefix decides whether the pool is worth its ~0.4 s spawn, so small
worlds stay in-process; a dead worker has its shard replanned in the parent, loudly.
The public library default is one in-process worker because implicit `spawn` can re-import and
execute an unguarded caller's top-level script. The guarded `analysis/run_lns.py` runner explicitly
passes `None` to opt into the adaptive automatic worker count.
Ruler planners also take `kernel_log2_min=18`: the `max_expansions`-derived g-hash/heap ceiling is
~470 MB, absurd for searches on an empty world, and dropping it measured **473 -> 214 MB per
worker and 7% faster** (cache residency) with bit-identical costs.

**3. The journals were 192 MB of tuples at 290 flights** — not 60, and `LNSState`'s own claim index
is the largest single block, not the occupancy services'. All of it is per-claim, so all of it is
linear in schedule size: the same structures measure **947 MB at 1,526 legs**, which is what makes
this a scaling limit rather than a footprint nicety.

| structure | before | after | how |
|---|---|---|---|
| `HexOccupancyService._rows` | 48.8 MB | 8.5 MB | flat int64 `(cell_id, s_lo, s_hi, code)`; cells interned once |
| `CompiledHexOccupancy._rows` | 31.6 MB | 4.3 MB | flat int64 `(key, packed_claim)` pairs |
| `CompiledHexOccupancy._claims` | 36.0 MB | 22.1 MB | one packed span per claim (`s0<<20 \| s1`); ownership stays in `_rows` |
| `LNSState._cells_of` | 43.6 MB | 12.0 MB | the DISTINCT cells, not one row per (cell, span) |
| `LNSState._claims` | 32.2 MB | 33.3 MB | unchanged (the shared cell tuples now bill here) |

`_cells_of`'s spans were never read — `_index_remove` bound them to `_s_lo`/`_s_hi` and dropped
them — and its 257,954 rows are only **78,705 distinct cells**, so both the removal filter and the
contention refresh were doing the same per-cell work 3.3x over.

**4. Two constants were sized for a different caller.** `_RANGE_CACHE_CAP = 128` matched the FCFS
sim's reuse window (one flight's volumes); under LNS the claim index is a third consumer that
rasterises only after ALL victims are committed, so a default 8-flight neighborhood evicted most of
its own rows first. 1024 (~1 MB) covers it. And the refcounted `_bump`/`_drop` (8.4 M calls per
60 iterations) built the `setdefault` default EAGERLY, allocating a throwaway dict on every hit;
inlined to `.get` + a None test.

### Follow-on: the compiled rasteriser (lever A), 2026-08-25

**Goal of this phase.** Rasterisation — turning a committed `Volume4D` into the hex cells it blocks
— is not on the path INTO the ledger (`commit()` stores volumes verbatim); it runs in the ledger's
subscribers, rebuilding the discrete obstacle map the next A\* search reads. That is a write-side
cost, and it is disproportionately an LNS cost: the FCFS sim rasterises each volume once, while LNS
re-commits a neighbourhood every iteration and, on a rejected one (79%), commits twice. This phase
made each rasterisation ~4.8x cheaper while keeping the numpy cell decisions exact.

| change | point | how it works | issues | outcome |
|---|---|---|---|---|
| `planner/hexgrid_kernel.py` (new) — `sweep_box`, `sweep_cyl` | 62 candidates per volume, 10 kept; ~10 numpy calls on 62-element arrays is ~3.4 us of dispatch each for a few flops | `@njit(cache=True, nogil=True)`, flat scalars in, caller-allocated arrays out, **emits only the kept cells** so the host never materialises the 84% discards | q-major/r-minor is load-bearing (a transposed nest keeps every cell but reorders `_rows`/`_claims`/`block_range` — nothing raises); `np.hypot` is not `sqrt(dx²+dy²)`; no rounding in the kernel (`_axial_round` is banker's, numba's differs) | post-boundary-repair seeded benchmark: 39.2 -> **8.2 us/volume (4.8x)** on `rasterize_volume_ranges` |
| `hexgrid._sweep_kept` + `_axial_rect` | three public rasterisers each open-coded "slack then mask" and would drift; the A/B and rollback need one switch | one helper returns the kept cells; the four-corner rectangle is lifted so both paths derive it identically; a context manager switches and restores the backend | `(q1-q0+1)*(r1-r0+1)` bounds output exactly; box cells near a threshold are resolved by the numpy oracle | `USE_COMPILED=False` is the old numpy path bit-for-bit; all public rasterisers get the kernel and fallback for free |
| `_levels_overlapped` reads `flat_aabb()` | allocated two throwaway `np.array`s per volume just to read two z scalars | one-line substitution; `flat_aabb` is pinned bit-for-bit against `aabb()` | none — the pinning test is the guarantee | two fewer allocations per volume per rasterise |

**The exactness argument, and why it is not bit-parity.** numpy's `(N,3) @ (3,3)` does not sum in
the order a register-scalar expression does. A measured scenario margin is not a correctness proof:
a corridor can put a cell exactly at an inflation threshold, and a two-ULP difference was reproduced
that made the raw scalar kernel omit the cell. The box kernel now computes a conservative roundoff
envelope from the absolute dot-product terms, emits scalar misses just outside the pad when they are
inside that envelope, and marks cells near either threshold. `_sweep_kept` re-evaluates only those
marked cells with the numpy oracle. Ordinary cells stay compiled; boundary decisions are exact by
construction. The cylinder path remains bit-identical (`np.hypot`).
`analysis/verify_rasteriser.py` is tracked and asserts ordered row equality per volume on both full
cuts; the suite also pins the concrete two-ULP regression and a short real cut.

**Measured, paired against `b9d42c3` (both arms back to back, trajectory sha identical):**

| | 120 s cut (290 legs) | | 600 s cut (1,526 legs) | |
|---|---|---|---|---|
| | main | branch | main | branch |
| total | 93.17 s | **70.43 s** (-24.4%) | 241.70 s | **158.68 s** (-34.4%) |
| state build | 11.24 s | 3.57 s (3.15x) | 55.98 s | 11.54 s (**4.85x**) |
| iteration loop | 81.93 s | 66.87 s | 185.72 s | 147.14 s |
| FCFS baseline sim | 21.6 s | 12.1 s | 67.9 s | 58.6 s |
| peak RSS | 1,084 MB | 913 MB | 2,996 MB | 2,303 MB |
| sha | `d72adb89b9315204` | same | `da73a15e18323cdc` | same |

The rasteriser alone (against the pre-A branch) is **-10.1%** at 290 legs and **-13.8%** at 1,526.
Suite: 1115 passed / 2 skipped; ruff clean.

### Next levers, re-measured AFTER the rasteriser

The loop is now genuinely dominated by search: `_plan_compiled`'s 36.6% profiled self-time is the
numba A\* kernel itself (cProfile cannot see into it, so it bills to the caller).

* **Occupancy insert loops — the new largest addressable block.** `HexOccupancyService.on_commit`
  7.9 s + `CompiledHexOccupancy.on_commit` 3.9 s of a 32.7 s profiled run: per-(step, cell) refcount
  bumps and 862 k `_Pool.block_range` calls. Now that the geometry is compiled, THIS is what a
  commit costs.
* **Restore-commit undo cache (~8.7% of loop).** Unchanged in share by A, and it attacks the same
  insert loops from the other end: 79% of iterations are rejected, and `_rewind` re-inserts exactly
  what the destroy just removed. Needs a two-generation identity-keyed stash (the reject path
  releases twice before restoring).
* **`_absorb` (5.4 s, one-time).** The repair planner rebinds and re-absorbs the whole schedule on
  its first plan. Only avoidable by handing over warm services, which the FCFS sim cannot supply
  (`track_removal=False`, eviction watermark advanced).

**Lever C is DROPPED.** Deriving the LNS claim index from the occupancy journal measured 1.9% / 4.3%
before A and **0.8% / 1.8% after** (`_rebuild_claim_index` 0.53 s / 2.89 s) — A is what made the
rasterisation C would have removed cheap. Not worth the coupling it required (reaching into the
service's journal, reproducing its eviction clamp, and only working with `incremental_release=True`).

## 7. Build order from here

1. ~~Offline core~~ (this change).
2. Offline measurement on `density_faa_wing_zipline` cuts (`--demand-duration` 300/600 s, then
   full FAA): anytime curve, % premium recovered vs iterations, wall split (absorb vs plans), and
   the colgen LP bound as the gap denominator where available.
3. Kernel tie-randomization A/B (§4.2) if no-op neighborhoods dominate the trajectory.
4. Online receding-horizon form: batch of arrivals in `[T, T+H)` = the newcomers, their
   collision partners (via `ledger.conflicts` of unimpeded probes, the conflict-set machinery) =
   the rest of the neighborhood `A_s`; PP-repair with the FCFS placement as fallback; freeze
   airborne + departing-within-epsilon flights via `frozen_flight_ids`. The offline transaction
   is reused unchanged — newcomers are victims with no incumbent (revert = FCFS fallback).
5. ~~Parallel LNS (DROP-LNS)~~ — §8. Built and measured: 2.25x (m=4) / 2.55x (m=8) per second of
   loop wall on FULL density_faa, but break-even-to-losing on the 120 s cut. `search_workers`
   stays defaulted to 1 only until the crossover between those two scales is pinned.

## 8. Parallel LNS — DROP-LNS (2026-08-25)

Build step 5 of §7. `context/drop-lns-parallel-2024.pdf` — Chan et al., *Anytime MAPF using
Operation Parallelism in LNS*. Both of the paper's synchronising variants are built (`sync` and
`drop`); DETA is not.

```
freespace_sim/planner/lns/parallel.py   WorkerSpec / TaskResult / _Changelog / LNSWorkerPool
                                        _loop_sync / _loop_drop / run_lns_parallel
freespace_sim/planner/lns/state.py      + LNSState.replica, + apply_delta, + _apply_in_memory,
                                        + unimpeded_cost= injection, RepairOutcome.{new_intents,envelopes}
freespace_sim/planner/lns/solver.py     + LNSConfig.{search_workers,parallel_mode,worker_kernel_log2}
                                        + LNSResult.{search_workers,parallel_mode,npo,auc,parallel_stats}
freespace_sim/planner/lns/neighborhood.py  agent_based_neighborhood(seed_fid=...)
analysis/prof_lns_replica_memory.py     the replica-memory gate
tests/test_lns_parallel.py              25 tests: parity, replica fidelity, apply_delta, pool, DROP
```

### The two departures from the paper, and why

**1. The private copy is a persistent replica synced by delta.** The paper's `P` is a list of
paths, so `P <- P_min` is free. Ours is a ledger + occupancy stack + claim index; building one is
the O(schedule) rebuild §3 measured at 3.74 s and removed. So a worker builds its replica ONCE and
is afterwards told "the incumbent moved" by a compacted diff — `LNSState.apply_delta`, riding the
`release_many`/`subscribe_release` machinery. **Incremental release is what makes DROP-LNS
affordable here at all.**

**2. No mutexes: a single-threaded coordinator IS `M_main` and `M_task`.** It holds its own
`LNSState` over the CALLER's ledger, so `apply_delta` is also the write-back and the seed
selection shares one implementation of `movable_ids`/`delay` with the workers. Workers are spawned
processes on one duplex pipe each (`colgen.pricing_pool`'s shape, for its documented reasons).
Processes not threads despite `astar_kernel` being `nogil`: §6 measured `_plan_compiled`'s mask
build at ~31% of profiled self-time, and `hexgrid._RANGE_CACHE` is a process-global `OrderedDict`
with a check-then-act eviction.

**The always-rewind rule** is what removes three-way merges: a worker never keeps its own accept,
so its `applied_version` is always a coordinator-blessed version. It costs one extra
`release_many` + k commits on the ~21-50% of tasks that accept.

### Two traps that byte-parity caught, and one that only measurement caught

* **The RNG stream position, not the seed.** `AdaptiveSelector.pick` consumes exactly one draw from
  the SAME generator the destroy operators then read (`solver.py:155-161`). A worker rebuilding
  `default_rng(SeedSequence([seed, i]))` starts a draw earlier and picks different victims from
  iteration 0. The coordinator ships `rng.bit_generator.state`.
* **`apply_delta` commits in the CALLER's order, not sorted**, so the ledger's `_vols`/`_fids`
  layout matches what the in-process repair would have produced.
* **The wholesale-overwrite guard's sign** (paper Alg. 2 line 23). With `C_base` the incumbent cost
  at the worker's base and `S` what everyone else has since removed from it, `c(P) < c(P_min)` iff
  `improvement > S`. Written backwards it becomes `improvement > -S`, which is true almost always:
  it fired 15 times in 60 tasks and left DROP at **-0.82%** while ACCEPTING TWICE AS OFTEN as SYNC's
  -1.39%. More accepts, worse schedule, because each one reverted real work. Fixed: -0.82% -> -2.12%,
  overwrites 15 -> 1. Nothing in the test suite caught this; the density readout did.

### Measured — and the honest verdict

Replica memory (`analysis/prof_lns_replica_memory.py`, 120 s FAA cut, 290 legs, 26,218 volumes):
**~350 MiB per worker, flat** (1079 / 1411 / 2075 / 3397 MiB of tree RSS at m = 1/2/4/8). Spawn is
~0.4 s per worker and parent-serial; the replica build is 1.8 s and concurrent.

**The verdict depends on SCALE, and the 120 s cut is the wrong place to read it.** Both tables
below are seed 0, neighborhood 8, every arm verified conflict-free, measured on top of §6
(`analysis/sweep_lns_workers.py`). The rate column is improvement per second of LOOP wall,
normalised to sequential.

**120 s cut — 290 legs, 26,216 volumes:**

| arm | tasks | accepted | improvement | loop wall | task/s | vs seq |
|---|---|---|---|---|---|---|
| sequential | 60 | 30 (50%) | 2.46% | 19.3 s | 3.11 | 1.00x |
| drop m=4 | 120 | 26 (22%) | 1.99% | 15.1 s | 7.97 | 1.03x |
| drop m=8 | 180 | 28 (16%) | 1.90% | 19.7 s | 9.14 | 0.76x |
| sync m=4 | 120 | 21 (18%) | 1.68% | 21.9 s | 5.49 | 0.60x |

**FULL density_faa_wing_zipline — 4,636 legs, 426,756 volumes:**

| arm | tasks | accepted | improvement | loop wall | task/s | vs seq |
|---|---|---|---|---|---|---|
| sequential | 2000 | 497 (25%) | 2.06% | 2000.1 s | 1.00 | 1.00x |
| drop m=4 | 2000 | 376 (19%) | 1.56% | 677.4 s | 2.95 | 2.25x |
| drop m=8 | 2000 | 314 (16%) | 1.14% | 435.8 s | 4.59 | 2.55x |
| **drop m=4** | 6000 | 903 (15%) | **2.64%** | 1721.3 s | 3.49 | 1.49x |
| **drop m=8** | 6000 | 747 (12%) | **2.11%** | **1046.1 s** | 5.74 | 1.96x |

**The wall-clock headline, matched on QUALITY rather than task count** (the 6000-task rows exist
for exactly this — no extrapolation):

* **m=8 reaches the sequential schedule in 1.91x less wall** — 2.11% in 1,046 s against 2.06% in
  2,000 s.
* **m=4 beats it outright**: 2.64% in 1,721 s, i.e. 28% more improvement in 14% less wall.

Note the rate column FALLS as the budget grows (m=8: 2.55x at 2000 tasks, 1.96x at 6000), because
LNS has diminishing returns and the parallel arm is further along its own curve. So "the speedup"
is not one number — it is a function of both instance size AND budget, and any single figure has
to name both.

**At full scale parallel LNS wins, and the ordering of m REVERSES.** At 290 legs m=4 was
break-even and m=8 was a 0.76x loss; at 4,636 legs m=4 is 2.25x and m=8 is 2.55x. Two things move
together to produce that:

* **Per-task cost grows with the schedule** — 3.11 tasks/s at 290 legs, 1.00 at 4,636 — so there
  is 3x more work per task for the pool to hide, and more of the loop is inside the worker rather
  than the coordinator. Throughput efficiency rises with it: 64% -> 74% at m=4.
* **Staleness falls relative to the work.** Discarded-stale is 67/2000 = 3.4% at full scale
  against ~10% on the cut, and clean merges roughly equal dirty ones (94 vs 95 at m=4) where the
  cut was 11 vs 13 out of far fewer tasks. A neighborhood of 8 collides far less often inside
  4,636 flights than inside 290 — which is exactly the "bigger instances" prediction, confirmed.

The redundancy story from the cut still holds, just weakened: value per task still falls (25% ->
19% accept at m=4) but only by 1.32x against a 2.95x throughput gain, where on the cut the two
cancelled almost exactly (2.3x vs 2.56x). m=8 pays more for it — dirty 201 vs 100 clean, 67%
dirty rate against m=4's 50% — and still nets out ahead because throughput is 4.59x.

**Caveat on how to read these:** the 2000-task rows are matched on TASK COUNT, so the parallel arms
reach a worse absolute schedule in much less wall — use the rate column, or the 6000-task rows,
which are matched on quality instead.
The `rss` column in the sweep is sampled AFTER `pool.close()`, so it is the coordinator's
footprint, not the peak — use `analysis/prof_lns_replica_memory.py` for per-worker cost.

`[[parallel-loses-on-density-scenarios]]` is NOT the same result and must not be quoted alongside
this one. That is the A* speculative runner losing 0.66-0.74x on density because its serial commit
floor dominates a compiled per-flight plan — a coordinator bottleneck that gets WORSE with scale.
The LNS pool's small-instance loss came from redundant neighborhoods instead, which gets BETTER
with scale, and does: 1.03x at 290 legs, 2.25x at 4,636. Same superficial shape on the cut,
opposite behaviour where it matters.

SYNC is worse than DROP wherever both were measured, exactly as the paper reports and for the
paper's reason: best-of-m discards m-1 results (`notsel=18` of 120 on the cut). It was NOT run at
full scale — every full-scale row above is DROP — so treat "SYNC loses" as a cut-only finding.
Its value here was never throughput anyway: it is DETERMINISTIC. DROP is therefore the default
mode when parallel search is explicitly enabled; `search_workers=1` still keeps LNS sequential by
default. Effective widths below two use the sequential engine directly because a private replica
cannot add concurrency in that case.

### Where it goes from here

Items 2 and 3 of the earlier list are now DONE and are what produced the full-scale table:
bigger instances and longer budgets were the lever, not a code change. What is left:

1. **Raise the default once the scale rule is pinned.** `search_workers` stays 1 because the
   verdict inverts between 290 and 4,636 legs and nothing yet says WHERE. The missing measurement
   is the 300 s (754 legs) and 600 s (1,526 legs) cuts, which bracket the crossover; with those,
   the default can become scale-dependent instead of conservative. Note this is the same shape as
   `[[lns-neighborhood-size-is-scale-dependent]]` — N=8 wins at 1,526 legs, N=2 at 4,636 — so the
   two knobs should be swept together rather than one at a time.
2. **Seed diversity, still unbuilt — but see §9.3 before spending on it.** §9 shows the repair's
   ceiling is the COLUMN POOL, not neighborhood choice, so diversifying seeds redistributes work
   that is individually capped. Still worth trying at m=8 (dirty 67% vs m=4's 50%), but it is no
   longer the top lever.
   Original note: m workers rank the same
   incumbent and so attack the same congested region; give slot j a seed from a different
   contention cluster. It should matter MORE at m=8, where the dirty rate is 67% against m=4's
   50% — that gap is the redundancy this would attack.
3. **Is the envelope too coarse?** `dirty_rate` is 50-67%, and the read-set test is an xy bbox plus
   hub discs — a conservative superset. Worth checking whether the neighborhoods genuinely collide
   before treating the dirty rate as a property of the problem rather than of the test.
4. **m > 8.** m=8 is still improving (2.55x vs m=4's 2.25x) at full scale, so the knee the paper
   reports at 4-8 threads has NOT been reached here. Sweep 16 and 32 before assuming it exists.

## 9. Repair quality: N, ordering, and where the ceiling actually is (2026-08-27)

§8 answered "does the pool pay". This answers "what is the pool doing the work ON", and finds that
the two knobs everyone reaches for first (neighborhood size, repair order) are worth more than the
repair *strategy*, which turns out to have almost no headroom at all.

All measurements: FULL `density_faa_wing_zipline` (4,636 legs, 426,756 volumes), seed 0, every arm
verified conflict-free. Harnesses live in `.context/perf/` (gitignored):
`probe_order_scale.py`, `probe_best_of_k.py`, `probe_neighborhood_ip.py`.

**Harness note — two caches make this affordable, and both are exact.** The FCFS baseline (~193 s
of A\*) is pickled once and each arm rebuilds a fresh ledger by committing the intents' own
`Volume4D` objects (the `LNSState.replica` / `verify` recipe, which also satisfies the
constructor's object-identity check). The unimpeded ruler — one A\* plan per movable flight, run by
`LNSState.__init__` EVERY time — is memoised by flight id across arms; it is a pure function of
`(request, cfg, static_terms)` and those are identical across arms, so this is exact, not an
approximation. Together they cut a 20-arm sweep from >5 h of redundant A\* to ~11 s of init per
arm. Also: `time_limit_s`'s clock starts BEFORE `LNSState` is built, so a naive 600 s cap spends
most of it on init — every arm below was given `init + 600` and `loop_s` landed at 605-608 s.

### 9.1 Neighborhood size dominates, and the paper's N=16 is the worst setting here

Improvement % at a fixed 600 s of LOOP wall (so the column is directly comparable):

| N | order | seq | sync m=2 | sync m=4 | drop m=4 | drop m=8 |
|---|---|---|---|---|---|---|
| 4 | premium | 1.56% | 1.51% | 1.60% | 1.87% | **1.93%** |
| 8 | premium | 0.91% | 0.90% | 1.16% | 1.45% | **1.48%** |
| 4 | random | 0.97% | 1.02% | 1.25% | 1.24% | 1.60% |
| 8 | random | 0.73% | 0.92% | 0.97% | 1.08% | 1.20% |
| 12 | random | 0.48% | 0.60% | 0.79% | 0.82% | 1.15% |
| 16 | random | 0.42% | 0.41% | 0.57% | 0.88% | 0.89% |

* **N=4 beats N=8 at every mode and both orders**, and under random ordering sequential loses 57%
  of its improvement going N=4 → N=16. DROP-LNS fixes N=16 and MAPF-LNS grid-searches
  N ∈ {2,4,8,16} offline; at this scale the optimum is at the small end of that grid, consistent
  with `[[lns-neighborhood-size-is-scale-dependent]]` (N=2 beat N=8 at 4,636 legs).
* **Parallel's edge WIDENS with N** under random ordering — drop m=8 vs seq is 1.65x at N=4 but
  2.40x at N=12 — because larger neighborhoods make each task costlier, so more work is hideable
  behind fixed coordinator overhead. The corollary is that the best cell (N=4, drop m=8) is the one
  where parallelism helps *least* proportionally; N and mode must be tuned together, not in
  sequence.
* **m=8's edge over m=4 collapses at N=16** (0.89% vs 0.88%) — the first sign of the paper's
  thread knee, appearing only where tasks are largest.
* **DROP > SYNC at matched m everywhere**, as the paper reports.

### 9.2 Ordering is worth 1.2-1.6x, but only as a POLICY, not as a search

Premium-descending vs random at N=4: seq **1.61x**, drop m=4 1.51x, drop m=8 1.21x. Note the gain
SHRINKS as m rises — the parallel arms were partly compensating for a weak repair order, and with a
good order sequential recovers most of that gap (accept rate 31.4% premium vs 24.0% random at N=4
sequential). Any future claim that parallelism is worth Nx must state the repair order.

But **searching over orderings is nearly worthless**. Best-of-k PP orderings on the SAME
neighborhood (1,526-leg cut, 150 neighborhoods, slot 0 = premium, slots 1-7 random, frozen
incumbent so all k see an identical schedule):

| k | accept rate | mean gain/nbhd | vs k=1 | work |
|---|---|---|---|---|
| 1 | 43.3% | 28.1 | 1.00x | 1x |
| 2 | 49.3% | 33.5 | 1.19x | 2x |
| 4 | 51.3% | 36.6 | 1.30x | 4x |
| 8 | 53.3% | 38.3 | **1.36x** | 8x |

8x the plans for 1.36x the improvement — far worse than spending those 8 workers on 8 different
neighborhoods (§8's 2.55x). Random rescued only 10% of the neighborhoods premium missed, and those
were low-value: accept rate rose 10 pp while gain rose 36%. Decisively: **every one of the 789
failures was `no_improvement`, with ZERO `denied` and ZERO `anchor`.** The neighborhoods are
perfectly feasible; PP simply cannot find a cheaper arrangement of those flights in any order.

### 9.3 Joint optimisation does not help either — the COLUMN POOL is the ceiling

`probe_neighborhood_ip.py` replaces PP with an exact set-partitioning IP over the neighborhood:
one route per victim, no conflicting pair, minimise summed cost. Columns are real A\* plans of each
victim against the background plus varying subsets of the other victims' incumbent routes — a
superset of every route ANY PP ordering could produce, so the IP's optimum upper-bounds best-of-k
PP by construction. Solved with `scipy.optimize.milp`, the same HiGHS `colgen.master.HighsBackend`
wraps, minus its revenue transform.

| stage | N=4 (K=5) | N=8 (K=6) |
|---|---|---|
| **IP solve** | **0.9 ms** | **0.9 ms** |
| conflict tests | 5 ms | 30 ms |
| column generation | 2.2 s | 7.2 s |
| PP repair (reference) | 0.42 s | 1.02 s |

**The IP is free — ~1 ms, flat in N — and it finds nothing.** IP = PP = incumbent on 6/6 at N=4;
at N=8 it beat PP on 1 of 6, by 0.07%. The reason is in the column counts: **11 columns for 8
flights**, barely one apiece. A\* is deterministic and cost-optimal, so replanning a victim against
the same obstacle set reproduces its incumbent route; a pool built from A\* replans is exactly the
set PP already explores.

So the bottleneck is not the repair STRATEGY (greedy vs joint) — §9.2 and §9.3 rule out ordering
and jointness respectively — it is the **column pool**. Improving requires columns A\* will never
return: routes that are *worse for that flight* but free a neighbour. That is precisely what colgen
prices, by REDUCED cost against LP duals rather than raw cost, and it is the piece this probe did
not borrow. Two supporting numbers from that side:
`[[colgen-priced-routes-are-free-lateral-swaps]]` measured 84.4% of priced routes as
delay-identical lateral swaps — exactly the free-swap columns missing here — and
`[[colgen-pricing-is-spatial-not-temporal]]` 99.3% as new routes rather than re-timings.

*Caveats*: 6 neighborhoods per N, and the IP's conflict model is pairwise only, so it is a
relaxation of true multi-way feasibility — a proposed triple could pairwise-check clean and collide
three-way. It never bit here (the IP essentially never left the incumbent) but any production use
needs a feasibility replay.

### 9.4 Why we are ~200x slower per operation than the papers

MAPF-LNS's `PathTable` is `vector<vector<int>>` indexed `[location][timestep]` -> agent id;
`constrained()` is 3-5 direct array reads, and insert/delete is one write per path step. Room
(32x32) is 1,024 cells x ~200 steps x 4 B ~= **0.8 MB, L2-resident**. Ours cannot be dense — 144k
cells x 3 levels x ~1,800 steps is ~777M space-time cells — so it is interval pools plus AABB
pruning over 426,756 `Volume4D`s.

| | DROP-LNS (Room k=300, m=8) | here (density_faa) |
|---|---|---|
| ops/s | ~1,030 | 0.65-11.0 |
| per single-agent replan | ~0.48 ms | **~110 ms** |
| parallel efficiency at m=8 | ~97% (7.75x over MAPF-LNS) | 72% at m=4 |
| accept rate | 1.9% | 12-31% |
| memory | 9.8-1,703 MB | ~14.9 GB tree at m=8 |

Measured on this box (pointer chase, random cycle): 2.0 ns at 16-64 KB, 10.7 ns at 4 MB, 28.8 ns at
8 MB, **136.8 ns at 256 MB**; L2 is 12 MB on the performance cluster. A worker's ~1.2 GB working set
overshoots L2 ~100x, so essentially every occupancy probe is a DRAM access — the likeliest reason
our m=8 efficiency is 72% where theirs is 97%. Their own data corroborates it: Den520d (65,792
cells ~= 131 MB table) is exactly where their thread knee drops to 4 and City/Den degrade past it.

**Ideas, ranked by measured leverage** (none built):
1. **Local dense occupancy per task** — materialise a dense `int32[cell, step]` owner array over
   just the neighborhood's bbox x step window, then `is_blocked` is one index. Their PathTable,
   applied locally: a repair reads a tiny region but currently probes a 1.2 GB global structure.
2. **Bound the replica to the task's envelope** rather than the whole schedule — same insight at
   the process level, and the direct attack on the 72% efficiency.
3. **Vectorise the mask build** (`astar/planner.py`, the `to_ok`/`land_ok` loops): ~1,800 Python
   iterations per plan with per-element service calls and scalar numpy stores. It is ~1/3 of
   HOST self-time — *not* 1/3 of plan wall, since kernel time is not in self-time; an earlier draft
   of this section made that error.
4. **N=4 over N=8** (§9.1) — free.
5. Their `goals[]` target-conflict trick: one array lookup instead of endpoint dwell volumes.

Realistic ceiling for 1-3 is maybe 3-5x, not 200x; the rest is problem structure (777M space-time
cells vs ~200k, W=4 footprints, terminal capacity, 3D levels) that they simply do not model.
## 9. The dense occupancy window (2026-08-27) — idea #1 from the MAPF-LNS code read

**Goal of this phase.** After comparing this world against MAPF-LNS's C++ (§8's follow-up), the
ranked list of ideas put **local dense occupancy** first and estimated it "large". This section is
that idea built, measured, and re-estimated: it works, it is byte-identical, and it is worth
**~1.08–1.14x**, not the 3–5x the ranking guessed. The reasoning behind the guess was wrong in a
specific and reusable way, which is most of what this section is for.

### The idea

Their `PathTable` is `vector<vector<int>>` over `[location][timestep]` and `constrained()` is 3–5
direct array reads. On a 32x32 Room map that table is ~0.8 MB. Ours cannot be global — 144k hexes x
3 levels x ~1,800 steps is ~777M space-time cells — so `kernel._blocked` instead walks two
free-interval pools per probe, plus a `static_col` read and an `ov_own_gen` read.

`planner/astar/window.py` builds the same table over the box ONE plan reads: a **bitmap**, one bit
per (cell, step), set iff the corridor pool blocks it OR a foreign column walls it. One bit suffices
because everything `_blocked` returns 1 for is an OR, and both per-cell terms fold in at build time
(`static_col` is step-independent; `ov_own_gen` is stamped with the same per-plan `gen`).

| change | point | how it works | issues | outcome |
|---|---|---|---|---|
| `planner/astar/window.py` (new) | give `_blocked` a cache-resident structure to answer from | `window_bounds` sizes a box around the origin hex, its exit lanes and the landing lanes; `build_window` fills a bit per (cell, step) | the build must not be O(window area) or it costs more than it saves — the first version was, and measured **0.866x** | 1-bit rows, byte-padded |
| `build_window`'s interval merge | make the build O(claims) | `free = corridor-free ∩ (own ∨ column-free)`: an untouched cell (one seed interval, no successor) writes NOTHING; a claimed cell is filled blocked and cleared back over a two-pointer merge of the two free lists | relies on the pools' ascending-sort invariant — which `_Pool.block_range`'s own early-exit already relies on, so it is not new risk | **0.866x -> 1.036x** on metro, and the same rewrite is what makes the density numbers positive |
| `kernel._blocked` + `_search` | the read side | three extra args (`win`, `wbox`, `win_stats`); in-window ⇒ one byte read, a shift, a mask; out-of-window ⇒ the original walk, unchanged | `wbox[W_STEPS] == 0` is the single OFF switch, known only to `_blocked` and `window.disable` | probes outside the window are exact, so undersizing costs speed and never correctness |
| `AStarPlanner.window_bytes`, `LNSConfig.window_bytes`, `WorkerSpec.window_bytes` | make the A/B expressible end to end | 0 = off, default 2 MB | `WorkerSpec` has no field defaults on purpose, so adding one broke a test's construction — which is the discipline working | the LNS loop A/B runs from a config flag, not a monkeypatch |

### Measured

Byte-identical everywhere: the paired A/B compares accept/deny, cost, every volume AND
`last_expansions`, and the LNS A/B compares the whole trajectory.

| measurement | arms | result |
|---|---|---|
| plan wall, density_faa, 120 flights vs the full 426,756-volume ledger | off / on | **1.144x** |
| plan wall, same, 60 flights at 1 / 4 / 8 concurrent processes | off / on | 1.075x / 1.058x / **1.088x** |
| **LNS loop**, density_faa, 300 iterations at N=8, `run_lns` end to end | off / on | **1.075x** (300 tasks, 96 accepted, cost 1,348,639 — identical in both) |
| plan wall, metro_2uss (155 legs, 4,464 volumes) | off / on | 1.036x |
| **ceiling**: window built from each plan's OWN recorded read bbox | off / oracle | **1.157x** |

### Three things measurement corrected

**1. The mechanism is the LIST WALK, not DRAM latency.** The idea was argued from "1.2 GB working
set against a 12 MB L2, so ~137 ns per probe against ~2 ns". That reasoning does not survive
contact: a plan touches only ~5,550 cells (`.context/perf/probe_read_window.py`), i.e. ~180 KB of
pool head slots, which is L2-resident after first touch. The 52 MB is the pool, not what one plan
streams through. What the window actually removes is the free-interval traversal (mean chain 4.4,
worse on hot cells) and two per-cell array reads — a real but bounded saving, and the measured
1.08–1.14x is the size of it.

**2. Concurrency does not amplify it.** `_packed`'s precedent (2.5x solo, 3.1x at 8 processes) was
the reason to expect more under `search_workers=8`. It does not repeat: 1.075x / 1.058x / 1.088x at
1 / 4 / 8. Consistent with (1) — if the structure was already effectively cache-resident per plan,
shrinking it further buys little even when eight processes share the cache.

**3. The bounds are not worth tuning.** The shipped heuristic answers 84% of probes from the window
and gives up entirely on 8% of plans (box over the cap), which looks like obvious headroom. It is
not: an oracle window built from each plan's own recorded `read_bbox` — 100% hits, no plan skipped,
and un-buildable in practice since capturing it costs an extra plan — is worth **1.157x against the
heuristic's 1.144x**, i.e. **1.011x**. Measuring the ceiling before tuning saved the tuning.

### One trap, caught by asking rather than by a test

`planner.py` imports `window` at module level (for `empty_wbox` / `window_bounds` / `disable`, which
are plain Python), but `AStarPlanner.__init__`'s numba fallback is an `except ImportError` around
`from .kernel import _search` — a DEFERRED import — and it cannot see a module-level one. So
`window`'s `from numba import njit` turned the documented "degrade to the pure-Python reference,
~5-7x slower, results identical" into "`import freespace_sim.planner.astar` raises
`ModuleNotFoundError`". Nothing in the 1,000-test suite caught it, because the suite runs WITH
numba; it surfaced from checking a sentence written in `astar/__init__.py`'s docstring claiming the
existing guard covered the new module. `window` now carries its own guard whose stand-in RAISES
rather than interpreting (a silently-interpreted `build_window` would be an unbounded slowdown), and
`test_window_module_imports_without_numba` reproduces a numba-less install in a subprocess.

A fourth expectation also failed, more mildly: the LNS loop was supposed to reward the window MORE
than a static ledger does, since destroy/repair re-fragments the same congested cells thousands of
times and §6 named that fragmentation as the run's slowdown. It measured 1.075x on the loop against
1.144x on plan wall — lower, not higher, because the loop also contains commit/release/index work
the window does not touch.

### Where it goes from here

The window is on by default because it is exact, cheap in memory (~0.1 MB typical, 2 MB capped) and
positive at every scale measured. But at 1.08–1.14x it is not the lever that closes the ~200x gap to
the paper, and (1) above says why the next lever should not be argued from cache size either. The
remaining ideas from that list are unaffected by this result — the mask build (idea #3, ~a third of
host self-time, a genuinely different cost) and N=4 over N=8 (idea #4, free and already measured at
1.3x) are both still open, and neither depends on the occupancy structure at all.

## 10. The footprint release (2026-08-27) — the destroy costs as much as the search

**Goal of this phase.** §9 ended by saying the next lever should not be argued from cache size.
This one was found by measuring instead: a stage-by-stage attribution of the LNS loop
(`.context/perf/prof_lns_loop.py`) put **`release_many` at 35% — as large as the A\* search itself**,
which is not on any prior list of ideas. This section is the measurement, the brainstorm it drove,
and the fix that came out of it: **1.127× on the loop, byte-identical trajectory**.

### Where an LNS operation's ~1 s actually goes

Exclusive time per stage, full `density_faa`, 120 iterations at N=8, dense window off. Two
measurement traps had to be removed first — `run_lns` stamps `wall_s` AFTER its closing `verify`
replay (4,636 commits landing in the `commit` bucket), and the repair planner's first plan
re-absorbs the whole schedule (**38 s, one-time**). Steady state after both:

| stage | ms/task | % of loop |
|---|---|---|
| A\* search (numba kernel) | 455 | **45.3%** |
| **destroy — `release_many`** | 354 | **35.3%** |
| commit (repair + rewind) | 124 | 12.4% |
| A\* host (mask, volumes, conflict check, overlay) | 44 | 4.4% |
| destroy heuristic + claim index | 19 | 1.9% |

Two conclusions land immediately. **Even a free search buys only 1.8×.** And **idea #3 from the code
read — "vectorise the mask build, est. 1.2–1.5×" — is dead**: the whole A\* host is 4.4% and the mask
is 1.5%. The "~31%" that motivated it was a share of cProfile *self*-time, a denominator that
excludes the kernel entirely, and cProfile inflates this code 3.6× (359M calls), over-weighting
exactly the call-heavy loops that idea targeted.

### Why the destroy is expensive, and the brainstorm it drove

`_Pool` stores **free** intervals, which can absorb a block but cannot subtract one, so
`on_release` is `reset_cell` plus a re-apply of every SURVIVING claim on each touched cell. The
reference `HexOccupancyService` holds the same information as **refcounts**, where removal is a
decrement — and costs 2.5% of the loop against the compiled pool's 24.4%. Same information, two
representations, an order of magnitude apart on removal.

Four designs could attack it, each sized by exactly one measurement, so Phase 0 measured before
designing (`.context/perf/probe_release_cost.py`):

| # | measurement | 1,526 legs | 4,636 legs | verdict |
|---|---|---|---|---|
| 1 | victim overlap inside one `release_many` | 1.15× | 1.17× | **batching nearly dead** |
| 2 | release+commit spent on tasks that REVERT | 94.3% | **95.1%** | **the lever** |
| 3 | rebuild vs the released flight's own footprint | 5.4× | **12.2×** | grows with congestion |
| 4 | a pool-less probe's scan vs today's interval walk | 2.8× | **3.3×** | gates dropping the pool |
| 5 | `on_release` share of the loop | 8.0% | **19.8%** | worsens with scale |

(1) killed the cheapest idea: `release_many` publishes per flight so a shared cell is rebuilt once
per victim, and the agent-based operator *deliberately* picks colliding flights — but they overlap
on only ~15% of cells, so batching is worth ~3% of the loop. (4) killed the most ambitious: the pool
is a query accelerator derived from `_claims`, and serving probes from `_claims` directly would make
release O(own) at the cost of scanning 3.3× more on the **query** side, which is 45% of the loop.
(3) is the one that should worry: the blowup is not geometry, it is congestion, and it more than
doubles between the two scales.

### What shipped: the reject-path undo journal

| change | point | how it works | issues | outcome |
|---|---|---|---|---|
| `_Pool.chain` / `_Pool.restore_cell` | get a cell back to a known earlier state in O(its own intervals) instead of O(its survivors) | `chain` walks out the live free intervals; `restore_cell` re-seeds the head and relinks the rest | `_alloc` can `_grow`, which REPLACES `iv`, so the array is re-read every step | restore is O(10.8) where the rebuild was O(35 `block_range` walks) |
| `CompiledHexOccupancy.begin_undo` / `rollback_undo` / `discard_undo` / `resume_undo` | 79–90% of tasks reject, and a rejected task ends where it started | copy-on-write: snapshot a cell's chain and claims the first time the transaction touches it; roll back by writing them straight back | what is NOT journalled needs an argument, not an omission — `col_owners` is a documented never-pruned superset, `_fids` is a monotone interning pool, and `evicted_before` cannot move (LNS pins `evict_floor=0.0`) | `_rewind` 284 → **63 ms** |
| `_suspended` on both ledger hooks | `_rewind` still restores the LEDGER — it is the source of truth — and those calls would re-do work the rollback already undid | after a rollback the service ignores callbacks until `resume_undo` | a service left suspended would silently ignore every later commit, so `resume_undo` is in a `finally`; and `on_release` would `KeyError` on the popped rows, so suspension is required rather than an optimisation | the ledger stays authoritative with no ledger-side seam |
| `LNSState._rollback_and_rewind`, `AStarPlanner.undo_journals` | one place that opens and closes the transaction | opened before the destroy, discarded AFTER the accept path's in-memory move (so a raise there still has a journal), rolled back on every other exit | only removal-mode services qualify — with `incremental_release=False` there is no release subscriber at all and suspending would desync | `LNSConfig.undo_journal`, on by default |

**Measured, paired at the loop level (200 iterations, full `density_faa`):** 215.1 → **190.9 s,
1.127×**, with both arms producing 200 tasks, 65 accepts and cost 1,350,294 — *identical*. The
bucketed profile confirms the win is where it was predicted rather than somewhere else: `_rewind`
inclusive 24.1 → 5.4 s, `release_many` 355 → 237 ms/task, loop 158.5 → 139.0 s.

Parity is pinned on ANSWERS, not bytes, because the restore is deliberately canonical — it drops
the empty `lo > hi` slots `block_range` leaves behind, so a restored chain can be shorter than a
rebuilt one while describing the same free-step set (`_Pool.reset_cell` already states that slot
identity never affects an answer). `test_undo_journal_is_answer_identical_to_release_and_recommit`
compares per-task outcomes, a `blocked_py` sample and the committed multiset; a second test forces a
raise mid-repair and asserts the journal is closed, the service unsuspended, and the occupancy
restored.

### What is left

* **The reference service is the rest of the rewind.** 63 ms still goes to `HexOccupancyService`'s
  release + re-commit, which is not journalled. Worth ~4% more, and harder: its state is refcount
  dicts, so a snapshot may cost as much as the work it saves. Measure before building.
* **Batching (1) composes** and is worth ~3%; it is now the cheapest remaining item.
* **The one-time `_absorb` is 38 s** — 24% of a 120-iteration run, amortising to ~2% at 2,000, but
  paid by EVERY parallel worker at startup.
* **The blowup still grows with congestion.** The journal removes it on the reject path only; a
  destroy that is accepted still pays 12.2×. That is the case a representation change (option 4)
  would fix, and (4)'s 3.3× query penalty is what would have to be bought off first — plausibly by
  making the §9 window mandatory rather than capped.

## 11. Areas 2, 1, 3 from the post-#120 profile — Phase A (2026-08-27)

**Goal of this phase.** §10's attribution left a ranked list; a plan-critic pass then found that the
obvious implementation of its smallest item breaks eight existing tests. This section is Phase A
built the way that review forced: **1.052×, byte-identical, and zero test edits.**

### The finding

`HexOccupancyService` maintains two per-(cell, step) maps. `pad` is read by `pad_clear`; `blocked`
is read by `is_blocked` — and `is_blocked` is called ONLY from `_plan_reference` and the
envelope-recording shim around it. `_plan_compiled` never calls it: measured over 20 LNS tasks,
`pad_clear` 62,537 calls and `is_blocked` **zero**. Yet **98.3% of `pad` bumps also bump `blocked`**,
so on a compiled planner the map is half this service's dict traffic, written and never read.

| change | point | how it works | issues | outcome |
|---|---|---|---|---|
| `HexOccupancyService(maintain_blocked=True)` | make the map optional at the ONE site that knows it will not be read | a `_blocked_live` flag hoisted into `add_volume`/`on_release` beside the existing `track`, extending the guards that were already there (`if in_blk` → `if in_blk and blk_live`) | **the default must be True.** Every test constructs the service directly and six files read `svc.blocked` or call `is_blocked` on one; a default-off parameter — or a gate on `track_removal` — breaks them. Default-on means the parameter changes behaviour at exactly one call site | zero test edits |
| `AStarPlanner._occupancy` passes `maintain_blocked=not self.compiled` | the planner is the only object that knows which search will answer the obstacle test | one argument at `planner.py:321` | a planner that later dispatches to the reference needs the map after all — hence the next row | the saving applies exactly where it was measured |
| `enable_blocked(ledger)`, armed at the top of `_plan_reference` | a fallback must get an EXACT map, not a stale one | re-derives `blocked` through `add_volume` with a new `_pad=False` selector | a fresh per-volume loop would drop the committing flight's own-column skip (making its own 90 m interior read as a wall), write sets where removal mode needs refcount dicts (`TypeError` on the next destroy), and miss the `evicted_before` clamp. Routing through `add_volume` inherits all three | one rasterise/skip/clamp path, two boolean selectors |
| `is_blocked` raises when the map is unmaintained | a silently stale oracle is compared against by every compiled-path parity gate | one guard | unreachable by construction, which is the point: it converts a future wiring mistake into a crash instead of a wrong parity result | — |

### Measured

**1.052×** — median of 3 alternating passes per arm, full `density_faa`, 120 iterations, identical
trajectory across all 6 (35 accepts, cost 1,352,414 every time). The `on` arm won every pass
(131.5→125.6, 134.5→129.3, 132.2→123.2). Profiler agrees: `commit.reference` 25.9→21.6 s,
`release.reference` 3.8→2.3 s, `_absorb` 36.6→34.1 s.

**A measurement bug worth recording, because it nearly shipped.** The first A/B reported **1.137×**.
It was wrong: a sed-based edit had left `undo_journal=on` in the arm config, so the "off" arm ran
with #120 disabled and the number was the two changes together. Two things caught it — the profiler
delta was only 1.037×, and this box's run-to-run spread is ~10% (the same configuration measured 933
and 1029 ms/task on different runs), which is larger than the effect. **A single pass per arm cannot
measure a 5% change here.** `probe_blocked_map.py` now alternates and reports medians, and its
docstring says the pin on the constructor is the ONLY difference between arms.

### What A did NOT do

`add_volume` still writes the `-2 if in_blk else -1` row code whether or not the map is live, so a
later `enable_blocked` never has to reconcile two row vintages — which means A does not save the
`_rows` append, nor the `rasterize_ranges`/`_intern` work those buckets also carry. That is why the
estimate was "~4–5.5%, not half the bucket", and why 1.052× is the estimate landing rather than the
implementation underperforming.

### Phase 0 for area 1 (the pool-less occupancy) — C SURVIVES, with a different shape

`release.compiled` is still 26.5% of the loop after #120, because the journal only fixed the reject
path: an accepted destroy still rebuilds every touched cell from its survivors. Removing that means
removing the free-interval pools, whose measured blocker was a **3.3x point-probe penalty** for
answering from `_claims`. A MANDATORY window has no point probes, so the question is the build.

Five measurements, and three of them moved the design:

| measurement | result | consequence |
|---|---|---|
| **gate** — does `_claims` hold what the pools hold? | equal over 10k+ (cell, step) points on a fragmented schedule, INCLUDING post-`on_release` state | **PASS** — checked while both exist, via `blocked_py_claims` vs `blocked_py` |
| build from a flat arena (`paint` alone) | 0.514 ms vs the pool build's 1.443 ms | **2.81x FASTER** — a claim IS a blocked span, so the paint replaces invert-and-merge |
| build keeping `_claims` as a dict of lists | 38.1 ms (flatten 37.6) | **26x slower — fatal.** The arena is not an optimisation for this phase, it IS the phase |
| `_claims` as a flat arena | **37 MB** against the pools' **52 MB** | **saves memory.** The plan feared 100-200 MB; that was the dict, and the arena replaces something bigger |
| window coverage INSIDE the LNS loop | 79.0% probe hit, and only **58.3% of plans** miss-free at the shipped bounds | the real risk — under a mandatory window one miss forces a widen-and-rerun, so plan-level coverage is the statistic, not probe-level |

The last one is the one that would have killed C, and it is a tuning problem rather than a design
problem:

| margin | cap | plans with ZERO misses | probe hit | plans with no window | peak |
|---|---|---|---|---|---|
| 12 (ships) | 2 MB | 58.3% | 60.9% | 7 | 1.7 MB |
| **24** | **8 MB** | **100.0%** | **100.00%** | **0** | 3.9 MB |
| 48 | 32 MB | 100.0% | 100.00% | 0 | 6.3 MB |

At margin 24 the widen-and-rerun path becomes a rare safety valve instead of the common case, and
the wider build is affordable precisely because the claims paint is 2.81x cheaper. Note the peak
window grows to 3.9 MB, which eight workers cannot hold in one 12 MB cluster L2 — but §9 already
established that the window's win is the LIST WALK and not cache residency, so that is a thing to
re-measure rather than a thing to fear.

**Revised scope for C, against what the plan said:** the arena is mandatory (not an "if flattening
is slow" contingency), the memory objection is inverted (it saves 15 MB rather than costing 150),
and the bounds must widen to margin 24 / 8 MB in the same change. The collateral is unchanged and
large: `_Pool`, the #120 undo journal and its five call sites, and four window tests that reach into
pool internals.

### Step 1 of the pool-less occupancy: the flat claim arena (2026-08-27)

**Goal of this step.** Phase 0 said the arena IS phase C (keeping `_claims` as a dict of lists makes
the window build 26x worse than the pool build it replaces). Step 1 builds it **alongside** the dict
and the pools, changes nothing, and answers one question: *does maintaining it cost more than the
dict it would replace?*

| structure | commit | release | total | note |
|---|---|---|---|---|
| **arena** | 0.03 s | 0.03 s | **0.06 s** | 43 ns per added claim, 55 ns per removed one |
| claim dict (what it replaces) | 0.15 s | 0.00 s | 0.15 s | upper bound — also charges `_record`'s row appends, which survive |
| pools (what deleting them removes) | 7.19 s | 0.28 s | **7.47 s** | 11.8% of the loop |

40 LNS tasks at N=8 on full `density_faa`, with `arena_matches_claims()` asserted after **every**
task, not sampled. The arena is **cheaper than the dict** and **125x cheaper than the pools**.

`claim_arena.py`: one int64 arena holding every claim, each cell's claims in one contiguous slab
described by `start`/`length`/`cap`. Removal is a **swap-remove** — order within a slab carries no
meaning, since the window paint ORs spans and `blocked_at` is a membership test — so removing a
flight costs its OWN footprint. `add_many` computes the tail it needs before writing anything, so a
caller can grow and retry with no risk of double-applying a partial batch.

**Three things went wrong, and each is the useful part of this step.**

1. **The first measurement said the arena cost 252 ms/task.** It was the step-1 shim: a rolled-back
   transaction re-derived the arena from `_claims`, i.e. **4.29 million re-adds per rejected task**.
   Replaced with an inverse replay of the transaction's OWN adds and removes — ~7,450 claims. That
   is what took `arena` from 10.09 s to 0.06 s; the structure was never the problem.
2. **The inverse replay silently did nothing at first.** `rollback_undo` clears `self._undo` before
   restoring (so the restore is not itself journalled), and `_arena_rollback` read that field — so
   it found `None` and returned. Caught by `test_arena_survives_the_lns_reject_path`, which compares
   the arena to the dict after every task; the journal is now passed in explicitly.
3. **Memory came out 4x the Phase 0 projection** — 150 MB against 37 MB. The projection assumed
   exact packing; real slabs are powers of two and the buffer doubles, so a bulk load settles at
   ~2.7x its live size with almost no garbage. A garbage-only compaction trigger never fires there.
   Adding a `tail > 2 x live` trigger and right-sizing the compacted buffer took it to **99 MB**
   holding 34 MB of claims.

The memory story is honest but not finished: 99 MB of allocation for 34 MB of live claims. It still
comes out ahead in the final design, which deletes **both** the 52 MB of pools **and** the ~186 MB
dict — but "the arena is smaller than the pools" (Phase 0's claim) is only true of the packed data,
not of the allocator, and a bulk-load path that sizes slabs exactly during `_absorb` is the obvious
next lever.

### Steps 2 and 3: the pool-less occupancy (2026-08-27) — 1.51x, identical schedule

**Goal.** Delete the free-interval pools, so that removing a flight costs its own footprint instead
of a rebuild of every cell it touched from that cell's survivors (12.2x, growing with congestion).

**Step 2 — the window becomes mandatory.** Bounds widened from margin 12 / 2 MB to **margin 24 /
8 MB**, chosen from the plan-level coverage sweep rather than the window's typical size: what matters
once the window is the only answer is not the hit RATE but the share of plans with ZERO misses (one
miss forces a re-run), and that is 58.3% at the old bounds and 100.0% at the new. A probe outside the
window raises a per-plan sticky flag; the host reads it once after the search, widens and re-runs
under a fresh `gen`, using none of the previous output — the FB_MASK discipline.

*Why a flag and not a new return code:* only one of `_search`'s five `_blocked` call sites inspects
the returned value (the neighbour test, looking for the out-of-box -1). The other four compare
against zero, so a new negative code would read as "blocked" at four of them — silently pruning a
legal edge without raising anything. One branch per plan beats five per probe, and cannot be misread.

**Step 3 — delete the pools, the claim dict, and #120's undo journal.** `build_window_claims` paints
from the arena: a claim IS a blocked span, so it is `win |= span`, where the pool build had to fill
each claimed row and clear back over a two-pointer merge of two free-interval lists. `_Pool` and its
`chain`/`restore_cell`/`block_range`/`blocked_at`/`reset_cell` are gone, and with them the undo
journal — which existed to make the pools' rebuild cheap on the reject path, and has nothing left to
undo now that a release is O(own footprint).

`window_bytes = 0` stops meaning "window off, same answers" and starts meaning "the kernel cannot
answer a probe", so it now raises; `compiled=False` is the off switch. A window that cannot be built
even at the widen ceiling dispatches to `_plan_reference` with `_fb_reasons["window-exhausted"]`.

**Measured**, like-for-like (both `--sub`, 120 iterations, full `density_faa`, same seed):

| bucket | with pools | pool-less |
|---|---|---|
| **loop** | 124.9 s | **82.5 s** |
| `release.compiled` | 24.8 s (15.79 ms/call) | **0.1 s (0.06 ms/call)** — 263x |
| `commit.compiled` | 23.2 s (3.73 ms) | 12.2 s (1.97 ms) |
| one-time `_absorb` | 34.1 s | 24.0 s |
| per task | 1041 ms | **688 ms** |

End to end via `run_lns` at 200 iterations: **186.5 -> 119.0 s, 1.57x**, and the SCHEDULE IS
IDENTICAL — 120/200 tasks, 35/65 accepted, cost 1,352,414 / 1,350,294 matching values recorded before
the pools were deleted, both verified conflict-free.

**What the parity gates became.** The window tests used to compare window-on against window-off;
that arm no longer exists, so they now compare the compiled path against the pure-Python reference —
strictly stronger, since the reference shares no occupancy structure with the window where the old
off-arm shared the pools the window was built from. `test_compiled_occupancy_matches_is_blocked` is
the external oracle for `blocked_py`, which is now an arena scan. Deleted: two `_Pool` unit tests,
the `block_range` free-set oracle, the two #120 journal tests, and three window tests that reached
into pool internals.

**What is left.** `commit.reference` is now the largest occupancy bucket at 24.3% — the reference
`HexOccupancyService`, which Phase A already halved and which exists to answer `pad_clear` at two
cells per plan. `_absorb` is still 24.0 s one-time, now almost entirely that service plus
`TerminalCapacity`. And the arena holds ~99 MB of allocation for 34 MB of live claims; sizing slabs
exactly during a bulk load is the obvious next lever.

### Phases 4 and 5: the absorb pass and the own-column test (2026-08-29) — 1.88x bind, 1.43x total

Two findings from re-profiling `_absorb` per call. Both are on the commit path, both are
mechanism-only, and neither was touched by #119-#123.

**The measurement that started it.** Both occupancy images absorb the SAME ledger with the SAME call
counts — 4,636 `on_commit`, 426,756 volume calls, 4,327,758 rasterizer rows each — so the gap between
them (16.09 s vs 12.02 s) is cost per row, not call count. The rasterizer emits 7.36 steps per row;
the reference expands that to per-step dict entries (31.9 M writes, 66.1 M `dict.get`), the compiled
one stores 4.29 M packed spans.

**Phase 4 — `_absorb` defeats the memo it was designed around.** `hexgrid.rasterize_ranges` memoizes
geometry by `id(vol)` behind a 1024-entry LRU precisely so a commit's several consumers share one
sweep; its comment says the cap "must exceed the reuse WINDOW", and SIPP's `_add` states the same
intent for three consumers. That holds on the LIVE commit path, where `ledger.commit` fans out to
every subscriber back to back. It does NOT hold in `_absorb`, which ran each service as its own full
pass — reuse window 426,756 volumes against a 1024 cap, so every consumer after the first missed and
re-swept. `_absorb_many` feeds each flight to every service before moving on. **1.118x** through the
real `plan()` bind path.

Absorb is an LNS-only cost: in a plain FCFS run the planner binds while the ledger is still empty
(measured: 3 `_absorb` calls, 0 volumes), and everything after arrives through the subscribe hook.

**Phase 5 — the own-column test was loop-invariant and computed per cell.** `_inside_a_column` existed
in three textually identical copies (both occupancy images and `LNSState._inside_own_column`, a fourth
via `enable_blocked`), each called per rasterized cell — 4.2 M times per consumer, every one
allocating a numpy array inside `hex_center`. The answer depends only on `(cols, R)`, both fixed for a
flight. `hg.column_hexes` resolves it to a frozenset once per flight, memoized across all consumers;
the three methods are deleted. **1.609x**.

    phase                          bind        loop       total
    before (per-service, per-cell) 25.62 s     21.33 s    46.95 s
    after  (one pass, per-flight)  13.61 s     19.12 s    32.73 s
                                   1.882x      1.116x     1.434x

Identical bind fingerprint (15 fields, both images plus `TerminalCapacity`) and identical repair
trajectory across all six alternating passes.

**Two things measurement corrected.** cProfile put `_inside_a_column` at 15-24% of each absorb, which
would cap Phase 5 near 1.3x; it measured 1.609x, because the profiler bills the predicate's own frames
but not the `np.array` allocation inside `hex_center` — 12.6 M allocations across three consumers.
And the two phases were expected to be sub-additive, since Phase 5 shrinks the same work whose
duplicate pass Phase 4 removes; the direct combined measurement (1.882x) slightly EXCEEDS their
product (1.799x), so the interaction runs the other way.

**Two traps caught in review, both real.** Applying the batch uniformly would have added a
`subscribe_static` to the shrink-rebuild branch, which never had one — it both appends a subscriber
and replays every hub, so each LNS release cycle would leak one, answer-neutral but not
state-identical. And both services are assigned to `self` and to the ledger BEFORE their absorb, while
no rebind guard can see a partially-fed service (`n_added` only under-counts, so the shrink tripwire
stays silent) — a raise inside the batched absorb would leave a hollow, permanently-bound occupancy.
`_plan_compiled` now drops both ledger identities on failure so the next `plan()` rebuilds.

The `column_hexes` cache settled at exactly 182 entries on `density_faa_wing_zipline`, which is that
scenario's `wing_hubs` — the key is per-HUB, not per-flight, so the cap is sized against
`density_future_wing_zipline`'s 476.
