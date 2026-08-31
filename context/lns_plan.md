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
