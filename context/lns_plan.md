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

## 6. Build order from here

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
5. Parallel LNS (DROP-LNS) — separate effort, separate branch.
