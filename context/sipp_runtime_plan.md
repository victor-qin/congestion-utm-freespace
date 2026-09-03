# Making SIPP's LNS runtime competitive again — profiling and PR plan

Branch: `victor-qin/sipp-runtime-profiling` (measurement + this record), off `4c81255` (#125 merged).
Prior record: `context/sipp_lns_plan.md` (the integration; §13 records the inversion this plan answers).

## 0. The question

`#125` shipped SIPP as a selectable LNS repair planner. When it was written it was ~1.1x *less* wall
than A\* at the same schedule quality. `#124` then rewrote A\*'s occupancy and the verdict inverted:
A\* is now **1.67x faster** (§2 re-measures this on the merged code; `#125` §13 recorded 1.50x, from
a run whose A\* arm had drifted high).

Nothing about SIPP got slower. **A\*'s ledger side got cheaper and SIPP's structures were not in the
rewrite.** This document re-measures that from scratch on the merged code, finds the term that
dominates, and proposes the PR series that removes it.

## 1. Measurement

`analysis/prof_sipp_ledger.py` (new). It absorbs a real `density_faa` schedule into each ledger
structure *individually*, then times `on_commit` and `on_release` over a held-out block of flights.
Per-structure rather than per-planner, because the two planners share `_svc` and the question is
which of the *differences* pays.

Full `density_faa` (4,636 flights / 426,773 volumes; 2,400 warm, 150 timed):

| structure | commit ms/fl | release ms/fl | total |
| --- | ---: | ---: | ---: |
| sipp `_svc` — hex dicts, `blocked` **ON** | 3.413 | 2.150 | 5.562 |
| astar `_svc` — hex dicts, `blocked` OFF | 2.529 | 1.226 | 3.755 |
| sipp `_scocc` — free-interval **POOL** | 3.577 | **5.995** | 9.572 |
| astar `_cocc` — claim **ARENA** | 1.720 | **0.034** | 1.753 |
| sipp `_sidx` — step-keyed dicts | 2.881 | 0.824 | 3.706 |
| **SIPP total** (3 structures) | 9.871 | 8.969 | **18.840** |
| **A\* total** (2 structures) | 4.249 | 1.260 | **5.509** |
| ratio | 2.32x | **7.12x** | **3.42x** |

All three structures bind on **every** plan at `density_faa` (`fixed_exit_lanes=True`, every flight
hub-to-hub): 1,526/1,526 calls to each of `_occupancy`, `_scompiled_occ`, `_sipp_index` and
`_sbuild_overlay`. None of this is a terminal-only tail.

### 1.1 The release term is O(congestion), not O(footprint)

A free-interval pool cannot be un-split in place, so `CompiledOccupancy.on_release` resets each
touched cell and **re-applies the survivors**. That multiplier is how many *other* flights share the
released cells, so it grows with the schedule:

| warm flights | own claims | cells | re-applied | amplification | release ms/fl |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 122,831 | 37,174 | 231,439 | 1.88x | 0.611 |
| 300 | 129,188 | 39,075 | 285,687 | 2.21x | 0.879 |
| 600 | 138,208 | 41,766 | 403,455 | 2.92x | 1.496 |
| 1,200 | 129,404 | 39,141 | 576,066 | 4.45x | 2.772 |
| 2,400 | 132,573 | 40,071 | 931,928 | **7.03x** | **6.105** |

Dead linear in congestion, and 2,400 is still only half of `density_faa`. A\*'s arena over the same
releases is **0.034 ms and flat** — a swap-remove costs the flight's own footprint, so there is no
multiplier to grow.

This is not an analogy to A\*'s old design. **It is the same algorithm.** `claim_arena.py`'s header
records what `#124` deleted: *"removing a flight then rebuilt each touched cell from its SURVIVORS —
measured at 12.2x the released flight's own footprint at density_faa scale, and growing with
congestion."* `CompiledOccupancy.on_release` is a deliberate port of that predecessor, made when it
was still A\*'s shipped design and its structural twin `_Pool.reset_cell` still existed. The twin is
gone; the port is not.

### 1.2 A plan reads 1.4% of what the pool maintains

Instrumenting `_splan_compiled`'s recorded read bbox over a full 1,526-flight SIPP run:

```
global box      : 741 x 386 q,r x 1 level = 286,026 cells, MAXS 4,106, pool 654,266 slots
read bbox cells : p50 3,910   p90 12,690   max 20,655
read steps      : p50 1,172   p90 1,284    max 1,317   (of 4,106)
COVERAGE        : p50 1.367% of cells x 28.5% of steps = 0.39% of the (cell, step) box
```

This is exactly the observation `#124` acted on for A\*: the global derived structure is maintained
in full so that each plan can read a thousandth of it. `window.py` already contains the answer —
materialise the region a plan actually reads, from claims, per plan.

### 1.3 `_sidx` maintains a global index to answer ten cells

`SafeIntervalIndex` costs 3.706 ms/flight and has exactly one consumer on the compiled path:
`_sbuild_overlay` calls `free_intervals(q, r, L, own, base, max_step, fixed)` for the flight's own
terminal lane cells — `o_lanes + d_lanes`, order ten cells per plan. A global step-keyed inverse
index, maintained on every commit and released on every destroy, to answer ten per-cell queries.

### 1.4 `needs_blocked_map` pays for a map nothing reads

`SIPPPlanner.needs_blocked_map = True` forces `HexOccupancyService` to maintain the `blocked` map
that compiled A\* switches off. Its only reader is `is_blocked`, and the only SIPP call sites are in
`_succ` — the **pure-Python reference** successor generator — plus the A\* fallback. `_splan_compiled`
never calls it; measured fallbacks on `density_faa` are zero. Cost of the map alone: **1.807 ms/flight**
(5.562 against 3.755).

A\* had the identical problem and solved it: build the service with `maintain_blocked=False` and call
`svc.enable_blocked(ledger)` at the top of the reference search — a sticky, one-shot O(schedule)
rebuild that arms the map before any `is_blocked` can run. SIPP declared the flag instead.

## 2. Where this leaves the end-to-end picture

Fresh paired A/B on `4c81255` (`analysis/ab_lns_repair_planner.py`, full `density_faa`, 4,636 legs,
N=8, 300 iterations, arms strictly sequential, baselines asserted identical):

| arm | loop s | plan s | ledger s | other s | improvement | accepted | release subs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astar | 165.8 | 135.7 | **23.5** | 6.6 | 0.46% | 86/300 | 3 |
| sipp | 276.8 | 146.3 | **124.1** | 6.4 | 0.54% | 88/300 | 4 |

SIPP is **1.67x slower** on the loop, and the ledger is 44.8% of it against A\*'s 14.2%. Quality is a
wash (+0.08 pp against a recorded 0.18 pp seed-to-seed spread), zero kernel fallbacks, both verified.

**Two corrections to the story as previously recorded.**

*First*, the ledger ratio here is 124.1 / 23.5 = **5.28x**, higher than the 3.42x §1 measures. The two
agree: §1's warm set is 2,400 flights and the real LNS state is 4,636, and §1.1 shows the release term
is linear in congestion. Extrapolating the amplification (7.03x at 2,400) to full scale puts SIPP's
release near 12 ms/flight and its total near 24.8 against A\*'s 5.5 — 4.5x, and the remainder is the
rejected-iteration release/recommit that both arms pay. **§1's numbers under-state the gap** because
they were taken at half the congestion; they are the conservative bound.

*Second, and it cuts against this plan:* SIPP's plan side is **not** ahead. `context/sipp_lns_plan.md`
§13 reported it as 1.05x faster, derived by subtracting ledger from loop. This run has an explicit
`t_plan_s` timer and it says 146.3 s against A\*'s 135.7 — SIPP's plan side is **7.8% slower**. The
older figure folded `other` into "plan" and compared against an A\* loop that had drifted high
(181.0 s then, 165.8 s here; A\*'s loop time is known to drift across invocations). Section 4.7 is
re-derived from these numbers, not those.

The ledger is still the whole *actionable* deficit — 100.6 s of the 111.0 s gap — and bringing it to
A\*'s cost is not a tuning exercise: it is adopting the structure `#124` built.

## 3. The PR series

Three PRs, stacked. Each is independently measurable and independently revertible; **PR 1 ships on
its own** and pays 9.6% of the ledger side for ~20 lines.

### PR 1 — `sipp: stop maintaining the write-only `blocked` map`

*The point:* delete the one cost SIPP pays purely because it declared a flag instead of reusing A\*'s
lazy re-arm. Nothing about the compiled path changes; the reference path arms the map itself.

**`freespace_sim/planner/sipp.py`**

* `SIPPPlanner.needs_blocked_map` — **REMOVED** (class attribute; inherits `False` from `AStarPlanner`).
* `SIPPPlanner._splan_reference` — **MODIFIED**. One line after the service is bound:

  ```python
  svc = self._occupancy(req, ledger, cfg)
  svc.enable_blocked(ledger)      # this search reads `blocked` via _succ; arm it before any probe.
                                  # Sticky (see HexOccupancyService.enable_blocked): a run that falls
                                  # back once pays the O(schedule) rebuild once, not per plan.
  ```

* `SIPPPlanner._fallback` — **UNCHANGED**. It delegates to `AStarPlanner.plan`, whose
  `_plan_reference` already calls `enable_blocked`.

**`tests/test_lns_sipp.py`**

* `test_sipp_does_not_maintain_the_blocked_map_on_the_compiled_path` — **ADDED**. Plan a compiled
  SIPP flight against a committed ledger; assert `planner._svc._blocked_live is False` and
  `planner._svc.blocked == {}` while the plan is still exact against the reference.
* `test_sipp_reference_arms_the_blocked_map` — **ADDED**. Force `_splan_reference` directly, assert
  `_blocked_live` flips to `True` and the resulting intent is byte-identical to one planned with the
  map maintained from the start. *This is the assertion that actually gates the change* — a test that
  only checks the flag is off would pass against a build that silently plans wrong.

*Expected:* `_svc` 5.562 -> 3.755 ms/flight; SIPP ledger 18.840 -> 17.033 (-9.6%). No plan-side change,
zero answer change.

### PR 2 — `sipp: build safe intervals per plan, from A*'s claim arena`

*The point:* delete the global free-interval pool and the global inverse index, and derive both from
the claim arena over the window a plan actually reads. This is `#124` applied to SIPP's side of the
house: it removes the O(congestion) release term, the `_sidx` maintenance, and the static-wall trap
in one move, and leaves SIPP maintaining exactly the two structures A\* maintains.

Split in two so the new kernel input is provable before anything is deleted.

#### PR 2a — the builder, gated against the current pool as an oracle

*The point:* a numba builder that turns arena claim slabs into the interval-chain layout the SIPP
kernel already consumes, verified byte-exact against the structure it will replace. Pure addition —
no planner touched, nothing deleted, so a parity failure is a red test rather than a wrong schedule.

**`freespace_sim/planner/sipp_window.py`** — **NEW FILE**.

* `window_bounds(cocc, wbox, *, q_cells, r_cells, base, max_step, lateral_margin, max_slots)`
  — **ADDED**. Mirrors `astar/window.window_bounds` with one deliberate difference: the step span is
  `[base, max_step]` **exactly**, not a heuristic tail.

  ```python
  # A* clips steps to base + n_gsteps + tail_steps because a bit outside the span is one probe.
  # A SIPP interval's `hi` answers "how long may I wait here", so a short span does not miss a
  # probe -- it silently shortens a wait and changes the plan. max_step already bounds the search
  # (the kernel skips arr > max_step), so clipping there is exact and removes a widen axis.
  q0, q1 = min(q_cells) - lateral_margin, max(q_cells) + lateral_margin
  r0, r1 = min(r_cells) - lateral_margin, max(r_cells) + lateral_margin
  clip to cocc's global box; s0, s1 = max(0, base), min(max_step, cocc.MAXS)
  n_wcells = (q1-q0+1) * (r1-r0+1) * n_levels
  return -needed if n_wcells + slack > max_slots else n_wcells   # recoverable, like A*'s -nbytes
  ```

* `build_window_intervals(arena, slab_start, slab_len, static_col, ov_own_gen, gen, qmin, rmin,
  rspan, n_levels, wbox, iv_lo, iv_hi, iv_nxt, s0_shift, span_bits, field_mask)` — **ADDED**, `@njit`.
  Argument-for-argument `astar/window.build_window_claims`'s signature (same arena accessors
  `cocc._arena.arena/.start/.length`, same `key = (c << 1) | pool_idx`, same packing constants), with
  the bitmap output swapped for the interval chain. One pass per in-window cell; the head slot of
  window-cell `w` IS slot `w`, overflow appended after `n_wcells` — the layout `sipp_kernel._search`
  already walks. Ownership is resolved per CELL by `ov_own_gen`, exactly as A* does, which is what
  §4.1 is about.

  ```python
  for w in range(n_wcells):
      iv_lo[w], iv_hi[w], iv_nxt[w] = s0, s1, -1        # start fully free over [base, max_step]
      c = global_cell_of(w)
      if static_col[c] and ov_own_gen[c] != gen:
          iv_lo[w], iv_hi[w] = 1, 0                     # foreign always-active wall: no free interval
          continue
      own = ov_own_gen[c] == gen
      for pool_idx in (0, 1):                            # 0 = corridor, 1 = column
          if pool_idx == 1 and own:
              continue                                   # own column is transparent (== the deleted overlay)
          k = (c << 1) | pool_idx
          for j in range(slab_start[k], slab_start[k] + slab_len[k]):
              s_lo, s_hi = unpack(arena[j])
              subtract [s_lo, s_hi] from cell w's chain   # same splice `block_range` does today
  ```

  Subtraction is the existing `block_range` logic, moved into numba and operating on a window-local
  chain. Claim order is irrelevant (subtraction commutes), which is what makes reading the arena's
  slabs directly sound.

**`tests/test_sipp_window.py`** — **NEW FILE**. The oracle gate:

* `test_window_intervals_match_the_global_pool` — **ADDED**. Commit a congested `density_faa` cut into
  BOTH `CompiledOccupancy` and `CompiledHexOccupancy`; for a sample of window boxes, assert every
  in-window cell's interval chain from `build_window_intervals` is **set-equal** to the global pool's
  chain clipped to `[s0, s1]`. Chains are compared as sorted `(lo, hi)` lists, not slot-by-slot: slot
  order is storage, and asserting on it would pin an implementation detail the kernel does not read.
* `test_window_intervals_handle_the_static_wall` — **ADDED**. A hub with
  `terminal_airspace_always_active`: assert the wall's cells are empty for a foreign flight and fully
  free for the owning hub. *This is the case `CompiledOccupancy` could only express by writing the
  wall into the same array as commit-derived blocks — the trap that bit twice in #125.*
* `test_window_intervals_own_column_matches_the_overlay` — **ADDED**. Assert the built chain for an own
  lane cell equals `SafeIntervalIndex.free_intervals(...)` for the same flight — i.e. the builder
  reproduces `_sbuild_overlay` before the overlay is deleted.

*Expected:* no runtime change (nothing calls it yet); the parity evidence PR 2b needs.

#### PR 2b — flip the planner, delete the two global structures

*The point:* make the window-local pool the kernel's only occupancy input, so the global pool and the
global index have no readers and can go.

**`freespace_sim/planner/sipp.py`**

* `SIPPPlanner._scompiled_occ` — **REMOVED**. Replaced by the inherited `AStarPlanner._compiled_occ`,
  which binds `CompiledHexOccupancy` (the arena) to the ledger. SIPP and its own A\* fallback then
  share one instance instead of maintaining two structures for the same claims.
* `SIPPPlanner._sipp_index`, `_sbuild_overlay`, `_overlay_slot` — **REMOVED**, along with the
  `_k_ov_*` overlay arrays in `_skernel_state`. Own-column transparency moves into the window build's
  `ov_own_gen` argument, which is A\*'s existing per-plan stamp.
* `SIPPPlanner._skernel_state` — **MODIFIED**. Work arrays size to the **window**, not `cocc.cap`:
  `front_head/tail/gen`, `goal_gen`, `goal_cost` become `max_window_slots`-sized. Grow-on-demand with
  the same `if self._k_cap < needed` guard.
* `SIPPPlanner._splan_compiled` — **MODIFIED**. The overlay build and the `cocc.cap` frontier give way
  to a widen loop that mirrors `AStarPlanner._plan_compiled`'s:

  ```python
  cocc = self._compiled_occ(req, ledger, cfg)          # the shared arena
  widen = 0
  while True:
      self._sgen += 1                                   # FRESH gen per attempt (see risk 3)
      self._mark_own_cells(cocc, gen, o_term, d_term)   # A*'s _stamp_own_overlay, reused verbatim
      n_slots = SW.window_bounds(cocc, wbox, lateral_margin=_MARGIN << widen, ...)
      if n_slots <= 0: return self._splan_reference(...)
      SW.build_window_intervals(cocc.arena, ..., wbox, ...)
      code, ... = sipp_kernel.search(iv_lo, iv_hi, iv_nxt, wbox, ...)
      if code != FB_WINDOW or widen == _WIDEN_MAX: break
      widen += 1
  ```

* `SafeIntervalIndex` — **KEPT**. It is still the pure-Python reference's occupancy (`_splan_reference`
  builds `_SafeIntervals` over it) and PR 2a's oracle. What goes away is its **ledger subscription**:
  it binds lazily on the first reference dispatch and absorbs from the ledger then, exactly as
  `enable_blocked` does for the `blocked` map. On a run with zero fallbacks it is never built.

**`freespace_sim/planner/compiled_occupancy.py`** — **FILE DELETED**. `CompiledOccupancy`,
`reset_cell`, `on_release`, `_record`, `_claims`, `_rows`, `_static_cells`, `_free` all go with it.

**`freespace_sim/planner/sipp_kernel.py`** — **far smaller than it looks**, and worth checking before
budgeting for it. The kernel's cell encoding is already fully parameterised by
`qmin, rmin, rspan, qspan, nlevels` (`cell = ((q - qmin) * rspan + (r - rmin)) * nlevels + L`), so
"window-local" is *passing the window's bounds instead of the global box's*. Three consequences:

* `_search` — **MODIFIED**, but only in its parameter list: `ov_lo/ov_hi/ov_nxt/ov_head/ov_gen/cap`
  are dropped (the overlay folds into the build, so the `sj >= cap` branch is dead) and `iv_*` become
  the window pool. The traversal, dominance, heap and cost code is untouched.
* **`FB_WINDOW` is not needed.** `_search` already returns `FB_OOB` on the *first* touch of a cell
  outside `[qmin, qmin+qspan) x [rmin, rmin+rspan)`, before reading any chain — exactly the
  "never a partial result" discipline §4.2 asks for, already implemented and already tested. A window
  miss reuses it verbatim; only the host's response changes, from "fall to the reference" to "widen,
  then fall to the reference at the ceiling".
* `_note_cell` — **UNCHANGED**. It is called with world `(q, r)` at every site (`_note_cell(read_bbox,
  nq, nr, Lc)`; the takeoff site reconstructs `iq0 + qmin` first), and the path output does the same
  (`out_r[m] = ccqr - ci * rspan + rmin`). The DROP envelope contract from `#125` therefore survives
  the change with no code touched — which is the one thing here worth verifying by test rather than
  by reading, and `tests/test_parallel_envelope.py` already does.

**Tests**

* `tests/test_lns_sipp.py` — **MODIFIED**. `test_sipp_subscriber_counts` drops from three release
  subscribers to two; the `CompiledOccupancy` refcount/journal/static-wall/slot-reclaim tests are
  **DELETED** with the structure they gate (their intent survives in `tests/test_sipp_window.py`).
* `tests/test_parallel_envelope.py` — **UNCHANGED and load-bearing**. The world-anchored soundness
  gate and the cell-exact `cell_bbox` test are what prove `_note_cell`'s coordinate change is right.
* `tests/test_sipp_window.py` — **MODIFIED**. Add `test_window_miss_widens_and_stays_exact`: force a
  deliberately undersized window, assert the plan widens and lands byte-identical to the unwidened
  reference.

*Expected:* SIPP's ledger structures become `_svc` (3.755) + shared `_cocc` (1.753) = **5.508 ms/flight**,
down from 18.840 — and the release term stops growing with congestion. New per-plan cost: one window
build, bounded by A\*'s measured 0.514 ms paint over the same box. Against 2 releases + 2 commits per
plan at LNS's ~4% accept rate, that trades ~19 ms/plan of ledger work for ~1 ms/plan of build.

Kernel memory falls with it: the frontier arrays are sized to `cocc.cap` = 654,266 slots today and to
a window (order 10-40k slots) after, which is the term `#125` flagged as deciding whether DROP m=8 fits.

## 4. Critical review of this plan

### 4.1 Own-column ownership: A\*'s model is *less* exact than SIPP's, and this plan adopts it

`SafeIntervalIndex.cell_blocked` resolves column ownership **per (cell, step)** — a cell may be under
an own column at step 10 and a foreign one at step 50, and SIPP gets both right. A\*'s `ov_own_gen` is
one boolean **per cell**, so it cannot express that; `_stamp_own_overlay` detects the own-∩-foreign
overlap via `col_owners` and falls back to the pure-Python reference for exactness (issue #3).

Adopting the arena means adopting that collapse. **This is a real behaviour change and must not be
buried.** Three things make it the right call anyway:

1. It makes SIPP *equally* exact to A\*, which is the shipped production default — not less exact
   than the thing it is being compared against.
2. The fallback is a fallback to the exact reference, not an approximation. No wrong answer is
   possible; the cost is wall-clock on the affected plans.
3. `demand.py` reject-samples hub spacing (#27, and #24 on the wider `exit_radius` extent), so the
   overlap is rare by construction.

**Required, not optional:** PR 2b must instrument the overlap rate on `density_faa` and report it. If
it is above ~1% of plans the trade is off, and the fix is to widen the arena's column claims to carry
the owning `tid` per claim (a parallel `int32` array beside the arena, or 8 bits stolen from the
20-bit `fid_code` field) so ownership stays per-(cell, step). Cheaper than it sounds, but it is
`#124`'s data structure being changed for SIPP's benefit, so do it only if measured necessary.

### 4.2 A window miss is not one wrong bit for SIPP

For A\* a probe outside the window is one `blocked` query. For SIPP the kernel **walks a cell's whole
interval chain**, so a cell outside the window has no chain at all and the search would silently see
it as free — a conflicting plan, not a slow one. The widen path is therefore load-bearing in a way
A\*'s is not.

*Resolution, and the first part is already built:*

* The kernel returns `FB_OOB` on the **first** out-of-box cell touch, before any chain is read
  (`sipp_kernel._search` line 252). Pointing it at a window box rather than the global box makes that
  the window-miss signal for free — no new code, no partial result. This is the single fact that
  makes PR 2b tractable.
* The widen ceiling falls through to `_splan_reference`, which is unbounded. Correctness never depends
  on the window being big enough.
* `tests/test_sipp_window.py::test_window_miss_widens_and_stays_exact` deliberately undersizes the
  window and asserts byte-identical output. **A test that merely checks "no exception" would pass
  against a build that plans through phantom-free cells** — this is precisely the failure mode that
  made two envelope tests in `#125` non-gates.

Margin: A\* uses 24 hexes for 100% zero-miss plan coverage. SIPP's measured read set is *tighter*
(dirty rate 61.5% against A\*'s 78.7% at 4 workers, because the interval collapse probes fewer cells),
so 24 should cover it — but "should" is not a measurement. PR 2b runs the same plan-level coverage
sweep A\* was calibrated with and reports the zero-miss share before picking the constant.

### 4.3 Step span: A\*'s heuristic tail would be a silent wrong answer here

A\*'s `window_bounds` clips steps to `base + n_gsteps + tail_steps`. Copying that would truncate SIPP
interval `hi` values, and `hi` is what answers "may I wait here that long" — the plan would come back
*feasible but worse*, with no fallback triggered and no test failing. The plan therefore fixes the
span at exactly `[base, max_step]`. Because intervals store endpoints rather than a dense time axis,
this costs essentially nothing in memory (unlike A\*'s bitmap, where the step span *is* the row size)
— which is why the two structures can honestly make opposite choices here.

### 4.4 Version stamps under window-local cell ids

`front_gen` / `goal_gen` give an O(1) per-plan reset by stamping `gen`. Window-local cell ids **change
meaning between plans**, so a stale `stamp == gen` would be a wrong answer rather than a stale one.
This is safe only because `gen` is bumped per attempt — and the widen re-run must bump it too, which
is why the pseudocode in PR 2b puts `self._sgen += 1` **inside** the loop. `AStarPlanner._plan_compiled`
already established this discipline for its FB_MASK re-run ("fresh `gen`, no partial output is ever
consumed"); the risk is copying the loop without copying that line.

### 4.5 The plan side could get slower and the ledger win could be eaten

The window build is new per-plan work that does not exist today, and SIPP's plan side is currently
*ahead* of A\*'s (148.1 s against 155.7 s). If the build costs more than A\*'s 0.514 ms paint —
plausible, since it emits chains rather than bits — the net could be smaller than section 3 projects.

*Resolution:* PR 2a's oracle test doubles as the cost gate. Time `build_window_intervals` over the
sampled boxes and require it under 1.5 ms p90 before PR 2b flips anything. If it is over, the fallback
design is the reference's own: build **lazily per cell on first probe** and memoise, which is exactly
what `_SafeIntervals` does in `_splan_reference`. That keeps the arena and kills the global structures
either way — only the build strategy changes.

### 4.6 A cheaper alternative I considered and rejected

**Lazy release: mark released cells dirty, rebuild on first read.** Attractive because it keeps the
pool and touches ~50 lines. Rejected on the workload's own shape: LNS destroys N flights and then
re-commits repaired versions whose routes are *near the originals*, so `block_range` touches almost
exactly the cells the release dirtied, immediately. The rebuild is deferred by microseconds and the
O(congestion) term survives intact.

**Arena-backed re-apply: keep reset-and-re-apply, but read survivors from a flat arena in numba.**
This is a genuinely smaller PR and would cut the constant, plausibly 5-10x (6.105 -> ~0.8 ms/flight at
2,400 warm). It is the right move *only if* PR 2 proves too invasive: it leaves the multiplier in
place, so the gap re-opens as scenarios grow, and it keeps both the global index and the static-wall
trap. Recorded here as the fallback, not the plan.

### 4.7 What this plan does not fix — and the projection is parity, not a win

Ledger parity is the ceiling of this work, and §2's corrected numbers put that ceiling *below* A\*:

```
sipp today      276.8 = plan 146.3 + ledger 124.1 + other 6.4
sipp at parity  176.2 = plan 146.3 + ledger  23.5 + other 6.4
              + ~2.4   the new per-plan window build (~1 ms x 2,400 plans)
              = ~178.6  against A*'s 165.8   ->  still ~1.08x SLOWER
```

So the honest projection is **SIPP lands within ~8% of A\*, not ahead of it.** The remaining gap is
plan-side (146.3 against 135.7) and this plan does not touch it.

That is still worth doing, for three reasons that do not depend on winning the wall-clock race:

1. It removes a term that **grows with scenario size**. At parity the two planners scale together; today
   the gap widens with every flight added, which makes every SIPP measurement scenario-specific.
2. It deletes two structures, ~570 lines, the static-wall trap that bit twice during `#125`, and the
   ~334 MB of kernel arrays sized to `cocc.cap` — the term `#125` flagged as deciding whether DROP m=8
   fits in memory.
3. SIPP's quality edge is real if small and consistent in sign (+0.08 pp here, and it accepted 88
   against 86). At 1.67x the wall that is not a trade anyone would take; at 1.08x it becomes a live
   question, and at parity-plus-a-plan-side-fix it becomes the default.

**Anyone hoping this restores a 1.1-1.2x SIPP win should not read that here.** If the goal is to beat
A\*, this plan is a prerequisite, not the answer — the answer needs a plan-side lever as well, and
`context/sipp_lns_plan.md` has no candidate for one.

## 5. Branch and review mechanics

| branch | contents | reviewable in isolation? |
| --- | --- | --- |
| `victor-qin/sipp-runtime-profiling` | this document, `analysis/prof_sipp_ledger.py`, `analysis/probe_sipp_read_window.py` | yes — measurement only, no planner change |
| `victor-qin/sipp-blocked-map` (off main) | PR 1 | yes — ~20 lines, 2 tests |
| `victor-qin/sipp-window-intervals` (off PR 1) | PR 2a | yes — new file + oracle tests, nothing wired |
| `victor-qin/sipp-arena-occupancy` (off PR 2a) | PR 2b | the deletions land here; parity comes from 2a |

PR 1 does not depend on PR 2 and should go first regardless of whether PR 2 is approved — it is the
only part that is pure removal of work nothing consumes.

### Gates every PR in the series must clear

1. **Full suite green** (`uv run pytest -q`; 1,307 passed / 2 skipped on `4c81255`).
2. **Exactness**: SIPP compiled output byte-identical to the SIPP reference on a congested cut, and
   zero kernel fallbacks on `density_faa` (`_sfb == 0`). A speedup with a nonzero fallback rate is
   measuring A\*, not SIPP.
3. **Paired A/B** via `analysis/ab_lns_repair_planner.py` at full `density_faa`, arms strictly
   sequential, baselines asserted identical. Report the `plan_s` / `ledger_s` split, not just the loop.
4. **Per-structure re-run** of `analysis/prof_sipp_ledger.py` at 2,400 warm, so the ledger claim is
   attributable rather than inferred from the loop.
5. **DROP re-check** at m=8 / N=2, since `#125`'s parallel numbers were taken against the structures
   PR 2 deletes and the envelope contract runs through `_note_cell`.

### What would falsify the plan

* PR 2a's build measures over ~1.5 ms p90 per window → the projected net win shrinks; take the lazy
  per-cell build in §4.5 or the arena-backed re-apply in §4.6 instead.
* The own-∩-foreign overlap rate (§4.1) exceeds ~1% of plans → the per-cell ownership collapse is too
  lossy; widen the arena's column claims to carry `tid` before proceeding.
* Zero-miss window coverage at margin 24 comes in materially under A\*'s 100% → SIPP's read set is
  wider than measured and the widen path becomes hot rather than rare; re-calibrate the margin on the
  coverage sweep, not on the miss rate.

---

## 6. Phase 1 result (`f4be4f0`)

Shipped: `needs_blocked_map` deleted, `svc.enable_blocked(ledger)` added to `_splan_reference`.

Paired against §2's baseline, same harness, same seed, full `density_faa` (4,636 legs, N=8, 300 iters):

| | `4c81255` | Phase 1 | delta |
| --- | ---: | ---: | ---: |
| loop s | 276.8 | **265.0** | −11.8 (−4.3%) |
| plan s | 146.3 | 144.8 | −1.5 |
| **ledger s** | **124.1** | **114.1** | **−10.0 (−8.1%)** |
| cost after | 1342247 | 1342247 | **identical** |
| accepted | 88/300 | 88/300 | identical |
| release subs | 4 | 4 | (Phase 3 drops this) |

§1 predicted −9.6% on the ledger side (1.807 of 18.84 ms/flight); measured −8.1%. The schedule is
**bit-identical** — same cost, same accept count, same improvement, zero kernel fallbacks, verified.
That is the property the change was designed for: it removes work, not information.

Against A\*'s 165.8 s loop, SIPP moves **1.67x → 1.60x slower**. The remaining 90.6 s of ledger gap is
the interval pool (§1.1) and the inverse index (§1.3), i.e. Phases 2-3.

Two things worth keeping from the implementation:

* **The obvious test is not a gate.** Asserting `needs_blocked_map` is gone, or that `_blocked_live` is
  False, passes against a build that plans wrong. The gate is comparing a lazily-armed planner's
  `blocked` map against one maintained eagerly from the first commit — `enable_blocked` REPLAYS the
  ledger, and replaying it wrong (double-counted refcounts, missed own-column skip) is the plausible
  bug. Both new tests were mutation-verified: restoring the flag fails one, deleting the
  `enable_blocked` call fails the other.
* **One existing test was silently at risk.**
  `test_compiled_terminal_path_never_routes_through_blocked` probes `sipp._svc.is_blocked` directly as
  an occupancy oracle. Post-change it would either raise, or pass because a stray fallback inside its
  1,200-event warm loop had armed the map stickily — a coin flip between the two. Pinned with an
  explicit `enable_blocked`.

---

## 7. Phase 3 result (`459a0c1`) — and the §4.7 projection was wrong

Shipped: `compiled_occupancy.py` deleted (407 lines), SIPP's compiled path builds safe intervals per
plan from A\*'s claim arena, `SafeIntervalIndex` unsubscribed, the own-lane overlay replaced by A\*'s
`ov_own_gen` stamp.

Paired A/B, same harness and seed, idle machine, full `density_faa` (4,636 legs, N=8, 300 iters):

| arm | loop s | plan s | **ledger s** | other s | improvement | accepted | release subs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| astar | 156.3 | 127.1 | **22.7** | 6.4 | 0.46% | 86/300 | 3 |
| sipp | **126.3** | **97.8** | **22.3** | 6.2 | 0.54% | 88/300 | 3 |

**SIPP is 1.24x FASTER than A\***, from 1.67x slower at `4c81255`. The schedule is **bit-identical**
to the pre-Phase-3 SIPP arm — same cost 1342247, same 0.54%, same 88 accepted, `_sfb == 0`, verified —
so this is a pure speedup, not a different answer.

| | `4c81255` | Phase 1 | Phase 3 |
| --- | ---: | ---: | ---: |
| loop s | 276.8 | 265.0 | **126.3** |
| plan s | 146.3 | 144.8 | **97.8** |
| ledger s | 124.1 | 114.1 | **22.3** |
| vs A\* | 1.67x slower | 1.60x slower | **1.24x faster** |

**Ledger parity landed exactly as designed** (22.3 against A\*'s 22.7, both maintaining the same two
structures), and the O(congestion) release term is not reduced but *gone* — there is no global
free-interval pool left to un-build.

### §4.7 was wrong, and in an instructive direction

§4.7 projected ~178.6 s, i.e. **still ~1.08x slower**, on the reasoning that "ledger parity is the
CEILING of this work" and "the remaining gap is plan-side and this plan does not touch it." The plan
side went **146.3 -> 97.8 s**, from 7.8% slower than A\* to 1.30x faster. The error was modelling
Phase 3 as a pure ledger change when the same rewrite also changes what the *kernel* reads.

**What I have not established is which term did it** — and two of the three candidates are now
measured and DEAD.

1. ~~**Working set.**~~ **FALSIFIED.** The per-slot arrays were still 5.7x over-allocated after
   Phase 3 (553,152 slots / 22.1 MB against a measured max usage of 97,228), because the frontier
   and goal arrays were sized from the interval build's conservative capacity ESTIMATE and then
   doubled. Sizing them from the build's actual tail took them to 3.9 MB — and moved kernel time
   **15.28 -> 15.16 ms/plan, i.e. nothing.** In hindsight that is what the design predicts: those
   arrays are version-stamped, so the kernel touches only the slots it visits and the allocation
   size never determined which cache lines were read. Kept anyway, as an 18.2 MB/planner memory win
   that matters under DROP (`299901c`'s sibling commit).
2. **Chain length.** UNSUPPORTED, not conclusive. Sampled mean intervals per cell: old global pool
   over `[0, MAXS]` **1.824**, new window pool over `[base, max_step]` **2.262** — the new chains are
   if anything LONGER. The populations differ (the old sample spans the whole box including empty
   regions, the new spans one busy window), so this is suggestive rather than decisive, but it does
   not support the hypothesis.
3. **Overlay indirection.** Still untested, and now the leading candidate by elimination. Three hot
   sites lost `ov_head[cell] if ov_gen[cell] == gen else cell` — two random reads and a branch per
   chain walk — plus an `sj >= cap` test per interval *inside* the walk. With a mean chain of ~2
   intervals, that per-walk preamble is comparable in size to the walk itself, which is the shape
   that would explain a large kernel-wide effect.

So: the *result* is solid and the *explanation* is still open, but narrower. Anyone citing a
mechanism should test (3) by re-adding the indirection behind a flag.

### Gates

* **Exactness:** cost bit-identical to the pre-Phase-3 arm at full scale; `test_sipp.py`,
  `test_compiled_replay_exact_metro`, `::_dallas_terminal` and `test_sipp_incremental_release_matches_rebuild`
  all green.
* **Fallbacks:** `_sfb`, `_sfb_oob`, `_sfb_overlap` all **0** on 1,526 compiled plans; zero window
  widens, zero buffer grows, zero reference dispatches, `_sidx` never bound. Margin 24 gives SIPP the
  same 100% zero-miss coverage A\* was calibrated to — so §4.2's worry about SIPP needing a wider
  window than A\* was unfounded.
* **Structure count:** 3 release subscribers for A\*, 3 for point-to-point SIPP, 3 for terminal SIPP.
  Was 3/3/4.
* **Suites:** 1,321 passed / 2 skipped, 71 under `-m slow`. Exactly 9 below the pre-Phase-3 1,330,
  matching the 9 deleted pool tests.

### Still outstanding

* **DROP re-check at m=8 / N=2.** `#125`'s parallel numbers were taken against structures this
  deletes, and the envelope contract runs through `_note_cell`.
* `build_window_intervals` is in neither `_warm_jit` nor `_swarm_jit` — a ~0.92 s cold compile that a
  spawned DROP worker would pay on its first repair. It matters for the DROP run above, not for the
  sequential one measured here.

### 7.1 DROP re-check (m=8, N=2, 600 s)

`#125`'s parallel numbers were taken against the structures Phase 3 deletes, and the DROP staleness
test runs through `_note_cell`, so this had to be re-run rather than assumed. Full `density_faa`,
`--search-workers 8 --parallel-mode drop --neighborhood 2 --time-limit 600`, arms sequential:

| | astar | sipp |
| --- | ---: | ---: |
| cost | 1349506 → 1293940 | 1349506 → **1291433** |
| improvement | 4.12% | **4.30%** |
| **tasks** | 27,791 | **31,983 (1.15x)** |
| accepted | 2,572 | 3,002 |
| accept rate | 9.3% | 9.4% |
| clean-merge | 830 | **1,056** |
| dirty rate | 40.8% | **39.3%** |
| verified | True | True |

**15% more tasks in the same wall clock, both conflict-free.** The read envelope survived the
window-local rewrite — the specific risk in re-running this — and in fact got *tighter*: dirty rate
below A\*'s and clean-merges up 830 → 1,056, consistent with `#125`'s measurement that SIPP's
interval collapse probes fewer cells (61.5% against 78.7% at 4 workers).

**The quality delta is ONE SAMPLE per arm and should not be quoted as a win.** +0.18 pp is exactly
the seed-to-seed spread the sequential A/B recorded, and the adaptive operator weights diverged
sharply between the two runs (`agent` 6.195 for astar against 0.296 for sipp), so they explored
materially different trajectories. The **throughput** figure is the robust one: it is mechanical.

This run also depends on `299901c`, which warms `build_window_intervals` in `_swarm_jit`. Without it
every spawned worker meets an uncompiled builder on its first repair — ~0.92 s cold, simultaneously,
billed to SIPP. That is the compile stampede `_swarm_jit` exists to prevent, and the measurement
above would have carried it.
