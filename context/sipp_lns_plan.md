# SIPP as an LNS repair planner — design record

Branch `victor-qin/lns-sipp-planner`, forked off #118 (`victor-qin/drop-lns-parallel-plan`).

## 0. The question, and the one number that answers it

LNS repair is `plan()` against a **nearly-full ledger**, which is exactly where SIPP wins: 1.86x A*
at lambda=4000, 2.82x at lambda=8000 (PR #113), because the interval collapse pays more the more
committed traffic there is to collapse. Repair is also the only thing in the loop that scales with
neighborhood size, so a faster repair is a directly larger iteration budget.

But LNS is also the most **commit-heavy** workload in this repo, and that is where SIPP has always
been weak. One iteration is:

```
release_many(victims)          # 1 release of k flights
k x plan()                     # the part SIPP is good at
k x commit()                   # the part SIPP is bad at
--- and on the 79% of iterations that reject: ---
release_many(victims)          # again
k x commit()                   # the old volumes, back
```

So a rejected iteration pays the commit side **twice** and the plan side once. SIPP keeps **four**
ledger-subscribed structures where A* keeps two (`_svc` + `_tcap`, inherited, plus its own `_sidx`
+ `_scocc`). That doubling is what `[[sipp-commit-side-overhead]]` measured as 1.36x slower
end-to-end before #117, and what #117 fixed on the *commit* side (190.9 s -> 66.1 s).

**#117 did not touch the release side, and LNS is the only caller that has one.** That is the whole
of this work.

The decisive quantity is therefore not "is SIPP faster than A*" — it is measured and yes, in
congestion. It is:

> per LNS iteration, does SIPP's plan-side saving exceed its extra release-and-recommit cost
> for two more structures?

Nothing in the repo answers that today, because SIPP cannot be used as a repair planner at all.
Sections 3-6 make it usable; section 8 measures it. **If the answer is no, the correct outcome of
this branch is a measured "no" and a default left at `astar`** — the same shape as #118's own
verdict on `search_workers`.

---

## 1. What LNS actually demands of a repair planner

Derived from the code, not assumed. Seven requirements; SIPP satisfies two.

| # | Requirement | Enforced at | SIPP today |
|---|---|---|---|
| R1 | `plan(req, ledger, cfg) -> OperationalIntent` | `state.py:527` (the only method `try_repair` calls) | OK |
| R2 | `evict_floor == 0.0`, so victims may be replanned in ANY priority order | `state.py:192` (raises) | attribute inherited; **its own two structures ignore it** (G3) |
| R3 | Not already bound to this ledger | `state.py:181` | OK (checks `_svc_ledger`/`_cocc_ledger`, both inherited) |
| R4 | Survive `detach_subscribers()` + epoch bump by REBINDING | `state.py:202`; SIPP's tripwires at `sipp.py:278`, `sipp.py:513` | OK — and pinned by `test_shared_sipp_occupancy_preserves_nonzero_ledger_epoch` |
| R5 | Un-absorb `release_many` in O(victim volumes) under `incremental_release=True` | `astar/planner.py:305-307`, `:797-798` | **absent for `_sidx` + `_scocc`** (G1) |
| R6 | Costs in the same currency as the unimpeded ruler and the incumbent | `state.py:171` (`_REPRODUCIBLE_PLANNERS`) | untested in congestion (G4) |
| R7 | For DROP with m>1: reset `last_envelope` per plan, set it when `record_envelope` | `parallel.py:715-727`, `:794` | **absent, and worse than absent** (G2) |

---

## 2. The gap — four findings, ranked

### G1 (blocker, performance). Two of SIPP's four ledger structures have no release hook.

`AStarPlanner._occupancy` subscribes `svc.on_release` and `_tcap.on_release` when
`incremental_release` is set (`astar/planner.py:305-307`), and `_compiled_occupancy` does the same
for `cocc` (`:797-798`). SIPP **inherits both**, so `_svc` and `_tcap` are already correct.

SIPP's own two — `_sipp_index` (`sipp.py:273`) and `_scompiled_occ` (`sipp.py:510`) — call only
`ledger.subscribe(...)` and `ledger.subscribe_static(...)`. There is no `on_release` on
`SafeIntervalIndex` or `CompiledOccupancy` at all, and `SIPPPlanner.__init__` never reads
`incremental_release` for them.

The consequence is not a wrong answer, it is a rebuild. After `release_many`, `ledger.n_volumes`
falls while their `n_added` does not, so the shrink tripwire fires on the next `plan()`:

```python
elif ledger.n_volumes < sidx.n_added:
    sidx.reset(); _absorb(sidx, ledger)      # sipp.py:288  — O(every committed volume)
elif ledger.n_volumes < cocc.n_added:
    cocc.reset(); _absorb(cocc, ledger)      # sipp.py:522  — again
```

`_absorb` walks `ledger.iter_committed()` in full. PR #109 measured the A* equivalent of ONE such
rebuild at 3.74 s — 94% of an iteration — which is exactly why `incremental_release` was built.
SIPP would pay TWO of them per iteration, and (because `_rewind` releases again) potentially four.
At 426,756 volumes x 2000 iterations this is not a slowdown, it is a non-starter.

### G2 (blocker for DROP m>1, and a silent correctness bug). No read-set envelope.

`SIPPPlanner.plan` (`sipp.py:294`) resets `last_expansions`, `_n_expansions` and `_air`. It does
**not** reset `last_envelope`, and `sipp_kernel.py` has no `read_bbox` accumulator (compare
`astar/kernel.py:149-179`, `:232-233`).

Two consequences, and the second is the serious one:

1. With `record_envelope=True`, every native SIPP repair reports `None`. `_read_set_is_clean`
   treats `None` as always-dirty (`parallel.py:718-722`), so DROP discards every stale result and
   degrades to SYNC. Correct, just pointless.
2. `SIPPPlanner._fallback` (`sipp.py:632`) calls `AStarPlanner.plan`, which **does** reset and set
   `last_envelope` (`astar/planner.py:392`, `:1153`). So: flight A falls back to A* and leaves an
   envelope; flight B plans natively in SIPP and does not clear it; `try_repair` appends
   `self.repair_planner.last_envelope` for B (`state.py:534`) and files **A's read set under B's
   name**. The coordinator then tests the wrong region, finds it clean, and merges a repair that
   is genuinely stale. Silent, and `verify.find_interflight_conflict` would not catch it — a stale
   merge produces a *worse cost*, not a conflict.

The one-line fix for (2) is unconditional and should land regardless of whether (1) is ever built.

### G3 (latent, currently answer-neutral). `evict_floor` is ignored.

A* computes `wm = req.t_request if self.evict_floor is None else min(req.t_request,
self.evict_floor)` (`astar/planner.py:332`, `:807`). SIPP's two call
`evict_before(int(req.t_request // cfg.dt_s))` unconditionally (`sipp.py:291`, `sipp.py:525`).

Today this changes nothing: `evicted_before` is **written but never read** in `sipp.py`,
`compiled_occupancy.py` or `sipp_kernel.py` — both docstrings say "storage reclaim is TODO". So
R2's guarantee holds by accident.

It stops being an accident the moment either (a) someone implements the reclaim, or (b) this plan
lands a journal-based `on_release` — because A*'s release path *clips spans by* `evicted_before`
(`compiled_hex_occupancy.py:364-365`) to avoid resurrecting an evicted step. Wiring `evict_floor`
through is three lines now and a silent exactness bug later.

### G4 (currency). The ruler is A*, the repair would be SIPP, and no congested test compares them.

- `_REPRODUCIBLE_PLANNERS = {"astar", "astar_ref"}` (`state.py:51`): `cfg.planner="sipp"` raises
  unless a `repair_planner=` is passed.
- `unimpeded._new_ruler` (`unimpeded.py:57-69`) hardcodes `AStarPlanner`.
- The only SIPP-vs-A* cost comparison is on an **empty world** at 1e-6
  (`test_sipp_compiled.py:61`). Every replay test (`:357`, `:381`) compares SIPP-compiled against
  SIPP-*reference* on an A*-committed ledger — never SIPP against A*.

There is a real argument that the A* ruler stays valid: both planners are exact optimizers of the
same weighted cost, and the ruler world is walls-only, so they must agree on the optimum. But
`accept_epsilon=0.0` means `try_repair` accepts on any strict improvement, so a systematic
1e-6-scale disagreement between the incumbent's currency and the repair's is worth pinning rather
than assuming. Note the incumbent and the repair are both SIPP once this lands, so the accept test
is self-consistent; only `delay()` (used for `premium` ordering and the agent-based destroy seed)
mixes rulers.

---

## 3. Phase 0 — merge `origin/main` (#117). Prerequisite, not housekeeping.

#118 forked at 9656faa (#115); main is 153ec4f (#117). A trial merge is **clean** (7 files, no
conflicts): `astar/compiled_hex_occupancy.py`, `astar/occupancy.py`, `compiled_occupancy.py`,
`hexgrid.py`, `sipp.py`, `tests/test_sipp_compiled.py`, `tests/test_sipp_kernel_3d.py`.

Merging first is a **design** prerequisite, not just a performance one:

- #117 gives `CompiledOccupancy.block_range(c, s0, s1)`, the SoA twin of `_Pool.block_range`. The
  removal journal Phase 1 ports packs a claim as `s0 << 20 | s1` — one int per *span*. Pre-#117
  `CompiledOccupancy` blocks one STEP at a time, so the same journal would carry ~8x the rows
  (#117 measured span median 7, max 17). Porting onto the old shape means porting it twice.
- #117 points both SIPP structures at `hg.rasterize_ranges`, whose memo already has a 1024-entry
  floor put there **for LNS specifically** ("the claim index reads victim rows rasterized across
  SEVERAL earlier commits"). The release path wants the same rows.

**Change:** `git merge origin/main` (no code edits expected). Then re-run `tests/test_sipp*.py`,
`tests/test_lns*.py` to confirm the union is green before anything is built on it.

---

## 4. Phase 1 — incremental release for SIPP's two structures. THE blocker.

**Point of the phase:** make `release_many` cost O(victim volumes) for SIPP, so an LNS iteration
stops paying two full-ledger rebuilds.

**The thesis, in one line:** *each of SIPP's two structures gets the removal treatment of its A\*
structural twin* — because they are structurally identical to those twins, and the two twins
already chose differently for good reasons.

| SIPP structure | shape | A* twin | removal technique |
|---|---|---|---|
| `SafeIntervalIndex` | `dict[cell, set[step]]` + `dict[cell, dict[step, set[tid]]]` | `HexOccupancyService` | **refcount** (`_bump`/`_drop`) |
| `CompiledOccupancy` | flat linked-list interval pool | `CompiledHexOccupancy._Pool` | **claim journal + per-cell rebuild** |

The split is not arbitrary. A step-keyed dict can be decremented in O(the victim's own rows). An
interval pool cannot be un-split in place — two flights blocking the same step produce one split,
and the second `block_range` is a no-op — so the only exact reversal is reset-the-cell and
re-apply the survivors, which is why A* keeps a per-cell claim multiset.

### 4.1 `SafeIntervalIndex` — refcounts (`freespace_sim/planner/sipp.py`)

**MODIFY `SafeIntervalIndex.__init__(self, cfg, track_removal=False)`**

```python
self.track_removal = track_removal
self._rows: dict[int, array] = {}     # fid -> flat int64 rows, mirroring HexOccupancyService
self._cells: list = []                # cell_id -> (q, r, L)   (interned once, as in the A* twin)
self._cell_ids: dict = {}
self._tids: list = []; self._tid_ids: dict = {}
# corr/cols unchanged in TYPE when track_removal is off; counter-dicts when on (see below)
```

**MODIFY `SafeIntervalIndex._add(self, vol, own_cols, _rows=None)`** — same body, but the two
insert sites become refcounted under the flag:

```python
for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(...):
    if not in_blk: continue
    cell = (q, r, L)
    if is_column:
        if self.track_removal:
            cid = self._intern(cell); code = self._tid_code(tid)
            _rows += (cid, s_lo, s_hi, code)
            for s in range(s_lo, s_hi + 1):
                d = self.cols.setdefault(cell, {}).setdefault(s, {}); d[tid] = d.get(tid, 0) + 1
        else:
            <unchanged set-based insert>
    elif not (own_cols and self._inside_a_column(q, r, own_cols)):
        if self.track_removal:
            cid = self._intern(cell)
            _rows += (cid, s_lo, s_hi, -1)
            d = self.corr.setdefault(cell, {})
            for s in range(s_lo, s_hi + 1): d[s] = d.get(s, 0) + 1
        else:
            <unchanged set-based insert>
```

**ADD `SafeIntervalIndex.on_release(self, flight_id, volumes)`**

```python
rows = self._rows.pop(flight_id)
for (cid, s_lo, s_hi, code) in rows:            # flat 4-slot rows
    if self.evicted_before is not None and s_lo < self.evicted_before:
        s_lo = self.evicted_before              # never resurrect an evicted step (A* twin, line 364)
    cell = self._cells[cid]
    for s in range(s_lo, s_hi + 1):
        <decrement corr[cell][s] or cols[cell][s][tid]; delete the key at zero;
         delete the empty inner dict so `if not corr` stays exact>
self.n_added -= len(volumes)
```

**ADD `_intern` / `_tid_code`** — verbatim in spirit from `occupancy.py:211-219`: intern each
`(q, r, L)` tuple once so a release reads back the tuple it inserted with, instead of re-packing
three ints per row.

**Why the type switch is reader-transparent — verified, not assumed.** Every reader of `corr` and
`cols` uses membership, iteration, or truthiness, all of which behave identically on `set` and
`dict`:

- `s in self.corr.get((q, r, L), ())` — `sipp.py:153, 154` — dict keys support `in`
- `any(t not in own for t in hubs)` — `sipp.py:149` — iterating a dict yields keys
- `cand.update(s for s in corr ...)` / `for s in cols` — `sipp.py:167, 169`
- `if not corr and not cols` — `sipp.py:163` — an empty dict is falsy

This is the same claim `HexOccupancyService` already makes in its own docstring ("Membership
queries are unchanged (`in` works on dict keys); flag off => the original set-based structures,
byte-for-byte"), which is the precedent for gating on the flag rather than converting outright.

**`static_cols` is untouched.** It is a separate `(q,r) -> {tid}` dict, read independently
(`sipp.py:146, 158`), and `reset()` deliberately preserves it — walls are infrastructure, not
commit-derived. No release ever touches it.

### 4.2 `CompiledOccupancy` — claim journal + per-cell rebuild (`compiled_occupancy.py`)

**MODIFY `_init_pool`** — add `self._free: list[int] = []`.

**MODIFY `_alloc`** — pop from `_free` before bumping `nslots`.

**ADD `reset_cell(self, c)`** — port of `_Pool.reset_cell` (`compiled_hex_occupancy.py:208`):

```python
slot = int(self.iv_nxt[c])
while slot != -1:                       # reclaim the overflow tail; head slot `c` is re-seeded
    self._free.append(slot); slot = int(self.iv_nxt[slot])
self.iv_lo[c] = 0; self.iv_hi[c] = self.MAXS; self.iv_nxt[c] = -1
```

The free list is **mandatory, not an optimization**, and A*'s own docstring is the argument:
`_alloc` is a pure bump allocator, so under LNS — which resets and re-applies the same hot cells
every iteration — `nslots` grows without bound and drags `cap` through repeated doubling for a
working set that never grows. Slot indices are pure storage (every reader walks the chain), so
which slot holds an interval never affects an answer.

**MODIFY `__init__(self, cfg, margin=48, track_removal=False)`** — add `_claims: dict[int,
list[int]]`, `_rows: dict[int, array]`, `_static_cells: set[int]`; raise if `track_removal and
MAXS >= 1 << 20` (the packing limit).

**MODIFY `_add(self, vol, own_cols, _rows=None)`** — record after the existing `block_range`:

```python
self.block_range(c, int(s_lo), int(s_hi))
if self.track_removal and c not in self._static_cells:
    self._record(c, int(s_lo), int(s_hi), _rows)   # key = c (ONE pool, unlike A*'s two)
```

**ADD `on_release(self, flight_id, volumes)`**

```python
rows = self._rows.pop(flight_id)
touched = set()
for i in range(0, len(rows), 2):        # flat (key, claim) pairs
    self._claims[rows[i]].remove(rows[i + 1])    # ValueError here IS the drift signal
    touched.add(rows[i])
for c in touched:
    self.reset_cell(c)                  # reclaims overflow slots onto _free
    if c in self._static_cells:         # <-- see "the one subtle thing", below
        self.iv_lo[c] = 0; self.iv_hi[c] = -1; self.iv_nxt[c] = -1
        continue
    survivors = self._claims.get(c)
    if survivors:
        for packed in survivors:
            self.block_range(c, packed >> 20, packed & 0xFFFFF)
    else:
        self._claims.pop(c, None)
self.n_added -= len(volumes)
```

**MODIFY `_wall_static_terminal`** — record each walled cell id into `self._static_cells` as it
writes the empty interval.

#### The one subtle thing: `CompiledOccupancy` is the only structure that stores always-active walls in the same array as commit-derived blocks

Everywhere else in the repo, statics live apart: A*'s `CompiledHexOccupancy.static_col` is a
separate bool array, `HexOccupancyService.static_term_cells` a separate dict,
`SafeIntervalIndex.static_cols` a separate dict. `CompiledOccupancy._wall_static_terminal` instead
writes `iv_lo=0; iv_hi=-1; iv_nxt=-1` **directly into the pool**.

A naive `reset_cell` therefore re-seeds a statically walled cell to `[0, MAXS]` — **fully free** —
and re-applying survivors only re-blocks their spans. The wall silently disappears for the rest of
the run, and the symptom is an LNS repair that flies through a hub's terminal airspace. `verify`
would catch it (a permanent ledger volume is a real conflict), but only at `verify_every`, and by
then thousands of iterations have been accepted against a wrong world.

"Just don't record claims on walled cells" is not sufficient on its own: the binding order in
`_scompiled_occ` is `subscribe(on_commit)` -> `_absorb(...)` -> `subscribe_static(...)`, so
`_absorb` records claims on cells that only become walled a moment later. Hence both halves — skip
recording once walled, **and** re-wall on release rather than rebuild.

(Note `block_range` on a walled cell is already a harmless no-op: head reads `a=0, b=-1`, so
`b < s0` sends the walk to `nxt == -1` and it returns. The bug is purely in the reversal.)

### 4.3 Wire the flag through (`sipp.py`)

**MODIFY `SIPPPlanner.__init__(self, max_expansions=1<<21, compiled=True, kernel_log2_min=None, incremental_release=False)`**

Forward both new kwargs to `super().__init__`. `kernel_log2_min` is needed independently of this
phase: `LNSState.replica` passes it (`state.py:322`) and `SIPPPlanner.__init__` currently drops it,
so a SIPP worker would silently run the A* fallback at the wrong array floor.

**MODIFY `_sipp_index` / `_scompiled_occ`** — two changes each:

```python
sidx = self._sidx = SafeIntervalIndex(cfg, track_removal=self.incremental_release)
...
if self.incremental_release:
    ledger.subscribe_release(sidx.on_release)
...
wm = req.t_request if self.evict_floor is None else min(req.t_request, self.evict_floor)
sidx.evict_before(int(wm // cfg.dt_s))          # G3: mirror astar/planner.py:332
```

**Expected outcome of Phase 1:** per-iteration cost stops containing `O(ledger)` terms. Whether the
remaining O(victims) release for four structures beats A*'s for two is section 8's question.

---

## 5. Phase 2 — select SIPP as the repair planner

**Point of the phase:** make the choice expressible, from the CLI down through a spawned worker,
without ever pickling a planner.

**ADD `LNSConfig.repair_planner: str = "astar"`** (`lns/solver.py:39`) — a *registry name*, not an
object. The parallel replica constructs its own planner inside a spawned process
(`state.py:322`), so the knob has to survive `WorkerSpec`'s "must be picklable" contract. A name
does; an `AStarPlanner` holding numpy pools and a bound ledger does not.

**ADD `lns/state.py::_new_repair_planner(name, *, incremental_release, kernel_log2_min=None, record_envelope=False)`** — the ONE construction site, so the sequential path, `LNSState.__init__`'s
default, and `LNSState.replica` cannot drift:

```python
if name in ("astar", "astar_ref"):   p = AStarPlanner(compiled=name == "astar", ...)
elif name in ("sipp", "sipp_ref"):   p = SIPPPlanner(compiled=name == "sipp", ...)
else: raise ValueError(f"repair_planner {name!r} is not a supported LNS repair planner")
p.evict_floor = 0.0        # R2, set HERE because the constructor is the owner
p.record_envelope = record_envelope
return p
```

Deliberately a small allowlist rather than `get_planner(name)`: the registry contains
`ShortcutRefiner` wrappers and the whole-schedule `colgen`, neither of which satisfies R1-R7 (a
wrapper has no `evict_floor` of its own — `state.py:190` documents exactly that trap). An explicit
list fails loudly on `astar_shortcut` instead of failing at `evict_floor` three frames later.

**MODIFY `LNSState.__init__`** — replace the hardcoded default (`state.py:208-209`) with
`_new_repair_planner(repair_planner_name, ...)`, adding a `repair_planner_name: str = "astar"`
kwarg alongside the existing `repair_planner=` object (which stays, for tests and for callers
passing a pre-warmed planner).

**MODIFY `LNSState.replica`** — take `repair_planner_name`, pass it to `_new_repair_planner`.

**MODIFY `WorkerSpec`** — add `repair_planner: str = "astar"`. It changes what a repair *is*, so
by that dataclass's own stated rule it belongs there.

**MODIFY `_build_lns_state`** (`solver.py:183`) and `parallel._make_spec` — forward `lns.repair_planner`.

**MODIFY `analysis/run_lns.py`** — `--repair-planner {astar,astar_ref,sipp,sipp_ref}`, default
`astar`.

---

## 6. Phase 3 — currency: the ruler and the parity gate

**Point of the phase:** make sure `delay()`, the accept test and the incumbent are all denominated
in seconds that mean the same thing.

**MODIFY `_REPRODUCIBLE_PLANNERS`** (`state.py:51`) — add `"sipp"`, `"sipp_ref"`, with the reason
written down: both planners are exact optimizers of the identical weighted cost over the identical
lattice, so on the walls-only ruler world they return the same optimum, and the A* ruler stays a
valid ruler for a SIPP incumbent. This is a *claim*, so it gets a test (below) rather than a
comment.

**Leave `unimpeded._new_ruler` on `AStarPlanner`.** Deliberate: the ruler world is empty, which is
SIPP's *worst* regime (no committed traffic to collapse into intervals, and `[[sipp-win-is-the-compiler]]` measured same-language SIPP 1.57x slower than A* with no congestion to exploit). Paying
SIPP prices for an A*-shaped workload to obtain the same number is a pure loss. Reconsider only if
the parity test below fails, in which case the ruler must follow the repair planner.

**ADD `tests/test_lns_sipp.py::test_sipp_and_astar_agree_on_unimpeded_cost`** — plan every request
of a small scenario on a walls-only ledger with both planners; assert costs equal to 1e-9 (not the
existing 1e-6). This is the test that licenses keeping the A* ruler.

**ADD `tests/test_lns_sipp.py::test_sipp_and_astar_agree_on_a_congested_ledger`** — the gap noted in
G4: replay a metro scenario, and at each step plan the request with BOTH against the same
A*-committed ledger, asserting equal accept verdict and equal cost. Routes may differ (ties break
differently); costs may not.

---

## 7. Phase 4 — DROP envelopes. Staged, and one part is not optional.

**Not optional, ship in Phase 1:** `SIPPPlanner.plan` must open with `self.last_envelope = None`,
matching `astar/planner.py:392`. That single line closes G2's silent-stale-merge bug. It costs
nothing and does not depend on anything else in this phase.

**Optional, and explicitly deferred:** a real SIPP read-set envelope. It needs a `read_bbox`
accumulator threaded through `sipp_kernel._search` (the A* kernel's is `astar/kernel.py:149-179`),
plus a `_mk_envelope` call on the SIPP compiled path. Deferred because:

- `record_envelope` is only ever set for `parallel_mode == "drop" and search_workers > 1`
  (`parallel.py:794`). SYNC and the sequential loop never read it.
- #118's own conclusion is that `search_workers` stays 1 pending a crossover measurement, so DROP
  is not the arm anyone is running yet.
- Until it exists, SIPP + DROP is **correct but degenerate**: every result reads as dirty, so DROP
  discards stale work instead of merging it. That is a performance floor, not a wrong answer —
  which is only true once the `last_envelope = None` reset above is in.

Consequently: allow `repair_planner="sipp"` with `parallel_mode="drop"`, but **log a warning once**
naming the degradation, rather than raising. Raising would make the honest combination
inexpressible for measurement.

---

## 8. What could go wrong — self-critique

**1. Phase 1 may not be enough, and the plan should say so up front.** Even with O(victims)
release, SIPP maintains four structures against A*'s two. The per-iteration ledger work is roughly
`2 x (release + k commits)` across all structures. If SIPP's structures cost about what A*'s do,
SIPP's commit side is ~2x and its plan side ~0.5x, so the net turns on the plan:commit ratio —
which `[[sipp-commit-side-overhead]]` says was 195 s commit vs a faster plan even *before* LNS
doubled the commit side. **This is the plan's central risk and it is not resolvable by design, only
by measurement.** Mitigation: section 9's A/B is the deliverable, and a measured "no" is an
acceptable outcome.

**2. A cheaper alternative exists and should be measured before Phase 1 is optimized.** SIPP's
`_sidx` + `_scocc` are only read by the *compiled* SIPP path. A repair planner that keeps A*'s two
structures and pays SIPP's two only when it plans is exactly what we are building — but nothing
forces `_sidx` and `_scocc` to be release-hooked *if the rebuild were cheap*. It is not (G1), so
Phase 1 stands. Worth stating so a reader does not re-derive it.

**3. `_claims[key].remove(...)` is O(list) and the list is per-cell.** A hot hub-mouth cell may
accumulate many claims. A* accepted this (`compiled_hex_occupancy.py:344`) because per-cell claim
lists are short relative to the schedule. If profiling shows otherwise, the fix is a
`collections.Counter` per key, not a redesign — but do not pre-optimize; A*'s measured experience
is the prior.

**4. The refcount conversion doubles `SafeIntervalIndex`'s memory in LNS mode.** `set[int]` ->
`dict[int, int]` is roughly 2x per entry, and `_rows` adds a journal linear in schedule size (A*
measured its own at 32 B/row after packing, 185 B unpacked — pack from the start). At 426,756
volumes this is real. Mitigation: pack `_rows` as `array("q")` from the first commit, exactly as
`HexOccupancyService` does, and measure RSS in the A/B — `[[colgen-lazy-rows-are-materialize-not-loads]]` warns that reading RSS from the probe process is a trap, so measure tree RSS.

**5. `_rewind` releases twice.** `try_repair`'s exception path calls `release_many(victims)` after
the loop already released them (`state.py:525` then `state.py:575`). A second `on_release` for the
same fid would `KeyError` on `_rows.pop`. **Checked: not a problem** — `release_many` skips fids
with no live runs (`ledger.py:285`), so the hook fires at most once per release. Worth a test
anyway, because it is invisible until an exception happens deep in a long run.

**6. Cost ties break differently, so there is no byte-parity gate against the A* LNS run.** Unlike
#118's `search_workers=1` parity gate, a SIPP repair produces a different (equal-cost) route, so
the LNS trajectory diverges from iteration 1. The available gates are weaker: equal *cost* per
plan (Phase 3's tests), conflict-freedom (`verify_every`), and the byte-parity of SIPP-with-release
against SIPP-with-rebuild (`--no-incremental`), which is the direct analogue of the gate PR #109
used and is the one that actually pins Phase 1.

**7. `[[analysis-scripts-are-not-maintained]]`** — `analysis/run_lns.py` gets a flag here; do not
repoint anything else in `analysis/`, and do not trust a neighbouring script's numbers.

---

## 9. Verification and measurement

**Correctness gates, in dependency order:**

1. Phase 0: existing `tests/test_sipp*.py` + `tests/test_lns*.py` green after the merge.
2. Phase 1: **`incremental_release=True` is byte-identical to `incremental_release=False`** for a
   SIPP repair — same trajectory rows, same final intents. This is the gate that pins the whole
   phase, and it is the same gate PR #109 used for A*.
3. Phase 1: a static-wall test — release a flight whose corridor touches an always-active hub's
   walled cells, then assert `free_intervals`/`blocked_at` still report the wall. This is the trap
   in section 4.2 and it must fail before the fix.
4. Phase 1: `_free` reclaim — assert `nslots` is bounded across many destroy/repair cycles on the
   same cells (the unbounded-growth failure has no other symptom until memory).
5. Phase 2: `repair_planner="sipp"` runs sequentially, `verify_every=1`, conflict-free.
6. Phase 3: the two cost-parity tests.
7. Phase 4: `last_envelope` is `None` after a native SIPP plan that follows an A*-fallback plan.
8. Parallel: `search_workers=1` with `repair_planner="sipp"` byte-identical to the sequential SIPP
   loop — #118's parity gate, re-run on the new arm.

**The measurement that decides the default** (`analysis/run_lns.py`):

Paired A/B, same seed, same `neighborhood_size`, same iteration budget, `--repair-planner astar`
vs `sipp`, on `density_faa_wing_zipline`. Report per arm: loop wall, iterations/s, plan-side vs
commit-side split, improvement %, and AUC. Two scales, because `[[lns-neighborhood-size-is-scale-dependent]]` and `[[drop-lns-parallel-implementation]]` both invert with instance size and a
single cut would mislead: the 1,526-leg cut (`--demand-duration 600`) and full 4,636.

Quote gains against the 19.3% that is actually delay, not against total cost
(`[[density-faa-delay-is-19pct]]`).

**Default stays `astar` unless SIPP wins on iterations/s at equal improvement-per-iteration.**
