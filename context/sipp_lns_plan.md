# SIPP as an LNS repair planner — design record

Branch `victor-qin/lns-sipp-planner`, forked off #118 (`victor-qin/drop-lns-parallel-plan`).

*Line citations are **post-merge** (i.e. against `origin/main` @ 153ec4f, which Phase 0 merges in).
`sipp.py` and `compiled_occupancy.py` shift by #117; `state.py`, `parallel.py` and `solver.py` are
untouched by the merge, so their branch numbers stand.*

## 0. The question, and the one number that answers it

LNS repair is `plan()` against a **nearly-full ledger**, which is exactly where SIPP wins: 1.86x A*
at lambda=4000, 2.82x at lambda=8000 (PR #113), because the interval collapse pays more the more
committed traffic there is to collapse. Repair is also the only thing in the loop that scales with
neighborhood size, so a faster repair is a directly larger iteration budget.

But LNS is also the most **commit-heavy** workload here, and that is where SIPP has always been
weak. One iteration is:

```
release_many(victims)          # 1 release of k flights
k x plan()                     # the part SIPP is good at
k x commit()                   # the part SIPP is bad at
--- and on the 79% of iterations that reject: ---
release_many(victims)          # _rewind, state.py:575
k x commit()                   # the old volumes, back
```

So a rejected iteration pays the commit side **twice** and the plan side once.

**Count the structures carefully — the plan's central ratio depends on it.** Compiled A* binds
THREE ledger-subscribed structures, not two: `_svc` + `_tcap` from `_occupancy`
(`astar/planner.py:303-307`) and `_cocc` from `_compiled_occ` (`:796-798`), all three
release-hooked under `incremental_release`. SIPP's compiled path binds `_svc` + `_tcap`
(`sipp.py:687`), `_scocc` (`:688`) and `_sidx` (`:722`) — **four**, of which only the two inherited
ones are release-hooked. A flight that falls back to A* additionally binds `_cocc`, making five.

So the commit-side ratio is **4:3 ≈ 1.33x**, up to 5:3 on a fallback-heavy arm — not the 2:1 a
casual "SIPP has two extra structures" reading suggests. That materially shrinks the headwind SIPP
has to overcome, and it is the right prior to size the A/B against.

`[[sipp-commit-side-overhead]]` measured the pre-#117 version of this as 1.36x slower end-to-end;
#117 fixed the *commit* side (190.9 s -> 66.1 s). **#117 did not touch the release side, and LNS is
the only caller that has one.** That is the whole of this work.

The decisive quantity is therefore not "is SIPP faster than A*" — measured, and yes, in congestion.
It is:

> per LNS iteration, does SIPP's plan-side saving exceed the extra release-and-recommit cost of a
> fourth structure?

Nothing in the repo answers that, because SIPP cannot be used as a repair planner at all. Sections
3-7 make it usable; section 9 measures it. **If the answer is no, the correct outcome of this branch
is a measured "no" and a default left at `astar`** — the same shape as #118's own verdict on
`search_workers`.

---

## 1. What LNS actually demands of a repair planner

Derived from the code, not assumed. Seven requirements; SIPP satisfies two.

| # | Requirement | Enforced at | SIPP today |
|---|---|---|---|
| R1 | `plan(req, ledger, cfg) -> OperationalIntent` | `state.py:527` (the only method `try_repair` calls) | OK |
| R2 | `evict_floor == 0.0`, so victims may be replanned in ANY priority order | `state.py:192` (raises, but only for a BORROWED planner — see §6) | attribute inherited; **its own two structures ignore it** (G3) |
| R3 | Not already bound to this ledger | `state.py:181` | OK (checks `_svc_ledger`/`_cocc_ledger`, both inherited) |
| R4 | Survive `detach_subscribers()` + epoch bump by REBINDING | `state.py:201`; SIPP's tripwires at `sipp.py:287`, `sipp.py:522` | OK — pinned by `test_shared_sipp_occupancy_preserves_nonzero_ledger_epoch` |
| R5 | Un-absorb `release_many` in O(victim volumes) under `incremental_release=True` | `astar/planner.py:305-307`, `:797-798` | **absent for `_sidx` + `_scocc`** (G1) |
| R6 | Costs in the same currency as the unimpeded ruler and the incumbent | `state.py:171` — but it reads `cfg.planner`, NOT the repair planner (§6) | untested in congestion (G4) |
| R7 | For DROP with m>1: reset `last_envelope` per plan, set it when `record_envelope` | `parallel.py:715-727`, `:794` | **absent, and worse than absent** (G2) |

---

## 2. The gap — four findings, ranked

### G1 (blocker, performance). Two of SIPP's four ledger structures have no release hook.

`AStarPlanner._occupancy` subscribes `svc.on_release` and `_tcap.on_release` when
`incremental_release` is set (`astar/planner.py:305-307`), and `_compiled_occ` does the same for
`cocc` (`:797-798`). SIPP **inherits `_occupancy`, and calls it on its hot path**
(`sipp.py:687`), so `_svc` and `_tcap` are already correct and Phase 1 does not touch them. `_cocc`
is likewise already correct — SIPP just doesn't bind it unless a flight falls back.

SIPP's own two — `_sipp_index` (`sipp.py:281`) and `_scompiled_occ` (`sipp.py:518`) — call only
`ledger.subscribe(...)` and `ledger.subscribe_static(...)`. There is no `on_release` on
`SafeIntervalIndex` or `CompiledOccupancy` at all, and `SIPPPlanner.__init__` never reads
`incremental_release` for them.

The consequence is not a wrong answer, it is a rebuild. After `release_many`, `ledger.n_volumes`
falls while their `n_added` does not, so the shrink tripwire fires on the next `plan()`:

```python
elif ledger.n_volumes < sidx.n_added:
    sidx.reset(); _absorb(sidx, ledger)      # sipp.py:296-298  — O(every committed volume)
elif ledger.n_volumes < cocc.n_added:
    cocc.reset(); _absorb(cocc, ledger)      # sipp.py:530-532  — again
```

`_absorb` walks `ledger.iter_committed()` in full. PR #109 measured the A* equivalent of ONE such
rebuild at 3.74 s — 94% of an iteration — which is exactly why `incremental_release` was built.
SIPP would pay TWO per iteration. At 426,756 volumes x 2000 iterations this is not a slowdown, it
is a non-starter.

### G2 (blocker for DROP m>1, and a silent correctness bug). No read-set envelope.

`SIPPPlanner.plan` (`sipp.py:302`) resets `last_expansions`, `_n_expansions` and `_air`. It does
**not** reset `last_envelope`, and `sipp_kernel.py` has no `read_bbox` accumulator (compare
`astar/kernel.py:149-179`, `:232-233`).

Two consequences, and the second is the serious one:

1. With `record_envelope=True`, every native SIPP repair reports `None`. `_read_set_is_clean`
   treats `None` as always-dirty (`parallel.py:718-722`), so DROP discards every stale result and
   degrades to SYNC. Correct, just pointless.
2. `SIPPPlanner._fallback` (`sipp.py:640`) calls `AStarPlanner.plan`, which **does** reset and set
   `last_envelope` (`astar/planner.py:392`, `:1153`). So: flight A falls back to A* and leaves an
   envelope; flight B plans natively and does not clear it; `try_repair` appends
   `self.repair_planner.last_envelope` for B (`state.py:534`) and files **A's read set under B's
   name**. The coordinator tests the wrong region, finds it clean, and merges a repair that is
   genuinely stale. Silent, and `verify.find_interflight_conflict` would not catch it — a stale
   merge produces a *worse cost*, not a conflict.

The one-line fix for (2) is unconditional and should land regardless of whether (1) is ever built.

### G3 (latent, currently answer-neutral). `evict_floor` is ignored.

A* computes `wm = req.t_request if self.evict_floor is None else min(req.t_request,
self.evict_floor)` (`astar/planner.py:332`, `:807`). SIPP's two call
`evict_before(int(req.t_request // cfg.dt_s))` unconditionally (`sipp.py:299`, `sipp.py:533`).

Today this changes nothing: `evicted_before` is **written but never read** in `sipp.py`,
`compiled_occupancy.py` or `sipp_kernel.py` — both docstrings say "storage reclaim is TODO". So
R2's guarantee holds by accident. It stops being an accident the moment someone implements the
reclaim. Wiring `evict_floor` through is three lines now and a silent exactness bug later.

*(Note on a tempting mis-citation: `compiled_hex_occupancy.py:364-365` clips spans by
`evicted_before` on the **commit** path, inside `_add`. Compiled A*'s `on_release` (`:330-350`)
clips nothing. The only release-side clamp in the repo is `HexOccupancyService.on_release`
(`occupancy.py:252-253`), and §4.1 explains why SIPP must not copy it.)*

### G4 (currency). The ruler is A*, the repair would be SIPP, and no congested test compares them.

- `unimpeded._new_ruler` (`unimpeded.py:51-69`) hardcodes `AStarPlanner`.
- The only SIPP-vs-A* cost comparison is on an **empty world** at 1e-6
  (`test_sipp_compiled.py:61`). Every replay test (`:357`, `:381`) compares SIPP-compiled against
  SIPP-*reference* on an A*-committed ledger — never SIPP against A*.
- `_REPRODUCIBLE_PLANNERS` (`state.py:51`) does **not** guard this. See §6: the check reads
  `cfg.planner` — the planner that produced the *baseline* — and never inspects the repair planner
  at all.

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

**Change:** `git merge origin/main` (no code edits expected), then `tests/test_sipp*.py` +
`tests/test_lns*.py` green before anything is built on it.

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

**The A\* removal journal has FOUR moving parts, and porting three of them is a hard failure on the
first iteration.** They are: `__init__` (allocate the journal), **`on_commit` (write it)**, `_add`
(collect rows), `on_release` (read and reverse it) — plus **`reset()` (clear it)**. `_add` is
called from nowhere but `on_commit` (`sipp.py:103`, `compiled_occupancy.py:122`), so a spec that
only threads `_rows` into `_add` leaves `self._rows` permanently empty and the first
`release_many` raises `KeyError`. Both sub-sections below therefore modify `on_commit` and
`reset()` explicitly.

### 4.1 `SafeIntervalIndex` — refcounts (`freespace_sim/planner/sipp.py`)

**ADD the import** `from array import array` to `sipp.py` (it has none;
`compiled_hex_occupancy.py:40` is the precedent).

**MODIFY `SafeIntervalIndex.__init__(self, cfg, track_removal=False)`**

```python
self.track_removal = track_removal
self._rows: dict[int, array] = {}     # fid -> flat int64 4-slot rows (cid, s_lo, s_hi, code)
self._cells: list = []; self._cell_ids: dict = {}     # (q,r,L) interning pool
self._tids: list = [];  self._tid_ids: dict = {}      # terminal id -> code
# corr/cols keep their TYPE when track_removal is off; counter-dicts when on (see below)
```

**MODIFY `SafeIntervalIndex.on_commit(self, flight_id, volumes)`** — the journal's only write site,
mirroring `occupancy.py:231-240`:

```python
hg.prepare_range_cache_for_commit(volumes)
own_cols = <unchanged>
rows = [] if self.track_removal else None
for v in volumes:
    self._add(v, own_cols, rows)
self.n_added += len(volumes)
if self.track_removal:
    entry = self._rows.get(flight_id)
    if entry is None: self._rows[flight_id] = array("q", rows)
    else:             entry.extend(rows)
```

The `extend` branch is **not** optional: `_absorb` groups by `itertools.groupby` over
`iter_committed` (`astar/planner.py:118`), which yields a second group for any fid whose volumes
are non-contiguous in the ledger — and `apply_delta`'s one-commit-per-flight discipline
(`state.py:618`) exists precisely because that contiguity is not guaranteed.

**MODIFY `SafeIntervalIndex._add(self, vol, own_cols, _rows=None)`** — same body, refcounted
inserts under the flag:

```python
for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(...):
    if not in_blk: continue
    cell = (q, r, L)
    if is_column:
        if self.track_removal:
            cell, cid = self._intern(cell)          # NOTE: 2-tuple — see below
            code = self._tid_code(tid)
            _rows += (cid, s_lo, s_hi, code)
            d = self.cols.setdefault(cell, {})
            for s in range(s_lo, s_hi + 1):
                e = d.setdefault(s, {}); e[tid] = e.get(tid, 0) + 1
        else:
            <unchanged set-based insert>
    elif not (own_cols and self._inside_a_column(q, r, own_cols)):
        if self.track_removal:
            cell, cid = self._intern(cell)
            _rows += (cid, s_lo, s_hi, -1)
            d = self.corr.setdefault(cell, {})
            for s in range(s_lo, s_hi + 1): d[s] = d.get(s, 0) + 1
        else:
            <unchanged set-based insert>
```

**ADD `SafeIntervalIndex._intern(self, cell) -> (canonical_cell, cell_id)`** — port of
`occupancy.py:211-220`. It returns a **2-tuple**, and the canonical *tuple* must become the
`corr`/`cols` key, not just the journal's `cid`. That sharing is the whole of §8.4's memory
mitigation: `occupancy.py:212-214` puts it at "the difference between 80 bytes per row and 80 bytes
per cell". Binding only `cid = self._intern(cell)` and keying off the freshly-built tuple silently
throws it away.

**ADD `SafeIntervalIndex._tid_code(self, tid)`** — the `_tids`/`_tid_ids` half of the same
interning, so a row packs a small int rather than a hub id.

**ADD `SafeIntervalIndex.on_release(self, flight_id, volumes)`**

```python
rows = self._rows.pop(flight_id)
for i in range(0, len(rows), 4):                 # flat 4-slot rows
    cid, s_lo, s_hi, code = rows[i:i + 4]
    cell = self._cells[cid]
    for s in range(s_lo, s_hi + 1):
        <decrement corr[cell][s], or cols[cell][s][tid] for code >= 0;
         delete the key at zero, and the emptied inner dict, so `if not corr` stays exact>
self.n_added -= len(volumes)
```

**Deliberately NO `evicted_before` clamp here, and this is the opposite of A\*'s choice.**
`HexOccupancyService.on_release` does clamp (`occupancy.py:252-253`), but it is sound there only
because its `evict_before` **physically deletes** the sub-floor buckets (`occupancy.py:298-300`)
*and* `add_volume` applies the identical clamp on insert (`:147-148`) — a matched pair.
`SafeIntervalIndex.evict_before` deletes nothing ("storage reclaim is TODO", `sipp.py:132`), so
sub-floor entries stay live and a clamped release would leave every step in
`[recorded s_lo, evicted_before)` permanently un-decremented: phantom blocked steps outliving the
flight. Latent today (LNS pins `evict_floor = 0.0`, so `wm = 0`), but wired straight into the new
structure. Record the full span, reverse the full span. If reclaim ever lands, clamp **both**
sides together or neither.

**MODIFY `SafeIntervalIndex.reset(self)`** — add `self._rows.clear()`. Keep `_cells`/`_cell_ids`/
`_tids`/`_tid_ids`: pure interning pools, value-identical across a rebuild, and never read except
through a live row (`occupancy.py:307-309` says exactly this). Without the clear, the shrink
tripwire this plan quotes at `sipp.py:296-298` leaves a journal describing a structure that no
longer exists, and the next `on_release` decrements counts a fresh `_absorb` just rebuilt —
phantom-free cells, wrong world, caught only at `verify_every`.

**Why the type switch is reader-transparent — verified, not assumed.** Every reader of `corr` and
`cols` uses membership, iteration, or truthiness, all identical on `set` and `dict`:

- `s in self.corr.get((q, r, L), ())` — `sipp.py:161`
- `any(t not in own for t in hubs)` — `sipp.py:157` — iterating a dict yields keys
- `cand.update(s for s in corr ...)` / `for s in cols` — `sipp.py:175, 177`
- `if not corr and not cols` — `sipp.py:171` — an empty dict is falsy

This is the same claim `HexOccupancyService` already makes in its own docstring ("Membership
queries are unchanged (`in` works on dict keys); flag off => the original set-based structures,
byte-for-byte"), which is the precedent for gating on the flag rather than converting outright.

**`static_cols` is untouched.** A separate `(q,r) -> {tid}` dict, read independently
(`sipp.py:155, 166`), and `reset()` deliberately preserves it — walls are infrastructure, not
commit-derived. No release ever touches it.

### 4.2 `CompiledOccupancy` — claim journal + per-cell rebuild (`compiled_occupancy.py`)

**ADD the imports.** `from array import array`, and extend the existing
`from .astar.compiled_hex_occupancy import schedulable_horizon_steps` (`compiled_occupancy.py:34`) to
also pull `_FIELD_MASK, _SPAN_BITS, _SPAN_LIMIT`. That module is already a dependency and does not
import back (`compiled_hex_occupancy.py:36-47`), so this adds no cycle — and **sharing the constants
is the point**: two journals with independently-defined packing widths is a silent-corruption bug
waiting for someone to widen one of them.

**MODIFY `_init_pool`** — add `self._free: list[int] = []`. (Note this is also what makes `reset()`
free-list-correct for nothing: `reset()` calls `_init_pool`, so `_free` is re-created there and
needs no separate clear.)

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
MAXS >= 1 << 20`.

**MODIFY `CompiledOccupancy.on_commit(self, flight_id, volumes)`** — same shape as §4.1, against
`compiled_hex_occupancy.py:312-324`: build `rows = [] if self.track_removal else None`, thread it
through each `_add`, then `array("q", rows)` / `entry.extend(rows)`. Flat **`(key, claim)` pairs**
here, not 4-slot rows.

**MODIFY `_add(self, vol, own_cols, _rows=None)`** — record after the existing `block_range`:

```python
self.block_range(c, int(s_lo), int(s_hi))
if self.track_removal and c not in self._static_cells:
    self._record(c, int(s_lo), int(s_hi), _rows)
```

**ADD `_record(self, c, s0, s1, _rows)`** — port of `compiled_hex_occupancy.py:391-407`, minus the
pool index (one pool, so `key = c` rather than `c << 1 | pool_idx`):

```python
if s0 < 0:  s0 = 0
if s1 > self.MAXS: s1 = self.MAXS        # record what block_range WILL apply, not the raw span
if s0 > s1: return
# NO per-row _SPAN_LIMIT raise, unlike A* — see below.
packed = (s0 << _SPAN_BITS) | s1
self._claims.setdefault(c, []).append(packed)
if _rows is not None: _rows += (c, packed)
```

**Why the clamp, and why — unlike A\* — there is no per-row raise.** The spans arrive raw from
`rasterize_ranges`, *before* `block_range`'s own internal `if s1 > self.MAXS` clamp. A committed
volume can outlive the box (a late return commits past `MAXS` and box-guards to the reference), so
recording them unclamped would pack `s_hi >= 1<<20` into the `s0` field, and on release every
*surviving* claim in that cell replays at a garbage span. Clamping first also makes the journal
exactly reversible: what was applied is what is replayed.

That clamp is precisely the **divergence from A\***, and it makes A*'s per-row guard
(`compiled_hex_occupancy.py:392-397`) dead code here. A* records the raw span, so its constructor's
`MAXS` check genuinely does not bound `_record` and the raise is load-bearing. Post-clamp,
`0 <= s0 <= s1 <= MAXS < _SPAN_LIMIT` holds by construction, so the same raise can never fire.
Keep the constructor's `MAXS >= _SPAN_LIMIT` check as the single bound and carry a one-line comment
recording why the two `_record`s differ — importing A*'s justification wholesale would assert
something the clamp has already made untrue.

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
        self._claims.pop(c, None)       # the non-static branch's cleanup, which `continue` skips
        continue                        # walled ⇒ content irrelevant, and a replay would leak slots
    survivors = self._claims.get(c)
    if survivors:
        for packed in survivors:
            self.block_range(c, packed >> _SPAN_BITS, packed & _FIELD_MASK)
    else:
        self._claims.pop(c, None)
self.n_added -= len(volumes)
```

Equal spans are a **fungible multiset**: the owner journal says how many identical spans to remove,
and which instance is removed cannot affect the rebuilt pool. That is A*'s stated contract
(`compiled_hex_occupancy.py:258-259`) and `tests/test_lns.py:180` is its direct pin — port it.

**MODIFY `reset(self)`** — add `self._claims.clear(); self._rows.clear()` (mirroring
`compiled_hex_occupancy.py:443-444`), before `_init_pool()`. `_static_cells` is repopulated by the
existing `_static_hubs` replay at `compiled_occupancy.py:226`, and `_free` by `_init_pool`, so
neither needs its own clear. Reachable via the shrink tripwire at `sipp.py:530-532`; without the
clear, `_claims[key].remove(...)` either raises `ValueError` (mis-read as drift) or succeeds and
re-blocks spans the fresh `_absorb` already applied — double-blocked cells, silently wrong.

**MODIFY `_wall_static_terminal`** — record each walled cell id into `self._static_cells` as it
writes the empty interval.

#### The one subtle thing: `CompiledOccupancy` is the only structure that stores always-active walls in the same array as commit-derived blocks

Everywhere else, statics live apart: `CompiledHexOccupancy.static_col` is a separate bool array,
`HexOccupancyService.static_term_cells` a separate dict, `SafeIntervalIndex.static_cols` a separate
dict. `CompiledOccupancy._wall_static_terminal` instead writes `iv_lo=0; iv_hi=-1; iv_nxt=-1`
**directly into the pool** — and it has no choice: `sipp_kernel._search` (`:67-80`) takes the
interval pool and the overlay and **no static array**, so the pool is the only channel through
which the kernel can learn about a wall. Compare `astar/kernel.py:149`, which takes `static_col`
explicitly.

The consequence: `reset_cell`'s blank slate is `[0, MAXS]` — **fully free** — while a walled cell's
correct blank slate is `[0, -1]`. The claim journal only ever describes commit-derived blocks, so a
naive rebuild reconstructs one meaning and destroys the other. The wall silently disappears for the
rest of the run and repairs route foreign traffic through a hub's terminal airspace. `verify` would
catch it (a permanent ledger volume is a real conflict), but `verify_every` defaults to 0.

Note the asymmetry that lets it hide: `block_range` on an already-walled cell reads head
`a=0, b=-1`, tests `b < s0` (true for any `s0 >= 0`), follows `nxt == -1` and returns — a harmless
no-op. **Blocking is idempotent against a wall; un-blocking is not.**

"Just don't journal walled cells" is necessary but not sufficient, because of the bind order in
`_scompiled_occ`: `subscribe(on_commit)` -> `_absorb(...)` -> `subscribe_static(...)`. The replay
in `subscribe_static` is load-bearing (services bind lazily, after `sim.run` has registered every
hub), so `_absorb` records claims on cells that only become walled a moment later — the guard sees
an empty `_static_cells` at exactly the moment it matters. Hence both halves: skip recording once
walled, **and** re-wall on release rather than rebuild.

And walled cells are *guaranteed* to carry claims, from two sources: a flight's own terminal column
volume is never skipped (`is_column` volumes always `block_range` — "A column is foreign-to-everyone
here"), and exit-lane corridor cells lie outside the own-column disc so they are blocked and
journaled — while `hg.terminal_cells` is defined as column `|` exit lanes (`hexgrid.py:228-235`).

One tempting shortcut to reject: a wall is `(lo=0, hi=-1)` and a fully-covered commit is
`(lo=1, hi=0)` (`compiled_occupancy.py:211-213`), so you *could* sniff the sentinel. That is a
fragile pun on two arbitrary values, and `reset_cell` destroys the evidence before you can read it.
An explicit set is the honest encoding.

**Alternative considered and deferred:** give `CompiledOccupancy` a separate `static_col` bool
array like A*, so the trap disappears rather than being guarded. That means adding an argument to
`sipp_kernel._search` and consulting it in the kernel's interval walk — a numba hot-path change, in
a phase whose job is to add a release hook. The guard is ~6 lines and testable in isolation. Note
it in the code as the right follow-up if `CompiledOccupancy` grows a third consumer.

### 4.3 Wire the flag through (`sipp.py`)

**MODIFY `SIPPPlanner.__init__(self, max_expansions=1<<21, compiled=True, kernel_log2_min=None, incremental_release=False)`**

Forward both new kwargs to `super().__init__`. `kernel_log2_min` is needed independently of this
phase: `LNSState.replica` passes it (`state.py:322`) and `SIPPPlanner.__init__` currently drops it,
so a SIPP worker would silently run the A* fallback at the wrong array floor.

Setting `incremental_release=True` **also** release-hooks `_svc`/`_tcap` for free, because SIPP
calls the inherited `_occupancy` on its hot path (`sipp.py:687` -> `astar/planner.py:305-307`).
Phase 1 does not touch those two — they are already correct.

**MODIFY `_sipp_index` / `_scompiled_occ`** — three changes each:

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
remaining O(victims) release for four structures beats A*'s for three is section 9's question.

---

## 5. Phase 2 — select SIPP as the repair planner

**Point of the phase:** make the choice expressible, from the CLI down through a spawned worker,
without ever pickling a planner.

**ADD `LNSConfig.repair_planner: str = "astar"`** (`lns/solver.py:39`) — a *registry name*, not an
object. The parallel replica constructs its own planner inside a spawned process
(`state.py:322`), so the knob has to survive `WorkerSpec`'s "must be picklable" contract. A name
does; an `AStarPlanner` holding numpy pools and a bound ledger does not.

**ADD `lns/state.py::_new_repair_planner(name, *, incremental_release, kernel_log2_min=None, record_envelope=False)`** — the ONE construction site, so the sequential path, `LNSState.__init__`'s
default and `LNSState.replica` cannot drift:

```python
if name in ("astar", "astar_ref"):   p = AStarPlanner(compiled=name == "astar", ...)
elif name in ("sipp", "sipp_ref"):   p = SIPPPlanner(compiled=name == "sipp", ...)
else: raise ValueError(f"repair_planner {name!r} is not a supported LNS repair planner")
p.evict_floor = 0.0        # R2, set HERE because the constructor is the owner
p.record_envelope = record_envelope
return p
```

**MODIFY `_validate_lns_config`** (`solver.py:116`) — add the name check there, next to the
`operators` check. **This is not a stylistic preference; the allowlist raise cannot live only in
`_new_repair_planner`.** That function is called from `state.py:207-210`, which is *after*
`ledger.detach_subscribers()` at `state.py:201` — so a typo'd `--repair-planner sipp2` would strip
the caller's ledger of every observer and release subscriber and bump its epoch, and only then
raise. That is exactly the failure the vet block above it names: "a constructor that raises must not
leave the caller's ledger stripped of its subscribers" (`state.py:178-179`). `_validate_lns_config`
exists for this — "Validate all arguments that must fail before `LNSState` takes over the ledger" —
and both entry points already call it first (`solver.py:256`, `parallel.py:773`).

**And hoist the construction above the detach.** Inside `LNSState.__init__`, call
`_new_repair_planner(...)` into a local *before* `ledger.detach_subscribers()` and assign
`self.repair_planner` after. The validator covers a bad *name*, but the constructor also runs the
SIPP kernel import and `AStarPlanner._warm_jit()` (`astar/planner.py:276-277`) — any of which can
raise inside the same post-detach window. Hoisting is the fix rather than try/except-and-resubscribe
because **the epoch bump is not reversible.**

Deliberately a small allowlist rather than `get_planner(name)`: the registry contains
`ShortcutRefiner` wrappers and the whole-schedule `colgen`, neither of which satisfies R1-R7 (a
wrapper has no `evict_floor` of its own — `state.py:190` documents exactly that trap). An explicit
list fails loudly on `astar_shortcut` instead of failing three frames later.

Also note what `_new_repair_planner` bypasses: `LNSState`'s vet block (`state.py:180-195`, which
checks `evict_floor` and prior binding) runs only `if repair_planner is not None`, i.e. only for a
*borrowed* object. A constructed one is never vetted, which is why the constructor sets
`evict_floor` itself rather than trusting a later check.

**MODIFY `LNSState.__init__`** — replace the hardcoded default (`state.py:208-209`) with
`_new_repair_planner(repair_planner_name, ...)`, adding a `repair_planner_name: str = "astar"`
kwarg alongside the existing `repair_planner=` object (which stays, for tests and for callers
passing a pre-warmed planner).

**MODIFY `LNSState.replica`** — take `repair_planner_name`, pass it to `_new_repair_planner`.

**MODIFY `WorkerSpec`** (`parallel.py:80-99`) — add `repair_planner: str = "astar"`. It changes
what a repair *is*, so by that dataclass's own stated rule it belongs there.

**MODIFY the inline `WorkerSpec(...)` construction at `parallel.py:783-794`** — forward
`lns.repair_planner`. (There is no `_make_spec` helper; the spec is built inline.)

**MODIFY `parallel._worker_main`'s `LNSState.replica(...)` call (`parallel.py:241-251`)** — add
`repair_planner_name=spec.repair_planner`, alongside the existing `spec.incremental_release` /
`spec.kernel_log2_min` / `spec.record_envelope` forwards. Without this the worker silently repairs
with A* while the coordinator believes it is running SIPP — precisely the failure mode
`WorkerSpec`'s docstring names ("a worker that silently differs from the coordinator's belief is
the failure mode with no symptom", `parallel.py:83-86`).

**MODIFY `_build_lns_state`** (`solver.py:173`, the `LNSState(...)` call at `:183`) — forward
`lns.repair_planner`.

**MODIFY `analysis/run_lns.py`** — `--repair-planner {astar,astar_ref,sipp,sipp_ref}`, default
`astar`; and read `len(ledger._release_subs)` / `len(ledger._observers)` once after
`run_lns_on_result` returns (the runner holds the same ledger object, so no hook is needed) into the
JSON output. That is the direct check on §0's 4:3 claim and it costs one line.

**ADD plan-side / commit-side timers**, because §9 needs a split no current readout can produce:
`LNSResult.summary()` carries only `wall_s`/`init_wall_s`/`auc` (`solver.py:95-113`), and a
trajectory row only cumulative `wall_s` (`solver.py:318-324`). Two accumulators on `LNSState`,
incremented in `try_repair` — one around `self.repair_planner.plan(...)` (`state.py:527`), one
around `self.ledger.commit(...)` (`state.py:531`) plus `_rewind`'s release+commit loop
(`state.py:575-579`) — surfaced through `_finalize_lns_result` into `LNSResult.summary()`. Without
these the number that decides the default has no instrument.

*(A `perf_counter` per call is exactly what `[[colclr-per-flight-scaling]]` warns against for
per-flight attribution. It is fine here: these are two counters over a whole run, read once, not a
per-flight ranking — but do not let them grow into one.)*

---

## 6. Phase 3 — currency: what actually guards it, and what does not

**Point of the phase:** make sure `delay()`, the accept test and the incumbent are denominated in
seconds that mean the same thing — and be honest that no existing check enforces it.

**Leave `_REPRODUCIBLE_PLANNERS` alone.** This is a change from an earlier draft, and the reason is
worth writing down. The guard is:

```python
if repair_planner is None and cfg.planner not in _REPRODUCIBLE_PLANNERS:   # state.py:171
```

It reads **`cfg.planner`** — the planner that produced the *baseline* — and never inspects the
repair planner. The §9 A/B baselines on A* (`analysis/run_lns.py:83` passes
`planner_name="astar"`), so `repair_planner_name="sipp"` passes this check today, with or without
an edit. Adding `"sipp"` to the set would therefore not gate anything Phase 2 introduces; what it
*would* newly license is the opposite mixing — a SIPP-baselined run silently defaulting to an A*
repair planner and an A* ruler — which is the untested congested-currency question G4 raises. If
`test_sipp_and_astar_agree_on_a_congested_ledger` (below) passes, that is the evidence that
separately justifies the edit, as a follow-up rather than part of this branch.

So the currency risk Phase 2 actually introduces — **baseline A\*, ruler A\*, repair SIPP** — is
pinned by the two tests below and by nothing else in the code. Say so plainly rather than implying
a guard covers it.

**Leave `unimpeded._new_ruler` on `AStarPlanner`.** Deliberate: the ruler world is empty, which is
SIPP's *worst* regime (no committed traffic to collapse into intervals, and
`[[sipp-win-is-the-compiler]]` measured same-language SIPP 1.57x slower than A* with no congestion
to exploit). Paying SIPP prices for an A*-shaped workload to get the same number is a pure loss.
Reconsider only if the first test below fails, in which case the ruler must follow the repair
planner.

**ADD `tests/test_lns_sipp.py::test_sipp_and_astar_agree_on_unimpeded_cost`** — plan every request
of a small scenario on a walls-only ledger with both planners; assert costs equal to 1e-9 (not the
existing 1e-6). This is the test that licenses keeping the A* ruler.

**ADD `tests/test_lns_sipp.py::test_sipp_and_astar_agree_on_a_congested_ledger`** — the gap G4
names: replay a metro scenario and at each step plan the request with BOTH against the same
A*-committed ledger, asserting equal accept verdict and equal cost. Routes may differ (ties break
differently); costs may not.

---

## 7. Phase 4 — DROP envelopes. Staged, and one part is not optional.

**Not optional, ship in Phase 1:** `SIPPPlanner.plan` must open with `self.last_envelope = None`,
matching `astar/planner.py:392`. That single line closes G2's silent-stale-merge bug. It costs
nothing and depends on nothing else in this phase.

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
naming the degradation, rather than raising. Raising would make the honest combination inexpressible
for measurement.

**Emit it in `_validate_lns_config` (`solver.py:116`), guarded on
`lns.repair_planner.startswith("sipp") and lns.parallel_mode == "drop" and search_workers > 1`.**
The obvious home — `_new_repair_planner` — is the wrong process: it runs inside `LNSState.replica`
in a spawned worker (`parallel.py:241-251`), whose stderr is not the coordinator's log stream. A
warning nobody reads, fired once per worker, is functionally silent — which defeats the only reason
this warning exists.

---

## 8. What could go wrong — self-critique

**1. Phase 1 may not be enough, and the plan should say so up front.** Even with O(victims)
release, SIPP maintains four ledger-subscribed structures against compiled A*'s three — a **1.33x
(4:3)** commit-side ratio, or 5:3 on an arm with many kernel fallbacks (each binds `_cocc` too).
Against that, the plan side is ~0.5x in congestion. So the net turns on the plan:commit ratio,
which `[[sipp-commit-side-overhead]]` puts at 195 s commit against a faster plan *before* LNS
doubled the commit side. **This is the plan's central risk and it is not resolvable by design, only
by measurement.** The earlier "~2x" framing was wrong (it forgot A*'s `_cocc`) and overstated the
headwind; 1.33x is the honest prior, and it moves the expected answer toward "yes".

**2. A cheaper alternative exists and should be named so a reader does not re-derive it.** Nothing
forces `_sidx`/`_scocc` to be release-hooked *if the rebuild were cheap*. It is not (G1), so Phase 1
stands.

**3. `_claims[key].remove(...)` is O(list) and the list is per-cell.** A hot hub-mouth cell may
accumulate many claims. A* accepted this (`compiled_hex_occupancy.py:344`) because per-cell claim
lists are short relative to the schedule. If profiling disagrees, the fix is a `Counter` per key,
not a redesign — but do not pre-optimize; A*'s measured experience is the prior.

**4. The refcount conversion costs memory in LNS mode.** `set[int]` -> `dict[int, int]` is roughly
2x per entry, and `_rows` adds a journal linear in schedule size (A* measured its own at 32 B/row
packed, 185 B as tuples). Mitigations, both load-bearing: pack `_rows` as `array("q")` from the
first commit, and share the interned tuple between the journal and the `corr`/`cols` keys (§4.1's
2-tuple `_intern`). Measure **tree** RSS in the A/B —
`[[colgen-lazy-rows-are-materialize-not-loads]]` warns that reading RSS from the probe process is a
trap.

**5. `on_release` must tolerate a partial victim set, and it does.** `_rewind` has three call sites
(`state.py:558` exception, `:563` the **ordinary rejection path — the 79% of iterations §0 counts**,
`:638` `apply_delta`), so its `release_many(victims)` is a hot path, not an exceptional one. On the
common rejection path it is the *first* release of the newly committed volumes, so no double-fire.
The genuine double-release case is the `denied` break, where victims released at `state.py:525` were
never re-committed: `release_many` skips fids with no live runs (`ledger.py:285`), so `_rows.pop`
cannot `KeyError`. Correct, but invisible until it isn't — gate 5 in §9 pins it.

**6. Cost ties break differently, so there is no byte-parity gate against the A\* LNS run.** Unlike
#118's `search_workers=1` gate, a SIPP repair produces a different (equal-cost) route, so the
trajectory diverges from iteration 1. The available gates are weaker: equal *cost* per plan (§6's
tests), conflict-freedom (`verify_every`), and byte-parity of SIPP-with-release against
SIPP-with-rebuild (`--no-incremental`) — the direct analogue of PR #109's gate, and the one that
actually pins Phase 1.

**7. `[[analysis-scripts-are-not-maintained]]`** — `analysis/run_lns.py` gets a flag here; do not
repoint anything else in `analysis/`, and do not trust a neighbouring script's numbers.

---

## 9. Verification and measurement

**Correctness gates, in dependency order.** Gates 2-6 are unit-level and ported from existing A*
precedents on purpose: they fail *legibly*, whereas the full-trajectory gate 7 fails at iteration 1
with no locality.

1. **Phase 0:** existing `tests/test_sipp*.py` + `tests/test_lns*.py` green after the merge.
2. **Refcounts** — port `tests/test_lns.py:132` (`test_incremental_release_reference_service_refcounts`)
   to `SafeIntervalIndex`: two flights covering the same cells, removing one must NOT free them.
3. **Matches fresh absorb** — port `tests/test_lns.py:152` to `CompiledOccupancy`: after removing
   one flight, every `free_intervals_py` query matches a fresh instance that only ever saw the
   survivor. Run it per flight level.
4. **Equal spans are a fungible multiset** — port `tests/test_lns.py:180`. The direct pin for
   `_claims[key].remove()`'s semantics (§8.3).
5. **Slot reclaim** — port `tests/test_lns.py:204` (`test_pool_reset_cell_reclaims_overflow_slots`):
   `nslots` stays bounded across many destroy/repair cycles on the same cells. Unbounded growth has
   no other symptom until memory.
6. **New, no A\* precedent** — four failures this plan introduced and had to design around:
   - *static walls*: release a flight whose corridor touches an always-active hub's walled cells;
     assert the wall survives. Must fail before the §4.2 fix.
   - *`reset()` + journal*: with `track_removal=True`, commit two flights, force the shrink
     rebuild, release one; assert every query matches a fresh instance holding only the survivor.
   - *span overflow*: commit a volume whose step range exceeds `MAXS`; assert release leaves the
     pool identical to a fresh absorb of the survivors (the `_record` clamp, §4.2).
   - *evict residue*: commit, `evict_before(k)` above the flight's first recorded step, release;
     assert `corr`/`cols` are empty — no residue below `k` (the no-clamp decision, §4.1).
7. **Phase 1 headline:** `incremental_release=True` byte-identical to `incremental_release=False`
   for a SIPP repair — same trajectory rows, same final intents. PR #109's gate, re-run on SIPP.
8. **Phase 2:** `repair_planner="sipp"` runs sequentially, `verify_every=1`, conflict-free. Also
   confirm `tests/test_lns.py` raises no `ValueError` with `repair_planner_name="sipp"` —
   demonstrating (§6) that `_REPRODUCIBLE_PLANNERS` was never the gate.
9. **Bad name leaves the ledger intact:** construct an `LNSState` with
   `repair_planner_name="nope"`, assert `ValueError`, then assert `ledger._observers` and
   `ledger._release_subs` are still non-empty and `ledger.epoch` is unchanged. This is the pin for
   §5's validate-before-takeover ordering, and it fails silently without it.
10. **Phase 3:** the two cost-parity tests.
11. **Phase 4:** `last_envelope` is `None` after a native SIPP plan that follows an A*-fallback plan.
12. **Parallel:** `search_workers=1` with `repair_planner="sipp"` byte-identical to the sequential
    SIPP loop — #118's parity gate on the new arm. It is also what catches a worker that silently
    ran A* because the `replica(...)` forward (§5) was missed.

*Test hygiene, after Phase 1 lands:*

- Revisit whether gates 3+6 overlap gate 7 enough to move the slow trajectory run behind
  `@pytest.mark.slow`, as `test_compiled_replay_exact_big_dense_short_flights` already is
  (`tests/test_sipp_compiled.py:380`).
- Gate 6's *reset() + journal* case and gate 3's *matches fresh absorb* assert the **same oracle**
  (query-parity against a fresh instance holding only the survivor) by different entry paths. Keep
  both, but label the `reset()` one as the **defensive** unit test it is: under
  `incremental_release=True` that path is unreachable — `on_release` keeps `n_added` in lockstep, and
  `ledger.release` delegates to `release_many` whenever `_release_subs` is non-empty
  (`ledger.py:258-260`). A future reader must not mistake it for a live path and "simplify" the
  `reset()` clear back out.
- `tests/test_sipp_compiled.py:270` reads `sidx.corr.keys()` — the **only** cross-module reader of
  `corr` outside `sipp.py`. It is unaffected by §4.1's value-type switch (outer dict keys, and it
  builds a `track_removal=False` index), but re-run that file explicitly after §4.1 rather than
  trusting the full suite to surface it.

**The measurement that decides the default** (`analysis/run_lns.py`):

Paired A/B, same seed, same `neighborhood_size`, same iteration budget, `--repair-planner astar` vs
`sipp`, on `density_faa_wing_zipline`. Report per arm: loop wall, iterations/s, plan-side vs
commit-side split, improvement %, and AUC. Instrument `len(ledger._release_subs)` and
`len(ledger._observers)` after the first repair on each arm and report them alongside — that is the
direct check on §0's 4:3 claim, and it costs nothing.

Two scales, because `[[lns-neighborhood-size-is-scale-dependent]]` and
`[[drop-lns-parallel-implementation]]` both invert with instance size and a single cut would
mislead: the 1,526-leg cut and full 4,636. **`--demand-duration` and `--horizon` must be passed
together** — `analysis/run_lns.py:76-77` calls `ap.error` otherwise, so `--demand-duration 600`
alone aborts before the baseline runs. Follow the module docstring's paired shape
(`analysis/run_lns.py:3-8`, e.g. `--demand-duration 120 --horizon 1500`) and scale the horizon to
match.

Quote gains against the 19.3% that is actually delay, not against total cost
(`[[density-faa-delay-is-19pct]]`).

**Default stays `astar` unless SIPP wins on iterations/s at equal improvement-per-iteration.**
