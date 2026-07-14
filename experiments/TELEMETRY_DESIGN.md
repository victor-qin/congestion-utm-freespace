# freespace-sim — Permanent Run Telemetry: design + handoff spec

**Goal.** Make the per-run instrumentation a *first-class, opt-in, portable, tracked* feature of the
simulator — not chat-dependent monkeypatches — so any run (CLI, sweep, CI) can cheaply archive the
congestion telemetry alongside the existing run artifacts.

**Home: the issue-25 metrics branch `claude/metrics-issue-25-plan-98c366`** (decided by the maintainer).
That branch already has uncommitted work modifying the exact files this feature extends —
`freespace_sim/metrics.py`, `freespace_sim/runs.py`, `experiments/run.py`, `experiments/readouts/*` — so
telemetry should be built ON TOP of / merged with that work, following its conventions (see §7), NOT on
a separate branch. **Delivery status: CORE STREAMS IMPLEMENTED.** Landed: `freespace_sim/telemetry.py`
(`TelemetryCollector` + `terminal_frame`/`conflict_frame`/`filed_volume_frame`), the `_file_deny` capture
at BOTH compiled + reference deny sites in `astar.py`, `sim.run(telemetry=…)` + collector wiring, `save_run`
persistence (`terminal_telemetry`/`conflict_events`/`filed_volumes`/`ledger_end` parquets + `has_telemetry`
index col), the terminal-membership round-trip in `scenario_frame`/`load_run`, `--telemetry` on
`experiments/run.py`, and `tests/test_telemetry.py` (telemetry-off is byte-identical). **DEFERRED as separate
follow-ups:** gate attribution (§3b.2 — the `gate_reject` field/columns exist but emit 0 until the
plan-outcome binding-gate derivation lands) and §9 kernel-parity (a separate `--kernel-parity` CI audit).
Revised after the plan-critic review + the #28 rebase — see §2, §3, §5.6, §10.

---

## 1. What already exists (do NOT rebuild)

- `freespace_sim/metrics.py` — pure `SimResult → DataFrame`. `flight_frame` (one row per intent:
  delays, costs, denial_reason, detour, solve_time, stretch, reserved volume-seconds), `aggregate`,
  `per_uss_frame`. Post-hoc; no hooks.
- `freespace_sim/runs.py::save_run(result, …)` — writes a portable, tracked archive:
  `config.json`, `env.json`, **`git.json`** (SHA/branch), `summary.json` (= `metrics.aggregate`),
  `scenario.parquet`, `trajectories.parquet`, `reservations.parquet`, `flights.parquet`,
  `per_uss.parquet`, `replay.html`; appends one row per run to `results/index.parquet`.
- `ledger.subscribe(cb)` — an existing commit hook (`TerminalCapacity` already uses it). **Reuse this
  for dwell capture — no new ledger plumbing.**
- `AStarPlanner` per-plan **solver-health counters** (merged to `main` in #28, `85a138b`): `last_expansions`,
  `_fb` / `_fb_reasons` (compiled-kernel fallback count + reason histogram: `own-foreign-overlap`,
  `empty-air`, …), `_ref_dispatch` (pre-kernel reference-dispatch reasons: `legacy-terminal`, `box-guard`),
  `_remask` (FB_MASK widen count). Already maintained each `plan()` — **free to read after the call**; a
  run-level rollup (fallback-rate / remask-rate / ref-dispatch-rate) is pure capture, no new plumbing. These
  measure *solver health* (like §9's parity), not congestion — a natural companion stream to it.

**Conclusion:** ~80% of "instrumentation" is already permanent & portable. Three *genuinely-ephemeral*
streams need live capture (below). A **separate-category** stream — numerical byte-exactness parity —
measures the *solver*, not the airspace, and is documented separately in **§9** (keep it separately gated).

## 2. What's ephemeral vs recoverable  (re-triaged after the #28 rebase)

**Correction (plan-critic MOD):** the pre-#28 draft said dwells need live capture because
"`ledger.evict_before` deletes old volumes mid-run." That method **does not exist** — the
`ReservationLedger` is append-only (only the A\* *acceleration* services — `occupancy.py` /
`compiled_hex_occupancy.py` / `terminal_capacity.py` — evict, and that's transient live state, not the
persisted archive). Committed terminal columns are already in `reservations.parquet`. Re-triaged:

| Stream | Recoverable post-hoc? | Where it comes from |
|---|---|---|
| **Per-hub dwell occupancy** | **YES** — every committed column is a `CylinderSpec` in `reservations.parquet` (`terminal_id`, `t_start`, `t_end`); sweep-line for `peak_pad_occupancy`. **No live hook.** | `reservation_frame` (already saved) |
| **Gate attribution** (pad / air / lane) | **NO** — transient plan-time predicate outcomes, never stored | `admits` / `column_clear` / `exit_clear`, per accepted flight-endpoint |
| **Conflict culprits** (the blocker) | **NO** — the blocking volume is known only at the deny site | `ledger.conflicts(volumes)` at the deny check |
| **Filed volumes for ANY built-corridor denial** (the *rejected corridor itself*) | **NO** — discarded at every deny site that ran `_build` | the `volumes` in scope at each corridor-building `_deny` |
| **Per-flight terminal membership** (which hub each flight used) | **NO** for denied flights — not in `scenario.parquet`, not round-tripped by `load_run` | `request.origin_terminal`/`dest_terminal` |

Only these need first-class handling. The filed-volumes stream — **the flight's own filed-but-rejected
volumes** — is NEW (answers "track the volumes filed, at least for errors") and must cover **every** deny
site that built a corridor, not just `conflict_filed` (see §3b.3): the culprit tells you *what blocked it*,
the filed volumes tell you *what it tried to fly*. The terminal-membership row is a fix to the EXISTING
persistence, independent of telemetry. Full gap→mechanism matrix in §10.

## 3. Design

### 3a. New module: `freespace_sim/telemetry.py`
```python
@dataclass
class TelemetryCollector:
    enabled: bool = True
    # terminal metadata SNAPSHOT — captured at sim.run setup where the demand model is in scope (§3c),
    # because save_run only receives a demand *string*, not the model (plan-critic MOD).
    terminals: dict[Hashable, dict] = field(default_factory=dict)   # tid -> {cx,cy,capacity,radius,type}
    # gate attribution PER ACCEPTED FLIGHT-ENDPOINT (NOT per search node — plan-critic MOD): the binding
    # constraint over the waited interval. kind: 0=pad(admits) 1=air(column_clear) 2=lane(exit_clear).
    gate_reject: dict[Hashable, list[int]] = ...   # tid -> [pad, air, lane]
    conflict_events: list[dict] = field(default_factory=list)       # one row per CULPRIT (blocker)
    filed_volumes: list[dict] = field(default_factory=list)         # the REJECTED corridor's own volumes

    # NOTE: no on_commit/dwells hook — per-hub dwell occupancy is recovered post-hoc from
    # reservations.parquet (see §2, §3e). This collector captures only the non-recoverable streams.

    def on_gate_reject(self, tid, kind):                # called once per accepted flight-endpoint, §3b.2
        self.gate_reject.setdefault(tid, [0, 0, 0])[kind] += 1

    def on_deny(self, flight_id, reason, volumes, hits=None):   # EVERY deny that built a corridor
        for j, v in enumerate(volumes or []):           # the FILED (rejected) corridor — error forensics
            self.filed_volumes.append({"flight_id": flight_id, "reason": reason,
                                       "vol_idx": j, **_vol_row(v)})     # runs.reservation_frame schema
        for fid, vol in (hits or []):                   # the culprit(s) — conflict denials only
            self.conflict_events.append({
                "flight_id": flight_id, "culprit_fid": int(fid),
                "culprit_tid": vol.terminal_id, "shape": type(vol.shape).__name__,
                "t_start": vol.t_start, "t_end": vol.t_end})
```
`_vol_row(v)` is the per-`Volume4D` geometry dict already used by `runs.reservation_frame` (box: center/rot/ext;
cyl: cx/cy/radius/z_lo/z_hi) — factor it out of `runs.py` so filed volumes reuse the exact same schema.

### 3b. Emit points (small, guarded, zero-overhead when disabled)

1. **Dwells — NO live hook** (plan-critic MOD). Recovered post-hoc from `reservations.parquet`; the
   sweep-line lives in `terminal_frame` (§3e). Removed the `ledger.subscribe(on_commit)` wiring.
2. **Gate attribution — at flight-endpoint granularity, NOT per predicate call** (plan-critic MOD).
   `admits`/`column_clear`/`exit_clear` are evaluated *speculatively per A\* search node* (dozens–hundreds
   per delayed flight), so hooking their `False` returns counts search effort, not ground holds. Instead,
   emit once when the plan resolves an accepted takeoff/landing step: record which gate was the *binding*
   constraint over the waited interval — pad (`admits`), foreign-air (`column_clear`), or exit-lane
   (`exit_clear`) — via `self._tele.on_gate_reject(tid, 0|1|2)` at the plan-level outcome in
   `_plan_reference`/`_plan_compiled`. (`TerminalCapacity` gets `self._tele=None` only if a probe-level
   counter is also wanted; the *authoritative* per-flight attribution is at the plan outcome.)
3. **Filed volumes + culprits — at EVERY corridor-building deny site.** A denial discards a *built*
   corridor at three site kinds (all AFTER `self._build(...)`): the detour-factor `budget_exceeded`
   (built, no blocker), the reference `conflict_filed` (`_plan_reference` ~403-404), and the compiled
   `conflict_filed` (`_plan_compiled` ~892-893) — BOTH conflict sites live post-#28 (CRIT, §5.6). Route
   all of them through one helper that records geometry (+ blocker for conflicts) then denies unchanged:
   ```python
   def _file_deny(self, req, reason, volumes, ledger=None):
       if self._tele is not None:
           hits = ledger.conflicts(volumes) if reason is DenialReason.CONFLICT_FILED else None
           self._tele.on_deny(req.flight_id, reason.value, volumes, hits)
       return _deny(req, reason)          # denied intent UNCHANGED → verify/reservations stay byte-exact
   ```
   The no-goal `budget_exceeded`/`search_exhausted` sites built no corridor → nothing to file (their
   `flights.parquet` row is the record). Because the returned intent still carries `volumes=None`,
   telemetry-off is byte-identical and `verify`/`reservation_frame` never see a rejected corridor. *(A
   mirror hook in `mechanism.py` covers the future multi-USS `conflict_at_commit` — inert in
   single-threaded v0, where the commit re-check never fails.)*

### 3c. Wiring
`sim.run(cfg, *, telemetry: bool|TelemetryCollector = False, …)`:
- build `collector` if truthy.
- **snapshot terminals** into `collector.terminals` here — this is the ONE place `demand.terminals(cfg)`
  is in scope (sim.py already reaches it for `terminal_airspace_always_active`); fall back to harvesting
  `request.origin_terminal`/`dest_terminal` off accepted intents for non-hub demands (omits zero-traffic
  hubs). This is why `terminal_frame` can't call `demand.terminals` later — save_run only has a string.
- **attach `_tele` to EVERY nested `AStarPlanner`** (plan-critic MOD), not just the top-level planner:
  reuse `sim.py`'s existing `_reaches_astar` walk (it already descends `.inner`/`.warm_planner`) so
  `astar_shortcut` / `opt_astar` / `astar_milp` also emit. Each A\* forwards `_tele` to its `_tcap` at build.
- **no `ledger.subscribe`** — the dwell hook is gone (§3b.1).
- `SimResult` gains `telemetry: TelemetryCollector | None = None` (default None → today's behavior).

### 3d. Persistence (extend `runs.py::save_run`)
Guarded by `if result.telemetry is not None:` — write the new parquet artifacts (all `compression="zstd"`):
- **`terminal_telemetry.parquet`** — one row per hub, joining `collector.terminals` (the run-time
  snapshot) + a sweep-line over the accepted terminal-tagged cylinders (from `reservation_frame`) +
  per-hub flight stats: `tid, type, cx, cy, pads, radius, dist_to_edge_m, n_departures, n_arrivals,
  peak_pad_occupancy, pad_reject, air_reject, lane_reject, mean_ground_delay_s, max_ground_delay_s`.
- **`conflict_events.parquet`** — one row per culprit: `flight_id, culprit_fid, culprit_kind
   (static_wall|sibling|foreign), culprit_tid, shape, t_start, t_end`.
- **`filed_volumes.parquet`** — the rejected corridor geometry for every `conflict_filed` flight (error
   forensics; same geometry schema as `reservations.parquet`, keyed by `flight_id`, `vol_idx`). Answers
   "track the volumes filed, at least for errors" — with this + `conflict_events` you can render the
   denied corridor AND the blocker it hit, from disk, no re-run.
- **`ledger_end.parquet`** — the always-active terminal WALLS (`ledger._static_vols`), which
   `reservation_frame` does NOT capture (it only walks accepted intents). Accepted-flight volumes (already
   in `reservations.parquet`) + these walls == the FULL end-of-run ledger. See §10.
- (Optional, debug-only, **non-portable**) `ledger.pkl` — see §10 for why parquet is the tracked format
   and pickle is a local-debug escape hatch only.

### 3e. Rollup helpers (`telemetry.py` or `metrics.py`)
- `terminal_frame(result) -> DataFrame` — the per-hub table above. `peak_pad_occupancy` = sweep-line max
  overlap of the accepted **terminal-tagged cylinders** (from `reservation_frame`, grouped by
  `terminal_id` — NOT a live `dwells` map). `dist_to_edge_m` = `min(cx, W-cx, cy, H-cy)` from `cfg`
  region. Delay/gate stats over the #25 steady-state window (§7).
- `conflict_frame(result) -> DataFrame` — one row per culprit; `culprit_kind = static_wall if
  fid==STATIC_WALL_FID (ledger.py:65 — now LIVE post-#28) else sibling if culprit_tid==the filed flight's
  hub else foreign`.
- `filed_volume_frame(result) -> DataFrame` — the rejected corridors (error forensics), joinable to
  `conflict_frame` on `flight_id`.

### 3f. Entry point (chat-independent, reproducible)
**EXTEND the existing `experiments/run.py`** (the metrics branch already modifies it) — add a
`--telemetry` flag that flips `sim.run(..., telemetry=True)` and lets `save_run` persist the extra
artifacts:
```
uv run python -m experiments.run --scenario dallas_full --planner astar --telemetry --tag df_astar
```
Reproducible from `git.json` + `config.json`. Telemetry readouts (per-hub delay bars, pad-vs-air split,
edge-hub scatter) belong as new scripts under `experiments/readouts/` next to `compare.py` / `curve.py` /
`histograms.py`.

## 4. Exact change list  (updated for the plan-critic findings + #28 rebase)
- **ADD** `freespace_sim/telemetry.py` (`TelemetryCollector` + `terminal_frame` + `conflict_frame` +
  `filed_volume_frame`).
- **EDIT** `freespace_sim/runs.py`: factor a `_vol_row(v)` geometry helper out of `reservation_frame`
  (reused for `filed_volumes`); `save_run` writes `terminal_telemetry` / `conflict_events` /
  `filed_volumes` / `ledger_end` parquets (zstd) when `result.telemetry`; index gains `has_telemetry`.
- **EDIT** (prerequisite, NOT telemetry-gated) `freespace_sim/runs.py`: `scenario_frame` writes
  `origin_terminal`/`dest_terminal` (id + capacity + radius); `load_run` restores them onto the rebuilt
  `FlightRequest`. Without this, **denied** hub flights lose their hub membership, and even a re-run from a
  loaded folder can't reconstruct it (a completeness gap in the existing archive, independent of telemetry).
- **EDIT** `freespace_sim/planner/astar.py`: add `self._tele=None`; a `_file_deny(req, reason, volumes,
  ledger)` helper at the **three corridor-building** deny sites (detour `budget_exceeded`; `conflict_filed`
  reference 403-404 AND compiled 892-893 — CRIT); keep bare `_deny` at the no-goal sites (nothing built);
  emit per-flight gate attribution at the plan outcome; forward `_tele` to `self._tcap`.
- **EDIT** `freespace_sim/sim.py`: `run(..., telemetry=…)`; build collector; **snapshot terminals**
  (only place `demand.terminals(cfg)` is in scope); attach `_tele` to every nested A\* via `_reaches_astar`;
  put collector on `SimResult`. (No `ledger.subscribe` — dwell hook dropped.)
- **EDIT** `freespace_sim/planner/terminal_capacity.py`: OPTIONAL `self._tele=None` only if a probe-level
  counter is also wanted (the authoritative attribution is at the plan outcome — §3b.2).
- **EDIT** `experiments/run.py`: add the `--telemetry` flag (merge with the #25 edits, don't clobber).
  New readouts under `experiments/readouts/` (per-hub bars, pad/air/lane split, edge scatter, conflict map).
- **ADD** `tests/test_telemetry.py`: telemetry-off == byte-identical `flights.parquet` (parametrized over
  `{astar, astar_ref}`); conflict + filed-volume capture fires on the **default compiled** path (guards the
  CRIT); `peak_pad_occupancy` from `reservation_frame` == brute-force overlap (no live hook); culprit
  classification incl. `static_wall`; `terminal_frame` includes a zero-traffic hub (proves run-time
  snapshot); save_run/load_run round-trips the parquets.

## 5. Self-review (risks → mitigations)
1. **Hot-path overhead** (admits/column_clear called millions of times). → Disabled by default; the guard
   is one `is None` check when off (≈free). Only telemetry runs pay the increment. Measure: A/B a
   telemetry-on vs -off dallas_full; expect <10% and only when on.
2. **Byte-exactness** — hooks must be pure observers. → They only read + increment; no control-flow
   change. `compiled==reference` and node-parity tests must still pass. The `_file_conflict` refactor is
   behavior-identical (same `_deny`). Regression test: telemetry-off produces identical `flights.parquet`.
3. **Dwells are recoverable, not ephemeral** (corrected) — no `ledger.evict_before` exists; committed
   columns are in `reservations.parquet`, so `peak_pad_occupancy` is a post-hoc sweep-line: no live hook,
   no added byte-exactness surface. ✓
4. **Culprit + filed-volume correlation** — emitted at the conflict site with `req` and `volumes` in scope
   → exact (no ordering guess, unlike the monkeypatch); captures both the blocker and the rejected corridor. ✓
5. **Static walls — culprit path now LIVE post-#28.** `register_static_terminal` files walls into
   `ledger._static_vols` (NOT via `commit`), and `conflicts()` surfaces wall hits as
   `(STATIC_WALL_FID=-1, wall_vol)` ([ledger.py:65](../freespace_sim/ledger.py),
   [:186](../freespace_sim/ledger.py)). So `_file_conflict` correctly attributes `culprit_kind=static_wall`.
   (Pre-#28 the constant didn't exist — the earlier plan-critic flagged it dead; the rebase makes it real.)
   Walls are excluded from dwell occupancy (dwells come from accepted intents in `reservations.parquet`,
   not the static set), and they ARE captured for the end-of-run ledger via `ledger_end.parquet` (§3d,§10). ✓
6. **Conflict-site coverage — #28 is ALREADY MERGED (both sites live).** ⚠️ CRIT (plan-critic). This
   branch was rebased onto `main` `85a138b` (PR #28), so **BOTH** `conflict_filed` sites now exist and
   MUST be wired in this change: the reference `_plan_reference` site (`astar.py:403-404`) AND the
   compiled `_plan_compiled` site (`astar.py:892-893`). **Compiled is the DEFAULT**
   (`AStarPlanner(compiled=True)`), so wiring only the reference site would leave
   `conflict_events.parquet`/`filed_volumes.parquet` EMPTY for the plan's own example command
   (`--planner astar` → compiled → line 893). `_file_conflict` is a method on `AStarPlanner`, so
   `self._file_conflict(req, ledger, volumes)` resolves at both sites — apply it to both, no follow-up.
7. **Overhead-free default** — `telemetry=False` path is byte-identical to today; existing analysis
   scripts unaffected.

## 6. Persisted schema quick-reference (what gets saved)
```
results/<run>/terminal_telemetry.parquet   # per hub
  tid  type  cx  cy  pads  radius  dist_to_edge_m
  n_departures  n_arrivals  peak_pad_occupancy
  pad_reject  air_reject  lane_reject         # binding gate per accepted flight-endpoint
  mean_ground_delay_s  max_ground_delay_s
results/<run>/conflict_events.parquet        # per culprit of a conflict_filed
  flight_id  culprit_fid  culprit_kind  culprit_tid  shape  t_start  t_end
results/<run>/filed_volumes.parquet          # the REJECTED corridor geometry (error forensics)
  flight_id  vol_idx  kind  cx cy cz  rot ext  radius z_lo z_hi  terminal_id  t_start t_end
results/<run>/ledger_end.parquet             # always-active terminal WALLS (rest of the end ledger)
  <same geometry schema as reservations.parquet>
results/index.parquet  += has_telemetry
```
Everything else (per-flight delays/denials/detour/cost, scenario, trajectories, `reservations` = accepted
volumes, config, git SHA) is ALREADY written by `save_run` today. `reservations.parquet` (accepted) +
`ledger_end.parquet` (walls) == the full end-of-run ledger — see §10.

## 7. Composition with issue-25 (steady-state window) metrics — why this is the right home

Issue #25 gives `metrics.py` a **twin-window** convention: report each headline over BOTH the whole run
AND a density-plateau (steady-state) window (adaptive smoothing on median trip width), because delay only
exceeds the whole-run mean in the congested regime. The telemetry rollups must follow the same
convention or they'll disagree with the branch's headline numbers:

- **`terminal_frame` per-hub delay + gate-attribution should be computed over the SAME steady-state window**
  the branch uses for `aggregate`, not just whole-run — a hub's pad/air split during ramp-up differs from
  its plateau split. Reuse the branch's window-selection helper (whatever `metrics.py`/`experiments` grow
  for #25) rather than re-deriving the plateau.
- **`peak_pad_occupancy`** is inherently whole-run (the max is the max); report it as-is, but ALSO report
  plateau-mean occupancy for the twin.
- **`conflict_events`** are timestamped (`t_start`); tag each with in-window / out-of-window so the
  conflict_filed rate can be read on the plateau too.
- The two new parquets slot into the branch's `save_run` rework and its `experiments/readouts/` scripts
  (a `readouts/terminal.py` for the per-hub bars + edge scatter, mirroring `histograms.py`).

**Merge discipline:** the branch has UNCOMMITTED edits to `metrics.py`, `runs.py`, `experiments/run.py`.
Build telemetry as additive functions/artifacts on top of those edits — do not revert or duplicate the
#25 window logic; call into it.

## 8. Where the current (throwaway) instrumentation lives — for reference / to port
The chat-session scratchpad monkeypatch scripts capture the SAME data ad-hoc and are the reference impl:
- `run_df1800_astar_instrumented.py` — the four wrappers (commit→dwells, admits/column_clear→gate
  attribution, plan+any_conflict→conflict culprits) + the per-flight/hub/region capture. Port these
  wrappers into `TelemetryCollector` hooks (§3) — the capture logic is identical; only the plumbing
  becomes first-class.
- `df1800_astar_instr.pkl` / `df1800_shortcut.pkl` — the emitted data shape the parquets should match.

## 9. Numerical-parity telemetry (kernel byte-exactness) — a SECOND, distinct category

Streams 1–3 measure the **simulation** (congestion physics). This stream measures the **solver's
numerical fidelity**: how often floating-point associativity in the A* g-value accumulation could break
the `compiled == reference` byte-exactness guarantee. Different audience (kernel engineering / CI),
different gating, and it only makes sense on the **reference** planner (the oracle the kernel is checked
against). Fold it into the same `TelemetryCollector` framework (opt-in, observer-only) but as a separate
stream behind its own flag — do NOT bundle it with the congestion telemetry.

Originating from a parallel session's throwaway instrumentation (now in `astar-speedup` `stash@{0}`); this
section makes it first-class and fixes its bugs.

### 9a. The question it answers ("A1" associativity)
IEEE-754 `+` is not associative. **Takeoff edges are the only edges built from two non-trivial operands:**
`a = takeoff_cost[L]` (climb-to-level cost) and `b = cost_air_lateral_per_m·(lane.dist − o_r)` (exit-lane
lateral cost). The reference forms `cost = a + b`, then `ng = base_g + cost` — i.e. the grouping
`base_g + (a + b)`. A compiled kernel that instead accumulates `(base_g + a) + b` differs by ≤1 ULP on
some edges; that 1-ULP delta can flip the `if ng < g.get(nst)` relaxation test or the `(f, counter)` heap
tie-break → a different but equal-cost path → byte-exactness fails. The telemetry counts the mismatch
rate to **bound the risk surface** and prove the kernel must replicate the reference's exact grouping. A
parallel session measured **~3.7% of takeoff edges** differ — non-negligible. (This is the empirical
backing for the plan's #1 kernel-correctness risk: tie-break / float parity.)

**Calibration (measured on `astar-speedup` before merge).** That ~3.7% is the *risk surface* — edges where
the two groupings differ by ≤1 ULP — **not** a realized-failure rate. Two levels of measurement:
- *arithmetic sweep* (isolated, over realistic `base_g` / `takeoff_cost` / lane-offset ranges): **0.71%** of
  triples differ, by ~2×10⁻¹³;
- *in-search* (saturated `region=(8000,6000)`, `λ=9000`, `pads=1`, seed 1): **10,288 / 278,997 = 3.7%** of
  takeoff-lane relaxations differ → exactly the `_STATS[1]/_STATS[0]` this stream reports.

But a direct compiled-vs-reference A/B over **2,650 saturated flights flipped ZERO heap `(f, counter)`
ties** — identical `last_expansions` / centerline / cost. So `assoc_mismatch_rate` reads ~3.7% **even on a
correct kernel**: it bounds what a wrong grouping *could* cost and is the number to watch if the cost
structure ever makes exact ties common — it is **not itself a divergence count**. (#28's merged kernel now
uses the reference's exact `base_g + (a + b)` grouping, so realized divergence is 0 by construction — keep
this stream as a periodic CI *audit* of that surface per §9d, not a pass/fail gate.)

### 9b. Reference implementation (verbatim, from `astar-speedup` `stash@{0}`)
Two coupled edits in `_plan_reference`. **Producer** at the fixed-lane takeoff-edge generation — splits the
edge cost into its two operands and stashes them keyed by the successor state (the emitted cost `a + b` is
byte-identical to the original single expression, so reference behavior is unchanged):
```python
# was: out.append(((\"a\", lq, lr, L, ts),
#                  takeoff_cost[L] + cfg.cost_air_lateral_per_m * (lane.dist - o_r)))
_a_ = takeoff_cost[L]; _b_ = cfg.cost_air_lateral_per_m * (lane.dist - o_r)
_AB[(\"a\", lq, lr, L, ts)] = (_a_, _b_)
out.append(((\"a\", lq, lr, L, ts), _a_ + _b_))
```
**Consumer** at the relaxation step — looks the operands back up and compares the two association orders:
```python
ng = base_g + cost                 # cost == a + b for this takeoff edge
_ab = _AB.get(nst)
if _ab is not None:                # successor came from an instrumented takeoff edge
    _STATS[0] += 1                 # count: takeoff-edge relaxations examined
    if ((base_g + _ab[0]) + _ab[1]) != (base_g + (_ab[0] + _ab[1])):
        _STATS[1] += 1             # count: (base_g+a)+b  differs from  base_g+(a+b)
```
`_STATS[1] / _STATS[0]` is the mismatch rate.

### 9c. Making it first-class (and fixing the 3 bugs in the stashed version)
- **Collector fields:** `assoc_examined: int`, `assoc_mismatch: int` — plus a **plan-local** operand map
  (NOT module-global) reset at the top of each `_plan_reference`.
- **Emit points:** the producer records `(a, b)` into the plan-local map; the relaxation compares +
  increments the collector. Guard both with `if self._tele is not None and self._tele.kernel_parity:` so
  they're free when off.
- **Bugs this fixes** (all present in the stashed version):
  1. `NameError` — `_AB`/`_STATS` were never module-initialized; a first-class collector field can't be
     undefined.
  2. **Unbounded growth** — the module-global `_AB` was never cleared, so it accreted every takeoff state
     across all 34k plans. A plan-local map bounds it to one plan.
  3. **Cross-flight key reuse** — the same `("a", q, r, L, ts)` key recurs across flights with *different*
     `lane.dist`, so a global `_AB` could compare a relaxation against another flight's operands. Plan-local
     scoping makes producer→consumer within one plan exact.
- **Optional richness:** bucket `assoc_mismatch` by flight level `L` and record the max ULP gap, to see
  whether mismatches concentrate at a level or a magnitude regime.
- **Reference-only:** it validates the kernel, so it runs under `astar_ref` (`compiled=False`); on the
  compiled path there is no `_plan_reference` accumulation to instrument.

### 9d. Persistence + gating
- Scalar summary, not per-flight → write **`kernel_parity.json`** (`{n_examined, n_mismatch, mismatch_rate,
  by_level?}`); add an **`assoc_mismatch_rate`** column to `index.parquet` for cross-run/CI tracking. No
  parquet table needed.
- **Separately gated** (`--kernel-parity`, or `telemetry="parity"`), **off by default**, and NOT run every
  experiment — it's a periodic/CI byte-exactness *audit* on the reference planner, not a per-experiment
  research metric. Its hot-path overhead lives entirely on the (already-slow) reference path, so it never
  touches production/compiled runs.

### 9e. Self-review
- **Observer-only invariant:** the producer must keep `out.append(..., _a_ + _b_)` — the split into
  `_a_`/`_b_` is naming only; the emitted edge cost stays `a + b`, so the reference remains a faithful
  oracle. A reviewer must confirm the refactor didn't change the emitted value (it doesn't).
- **Generalization:** today only takeoff edges combine two operands; if a future edge type gains multiple
  cost terms, extend the same operand-stash pattern (or the audit will silently miss that edge's
  associativity).
- **It does not replace the exactness TESTS** — `compiled == reference` byte-identity + node-parity remain
  the gate. This telemetry *explains and sizes* a divergence risk; it doesn't certify absence of one.

## 10. Error forensics, end-of-run ledger & storage format  (answers to the design questions)

### 10.0 Gap-coverage matrix — every persistence gap → its mechanism

Audited against the current-archive completeness analysis (accepted flights are ALREADY fully captured —
request/departure times, reserved volumes, flown centerline, outcomes; the gaps are denials, walls, gate,
and terminal membership):

| Gap (not stored today) | Captured by | Artifact |
|---|---|---|
| Denied corridor geometry — `conflict_filed` (ref **and** compiled sites) | `_file_deny` → `on_deny(volumes, hits)` | `filed_volumes.parquet` |
| Denied corridor geometry — detour `budget_exceeded` (built then over-detour) | `_file_deny` → `on_deny(volumes)` | `filed_volumes.parquet` |
| Denied corridor geometry — `conflict_at_commit` (multi-USS; v0-inert) | `mechanism.py` mirror hook | `filed_volumes.parquet` |
| The blocker(s) a conflict hit | `ledger.conflicts(volumes)` at the deny site | `conflict_events.parquet` |
| Always-active terminal **walls** | `ledger._static_vols` dump | `ledger_end.parquet` |
| Gate attribution (pad / air / lane) | per-flight-endpoint `on_gate_reject` | `terminal_telemetry.parquet` |
| Per-hub metadata (pads, radius, center), incl. **zero-traffic** hubs | run-time terminal snapshot (§3c) | `terminal_telemetry.parquet` |
| **Per-flight terminal membership** | `origin_terminal`/`dest_terminal` columns | `scenario.parquet` (+ `load_run` restore) |
| No-goal `budget_exceeded` / `search_exhausted` | (no corridor was ever built) | `flights.parquet` (already) |

With these, a saved run captures **everything that happened — accepted and denied — without a re-run.** The
last two non-accepted denial reasons genuinely have no geometry to store (search never reached a goal), so
their `flights.parquet` row + `denial_reason` is the complete record.

---

**Q: Will it track the volumes filed, at least for errors?** **Yes** — `filed_volumes.parquet` (§3d) stores
the *rejected corridor's own* `Volume4D`s for every `conflict_filed` flight (the geometry A\* built, then
had to deny), keyed by `flight_id`+`vol_idx`, in the same analytical schema as `reservations.parquet`.
Paired with `conflict_events.parquet` (the blocker), a readout renders **both** the denied corridor AND
what it collided with — from disk, no re-run. (Only `conflict_filed`/`conflict_at_commit` reach a *built*
corridor; `budget_exceeded`/`search_exhausted` never found a goal, so they have no filed volumes — their
`flights.parquet` row + denial reason is the whole record.)

**Q: The ledger state at the end of the run?** Fully captured, in two pieces:
- **accepted flight volumes** — already in `reservations.parquet` (every committed corridor + hover column;
  the ledger is append-only, so nothing is ever deleted — §2).
- **always-active terminal walls** — NEW `ledger_end.parquet` (`ledger._static_vols`), which
  `reservation_frame` doesn't walk.

`reservations.parquet` ∪ `ledger_end.parquet` == the exact `ReservationLedger` `_vols` + `_static_vols` at
run end. A `load_ledger(run_folder)` helper can rebuild a live `ReservationLedger` from those two (reusing
the existing `_volume_from_row`), so you can replay `any_conflict`/`conflicts` against the final airspace.

**Q: What storage should we expect?** Structured columnar parquet (zstd); sizes dominated by the ALREADY-
existing `reservations.parquet`:
- `reservations.parquet` (existing): ~accepted × ~40 volumes/flight. dallas_full (~30-35k accepted) ⇒ ~1M+
  rows, tens of MB compressed. Telemetry adds little on top of this.
- `filed_volumes.parquet`: ~(conflict_filed denials) × ~40 volumes → a few hundred-k rows, single-digit MB.
- `conflict_events` / `terminal_telemetry` / `ledger_end`: thousands of rows or fewer — sub-MB each.

Net: a dallas_full telemetry run grows the folder by ~single-digit MB over today's ~tens of MB.

**Q: Can we write things to pickle (e.g. the ledger as a pickle)?** Technically yes (`Volume4D`/`BoxSpec`/
`CylinderSpec` are plain dataclasses), but **do NOT use pickle for the tracked archive** — for the same
reasons `save_run` already serializes analytical geometry to parquet:
- **Not portable / not durable** — pickle is Python-version- and class-definition-specific; the moment
  `Volume4D` or a geometry spec changes, old pickles fail to load. The HF-Hub run store (issue #12/#13) and
  its in-browser Dataset Viewer only understand parquet/json.
- **Not inspectable + a security risk** — `pickle.load` executes arbitrary code; a *synced* store of
  pickles is a liability. Parquet is inert, columnar, queryable without our code.
- **Redundant** — the ledger is already fully reconstructible from parquet (accepted reservations +
  `ledger_end`), so a ledger pickle stores nothing new, just more fragilely.

**Escape hatch:** a `--dump-ledger` debug flag MAY write `results/<run>/ledger.pkl` for a quick local
`pickle.load` mid-debugging — but it's explicitly non-portable, excluded from the synced store, and never
the source of truth. Prefer `load_ledger(folder)` (parquet → live ledger) for anything reproducible.
