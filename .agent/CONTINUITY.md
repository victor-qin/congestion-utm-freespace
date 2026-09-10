# CONTINUITY

## [PLANS]

- 2026-09-08T22:10Z `[USER]` Replace paired outbound/return requests with a single `FlightRequest`
  itinerary; remove `_est_trip_s`. Sequenced after PR #128 so #128 acts as the acceptance test.
- 2026-09-08T22:10Z `[USER]` Turnaround is a held pad reservation: full cylinder for the descent and
  the climb, LOW ground box for the dwell between them.
- 2026-09-09T01:30Z `[USER]` SUPERSEDES the deferral: the legacy two-request path is DELETED, not
  kept for archived runs. `paired_outbound_id`, `return_anchor` (both modes, the CLI flag, the
  coupling loop), `demand_turnaround_s`, the LNS anchor guard and every paired-precedence check in
  `verify` are gone. Archived runs still load — as two independent one-way flights, which is what
  they were; only the link is lost, and nothing outside the deleted code read it.

## [DECISIONS]

- 2026-09-08T22:12Z `[ASSUMPTION]` Legs are CHAINED, not searched jointly: leg 2 is planned against
  leg 1's outcome. A joint search needs a leg dimension in the A* state that the compiled kernel,
  the fixed-lane gates and the heuristic's admissibility corrections all key off. Costs joint
  optimality across legs — which the two-request scheme did not have either.
- 2026-09-08T22:40Z `[CODE]` `get_planner` wraps every per-flight planner in `ItineraryPlanner`;
  whole-schedule planners are not wrapped. Reach an inner planner via `iter_planner_chain`, not by
  peeling a fixed number of `.inner`s.
- 2026-09-08T23:05Z `[CODE]` `colgen` REFUSES a round-trip itinerary rather than pricing the outbound
  and dropping the return silently. `colgen_test` is one-way as a result.
- 2026-09-08T23:20Z `[ASSUMPTION]` The legacy two-request path still LOADS so archived runs work;
  `test_demand_hub` builds those inputs directly via `_LegacyPairModel`.

## [PROGRESS]

- 2026-09-08T22:55Z `[TOOL]` First `_replay` fixture attempt kept the pre-itinerary flight ids and
  failed with `KeyError`. Verified the change is a PURE RENUMBERING before remapping: all 53
  `dallas_hub_2uss_large` deliveries match old fid 2k on (t_request, origin, dest), because that
  scenario sets no `departure_offset_s` and the deleted return-lead draw consumed no randomness.
- 2026-09-08T23:30Z `[TOOL]` `colgen_test` going one-way halves its load (49 requests vs 98). Three
  colgen tests failed on the SAME root cause; the density-miniature one is now a strict xfail rather
  than a lowered threshold, because re-tuning λ is a research decision.

## [DISCOVERIES]

- 2026-09-08T22:35Z `[TOOL]` The joined centerline has a hole: the segment across the dwell is not
  flown. Zero-length in space, so distance sums and position interpolation were already correct, but
  the replay's per-segment corridor rebuild invented a box and fell back to explicit polygons on
  22/22 flights. Fixed by `OperationalIntent.leg_starts`.
- 2026-09-08T22:20Z `[TOOL]` Round trips accept at roughly half the one-way rate on a saturated
  2-hub world (28 vs 58 of 200) — expected, since a trip needs BOTH legs to fit. UNCONFIRMED at
  density_faa scale; that is the number deciding whether the model costs throughput.
- 2026-09-08T21:50Z `[CODE]` An itinerary's two legs are the same aircraft, so leg 2 is NOT
  deconflicted against leg 1. Under the two-request scheme the ledger policed them against each
  other, which was never physically meaningful.

- 2026-09-09T00:40Z `[TOOL]` The itinerary model's central claim was UNTESTED — no test asserted a
  planned return leg departs after its own arrival. Added four, plus a fixture assertion pinning the
  congested regime (29/29 returns held past service, up to 185.5 s); without it the test would pass
  vacuously the moment the fixture stopped congesting.

- 2026-09-09T01:35Z `[TOOL]` The itinerary change was +634/-203 before this: the model was added
  while the scheme it replaces was kept alive. Deleting it takes the branch to +610/-1200, a net
  -590. `realized_takeoff_s` survives — `ItineraryPlanner` uses it to size the ground box.

- 2026-09-09T02:10Z `[USER]` `hover_time_s` 30 -> 16 s, and `service_time_s` 16 s with `SimConfig`
  owning the number (`HubRadiusDemand.turnaround_s=None` inherits it). Every shipped scenario had
  pinned `turnaround_s=0.0`, so the delivery took ZERO time; they now inherit. A column is
  `hover_time_s + column_dwell_s` at BOTH ends of every leg, so the change moves every result.

- 2026-09-10T01:20Z `[TOOL]` xhigh review of #129 found 10 issues, all fixed. Two were severe:
  LNS repaired a round trip as its outbound leg alone (36/36 stranded, reported as a 66.93% gain,
  verified=True) and the detour metric measured two legs flown against a one-leg reference (11.2x).
  Both re-measured clean: 36/36 legs kept, gain 8.67%, detour exactly 1.0x.
- 2026-09-10T01:25Z `[CODE]` `reject_itinerary` now guards `AStarPlanner.plan` / `SIPPPlanner.plan`,
  so a round-trip request reaching an unwrapped planner raises. It immediately caught a third site
  a hand-audit had missed: `lns/unimpeded.py:_new_ruler`, whose ruler halved every round trip's
  unimpeded cost and so inflated the `delay()` premium that picks victims.

- 2026-09-10T15:05Z `[TOOL]` RE-DERIVED, already documented in issue #131's "What this does not
  fix" list. The pad dwell box is INVISIBLE to the A* lattice by construction, not by oversight. `hexgrid.rasterize_volume` yields `(q, r, L, s)` and `_levels_overlapped` keeps a level
  only if the volume's z-AABB reaches `z_L ± corridor_height/2`; the ground box is z=[0, 5] and
  `config.py:263` REQUIRES `lv[0] - corridor_height/2 >= ground_level_m`, so a mandatory gap exists
  between ground and the lowest level and the box lives inside it. Measured: 0 rasterized cells at
  both shipped level configs ((75,) and (30,70,110)); a hover column gives 77 / 210.
- 2026-09-10T15:05Z `[TOOL]` Tagging the dwell box to make it a column is a DOUBLE trap. (1) The
  occupancy arena is `2 * NC` — corridor `(c<<1)` and column `(c<<1)|1` — and `is_column` only routes
  cells to the other pool; it does not bypass `_levels_overlapped`, so a tagged ground box STILL
  rasterizes to zero. (2) `volumes_conflict` measured: untagged dwell vs a foreign same-pad landing
  column = True (correct), tagged = False. Tagging would blind the ledger check that works without
  fixing the lattice check that does not.
- 2026-09-10T15:05Z `[CODE]` Precedent for a level-free claim is `static_col` (bool[NC] marked at
  EVERY level per terminal hex, `compiled_hex_occupancy.py:391`), but it is deliberately
  time-INVARIANT — `window.py:16` folds it in exactly because it is step-independent. A foreign
  column claim is a hard wall in `blocked()` (`:455`), so columns block OVERFLIGHT; marking the dwell
  the same way would close the airspace above a parked aircraft for its whole turnaround, which is
  what the low box exists to prevent.
- 2026-09-10T15:05Z `[TOOL]` Rebased onto `ae797d2` (#130, 105 files): 21 hunks / 13 files. The
  dangerous ones were NOT conflicts — #130 wrote `Parameters` docs for `turnaround_s` / `return_anchor`
  on four functions whose parameters this branch deleted; different lines, so git merged them
  silently and left docs for arguments that do not exist (`lns/solver.py`, `lns/state.py`,
  `lns/parallel.py`, `sim.py`). #130 also consolidated the standalone sipp-shortcut envelope test into
  the parametrized one, but its `inner = sc.inner` is broken by `ItineraryPlanner`'s extra wrapper
  layer; replaced with `iter_planner_chain`. 262 passed on the touched surface.
- 2026-09-10T15:05Z `[USER]` SUPERSEDES "the cited figure is now tracked": PNGs are gitignored
  repo-wide and only `make_figures.py` is tracked. The `!context/figures/*.png` un-ignore is reverted,
  the PNG untracked, and the generator now lives in `make_figures.py` as `fig_itinerary_reservation`
  (a schematic with the real geometry baked in, matching every other `fig_*`).- 2026-09-10T15:40Z `[CODE]` CORRECTION: issue #131 is a REPRESENTATION change (a `Leg` dataclass,
  `ground_holds`, the stored schema, the replay payload) — it does NOT put legs in the A* search
  state, and it explicitly lists the dwell planning-order bug as out of scope. No open issue owns the
  fix that would let a search DELAY leg 2 instead of denying the trip. #131 + #11 do dissolve the
  tagging trap: with `DeconTag(uss, hub, kind)` and a `GROUND_PAD` kind, the hold can be tagged (so
  `TerminalCapacity` counts it) without inheriting the cylinder-vs-cylinder transparency, which today
  is keyed on shape alone.

## [OUTCOMES]

- 2026-09-10T16:55Z `[TOOL]` PR #129 is MERGEABLE / CLEAN: rebased onto `ae797d2`, three commits
  added (rebase-scar repair, the remaining review findings, docs), force-pushed over `0ab5ac4`.
  53 files, +1221/-1320 vs main. Full suite green on the rebased tree: 1,225 passed / 2 skipped
  (`-m "not slow"`) and 67 passed / 1 strict xfail (`-m slow`), both exit 0; `ruff check` clean.
  `git merge-tree --write-tree origin/main HEAD` reports no conflicts.
- 2026-09-10T16:55Z `[USER]` The PR BODY was as stale as the code: it still promised "the legacy
  two-request scheme still loads", which `bc52e99` deleted, and quoted a reservation block from the
  pre-`hover_time_s`-16 geometry. Rewritten against re-measured numbers (congested boxes 183.5-327.5 s
  over a 180 s turnaround, 37/37 trips, verified=True) plus sections the reviewers had never seen:
  the seven review findings, the `hover_time_s` 30->16 config move, the v1->v2 stored-schema
  migration, and what #130's rebase merged silently.

- 2026-09-08T22:07Z `[TOOL]` PR #128 merged: one predicate for paired-leg precedence. Armed nominal
  LNS 0.16% → 0.46%, anchor-rejects 1427 → 707, zero violations added.
- 2026-09-09T00:05Z `[TOOL]` PR #129 open: itinerary model. 1,333 passed, 2 skipped, 1 strict xfail.
  0 precedence violations structurally on a congested hub world.
- 2026-09-10T03:40Z `[TOOL]` Max-effort review of #129 (10 finder angles + sweep) surfaced 15
  findings, all measured. Six are subsumed by the legs model (issue #131); the four live-behavior
  ones are fixed on this branch and pinned by tests that were confirmed to FAIL on the unfixed code.
- 2026-09-10T03:45Z `[TOOL]` The pad hold was invisible to every planner's occupancy but visible to
  the ledger: `_levels_overlapped` returns `[]` for the `z=[0,5]` box, `ledger.any_conflict` returns
  True for a column over the same pad. `_compose` now checks it and denies `CONFLICT_FILED`.
- 2026-09-10T03:50Z `[TOOL]` Measured, unfixed and owned by #131: an unimpeded round trip on a single
  flight level books 200 m of phantom "traffic-forced" excess altitude (11/11 flights, `total_delay_s`
  10.9 -> 58.2 s) because `nominal_altitude_change_m` takes no leg count; a reloaded run loses
  `leg_starts` (0/21) and its detour goes 1.038x -> 2.953x.
- 2026-09-10T04:20Z `[CODE]` `_SPEC_SCHEMA_VERSION` is 2. An unstamped payload now reads as v1 rather
  than as current, and a v1 recipe carrying `paired_return_request=True` is refused — replaying it
  under v2 would halve the flight set and renumber every id.


- 2026-09-10T04:30Z `[TOOL]` #129 review fixes landed: 1242 passed, 2 skipped on the non-slow suite.
  README's `--return-anchor` section (a deleted flag) rewritten to the itinerary model; seven
  `analysis/` call sites of the removed parameter repaired; the cited figure is now tracked
  (`!context/figures/*.png`).
- Remaining, all owned by issue #131 or listed there as out of scope: the leg model + stored schema;
  round-trip throughput at density_faa scale; re-tune `colgen_test` λ.
