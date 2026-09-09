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

## [OUTCOMES]

- 2026-09-08T22:07Z `[TOOL]` PR #128 merged: one predicate for paired-leg precedence. Armed nominal
  LNS 0.16% → 0.46%, anchor-rejects 1427 → 707, zero violations added.
- 2026-09-09T00:05Z `[TOOL]` PR #129 open: itinerary model. 1,333 passed, 2 skipped, 1 strict xfail.
  0 precedence violations structurally on a congested hub world.
- Remaining: measure round-trip throughput at density_faa scale; re-tune `colgen_test` λ; delete the
  legacy two-request path.
