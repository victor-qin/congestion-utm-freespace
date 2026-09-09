# HISTORY

Durable record of mistakes likely to recur between PRs. Format follows `.agent/CONTINUITY.md`.

- 2026-09-08T20:30Z `[TOOL]` A guard keyed by the wrong flight id fails SILENTLY through
  `dict.get()`, and a high rejection count is not evidence it guards the right thing. The paired-leg
  precedence guard held two dicts keyed by two different legs; the branch meant to catch an early
  RETURN matched 0 of 2,318 while the other branch over-fired. Keep ONE predicate and have every
  call site use it. Files: `freespace_sim/verify.py` (`pair_precedence_shortfall`),
  `freespace_sim/planner/lns/state.py` (`try_repair`).

- 2026-09-08T20:35Z `[TOOL]` A COUNT-based ratchet is identity-blind: LNS took precedence violations
  85 → 336 while worsening 256 pairs, 5 of which were already violating and 16 of which improved —
  none expressible as a count. Ratchet PER PAIR. Files:
  `freespace_sim/planner/lns/solver.py` (`_worsened_pairs`).

- 2026-09-08T20:40Z `[TOOL]` `verified` means SEPARATION only. A precedence violation holds disjoint
  pad windows (measured 49.3 s apart on pair 3906/3907), so no separation replay can see it and
  `verified=True` on every unflyable schedule. Do not read `verified` as "flyable". Files:
  `freespace_sim/verify.py`, `freespace_sim/sim.py`.

- 2026-09-08T20:45Z `[TOOL]` A checker gated on a knob that defaults to off runs never.
  `assert_incumbent_ok` was gated on `verify_every` (default 0) and `_finalize_lns_result` ran the
  separation replay but not the precedence one, so every LNS run ended with ZERO precedence
  verification. Files: `freespace_sim/planner/lns/solver.py`.

- 2026-09-08T22:55Z `[TOOL]` Changing how many requests a demand model emits RENUMBERS every flight
  and breaks fixtures that pin ids. Verify it is a pure renumbering (match old ids on t_request /
  origin / dest) before remapping, and check whether the deleted code consumed RNG draws. Files:
  `tests/test_hub_conflict_filed.py`, `freespace_sim/demand.py`.

- 2026-09-08T23:00Z `[TOOL]` An A/B script run from another directory silently imports the WORKSPACE
  tree. Assert `freespace_sim.__file__` names the intended checkout. Files: `.context/perf/*`.

- 2026-09-09T00:10Z `[ASSUMPTION]` Run only the tests that pertain to a change; the full suite is
  ~25 min single-core, 71% of it in 25 tests (`test_demand_hub` alone is ~28%). Use
  `-m "not slow"` while iterating and the full suite only before merge. Files: `pyproject.toml`.

- 2026-09-09T00:35Z `[TOOL]` A fixture that makes two definitions identical turns every test of their
  difference into a tautology. `test_realized_takeoff_is_the_column_start_not_the_first_waypoint`
  built `centerline[0][1] == volumes[0].t_start`, so `x < x + 1e-9` passed for any implementation.
  Build the fixture so the two CAN differ, and pin the regime a behavioural test needs. Files:
  `tests/test_paired_precedence.py`.
