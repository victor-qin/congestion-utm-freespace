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

- 2026-09-09T01:35Z `[USER]` A replacement that keeps the thing it replaces is not a replacement.
  The itinerary model landed +634/-203 because the two-request path was kept "so archived runs load"
  — but archived runs load fine without it, as the two independent flights they always were, and
  nothing outside the deleted code read the link. Check what actually depends on a legacy path before
  paying to keep it. Files: `freespace_sim/sim.py`, `freespace_sim/verify.py`,
  `freespace_sim/planner/lns/{state,solver,parallel}.py`, `freespace_sim/types.py`.

- 2026-09-09T03:10Z `[TOOL]` A failing assertion's MESSAGE is a hypothesis, not evidence.
  `test_lns_parallel` said "no accepted repair in 60 tries — pick a denser world"; density was not
  the cause (λ=1000 with 51/67 flights held still failed). The fixture picked victims by flight id,
  catching flights with delays [68,0,20,20,0,44] while the most-delayed six had [104,68,60,60,52,48].
  Measure what the fixture actually selected before believing what it says about the world. Files:
  `tests/test_lns_parallel.py`.

- 2026-09-09T03:15Z `[TOOL]` A test fixture that pins some config but inherits the rest breaks on any
  default change. `tests/test_colgen_solver._cfg` pinned flight levels, region and ground-delay cap
  but inherited `hover_time_s`, and its hand-derived step counts are computed on the column window
  that sets. Pin every knob an expectation was derived against. Files: `tests/test_colgen_solver.py`.

- 2026-09-09T03:20Z `[CODE]` `freespace_sim/planner/colgen/__init__.py` must keep `run_batch` /
  `ColGenSolver` behind its `__getattr__`. Importing them eagerly pulls SciPy into the geometry
  surface (against the module docstring) and defeats `monkeypatch.setattr(batch, "run_batch", ...)`,
  so `test_colgen_batch` runs the real Gurobi path and fails on a missing `gurobipy`. Files:
  `freespace_sim/planner/colgen/__init__.py`, `tests/test_colgen_batch.py`.
