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

- 2026-09-10T01:25Z `[TOOL]` When a request type needs a wrapper to be planned correctly, ask "who
  constructs a planner WITHOUT going through `get_planner`?" and guard the shared entry point rather
  than the one planner you were thinking about. A hand-audit found two of the three sites; the guard
  in `AStarPlanner.plan` / `SIPPPlanner.plan` found the third (`lns/unimpeded.py:_new_ruler`) on its
  first run. Files: `freespace_sim/planner/itinerary.py` (`reject_itinerary`),
  `freespace_sim/planner/lns/{state,unimpeded}.py`.

- 2026-09-10T01:30Z `[TOOL]` A cost-comparing search will ADOPT a plan that silently lost work: the
  one-way plan LNS produced for a round trip cost about half, so `try_repair` read the deleted return
  leg as a large improvement and reported 66.93%. A dropped-work bug inside an optimiser presents as
  a win, not as a failure. Files: `freespace_sim/planner/lns/state.py`.

- 2026-09-10T01:32Z `[CODE]` A metric pair must measure the same thing: `_flown_horizontal_m` summed
  a two-leg centerline while `_straight_horizontal_m` used one origin->dest pair, reporting 4,332 m
  of detour where 387 m existed. Both now split at `leg_starts`. Files: `freespace_sim/metrics.py`.

- 2026-09-10T04:10Z `[TOOL]` A regex/sed-style deletion of one field left scars in six files that no
  linter catches, and one of them silently removed test coverage: deleting a test took its body but
  left its `@pytest.mark.slow`, which re-bound to the NEXT function — a fast test on `main` became
  slow on the branch and dropped out of `-m "not slow"`, the mode this repo iterates with. Others
  were a docstring truncated mid-sentence (the one stating `WorkerSpec`'s invariant), two welded
  argument lists, and two runs of stray blank lines. After a mechanical delete, diff the marker →
  function bindings, not just the removed lines. Files: `freespace_sim/planner/lns/{parallel,solver}.py`,
  `freespace_sim/sim.py`, `tests/test_lns.py`, `tests/test_lns_parallel.py`.

- 2026-09-10T04:12Z `[CODE]` A wrapper whose `__getattr__` forwards EVERYTHING breaks two invariants
  at once. Forwarding `warm_planner` made `iter_planner_chain` yield the inner planner's warm planner
  at the wrapper's own depth, reordering a chain whose order is load-bearing (`astar_milp` went
  `[MILP, AStar]` -> `[Itinerary, AStar, MILP]`; `_terminal_capacity_for` takes the FIRST match).
  Forwarding `inner` made `copy`/`pickle` recurse to a stack overflow, because both build an instance
  with an empty `__dict__` before restoring state. A forwarding wrapper must refuse its own
  structural attribute names. Files: `freespace_sim/planner/itinerary.py`.

- 2026-09-10T04:14Z `[CODE]` A volume synthesized AFTER the searches that produced it is invisible to
  the thing that would have routed around it. The round-trip pad hold is built in `_compose` from
  both legs' results, and it rasterizes to nothing in every planner's hex occupancy (`z=[0,5]` vs a
  30 m ladder floor with `corridor_height_m` 30 ⇒ `_levels_overlapped` returns `[]`) while
  `ledger.any_conflict` sees it — so FCFS denied it at commit as a lost race, and the LNS commit path
  (which re-checks nothing) committed a real conflict. Anything added to `volumes` after planning
  must be conflict-checked before the intent is returned ACCEPTED. Files:
  `freespace_sim/planner/itinerary.py`, `freespace_sim/mechanism.py`, `freespace_sim/planner/lns/state.py`.

- 2026-09-10T15:45Z `[USER]` `.gitignore` line 8 is `*.png` and stays that way — figures are not
  committed, so a figure citation names a path the reader REGENERATES. That makes the citation honest
  only if `context/figures/make_figures.py` holds a `fig_*` whose `_save` name matches the cited file
  AND is registered in `FIGURES`. Three citations added here pointed at a PNG rendered by a throwaway
  script, so nobody but its author could produce it; the fix is to move the generator into
  `make_figures.py`, never to un-ignore the image. Files: `context/figures/make_figures.py`, `.gitignore`.

- 2026-09-10T18:10Z `[CODE]` A schema-version bump that ENUMERATES the keys it knows changed will
  miss the ones it does not. `_SPEC_SCHEMA_VERSION` 1->2 refused v1 payloads carrying
  `paired_return_request`, but that flag only chose which of v1's two filing schemes ran — BOTH
  emitted two requests per delivery, so the switch was `return_flights`, which also defaults to True
  (an absent key is a round-trip recipe). A second key, `turnaround_s`, stayed a live dataclass field
  while changing meaning (v1: when the return was FILED; v2: the pad dwell), so every archived spec's
  stored `0.0` would have replayed as a zero-second dwell. Refuse on the VERSION plus the field that
  changes the flight SET, and check what an absent key defaults to. Files:
  `freespace_sim/scenarios/spec.py`.

- 2026-09-10T18:12Z `[CODE]` Leaving a volume untagged to keep it OPAQUE to other flights also makes
  it opaque to permanent terminal walls, which are not flights. `conflict.volumes_conflict` exempts a
  pair only when both carry the same `terminal_id`, so the untagged pad hold — untagged precisely so
  a same-hub flight cannot land on the parked aircraft — was checked against a wall its own tagged
  columns fly straight through, denying a trip for its own hub's airspace. `ledger.any_conflict`
  includes static walls; `ledger.conflicting_flights` excludes the `STATIC_WALL_FID` sentinel. Pick
  the one that matches what the check is FOR. Files: `freespace_sim/planner/itinerary.py`,
  `freespace_sim/ledger.py`, `freespace_sim/conflict.py`.

