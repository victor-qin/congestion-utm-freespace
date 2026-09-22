# HISTORY

Durable record of mistakes likely to recur between PRs. Format follows the `CONTINUITY.md` rules in `AGENTS.md`.

- 2026-09-08T20:40Z `[CODE]` `verified` means INTERflight separation only: `find_interflight_conflict`
  replays intents in order and checks each against the flights committed BEFORE it, never against
  itself. A round-trip itinerary files both legs and its pad hold as ONE flight, so anything wrong
  within a trip is invisible to it and the schedule still reports `verified=True`. Do not read
  `verified` as "flyable". Files: `freespace_sim/verify.py`, `freespace_sim/sim.py`,
  `freespace_sim/planner/itinerary.py`.

- 2026-09-08T20:45Z `[CODE]` A checker gated on a knob that defaults to off never runs. LNS's
  mid-search `assert_incumbent_ok` fires only under `verify_every`, which defaults to 0
  (`lns/solver.py:713`), so an invariant checked only there goes unexercised on every real run. Put
  one that must hold in the unconditional closing pass (`_finalize_lns_result`). Files:
  `freespace_sim/planner/lns/solver.py`.

- 2026-09-08T22:55Z `[TOOL]` Changing how many requests a demand model emits RENUMBERS every flight
  and breaks fixtures that pin ids. Verify it is a pure renumbering (match old ids on t_request /
  origin / dest) before remapping, and check whether the deleted code consumed RNG draws. Files:
  `tests/test_hub_conflict_filed.py`, `freespace_sim/demand.py`.

- 2026-09-09T00:35Z `[TOOL]` A test that cannot fail pins nothing. A fixture that made two
  definitions identical (`centerline[0][1] == volumes[0].t_start`) passed `x < x + 1e-9` for any
  implementation. The same trap reappeared in a review fix: the `PlanEnvelope.union` assertion passed
  with its fix reverted, because a real itinerary's legs span the same (q, r) box and cannot produce
  the case. Build the fixture so the two sides CAN differ, and revert each fix to confirm its test
  fails. Files: `tests/test_paired_precedence.py`
  (`test_the_arrival_and_departure_clocks_are_the_columns_not_the_waypoints`),
  `tests/test_parallel_envelope.py`.

- 2026-09-09T01:35Z `[USER]` A replacement that keeps the thing it replaces is not a replacement.
  The itinerary model landed +634/-203 because the two-request path was kept "so archived runs load"
  — but archived runs load fine without it, as the two independent flights they always were, and
  nothing outside the deleted code read the link. Check what actually depends on a legacy path before
  paying to keep it. Files: `freespace_sim/sim.py`, `freespace_sim/verify.py`,
  `freespace_sim/planner/lns/{state,solver,parallel}.py`, `freespace_sim/types.py`.

- 2026-09-09T03:10Z `[TOOL]` A failing assertion's MESSAGE is a hypothesis, not evidence.
  `test_lns_parallel` blamed a sparse world ("pick a denser world"), but a denser one still failed:
  the fixture picked victims by flight id, which caught lightly delayed flights while the most delayed
  went untouched. Measure what the fixture actually selected before believing what it says about the
  world. Files: `tests/test_lns_parallel.py`.

- 2026-09-09T03:15Z `[TOOL]` A test fixture that pins some config but inherits the rest breaks on any
  default change. `tests/test_colgen_solver._cfg` pinned flight levels, region and ground-delay cap
  but inherited `hover_time_s`, and its hand-derived step counts are computed on the column window
  that sets. Pin every knob an expectation was derived against. Files: `tests/test_colgen_solver.py`.

- 2026-09-09T03:20Z `[CODE]` `planner/colgen/__init__.py` keeps `run_batch` / `ColGenSolver` lazy, and
  its docstring says why; what it does not say is that importing them eagerly defeats
  `monkeypatch.setattr(batch, "run_batch", ...)`, so `test_colgen_batch` runs the real Gurobi path and
  fails on a missing `gurobipy`. Files: `freespace_sim/planner/colgen/__init__.py`,
  `tests/test_colgen_batch.py`.

- 2026-09-10T01:25Z `[TOOL]` A request type that needs a wrapper is only safe where the wrapper is
  applied, so ask "who builds a planner WITHOUT `get_planner`?" LNS's repair planner did, and repaired
  every round trip as its outbound alone — which a cost-comparing search ADOPTS, because the dropped
  return costs about half and reads as a 66.93% improvement with `verified=True`. A dropped-work bug
  inside an optimiser presents as a win, not a failure. A guard in the leaf planners beats a
  hand-audit: `reject_itinerary` found a third bypassing site (`lns/unimpeded.py:_new_ruler`) on its
  first run, and now covers every leaf planner. Files: `freespace_sim/planner/itinerary.py`,
  `freespace_sim/planner/lns/{state,unimpeded}.py`.

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
  the thing that would have routed around it, and must be conflict-checked before its intent is
  returned ACCEPTED. The round-trip pad hold is built in `_compose` from both legs' results; at
  `z=[0,5]` it sits under the lowest corridor band (15 m on the default ladder: level 30 minus
  `corridor_height_m/2`), so `_levels_overlapped` returns `[]` and no planner's hex occupancy sees it.
  FCFS caught it at commit as a lost race; the LNS commit path, which re-checks nothing, committed a
  real conflict. Then pick the ledger query that matches what the check is FOR: the hold is untagged
  so a same-hub flight cannot land on the parked aircraft, which also makes it opaque to permanent
  terminal walls its own columns fly through, so it uses `conflicting_flights` (excludes
  `STATIC_WALL_FID`) rather than `any_conflict`. Files: `freespace_sim/planner/itinerary.py`,
  `freespace_sim/ledger.py`, `freespace_sim/conflict.py`, `freespace_sim/mechanism.py`,
  `freespace_sim/planner/lns/state.py`.

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

- 2026-09-16T17:00Z `[CODE]` An invariant that lives only in a test's precondition is not enforced.
  `_sweep_kept` sized its candidate rectangle by `infl_pad` and took the corridor set as a subset,
  which holds only while the pad footprint is the WIDER of the two; nothing in the code said so, and
  the one place it was written down was `assert infl_p >= infl_b` at the top of a test. Changing a
  config default (a 10 m delivery pad, 79.3 m vs 99.3 m) inverted it and silently dropped the outer
  ring of every committed corridor — 20 hex cells became 16 — so every planner deconflicted against
  less than the corridor sweeps. Sweep the union and carry both memberships instead of assuming an
  order. When a footprint is derived from a knob, the ordering of two derived quantities is a knob
  too. Files: `freespace_sim/planner/hexgrid.py`, `freespace_sim/planner/astar/occupancy.py`.

- 2026-09-16T17:05Z `[CODE]` A journal that encodes a boolean cannot record a third state. The
  reference occupancy wrote `-1` pad-only / `-2` pad+blocked per released row, which assumed every
  corridor cell was also a pad cell — true only under the nesting above. Once a cell could be
  blocked-only, `on_release` decremented a `pad` refcount that was never incremented, corrupting the
  map under LNS destroy with no test failing at the point of damage. Any encoding that mirrors a
  membership pair has to change with it; grep the release path whenever the add path grows a case.
  Files: `freespace_sim/planner/astar/occupancy.py`.

- 2026-09-16T23:40Z `[CODE]` Check whether the other planner already owns the geometry predicate you
  are about to write. `claim_inflation` guarded its exact-claim branch with "pitch exceeds corridor
  width", which is too weak — two hops on neighbouring lattice rows are 103.9 m apart, so a 110 m
  corridor on a 120 m pitch would have claimed exactly while overlapping. colgen had carried the
  correct predicate for a year (`hop_box_stays_in_its_cells`: `width <= circumradius` AND the
  overhang corner within the inradius), because its capacity rows rest on the same containment. It
  now lives in `hexgrid` and both call it. Files: `freespace_sim/planner/hexgrid.py`,
  `freespace_sim/planner/colgen/windows.py`.

- 2026-09-22T16:00Z `[CODE]` Eighteen tests pinned the literal default ladder / box / ceiling and rotted
  the moment the defaults moved (30/70/110 m -> 70/85/100/115 m, 30 -> 10 m box, 125 -> 120 m
  ceiling). Tests about ladder MECHANICS now derive their expectation from the SimConfig they run on
  (`set(range(cfg.n_levels))`, `cfg.nearest_level(cfg.cruise_level_m)`, the validator's own headroom
  formula) or pin an explicit ladder when the default is too tight for the fixture's premise (a 15 m
  rung is a 1-step check that cannot tell ceil from max(1, .); a planner has no vertical escape above a
  100 m warm plane under a 120 m ceiling). Exactly ONE test pins the literal defaults
  (`test_default_config_is_multilevel`), so a default change fails in one place. Files:
  `tests/test_config.py`, `tests/test_astar.py`, `tests/test_planner_milp.py`.

- 2026-09-22T15:35Z `[CODE]` The MILP planner family (`milp`, `astar_milp`, `astar_milp_shortcut`;
  `planner/milp.py`, the `pulp` dependency) was REMOVED: superseded by colgen as the bound and by the
  shortcut refiner as the geometric polish, and the largest source of default-sensitive fixture rot
  (its climb-over-wall test has no vertical escape on a 70-115 m band). Pitfalls that outlive it: an
  archived run whose config names a `milp*` planner no longer resolves in `get_planner`, and
  `plans_terminal_airspace` is now declared only by colgen, so `sim._wall_aware` admits A*-reaching
  chains and colgen and nothing else. Files: `freespace_sim/planner/__init__.py`, `freespace_sim/sim.py`,
  `freespace_sim/metrics.py`.

- 2026-09-22T18:40Z `[CODE]` Two fixture traps from moving the temporal claim onto half-open time (#136).
  A ZERO-DURATION volume at a grid instant, `Volume4D(shape, k·dt, k·dt)`, overlaps no period: with no
  buffer it claims only step `k` (the old `floor` bound handed it `k` and `k + 1`), and a dwell opening at
  step `k` reads from `k + 1`, so a test that used one as a pad blocker stopped blocking. An arrival-step
  probe reads the period BEFORE the step and its own, so a window that starts at the step a column opens
  (`pad_clear(s0, ...)`) begins at `s0 + 1`, and a zero-length dwell checks nothing. Give blockers real
  duration and dwells `>= 1`.
  Files: `freespace_sim/planner/hexgrid.py` (`_step_range`), `freespace_sim/planner/astar/occupancy.py`
  (`pad_clear`), `tests/test_sipp_compiled.py`, `tests/test_occupancy.py`.
