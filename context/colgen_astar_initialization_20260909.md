# Current initialization experiment and coefficient definitions

Worktree: /private/tmp/congestion-colgen-lns-20260905, branch codex/colgen-lns-main.

## Resource coefficients and fixed occupancy

A row r=(q,ell,tau) identifies a hex cell q (itself an axial coordinate pair), cruise-level index ell, and integer time period tau. In the current scenario each period lasts Delta=4 seconds and a cell row has capacity 1.

For flight f and column p, let C_fp be the union of every resource row reserved by its canonical trajectory. Then

\[
a_{rfp}=\mathbf 1\{r\in\mathcal C_{fp}\}.
\]

For a cruise path q_0,...,q_H, with integer departure d_p, the visit clock is v_j=d_p+n_climb+n_origin_lane+j. The geometry-derived offsets are currently (-2,1), so the cruise contribution is

\[
a^{\rm cruise}_{(q,\ell,\tau),fp}
=\mathbf1\{\exists j:q_j=q,\ \ell_p=\ell,\ v_j-2\le\tau\le v_j+1\}.
\]

Customer endpoint cylinders add nearby cell rows across flight levels over their dwell windows. Terminal takeoff/landing dwells add terminal/pad rows. The union is deduplicated: overlapping contributions from one flight still produce coefficient 1, not 2. The implementation is column_claims in network.py; windows.py derives the time offsets from the reservation geometry.

For an externally committed set G,

\[
l^{\rm fixed}_{(q,\ell,\tau)}
=\sum_{g\in G}\mathbf1\{(q,\ell,\tau)\in\mathcal C_g\}.
\]

It is a constant occupancy COUNT from flights outside the decision set, not a variable, penalty, route length, or initial schedule. The capacity constraint is l_fixed + sum_fp a_rfp x_fp <= B_r. The solver receives one fixed_claims set per committed flight and sums these sets with a Counter. In these batch simulations the dynamic ledger starts empty, so l_fixed=0 on every row. Warm-start columns remain decision variables and do not contribute to l_fixed. Permanent terminal airspace is excluded geometrically rather than represented by fixed occupancy.

## Measured use of the old departure ladder

Neither A* nor SIPP seeded the previous standard/cheap benchmarks. Nominal routes, 20 later copies at four-second increments, and a greedy ground-shift schedule initialized the master.

In the completed standard 1200-second outbound run, the first IP selected 626 nominal columns, 596 departure-ladder columns, 97 greedy columns beyond the ladder, and 218 priced columns. Its final selection used 340 nominal, 116 ladder and 1081 priced columns; no beyond-ladder greedy columns remained. The 116 ladder departures were delayed by 4–56 seconds, with 59 at only 4 or 8 seconds. Thus the ladder is used by actual integer schedules; this does not establish that all 20 alternatives are necessary. See analysis/colgen_cheap_pricing_1200s_20260909/summary/seed_usage.json.

## Centered FCFS A* seeds requested by the user

Interpret 10 as ten departure lattice steps, or 40 seconds. For an imported A* departure d_f^A and requested lattice departure b_f, define the latest legal departure d_f^max for this route and

\[
K_f=\{-\min(10,d_f^A-b_f),\ldots,\min(10,d_f^{\max}-d_f^A)\},
\quad
\mathcal P_f^0=\{\mathsf T_k p_f^A:k\in K_f\}.
\]

This gives up to 21 columns INCLUDING the A* center. A nominal-departure source gets 11 columns (center plus ten later). Missing earlier copies are not backfilled with more late copies. Times and costs shift; route geometry is unchanged. Earlier copies stop at b_f; later copies respect both ground-delay and path-arrival limits.

The treatment uses warm_start_planner='astar', seed_nominal_routes=False, provided_seed_ladder_steps=10, warm_start_max_shift_steps=0. The CLI mirrors these as --colgen-warm-start astar --colgen-no-nominal-seeds --colgen-provided-seed-ladder 10 --colgen-warm-start-max-shift 0. It skips nominal route construction, its old ladder, both greedy bootstrap passes, and import conflict repair. Individually valid source columns remain available even if their centers conflict; the IP must combine alternatives. A source-center selection becomes an incumbent only if it is jointly CG-row feasible.

The converter now rejects air holds explicitly (previous deduplication silently erased them), rejects routes beyond the pricing hop budget, and uses actual request-time/flight-ID FCFS ordering when repair is enabled. Defaults retain normal initialization.

Tests: 192 targeted tests passed, followed by 27 focused checks after the skip-greedy/trace refinement and 4 centered-window checks after final canonicalization. Both 12-flight centered/control smoke simulations physically verified and achieved the same objective.

The original seed-only full run was cancelled before pricing when the user requested centered alternatives. Its source A* prepass completed and was verified, but it is not a completed CG benchmark. The new paired run is analysis/colgen_astar_centered_1200s_20260909: 1537 outbound flights, 12 pricing workers, 4 Gurobi threads, standard exact pricing, LP target .001, IP30s per round, final IP600s/reserve660s, max30 rounds, CG budget14460s. Measured arms run sequentially; source/harness snapshots are frozen. A* planning/conversion are included in end-to-end time, although the CG budget starts afterward.

## A* airborne holds observed in the source schedule

Only 3/1537 flights held in the air, for 16 seconds total, in addition to ground delay and route overhead:

| Flight | Ground delay | Hold | Extra distance vs reference | Hold time | Distance from hold to customer |
|---|---:|---:|---:|---|---:|
| 2628 | 16 s | 4 s | 1323.256 m | 1808–1812 s | 13.89 km |
| 2072 | 8 s | 4 s | 272.783 m | 1952–1956 s | 2.91 km |
| 818 | 4 s | 8 s | 1085.373 m | 2092–2100 s | 1.03 km |

All holds are stationary cruise states at 100 m; they are distinct from mandatory terminal/customer dwell. Extra distance includes lattice overhead and should not all be called congestion detour.

A* offers ground-wait, lateral and hover edges in the same shortest-path search. Ground waiting costs 1/s; hovering costs 3/s; an extra 120 m lateral hop costs 12. Thus a four-second hold costs 12, versus 4 for four seconds of ground delay. A hold changes only downstream timing while a later departure shifts the entire earlier route and the takeoff slot. That can make a short local hold less costly than the feasible ground/detour alternatives. The exact blocking flights for these three choices have not been counterfactually replayed yet; do not claim their particular causal conflicts have been proven. Relevant code: astar/planner.py _edges and astar/kernel.py hover/ground expansions.

## Live benchmark checkpoint

The active full paired benchmark is exec session 35931. Its parent runs A* centered first and automatically runs normal initialization second, sequentially. Do not alter production files or benchmark harness/trace until both finish. Progress files are under analysis/colgen_astar_centered_1200s_20260909/density_faa_wing_zipline_seed0_iteration_ip_astar_centered; tail process.log and inspect iterations.json/ip_calls.json. The previous session 98968 was interrupted and is explicitly marked cancelled.

Source A* on the active run: 53.962 s, accepted1537, verified, cost168540.5537095. Conversion15.364 s, 1521 imported routes, 16 rejected (3 holds,3hopbudget,1degenerate,9outsidecorridor), 11 overloaded rows among source centers. No greedy import filtering or holds. Initial pool18782; initial centers were not jointly feasible, so no starting incumbent was claimed. All 1521 source paths were checked for recent cell revisits; none revisit within three hops.

Round1: pricing27.313s, IP native5.816s, covers1534, selectedcost161893.306 +30000denialpenalty; end-to-end available~181.53s. Selected origins: 860A*center,240earlier,175later,259priced.
Round2: IP native30.016s, covers1536, penalizedcost170455.090; origins680center,191earlier,144later,521priced.
Round3: all1537covered, cost159542.458.
Round4: cost158891.885, end-to-end471.36s.
Round5: cost158630.485.
Round6: cost158166.579; best global bound finally improving, still far from0.1%LPtarget. Round7 IP starting at last inspection.

A dedicated reporter exists at analysis/summarize_colgen_astar_centered.py, Ruff passed but it has not run against completed results yet. After completion run it with the benchmark root using the project venv and MPLBACKEND=Agg. Fix actual reporter assumptions if necessary, audit source/input hashes and budget/cost/column usage, inspect generated comparison/convergence PNGs, then write final results into this context note. Reporter reads correct result field global_cost_gap (not global_integer_gap). Its final ladder assertion allows ground-delay horizon; source hop checks mean path horizon should not truncate sooner here. Generic load_case is used for shared audits, but the new reporter rebuilds quality curves with actual initial/iteration simulation availability times including A* overhead.

Two explanatory PNGs already generated and visually QA passed: explanation/astar_holds.png and explanation/centered_departures.png under the active benchmark root. The first shows actual source routes and zoomed stationary waits. Both can be embedded with absolute paths. Holds image already shown to user. The counterfactual reason for a particular hold has not been replayed; the general objective/feasibility tradeoff was explained in commentary without asserting a proven per-flight cause.

Outstanding final response must cover the original coefficient/fixed-load questions, measured use of previous and new departure alternatives, actual centered-initialization benchmark outcome, and the airborne-hold explanation as appropriate to further user steering. No final has been sent. No commits/pushes/agents/goals/automations. Keep active task running unless user cancels or changes it.

## Completed A* centered arm; normal control still running

The treatment finished: 2517.80688375 s =41.96345 min including A*; 30 rounds, iteration_limit (NOT LP convergence). All1537 accepted, verified; no repair additions, no rounding/LNS, no kernel fallback (2label restarts). Final congestion cost156579.69574362953, grounddelay1260s, 38855columns (18782initial +20073priced). Pricing1481.06220838s; per-round IP wall528.26101167s/native467.08404899s; 23/30 round IPs optimal. Final IP wall13.568s/native13.10662484s, optimal withinpool. Certified LPgap11.6450129%; globalintegergap12.4369814%. Final usage:362A*center,93earlier,38later,1044priced. Initial source choices were not jointly CG-row feasible, so no initial incumbent was claimed. load_case audited this arm successfully.

The parent session35931 has automatically started the second arm iteration_ip_eager under the same root. It is still running. Keep production/harness/trace frozen until control completes. Poll its process.log (not the completed centered log), plus parentsession as needed. Sourcecase result.json/planner_stats.json/selections.json are ready for analyses.

The pricing diagnostic was checked: summing reduced costs over actually inserted columns matches rc_sum within less than1e-6 each sweep. Excess rc_n_positive counts are floating-point values under insertion tolerance, plus negligible nearzero duplicates. There is no lost substantial reduced cost. From rounds22–29 the restricted LPcost moves155176.699→155176.240 (less than0.5), while pricing residual sums remain tensofthousands; likely dual degeneracy/tailingoff, explicitly an inference, not an established code bug. No production changes were made for this diagnostic. Reporter now audits residuals<1e-3 and plots only full-service schedules in the main quality chart to avoid an all-denied initial point obscuring useful differences; full penalized curves remain in audit.json for time-to-quality calculations.

The user has already been told treatment took42.0minutes, accepted/verified1537, cost156579.7, finalIPoptimal~13.5s, hit30roundcapwithoutLPtarget; normal12-workercontrolnowrunning. Await its result before claiming a speedup or slowdown.

## FINAL: both runs complete and audited

The paired parent session35931 completed successfully. No benchmark processes remain from it. Reporter completed successfully; all input/source hashes, source archive, costs, IP budgets, centered time bounds, selected-column provenance and physical verification checks passed. Both comparison.png and convergence.png were visually inspected and passed. Only explanatory comments in params.py/warm_start.py were clarified after both measured simulations finished; measured executable behavior matches the current code. Ruff passed.

# A* centered initialization benchmark

Replacing nominal seeds with FCFS A* routes and centered departure alternatives took 51.1% longer in this paired run. Final cost changed by -0.0861%.

A* centered met the 0.1% LP target: False. Normal initialization met it: True. Pool-IP optimality is reported separately below.

| Metric | A* centered | Normal initialization |
|---|---:|---:|
| Total wall time (s) | 2517.81 | 1666.47 |
| Final penalized cost | 156579.696 | 156714.567 |
| Accepted / verified | 1537 / True | 1537 / True |
| CG rounds / stop | 30 / iteration_limit | 13 / lp_gap |
| Initial pool columns | 18782 | 32421 |
| Final pool columns | 38855 | 45879 |
| Pricing time (s) | 1481.06 | 877.97 |
| Per-round IP wall / native (s) | 528.26 / 467.08 | 402.23 / 366.06 |
| Final IP wall / native (s) | 13.57 / 13.11 | 19.70 / 19.18 |
| Final restricted IP optimal | True | True |
| Certified LP gap (%) | 11.64501 | 0.03844 |
| Global integer gap (%) | 12.4370 | 1.0185 |
| Ground delay (s) | 1260.0 | 1688.0 |

Normal/A* elapsed-time ratio: 0.662x. A* final cost change: -0.0861%.

These are one paired seed-0 run, not repeated timing estimates. Both arms use the same requests, source snapshot, 12 pricing workers and 4 Gurobi threads. A* planning and conversion are included in wall time; the CG budget begins after that prepass. The 0.1% target applies to the LP certificate, not integer optimality.

## Time to fully served solution quality

| Cost at most | A* centered (min) | Normal (min) |
|---:|---:|---:|
| 160000 | 6.35 | 15.04 |
| 158000 | 12.13 | 18.64 |
| 157000 | 19.41 | 22.24 |

## A* + centered ladder

Final selected-column provenance:

| Origin | Selected columns |
|---|---:|
| priced | 1044 |
| A* center | 362 |
| later A* variant | 38 |
| earlier A* variant | 93 |

Time to each arm's final cost, including initialization:

| Target | Minutes |
|---|---:|
| A* + centered ladder final cost | 41.59 |
| Nominal + later ladder + greedy final cost | 28.23 |

FCFS A* itself accepted and verified 1537 flights in 53.96 seconds, at cost 168540.554. Conversion took 15.36 seconds and retained 1521 routes. Their centers overloaded 11 CG rows, so they were pool alternatives rather than an initial incumbent.

| Import rejection | Flights |
|---|---:|
| air hold is not representable | 3 |
| degenerate path | 1 |
| path exceeds pricing hop budget | 3 |
| Outside the CG corridor | 9 |

## Nominal + later ladder + greedy

Final selected-column provenance:

| Origin | Selected columns |
|---|---:|
| priced | 1083 |
| nominal | 338 |
| later ladder | 116 |

Time to each arm's final cost, including initialization:

| Target | Minutes |
|---|---:|
| A* + centered ladder final cost | Not reached |
| Nominal + later ladder + greedy final cost | 27.29 |

The A* arm imports all individually valid centers without a greedy feasibility pass. It adds up to ten earlier and ten later four-second alternatives per center. Earlier alternatives stop at requested departure. Airborne holds and routes outside the CG domain cannot be imported; pricing may recover those flights. Original A* centers are used as an incumbent only if jointly feasible under CG rows.

The standalone A* marker is an operationally verified reference, available after its prepass. It contains holds and routes outside the CG domain, so it is not counted as a feasible CG-pool incumbent or used in the CG time-to-target calculations.

Input/source hashes, native IP budgets, column uniqueness, initial ladder bounds, objective recomputation and physical verification were audited. See audit.json for per-round column usage and solution availability.

![Runtime and quality](comparison.png)

![LP convergence](convergence.png)


Interpretation: centered A* seeds are useful for early good integer schedules, but this exact replacement is not a faster convergent CG default. It ran51.1%longer and missedLPtarget, for0.0861%lowercost. It reached160000cost in6.35min vs15.04min;158000 in12.13vs18.64;157000 in19.41vs22.24. The control reached its finalcost earlier too (27.29min vs28.23 for A* to match it). Keep normal initialization as default; the new centered option remains available for short-budget experiments. The experiment changes both initial spatial route and time coverage, so it does not isolate removing greedy, centered times, or nominal-route removal. An informative next ablation would keep21timeoptions per flight by backfilling missing early slots with more late slots, or retain one nominal reference column alongside A* seeds; these have NOT been run.

The new control with12workers produced the same first-sweep LPobjective, reduced-cost sum,1227columns and first-IPcost as the previous4-worker run. Firstpricing413.45299→153.86795s (2.687×); complete standardrun2759.42371→1666.4748s (~39.6%less), finalcost identical156714.56744. Final selections differ among equal-cost routes (new338nominal+116ladder+1083priced; old340+116+1081).

Final response still pending. Must be self-contained and answer latest airborne-hold question plus completed benchmark and earlier a_rfp/l_fixed and seed-usage questions, using concise text and links to full notes/report. Airborne-hold causal blocking flights were not counterfactually tested; only actual positions/times and the shared A* objective/edge rules were verified. No SIPP seed run was performed. No commits/pushes/agents/goals/automations were created.
