# Shortcut strategy benchmark

Run on 2026-08-02 with:

```bash
.venv/bin/python analysis/bench_shortcut_strategies.py --repetitions 3
```

The deterministic workload contains 42 uniform requests in a 4 km × 4 km region over 600 seconds.
Each measurement uses a fresh process and ledger; the compiled A* kernel is warmed before timing.
Values below are medians of three unprofiled repetitions.

**Scope:** the workload has **no terminals**, so `_rebuild`'s pad-capacity gate
(`TerminalCapacity.reservation_admitted`) is a no-op throughout and contributes nothing to these
timings. That gate's coverage lives in `tests/test_shortcut.py` and `tests/test_terminal.py`, which
parametrize the terminal fixtures across all three strategies. Read the numbers below as en-route
refinement cost only.

| planner | rebuilds | success / failure | rebuild time | full run | verified | plans changed vs legacy | exact heading skips |
|---|---:|---:|---:|---:|:---:|---:|---:|
| `astar_shortcut` | 935 | 888 / 47 | 145.02 ms | 324.11 ms | yes | 0 / 42 | 0 |
| `astar_heading_shortcut` | 519 | 472 / 47 | 79.64 ms | 268.28 ms | yes | 0 / 42 | 416 |
| `astar_batched_shortcut` | 867 | 405 / 462 | 138.55 ms | 395.12 ms | yes | 17 / 42 | 0 |

On this workload, the exact-heading variant reduced rebuild calls by **44.5%**, rebuild time by
**45.1%**, and end-to-end wall time by **17.2%**. Its serialized `OperationalIntent` values were
byte-identical to `astar_shortcut` after normalizing only `solve_time_s` in all three repetitions;
both variants intentionally use the same `astar_sc` intent label.

The batched algorithm reduced rebuild calls by 7.3%, but its extra failed maximal/fallback probes made
this particular workload 21.9% slower end-to-end. It intentionally changed 17 plans, so its result is
a route-quality/performance tradeoff rather than a drop-in implementation optimization. Earlier
turn-friendly samples favored batching; the comparison is workload-sensitive.

| planner | accepted / denied | mean cost | mean detour | mean altitude | mean delay | mean arrival | ledger queries | rebuilt volumes | committed volumes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `astar_shortcut` | 42 / 0 | 75.25 | 107.76 m | 60.00 m | 24.48 s | 279.60 s | 1,019 | 21,445 | 911 |
| `astar_heading_shortcut` | 42 / 0 | 75.25 | 107.76 m | 60.00 m | 24.48 s | 279.60 s | 603 | 11,463 | 911 |
| `astar_batched_shortcut` | 42 / 0 | 71.72 | 100.10 m | 60.00 m | 21.71 s | 276.58 s | 951 | 9,723 | 911 |

## Why the fast skip is stricter than heading equality

`build_reservation_from_corners` resamples each logical chord independently. Two perfectly collinear
legs can therefore produce different box boundaries and time windows when merged. A regression
fixture demonstrates a split `A→B→C` reservation passing while a heading-only `A→C` merge conflicts
with a time-local obstacle.

`astar_heading_shortcut` bypasses `_rebuild` only when:

1. both legs have the same 3D heading; and
2. the exact IEEE-754 `(segment_start, segment_end)` stream emitted before and after merging is
   identical.

Otherwise it executes the original candidate rebuild in the original order.
