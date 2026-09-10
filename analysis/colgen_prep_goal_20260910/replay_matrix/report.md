# Matched pricing replay matrix

All twelve fresh-process sweeps completed. The nine candidate/repeat comparisons
matched the corresponding baseline for all 1,537 flights on output, column
identity, claims, cost/reduced cost, and recorded search work. The combined arm
used the compiled goal search for every baseline goal invocation: 1,229 in round
1, 388 in round 6, and 241 in round 11 (including all 241 in the reverse repeat),
with no declines.

| Round | Baseline wall (s) | Prep wall (s) | Prep reduction | Combined wall (s) | Combined vs prep |
|---:|---:|---:|---:|---:|---:|
| 1 | 85.007 | 85.580 | -0.67% | 84.072 | 1.76% |
| 6 | 90.678 | 86.576 | 4.52% | 86.094 | 0.56% |
| 11 | 91.051 | 86.688 | 4.79% | 84.998 | 1.95% |

The three forward rounds total 266.737 seconds for baseline, 258.844 seconds for
prep-only (2.96% less), and 255.164 seconds for combined (1.42% less than
prep-only and 4.34% less than baseline).

The reverse round-11 order measured 84.165 seconds combined, 87.009 seconds
prep-only, and 89.875 seconds baseline. Forward/reverse round-11 averages are
90.463, 86.848, and 84.581 seconds respectively: prep-only saves 4.00%, the
compiled goal search saves another 2.61%, and combined saves 6.50% versus
baseline. Round-11 order drift was -0.98% combined, +0.37% prep-only, and -1.29%
baseline. Worker task-time averages show the same direction: 946.098, 904.346,
and 893.817 seconds (4.41%, then 1.16%, for 5.53% total).

Prep-only round 1 was flat: worker task time increased 0.21% and wall increased
0.67%. Its recorded compiled phase fell from 224.431 to 183.744 worker-seconds,
and task time outside bootstrap fell from 252.576 to 213.616 seconds, while total
bootstrap time rose from 696.596 to 737.583 seconds. The measurement does not
identify whether that bootstrap increase is memo fill/retention cost or ordinary
timing drift. In both baseline and prep-only, the first packing occurs in the
restricted DP; prep-only then reuses it for the main DP.

Combined reduces the recorded goal phase versus prep-only by 17.60% in round 1,
13.01% in round 6, and 16.02% in round 11. This timer is not a pure kernel timer:
the combined arm first packs the problem inside the native goal phase, while
prep-only first packs it during restricted DP.

Each replay starts a fresh 12-worker pool. This preserves matched cold inputs and
includes worker graph/preparation setup, but it does not reproduce the full
solver's solve-scoped warm worker caches. The full baseline capture's persistent
pool took 46.466 seconds in round 6 and 47.160 seconds in round 11; the cold
baseline replays took 90.678 and 91.051 seconds. Speed findings are one machine,
one seed, one forward pass, plus one reversed late-round repeat.
