# Full shared-preparation and compiled-goal comparison

The frozen combined candidate completed in 1,227.099 seconds versus 1,324.508
seconds for the matched frozen baseline, a reduction of 97.409 seconds (7.35%).
Subtracting the three measured capture operations gives 1,223.043 versus
1,320.360 seconds, a 97.316-second (7.37%) supporting comparison. Capture work
still advanced each solver's absolute clock, so the adjusted value does not undo
possible scheduling effects.

Pricing wall fell from 683.769 to 595.712 seconds, a reduction of 88.057 seconds
(12.88%). Worker task time fell from 5,965.637 to 5,068.288 seconds (15.04%).
Peak process RSS was 9,519.6 versus 9,542.0 MiB, an increase of 22.4 MiB (0.24%).
The candidate's compiled-goal warmup was outside the simulation clock and took
0.221 seconds, producing one compiled signature. The ordinary benchmark warmups,
also excluded, took 0.843 and 0.766 seconds.

All-call per-flight distributions treat a missing stage as zero. Times are seconds.

| Stage | Baseline mean / median / p95 / max | Combined mean / median / p95 / max |
|---|---:|---:|
| Task wall | 0.35285 / 0.27102 / 0.97894 / 4.99279 | 0.29977 / 0.23018 / 0.84918 / 4.30711 |
| Bootstrap | 0.16950 / 0.06629 / 0.72374 / 4.47092 | 0.15257 / 0.06577 / 0.61823 / 4.06757 |
| Goal | 0.04790 / 0 / 0.26802 / 4.23190 | 0.03623 / 0 / 0.05671 / 3.95001 |
| Main compiled | 0.17945 / 0.16614 / 0.34613 / 4.36989 | 0.14242 / 0.13329 / 0.25913 / 4.10198 |

"Main compiled" is the inclusive `compiled_s` phase: Python preparation, native DP
and host sink checks. It is not a measurement of pure native DP time. Bootstrap and
goal are also inclusive; goal is part of bootstrap, so their times must not be added.

Conditioned on a measured goal invocation, baseline has 6,572 calls with mean /
median / p95 / max 0.12323 / 0.00796 / 0.77580 / 4.23190 seconds. Combined has
6,570 calls at 0.09322 / 0.01204 / 0.59656 / 3.95001 seconds. The call sets differ
after intermediate-IP divergence, so these conditional values are descriptive,
not a paired per-flight estimate.

Both runs used identical input and algorithm-parameter hashes. Each accepted all
1,537 flights, denied none, passed final conflict verification, terminated after
11 rounds at the LP-gap condition, and returned an optimal final pool IP with
congestion cost 156,817.30255254533. Both recorded 42,879 timed columns and
11,962 spatial routes, no label restart, no compiled fallback or budget decline,
and no lost pricing worker. The final LP cost gap was
0.0008520122267288545 and the configured-domain integer gap was 1.1826% in both.

Every recorded round-level LP and pricing aggregate matched exactly, including LP
objective, pool size, columns added, reduced-cost sum/max, bounds, and gap. The
complete `columns.jsonl` and `paths.jsonl` traces are byte-identical, including
insertion order and exact costs. Full-run capacity claims were not stored by the
trace. The separate matched replay matrix compared all 1,537 per-flight claim
hashes in rounds 1, 6, and 11 and found no mismatch for prep-only or combined.

The candidate made 6,570 measured goal-search calls, all through the compiled
kernel, with zero declines. The baseline recorded 6,572 calls. The difference
comes after the time-limited IP incumbents diverge, which changes later known-column
cutoffs. Excluding timing, worker/PID, and the candidate-only native diagnostics,
per-flight records are identical through round 7. Later differences are confined
to entry cutoff and small label-count changes. Total main labels were
1,251,830,633 versus 1,251,816,978; total bootstrap labels were 438,691,513 versus
438,258,149.

The time-limited intermediate IP first selected a different incumbent in round 7.
It converged back to the same aggregate final cost, delay, detour, and p95 delay;
423 final selected column indices differ, so per-flight trajectories are not
claimed identical. The final restricted-pool IP gap differs only by floating
roundoff (8.31e-14 versus 1.66e-13).

This is one serial pair on one machine and one demand seed. The result supports a
speed improvement for the frozen shared-preparation plus compiled-goal candidate,
but does not provide a statistical interval. Completed-route screening was added
after this source was frozen and is outside this full comparison.
