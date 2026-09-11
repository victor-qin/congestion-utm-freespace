# Capacity scan and parent-validation benchmark

The combined V2 snapshot improved full wall time on both matched FAA outbound
seeds while preserving the final cost, aggregate metrics, and pricing pool exactly.

| Seed | Flights | Raw wall baseline → V2 | Capture-adjusted wall | Pricing | Final LP gap | Cost | Columns |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,537 | 1198.650 → 1065.784 s (-11.08%) | 1195.169 → 1061.700 s (-11.17%) | 560.800 → 472.031 s (-15.83%) | 0.000852012 both | 156817.302553 both | 42,879 both |
| 1 | 1,545 | 1499.890 → 1345.489 s (-10.29%) | 1493.946 → 1342.255 s (-10.15%) | 746.825 → 644.577 s (-13.69%) | 0.000717475 both | 161135.996334 both | 45,568 both |

Every arm accepted and verified every request, had no repair, fallback, restart,
worker loss, or native-goal decline, and finished with an optimal restricted IP.
All input hashes matched within each seed. Path and column identity/numeric hashes
matched on both seeds. The semantic iteration trace matched on seed 0. Seed 1's
finite iteration-IP incumbents and final selected trajectories differed, while its
final cost, aggregate metrics, and LP/pricing pool remained exact.

The capacity-row stage fell by 63.942 seconds (-86.78%) on seed 0 and 87.519
seconds (-96.53%) on seed 1. Parent validation of priced columns fell from 91.886
and 106.465 seconds to a 0.0046 and 0.0058 second certificate check. That check
timer excludes receipt issuance and worker certification, which are already inside
the full pricing time. All 10,457 and 12,998 priced columns were certified, with
zero candidate revalidation. Some
work moved into selected-column checking: `canonical_iteration_ip` increased by
21.974 and 29.349 seconds. Worker task totals did not fall, and per-flight task
means rose slightly; the end-to-end gain comes from removing parent-side scans
and validation rather than faster pricing workers.

Matched round-6 replays confirmed exact raw and cold-canonical parity. Pricing plus
parent acceptance changed from 92.479 to 77.978 seconds (-15.68%) on seed 0 and
94.977 to 91.018 seconds (-4.17%) on seed 1. Four synthetic row checks using one
real incumbent column per flight showed exact load and row parity; initial row
scans fell from 0.291–0.297 seconds to 0.00085–0.00104 seconds. All four cases
had zero violated rows, so their coefficient parity check was vacuous. Focused
unit tests cover cases that add rows. These row checks do not reconstruct the full
multi-column restricted master.

A 25-second iteration-IP policy was also tested on the same V2 source. Seed 0 kept
the same 11-round pool and final cost and improved wall time by 45.695 seconds
(-4.29%). Seed 1 required 15 rounds instead of 14, increased wall time by 12.764
seconds (+0.95%), built 46,885 rather than 45,568 columns, and found a cost 73.863
lower (-0.0458%). Its first LP/RC/column divergence occurred in round 8. Because
the runtime effect was inconsistent and seed 1's quality difference came from a
different, larger pool, the evidence supports retaining the 30-second default.

The seed-0 policy run had a large round-10 worker timing spike with only tiny work
counter changes. Even the 1,521 of 1,537 round-10 flights with exactly matching
counter vectors were 52.5% slower, while exactly matched-counter flights across
all seed-0 rounds were only 1.05% slower and those on seed 1 were 0.86% faster.
This is unexplained serial timing variation; the small label-count differences do
not account for the spike.

The primary and policy comparisons are one serial run per arm and seed, so they do
not estimate timing variance. Capture writes were included in raw wall and
separately subtracted only for the supporting adjusted-wall figures. Both frozen
297-file source snapshots reverified byte-for-byte after all runs.
