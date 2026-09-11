# 25-second iteration-IP policy

| Seed | Raw wall 30→25 | Iteration IP 30→25 | Rounds | Pool | Final cost | Final LP gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1065.784→1020.090s (-4.29%) | 374.122→320.962s | 11→11 | 42879→42879 | 156817.302553→156817.302553 | 0.000852012→0.000852012 |
| 1 | 1345.489→1358.253s (0.95%) | 446.659→412.488s | 14→15 | 45568→46885 | 161135.996334→161062.133254 | 0.000717475→0.000689763 |

Both runs accepted and verified every request, reached the LP-gap stop, certified every priced column without parent revalidation, and ended with an optimal restricted final IP. Seed 0 kept the exact LP, reduced-cost, and ordered-column trace, although recorded search work was not all identical. Seed 1 first changed priced-column counts in round 8, needed round 15, and produced a larger different pool; its final cost is therefore not a same-pool IP comparison. One serial run per policy and seed does not estimate timing variance.

The 25-second cap saved 45.695 seconds on seed 0 but added 12.764 seconds on
seed 1 because the changed trajectory required an extra round. This does not
support changing the 30-second default. The seed-1 final cost was 73.863 lower,
but it came from 1,317 additional columns and one additional pricing round.

The seed-0 round-10 worker spike is mostly timing variation rather than measured
search-work growth. Search counters changed by only +0.182% for bootstrap labels
and +0.004% for main labels, while task time rose 52.6%. Even among the 1,521 of
1,537 flights with exactly matching counter vectors, task time rose 52.5%.
Across all seed-0 rounds, exactly matched-counter flights were 1.05% slower; on
seed 1 they were 0.86% faster. These serial observations do not isolate machine
drift from other unrecorded work.
