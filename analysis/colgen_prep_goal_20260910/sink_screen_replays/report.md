# Compiled completed-route screening replays

The screened frozen source passed exact output, identity, claim, cost/reduced-cost,
and label-work parity for all 1,537 flights in captured rounds 1, 6, and 11. It used
the compiled goal kernel for every expected call and recorded no decline.

| Round | Old combined wall (s) | Screened wall (s) | Reduction | Goal seconds old / screened | Asked / skipped labels |
|---:|---:|---:|---:|---:|---:|
| 1 | 84.072 | 80.194 | 4.61% | 16.163 / 11.741 | 1,471 / 13,268 |
| 6 | 86.094 | 77.003 | 10.56% | 63.130 / 11.622 | 1,119 / 121,616 |
| 11 | 84.998 | 77.033 | 9.37% | 66.179 / 9.928 | 1,077 / 136,781 |

A fresh reverse round-11 pair measured old combined at 84.753 seconds and
screened at 77.299 seconds. Averaging the forward and reverse pairs gives 84.875
versus 77.166 seconds, a 7.709-second (9.08%) reduction. Average worker task time
fell from 895.056 to 836.546 seconds (6.54%), and goal time fell from 66.528 to
9.961 seconds (85.03%). Both reverse runs passed the same exact parity checks.

Asked/skipped diagnostics count finished labels. A finished label can have more
than one destination-lane callback, so these counts are not callback counts. These
are fresh-process pricing sweeps; they include worker startup and cold graph/prep
setup and do not measure a full master/IP solve.
