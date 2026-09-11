# Full validation benchmark

| Seed | Input match | Verified B/C | Accepted B/C | Final LP gap B/C | Adjusted wall B → C | Pricing B → C |
|---:|:---:|:---:|---:|---:|---:|---:|
| 0 | True | True/True | 1537/1537 | 0.000852012/0.000852012 | 1195.169s → 1061.700s (-11.17%) | 560.800s → 472.031s (-15.83%) |
| 1 | True | True/True | 1545/1545 | 0.000717475/0.000717475 | 1493.946s → 1342.255s (-10.15%) | 746.825s → 644.577s (-13.69%) |

| Seed | Arm | Task mean / median / p95 / max | Bootstrap mean / median / p95 / max | Main compiled mean / median / p95 / max | Worker task total | Utilization |
|---:|---|---:|---:|---:|---:|---:|
| 0 | baseline | 0.2757 / 0.2231 / 0.6834 / 4.5040 | 0.1254 / 0.0667 / 0.4407 / 2.6686 | 0.1455 / 0.1364 / 0.2642 / 4.2962 | 4661.308s | 0.693 |
| 0 | combined | 0.2770 / 0.2267 / 0.6801 / 4.4067 | 0.1241 / 0.0666 / 0.4376 / 2.6036 | 0.1450 / 0.1353 / 0.2645 / 4.2092 | 4683.414s | 0.827 |
| 1 | baseline | 0.2718 / 0.2178 / 0.6447 / 9.1697 | 0.1141 / 0.0641 / 0.3751 / 2.0936 | 0.1530 / 0.1348 / 0.2698 / 7.6546 | 5878.898s | 0.656 |
| 1 | combined | 0.2742 / 0.2218 / 0.6408 / 8.9623 | 0.1133 / 0.0637 / 0.3724 / 2.0648 | 0.1532 / 0.1343 / 0.2739 / 7.3309 | 5931.347s | 0.767 |

Named stage totals and exact quality fields are in `summary.json`. Capture overhead is
reported by each run; adjusted wall is supporting evidence, while matched sweep and row
replays isolate the two changes. Single runs at each seed do not estimate variance.
