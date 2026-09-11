# Component ablations

| Case | Worker sweep B→V2 | Parent acceptance B→V2 | Total B→V2 | Exact parity |
|---|---:|---:|---:|:---:|
| seed0_round6 | 76.947→77.977s | 15.532→0.000752s | 92.479→77.978s (-15.68%) | True |
| seed1_round6 | 90.633→91.017s | 4.344→0.000518s | 94.977→91.018s (-4.17%) | True |

| Case | Initial row scan B→row | Repeated median B→row | Exact parity |
|---|---:|---:|:---:|
| seed0_round1 | 0.290527→0.000938s (-99.68%) | 0.291989→0.000551s (-99.81%) | True |
| seed0_round11 | 0.291124→0.000924s (-99.68%) | 0.290378→0.000531s (-99.82%) | True |
| seed1_round1 | 0.293918→0.000852s (-99.71%) | 0.292111→0.000535s (-99.82%) | True |
| seed1_round11 | 0.296639→0.001041s (-99.65%) | 0.296464→0.000854s (-99.71%) | True |

Cold canonical validation is excluded from measured acceptance time. Row scans use one captured incumbent column per flight and do not reconstruct the full restricted master.
