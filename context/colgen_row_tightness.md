# colgen capacity rows: sound, but not tight (GitHub issue #101)

> Measured 2026-08-18/19 on `density_faa_wing_zipline`. Companion to `colgen_plan.md` §2
> ("Verified ledger geometry") and user decision #1 (row construction). The plan established
> that the rows **cover** the ledger; this document measures how much they over-cover, why,
> and what it costs. Gate: `tests/test_colgen_constraint_tightness.py`.

## The contract we have, and the one we do not

`windows._cross_check_conflicts` states it exactly: *"a geometric conflict must imply an
intersecting row claim."* That is **soundness**, and it is what priority 1 of the plan needs —
any row-feasible integer solution files with 0 denials. Its scan opens with

```python
if not volumes_conflict(first, shifted):
    continue          # every NON-conflicting pair is discarded here
```

so the converse — an intersecting row claim implies a real conflict, i.e. **tightness** — has
never been evaluated anywhere in the tree. The asymmetry is safe by design: a model that
over-claims refuses legal schedules but never files a conflicting one. It is not free.

## How much is over-claimed

### En-route arcs — run the shipped scan in both directions

```
template pairs scanned          8,228
real conflicts                  2,772
claims intersect, NO conflict   1,408   = 17.1% of all pairs
conflict but claims disjoint        0   <- soundness intact
of the 4,180 pairs colgen BLOCKS, 33.7% are not real conflicts
```

### Endpoint dwell — a closed-form over-reach

Two hover cylinders conflict iff their centres are within `2 * effective_hover_radius_m`
(**120 m** shipped). Their claim sets keep intersecting out to **309.28 m** — **2.58x**. The
cause is arithmetic in `endpoint_claim_cells`' own docstring: the claim radius per endpoint is

```
hover 60 + max(corridor_width 60, hover 60) + circumradius 69.28  =  189.28 m
```

sized for the cylinder-vs-transit-**box** worst case, then applied uniformly — including to
cylinder-vs-cylinder, which needs only 120 m. **Every pair in `(120 m, 309 m]` is one the
ledger accepts and the LP refuses.**

## What it costs on a real instance

A* FCFS at 600 flights produces a schedule the ledger cleared in full. Converting it to columns
(600/600 convert) and loading the claims:

```
rows claimed                    281,908
rows OVER CAP                        89   (0.032%)   — all `cell`, zero terminal
excess  p50 / p90 / max           1 / 1 / 1          — always exactly two aircraft
flights touching one                 16 of 600       (2.67%)
```

Every one of the 10 distinct flight pairs was checked against `conflict.volumes_conflict`, the
ledger's own ASTM §3.2.8 arbiter:

```
pairs whose filed volumes CONFLICT ....  0 of 10
pairs separated above the 103.92 m floor  10 of 10
minimum separation  min/median/max ....  207.85 / 207.85 / 623.54 m
contested cell on BOTH flights' paths .  0 of 10
```

**Zero are real conflicts.** Repairing them by ground shift costs 728 s across 19 flights
(~1% of the schedule's cost). For scale, repairing colgen's own geodesic seeds the same way
costs 20,152 s across 354 flights — 28x more.

### Where the over-cap claims come from

Attributing all 178 over-cap flight-row claims to the function that emitted them:

| source | claims | share |
|---|---|---|
| `_endpoint_claims`, origin dwell | 79 | **44%** |
| `_endpoint_claims`, destination dwell | 71 | **40%** |
| `_visit_claims`, en-route | 25 | 14% |
| origin dwell + en-route | 3 | 2% |

**84% is endpoint dwell**, not en-route flight. Two hypotheses that the data refutes: the
pairs are *not* sharing a hex (they are 208–624 m apart, wider than the 138.56 m cell), and
this is *not* the terminal own-column exemption leaking (0% of the rows are inside a hub
column; they sit 604 m – 3.6 km from the nearest hub). Reconstruction drift was checked and
cannot account for it — worst per-cell step delta is 1, and clearing a 4-step claim window
needs >=4 steps of separation.

## Why this is a performance problem, not only a correctness one

`RestrictedMaster.solve_ip` is a separation loop: solve the binary RMP, find
`violated_claim_rows`, materialize them, re-solve. Spurious rows are rows the integer solution
can violate, so they buy extra rounds — and rounds are the binding constraint at scale. Measured
at 2,000 flights the per-round MIP cost runs 22 s, 1,246 s, 2,328 s, so the loop completes 3
rounds and times out; at 1,500 it completes 11. That round count then sets everything
downstream: repairing the timed-out last round costs a 6.8% feasibility tax at 11 rounds and
**21.4%** at 3.

So tightening the rows is not a ~1%-of-cost cleanup. It is the lever on the round count, which
is the lever on whether colgen delivers a flyable schedule at all at 2,000 flights.

## Direction of the fix

1. **Endpoint dwell first** — 84% of the binding over-constraint, and the over-reach has a
   closed form (189.28 m of claim radius against a 120 m real conflict). Ask whether the
   cylinder-vs-box worst case needs to be charged to cylinder-vs-cylinder pairs, and whether
   the footprint *around* an endpoint needs cap 1 at all when `RowIndex.cap` already returns a
   real pad count for `term` rows.
2. **En-route chords second** — 14% here, but 33.7% of blocked template pairs. A cap-1 cell row
   is a clique over every chord through the hex; the true within-cell conflict graph is not a
   clique. Key rows on `(entry side, exit side)` and emit one row per maximal clique of
   mutually-conflicting chords. Cheap because the lattice is translation-invariant: whether two
   chords conflict depends only on the pair and their step offset, so the compatibility table is
   one small precomputation reused at every row.

**Soundness is the invariant.** Today's model is a cover; any refinement must stay one, and
`windows.py`'s FCL coverage guard plus the soundness halves of the new tightness tests are the
acceptance gate. Getting this wrong files schedules that genuinely collide, which is far worse
than being conservative.
