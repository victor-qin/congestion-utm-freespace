# Archived runs — registry and comparability notes

## ⚠️ Separation-semantics change, 2026-08-14 — read before comparing any delay number

Corridor transit volumes are now filed with a **leading-only** time pad, `[t0, t1 + time_buffer_s)`,
replacing the symmetric `[t0 - buf, t1 + buf)`. Ledger conflict is window overlap, so the enforced
gap between two conflicting transits is the *sum* of their facing pads: it went **8 s → 4 s**, and the
same-lane minimum headway through a cell went **16 s → 12 s**.

That is a real capacity change, not a bookkeeping one.

**Every run archived before this change measured a different airspace.** Delays, denials, objective
values, and any derived curve (delay-vs-λ in particular) are **not comparable** across it — the newer
side has strictly more same-lane throughput and will read lower on delay at fixed λ. Re-run any
baseline you are still actively comparing against; do not mix the two rulers in one figure or table.

This is the same class of hazard as the en-route ruler change in `d29420c` (PR #51): the numbers
still parse, so nothing fails loudly — you just get a wrong answer.

Rationale, the conservation law, and the accepted trade (a flight running *early* is uncovered by
design) are documented at `freespace_sim.volumes.corridor_segment_volume` and in
`context/ASTM_NOTES.md` §Buffers.

## Registry

<!-- One row per archived large run: scenario, planner/mode, commit SHA, cloud URL. -->
<!-- Runs recorded here must state whether they predate the 2026-08-14 filing change above. -->
