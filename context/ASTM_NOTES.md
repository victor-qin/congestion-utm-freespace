# ASTM F3548-21 — distilled notes for this simulator

Source: `context/F3548-21.pdf` (UAS Traffic Management (UTM) UAS Service Supplier (USS)
Interoperability). These are the sections that drive the code.

## The one predicate everything rests on — *conflict* (§3.2.8)
> "a situation where two operational intents intersect in space and time. For operational
> intents to intersect both in space and time, at least one 4D volume from each operational
> intent must intersect. For two 4D volumes to intersect, the spatial dimensions of the 4D
> volumes must share at least one point **and** the start/end time range for the two 4D volumes
> must overlap."

→ `conflict.volumes_conflict(a, b)` = **time-window overlap AND 3D spatial intersection**.
We implement the spatial test with python-fcl (exact Box/Cylinder collision).

## 4D volumes (§3.2.1–3.2.2)
- **3D volume** = a volume of airspace (lat, lon, altitude).
- **4D volume** = a 3D volume + a start and end time.
→ `volumes.Volume4D` = an fcl 3D shape (Box or Cylinder, with pose) + `[t_start, t_end]`.

## Operational intent = the reservation (§4.3.1, §4.3.5–4.3.8)
- An **operational intent** is one or more contiguous/overlapping 4D volumes bounding the flight.
- **Trajectory-based** intent (§4.3.5): "a series of volumes that follow the desired flight path
  and overlap in space and time." Lateral/vertical dimensions buffer the centerline.
  → our **corridor**: one oriented `Box` per timestep, consecutive boxes overlap (contiguity).
- **Area-based** intent (§4.3.5): a single volume not tied to a path, e.g. "starting/stopping on
  the surface or starting/stopping in the air."
  → our **hover reservation**: a `Cylinder` at takeoff/landing covering the climb/descent.
- A single operational intent may contain **both** (§4.3.5): hover-cylinder + corridor + hover-cylinder.

## Buffers — Total System Error + time buffer (§4.3.8–4.3.11)
- Lateral/vertical buffer = Total System Error = PDE + FTE + NSE (§4.3.8, Fig. 2).
  → `corridor_width_m`, `corridor_height_m` (knobs).
- Time buffer absorbs timing inaccuracy (wind, departure-time uncertainty) (§4.3.11).
  → `time_buffer_s` (makes consecutive boxes overlap in time, too).

## Strategic Conflict Detection — method NOT prescribed (§4.2.4)
> "The manner in which a USS finds a conflict-free route during planning or resolves a conflict
> that arises need not be prescribed and should allow for innovation."

→ This is the open door for RRT* / decoupled / MILP. The standard only constrains the **output**
(committed volumes must not conflict), not the algorithm. Hence our pluggable `Planner` protocol.

## Priority within a level = FCFS (§4.2.5)
> "Where conflicts are not allowed within the same priority level, the first-planned operation is
> given priority over subsequent operations."

→ `mechanism.FCFSMechanism`: earlier-filed intents are committed and become obstacles; the
newcomer must plan around them or be denied. (This is the core inefficiency the research probes.)

## Operational intent states (§4.4, Fig. 4)
Accepted → Activated → (Nonconforming ↔ Activated) → Contingent → Ended.
→ `types.IntentStatus`. v0 (strategic only) uses ACCEPTED / REJECTED / ENDED; the off-nominal
states (NONCONFORMING, CONTINGENT) are placeholders for the future BlueSky tactical layer, which
is where conformance monitoring and off-nominal 4D volumes (§4.4.4) come in.

## Scope boundaries the standard itself draws (relevant to our scope)
- §1.2 / §1.13.2: this version is **strategic** only; tactical conflict detection is explicitly
  out of scope. Matches our "strategic management only" focus.
- §1.11: the standard does **not** establish fairness/equity requirements — exactly the gap the
  FCFS-congestion experiment is meant to illuminate.
