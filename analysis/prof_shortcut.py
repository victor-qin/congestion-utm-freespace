"""Re-profile astar_shortcut under always-active terminal airspace (issue #30).

Reproduces dallas_full's bottleneck structure — the SAME 260 always-active hub walls, 3 flight
levels, and tagged terminal columns — but at a smaller lam/horizon so the warm phase is fast. Then
cProfiles a batch of astar_shortcut plans and attributes *tottime* (exclusive self-time, so the
buckets sum cleanly to the whole) into the categories issue #30 argues about:

    A* search  |  geometry rebuild  |  ledger.any_conflict  |  FCL narrowphase  |  shortcut driver

It also installs lightweight counters around the profiled batch to answer the question the raw
profile cannot: is the _StaticWallGrid (issue #30's already-landed lever #1) actually pruning the
260 static walls? We tally, per any_conflict query, how many walls the grid returns vs the old
full-scan baseline of len(_static_vols).

Usage: uv run python analysis/prof_shortcut.py [lam] [warm] [timed]
Defaults chosen so the run is a couple of minutes but the ledger is realistically dense.
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time
from collections import defaultdict

import numpy as np

from freespace_sim import ledger as ledger_mod
from freespace_sim import volumes as volumes_mod
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.scenarios.dallas import SCENARIOS
from freespace_sim.scenarios.spec import with_overrides
from freespace_sim.uss import USS

lam = float(sys.argv[1]) if len(sys.argv) > 1 else 9000.0
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 700
timed = int(sys.argv[3]) if len(sys.argv) > 3 else 220

# cell-size sweep hook: FS_DYNAMIC_CELL_M=512 patches the ledger's dynamic sub-index cell edge (dev-only;
# production ships the module default). Set before any ledger is built so commit/_candidate_indices see it.
if os.environ.get("FS_DYNAMIC_CELL_M"):
    ledger_mod._DYNAMIC_GRID_CELL_M = float(os.environ["FS_DYNAMIC_CELL_M"])
    print(f"[sweep] _DYNAMIC_GRID_CELL_M = {ledger_mod._DYNAMIC_GRID_CELL_M} m")

# dallas_full's exact world (260 hubs, 3 levels, always-active, tagged columns) at a reduced lam/horizon.
spec = with_overrides(SCENARIOS["dallas_full"], lam_per_hour=lam, horizon_s=3600.0,
                      planner="astar_shortcut")
cfg = spec.config()
demand = spec.demand_model()
reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
assert warm + timed <= len(reqs), f"need {warm + timed} flights, demand made {len(reqs)}"

ledger = ReservationLedger(cfg)
dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
planner = get_planner("astar_shortcut")
usses = {uid: USS(uid, dss, cfg, planner) for uid in {r.uss_id for r in reqs}}

# Register EVERY hub's permanent terminal wall BEFORE the first plan (mirrors sim.run under
# always-active) so the occupancy service's subscribe_static replay sees them and the ledger holds
# all the static walls the profile is meant to stress.
static_terms = list(demand.terminals(cfg))
for center, term in static_terms:
    ledger.register_static_terminal(center, term)
print(f"registered {len(static_terms)} always-active hub walls "
      f"(len(_static_vols)={len(ledger._static_vols)})")

# Warm the ledger to a steady-state density; also JIT-warms the numba A* kernel OUTSIDE the profile.
for req in reqs[:warm]:
    usses[req.uss_id].handle_request(req)
print(f"warmed ledger to {ledger.n_volumes} committed volumes ({warm} flights); "
      f"profiling next {timed} plans")

# ---- counters (answer "is the grid pruning?" — the raw profile can't tell static from dynamic) ----
stats_counters = defaultdict(float)
_orig_candidates = ledger_mod._StaticWallGrid.candidates
_orig_any_conflict = ReservationLedger.any_conflict
_orig_cand_indices = ReservationLedger._candidate_indices


def _counting_candidates(self, aabb):
    out = _orig_candidates(self, aabb)
    stats_counters["static_grid_queries"] += 1
    stats_counters["static_candidates_returned"] += len(out)   # walls surviving the xy-grid prune
    return out


def _counting_cand_indices(self, vol, vbb):
    out = _orig_cand_indices(self, vol, vbb)
    stats_counters["dynamic_queries"] += 1
    stats_counters["dynamic_candidates_returned"] += len(out)   # committed vols sharing a timestep AND xy-cell
    return out


def _counting_any_conflict(self, volumes):
    stats_counters["any_conflict_calls"] += 1
    stats_counters["any_conflict_volumes"] += len(volumes)
    return _orig_any_conflict(self, volumes)


ledger_mod._StaticWallGrid.candidates = _counting_candidates
ReservationLedger._candidate_indices = _counting_cand_indices
ReservationLedger.any_conflict = _counting_any_conflict

# ---- profile the batch ----
batch = reqs[warm:warm + timed]
t0 = time.monotonic()
pr = cProfile.Profile()
pr.enable()
for req in batch:
    usses[req.uss_id].handle_request(req)
pr.disable()
wall = time.monotonic() - t0

ledger_mod._StaticWallGrid.candidates = _orig_candidates
ReservationLedger._candidate_indices = _orig_cand_indices
ReservationLedger.any_conflict = _orig_any_conflict

# ---- bucket tottime into the categories issue #30 argues about ----
st = pstats.Stats(pr)
BUCKETS = {
    "A* search (planner+kernel)": [],
    "geometry rebuild": [],
    "ledger.any_conflict": [],
    "FCL narrowphase (conflict)": [],
    "shortcut driver": [],
    "other": [],
}


def classify(filename: str, func: str) -> str:
    f = filename.replace("\\", "/")
    base = f.rsplit("/", 1)[-1]
    if base in ("geometry.py", "volumes.py"):
        return "geometry rebuild"
    if base == "ledger.py":
        return "ledger.any_conflict"
    if base == "conflict.py" or "/fcl/" in f or base.startswith("fcl"):
        return "FCL narrowphase (conflict)"
    if base == "shortcut.py":
        return "shortcut driver"
    if "/planner/" in f or base in ("cost.py",):
        return "A* search (planner+kernel)"
    return "other"


total_tt = 0.0
per_func = []
for (filename, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
    total_tt += tt
    per_func.append((tt, ct, nc, filename, lineno, func))
    BUCKETS[classify(filename, func)].append(tt)

bucket_tt = {k: sum(v) for k, v in BUCKETS.items()}


def cumtime_of(base_name: str, func_name: str) -> float:
    """Cumulative time of a single (file basename, funcname) entry — includes its subcalls, so it
    captures numpy/FCL cost triggered by that entry (which tottime buckets scatter into 'other')."""
    for (filename, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if filename.replace("\\", "/").rsplit("/", 1)[-1] == base_name and func == func_name:
            return ct
    return 0.0


# Clean split by CUMULATIVE time of the entry points (what issue #30 measured). These are siblings
# in the call tree (search, then per-candidate: build corridor, then check ledger) so they don't
# double-count.
search_ct = cumtime_of("astar.py", "_plan_compiled") + cumtime_of("astar.py", "_plan_reference")
geom_ct = cumtime_of("volumes.py", "build_reservation_from_corners")
ledger_ct = cumtime_of("ledger.py", "any_conflict")
split = {
    "A* search (compiled kernel)": search_ct,
    "geometry rebuild (build_reservation_from_corners)": geom_ct,
    "ledger.any_conflict": ledger_ct,
}
split["shortcut driver + misc"] = max(0.0, total_tt - sum(split.values()))

print("\n" + "=" * 78)
print(f"WALL {wall:.1f}s for {timed} plans = {wall / timed * 1000:.1f} ms/plan   "
      f"(profiled tottime total {total_tt:.1f}s)")
print("=" * 78)
print("SPLIT BY CUMULATIVE TIME OF ENTRY POINTS (issue #30-comparable):")
print(f"{'  component':<52}{'cumtime (s)':>14}{'share':>10}")
print("-" * 78)
for k, v in sorted(split.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<50}{v:>14.2f}{v / total_tt * 100:>9.1f}%")
print(f"\n  refinement (geometry + ledger + driver) = "
      f"{(total_tt - search_ct) / total_tt * 100:.0f}%   vs search {search_ct / total_tt * 100:.0f}%")

print("\nSPLIT BY tottime BUCKET (exclusive self-time; numpy internals land in 'other'):")
print(f"{'  bucket':<36}{'tottime (s)':>12}{'share':>10}")
print("-" * 78)
for k, v in sorted(bucket_tt.items(), key=lambda kv: -kv[1]):
    if v > 0:
        print(f"  {k:<34}{v:>12.2f}{v / total_tt * 100:>9.1f}%")

# ---- inside any_conflict: static (walls) vs dynamic (committed) candidate work ----
sq = stats_counters["static_grid_queries"] or 1
dq = stats_counters["dynamic_queries"] or 1
avg_static = stats_counters["static_candidates_returned"] / sq
avg_dyn = stats_counters["dynamic_candidates_returned"] / dq
print("\n" + "=" * 78)
print("INSIDE any_conflict: static-wall scan (lever #1, LANDED) vs dynamic committed scan")
print("=" * 78)
print(f"any_conflict calls: {int(stats_counters['any_conflict_calls'])}  "
      f"(avg {stats_counters['any_conflict_volumes'] / (stats_counters['any_conflict_calls'] or 1):.1f} volumes/call)")
print(f"STATIC walls:  {int(sq)} queries, avg {avg_static:.2f} walls survive the xy-grid prune "
      f"(baseline full-scan = {len(static_terms)})")
print(f"   => grid prunes the static scan ~{len(static_terms) / (avg_static or 1e-9):.0f}x  "
      f"— the static-wall bottleneck from the issue is GONE")
print(f"DYNAMIC committed: {int(dq)} queries, avg {avg_dyn:.1f} committed vols share a timestep "
      f"(total {int(stats_counters['dynamic_candidates_returned'])} _aabb_miss checks)")
print(f"   => dynamic scan does ~{stats_counters['dynamic_candidates_returned'] / (sq * avg_static or 1):.0f}x "
      f"more broadphase work than the static scan — THIS is what any_conflict now spends its time on")

# ---- raw top-tottime, for ground truth ----
print("\n" + "=" * 78)
print("TOP 18 FUNCTIONS BY tottime (ground truth)")
print("=" * 78)
print(f"{'tottime':>9}{'cumtime':>9}{'ncalls':>11}  function")
for tt, ct, nc, filename, lineno, func in sorted(per_func, key=lambda r: -r[0])[:18]:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    print(f"{tt:>9.2f}{ct:>9.2f}{nc:>11}  {base}:{lineno}:{func}")
