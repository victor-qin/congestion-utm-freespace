"""A/B test the always-active column_clear short-circuit: BYTE-IDENTICAL check + speedup.

Runs one always-active scenario SERIALLY twice — baseline (legacy foreign-transit scan) vs patched
(skip the scan under `terminal_airspace_always_active`, since the permanent wall already excludes
foreign transit) — then:

  * PARITY: compares every committed plan (status + denial reason + each volume's t-span, terminal id,
    and AABB). The change is only sound if the two runs are byte-for-byte identical.
  * SPEED: reports wall time + ms/flight for each.

Usage: uv run python analysis/ab_column_clear.py [scenario] [--smoke] [--lam-scale F]
"""
from __future__ import annotations

import sys
from time import perf_counter

import freespace_sim.planner.terminal_capacity as tc
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run

_args = sys.argv[1:]
SCEN = next((a for a in _args if not a.startswith("-")), "density_faa_wing_zipline")
SMOKE = "--smoke" in _args
LAM = float(_args[_args.index("--lam-scale") + 1]) if "--lam-scale" in _args else 1.0

spec = get_scenario(SCEN)
if SMOKE:
    spec = with_overrides(spec, horizon_s=1800.0, demand_duration_s=300.0)
if LAM != 1.0:
    dov = {"lam_per_uss": {k: round(v * LAM, 2) for k, v in spec.demand.lam_per_uss.items()}} \
        if spec.demand.lam_per_uss else None
    spec = with_overrides(spec, lam_per_hour=round(spec.lam_per_hour * LAM, 2), demand_overrides=dov)
cfg, demand = spec.config(), spec.demand_model()
assert cfg.terminal_airspace_always_active, f"{SCEN} is not always-active — the short-circuit is a no-op"
print(f"scenario={SCEN} smoke={SMOKE} lam_scale={LAM}  always_active={cfg.terminal_airspace_always_active}",
      flush=True)


def _vol_sig(v):
    lo, hi = v.aabb()
    return (float(v.t_start), float(v.t_end), v.terminal_id,
            tuple(float(x) for x in lo), tuple(float(x) for x in hi))


def _intent_sig(it):
    return (it.status.value, getattr(it.denial_reason, "value", None),
            tuple(_vol_sig(v) for v in it.volumes))


def _timed_run(skip: bool):
    tc.SKIP_FOREIGN_WHEN_WALLED = skip           # toggle the module flag before the run
    t0 = perf_counter()
    res = run(cfg, demand=demand, parallel=None)  # serial → clean, deterministic, comparable
    return perf_counter() - t0, res.intents


t_base, base = _timed_run(skip=False)            # legacy: real foreign-transit scan
t_skip, skip = _timed_run(skip=True)             # patched: skip under always-active
tc.SKIP_FOREIGN_WHEN_WALLED = True               # restore default

# ---- parity ----
n = len(base)
assert len(skip) == n, f"flight count differs: {len(base)} vs {len(skip)}"
diffs = [i for i in range(n) if _intent_sig(base[i]) != _intent_sig(skip[i])]
acc_b = sum(1 for it in base if it.accepted)
acc_s = sum(1 for it in skip if it.accepted)
print(f"\nPARITY: {n} flights  accepted base={acc_b} skip={acc_s}  differing plans={len(diffs)}", flush=True)
if diffs:
    i = diffs[0]
    print(f"  FIRST DIVERGENCE at flight index {i} (fid={base[i].request.flight_id}):", flush=True)
    print(f"    base:  status={base[i].status.value} nvols={len(base[i].volumes)}", flush=True)
    print(f"    skip:  status={skip[i].status.value} nvols={len(skip[i].volumes)}", flush=True)
    print("  => NOT byte-identical: column_clear returned False for a real foreign transit under the "
          "wall (hex-edge case). Do NOT ship the skip; use the local-query variant instead.", flush=True)
else:
    print("  => BYTE-IDENTICAL ✓  every committed plan matches exactly.", flush=True)

# ---- speed ----
print(f"\nSPEED: baseline(scan)={t_base:.1f}s ({t_base / n * 1000:.1f} ms/flight)   "
      f"patched(skip)={t_skip:.1f}s ({t_skip / n * 1000:.1f} ms/flight)   "
      f"speedup ×{t_base / t_skip:.2f}  ({(1 - t_skip / t_base) * 100:.0f}% faster)", flush=True)
