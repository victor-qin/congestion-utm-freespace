"""Per-flight attribution harness — where does plan() time go as the ledger fills?

Runs a real named scenario (default ``density_faa_wing_zipline``) **serially** through ``sim.run`` and,
per window of flights, splits the mean per-flight ``solve_time_s`` into exclusive timed buckets and
normalizes the search bucket by node expansions:

    total_ms  =  direct_any_conflict_ms  +  column_clear_ms  +  exit_clear_ms  +  search_ms
    ns_per_expansion  =  search_ms / expansions_per_flight

That separates the two things that grow with the committed-flight count N — which is exactly the
"per-flight time rises as flights increase" symptom under investigation:

  * ``ns_per_expansion``  — rises with compiled-occupancy FRAGMENTATION (``_cocc.corr.nslots``): the
    kernel's ``_blocked`` walk traverses each hot cell's free-interval list from step 0, past every
    historical fragment the (watermark-only) compiled eviction never reclaimed. This is the
    *later-in-run* slowdown and it is FIXABLE (make the compiled pool reclaim like the reference
    ``HexOccupancyService`` does).
  * ``expansions/flight`` — rises with airspace DENSITY (more traffic ⇒ more reroutes). This is the
    *higher-density* slowdown and it is largely INHERENT deconfliction cost.

``any_conflict_ms`` is measured directly (not inferred) and strictly inside ``planner.plan`` so FCFS
commit and final verification cannot contaminate the solve-time attribution. Calls nested under
``exit_clear`` stay in that gate's bucket instead of being double-counted.

Serial on purpose: the per-flight *algorithmic* cost is identical serial vs parallel (each worker's
``plan()`` runs against a full replica ledger); parallel only rescales wall-time and adds
cache-contention noise (see ``analysis/prof_memory.py``), which would muddy the attribution.

Usage:
    uv run python analysis/prof_ledger_scaling.py [scenario] [--smoke] [--lam-scale F] [--window N]

    scenario     named registry scenario (default density_faa_wing_zipline)
    --smoke      shrink horizon+demand_duration for a fast harness check (~hundreds of flights)
    --lam-scale  scale the offered load (both lam_per_hour and per-USS rates) for an across-density point
    --window     flights per reported window (default 200)
"""
from __future__ import annotations

import sys
from collections import Counter
from functools import wraps
from time import perf_counter

import freespace_sim.sim as sim_mod
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.astar._packed import P_NXT
from freespace_sim.planner.terminal_capacity import TerminalCapacity
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run
from freespace_sim.types import IntentStatus

# ---- args ----
_args = sys.argv[1:]
SCENARIO = next((a for a in _args if not a.startswith("-")), "density_faa_wing_zipline")
SMOKE = "--smoke" in _args


def _opt(flag: str, default: float) -> float:
    return float(_args[_args.index(flag) + 1]) if flag in _args else default


LAM_SCALE = _opt("--lam-scale", 1.0)
WINDOW = int(_opt("--window", 200))

# ---- build the scenario spec (mirrors experiments/run.py:174-176) ----
spec = get_scenario(SCENARIO)
if SMOKE:
    # BOTH knobs, per ScenarioSpec.config's warning (spec.py:120-129): shrinking horizon alone leaves
    # the departure lead past the horizon and the run degenerates onto the box-guard fallback path.
    spec = with_overrides(spec, horizon_s=1800.0, demand_duration_s=300.0)
if LAM_SCALE != 1.0:
    dov = {}
    if spec.demand.lam_per_uss:                     # density_* scenarios drive per-USS streams
        dov["lam_per_uss"] = {k: round(v * LAM_SCALE, 2) for k, v in spec.demand.lam_per_uss.items()}
    spec = with_overrides(spec, lam_per_hour=round(spec.lam_per_hour * LAM_SCALE, 2),
                          demand_overrides=dov or None)

cfg = spec.config()
demand = spec.demand_model()
if demand is None:
    uss_ids = ("default",)
elif hasattr(demand, "n_hubs_per_uss"):
    uss_ids = tuple(demand.n_hubs_per_uss)
else:
    uss_ids = tuple(getattr(demand, "uss_ids", ("default",)))
if len(uss_ids) > 1:
    raise ValueError(
        f"{SCENARIO} has {len(uss_ids)} USSs ({uss_ids}); this profiler attributes one planner only. "
        "Choose a single-USS scenario."
    )
print(f"scenario={SCENARIO} smoke={SMOKE} lam_scale={LAM_SCALE} window={WINDOW}\n"
      f"region={cfg.region_size_m} horizon={cfg.horizon_s}s demand_window={cfg.demand_duration_s}s "
      f"lam={cfg.lam_per_hour}/h planner={cfg.planner} always_active={cfg.terminal_airspace_always_active}",
      flush=True)


# ---- free-interval list-length sampler (read-only; the walk we actually pay in the kernel) ----
def _list_stats(pool, max_heads: int = 20000):
    """(max, mean, n_fragmented) list length over a stride-sample of cell heads. nslots-NC already
    gives TOTAL splits for free; this exposes the WORST single-cell walk, which is what a query pays."""
    iv, nc = pool.iv, pool.NC
    stride = max(1, nc // max_heads)
    mx = tot = frag = seen = 0
    for c in range(0, nc, stride):
        n, slot = 0, c
        while slot != -1 and n <= 1_000_000:
            n += 1
            slot = int(iv[slot, P_NXT])
        mx = max(mx, n)
        tot += n
        frag += n > 1
        seen += 1
    return mx, tot / max(1, seen), frag


# ---- capture the planner and scope counters to its plan() call (not commit or verification) ----
captured: list = []
_orig_get = sim_mod.get_planner
SCOPE = {"plan": 0, "tcap": 0}


def _capturing_get(name):
    planner = _orig_get(name)
    original_plan = planner.plan

    @wraps(original_plan)
    def _scoped_plan(*args, **kwargs):
        SCOPE["plan"] += 1
        try:
            return original_plan(*args, **kwargs)
        finally:
            SCOPE["plan"] -= 1

    planner.plan = _scoped_plan
    captured.append(planner)
    return planner


sim_mod.get_planner = _capturing_get

# ---- exclusive planner-time buckets (cumulative; per-window deltas) ----
AC = {"t": 0.0, "n": 0}
_orig_ac = ReservationLedger.any_conflict


def _timed_ac(self, volumes):
    # FCFS commit and final verification also call this method but sit outside planner.plan. Calls nested
    # inside exit_clear belong to that terminal-capacity bucket, not here.
    if not SCOPE["plan"] or SCOPE["tcap"]:
        return _orig_ac(self, volumes)
    t0 = perf_counter()
    try:
        return _orig_ac(self, volumes)
    finally:
        AC["t"] += perf_counter() - t0
        AC["n"] += 1


# Split the gate: column_clear (foreign-transit tail-scan) vs exit_clear. The tcap depth makes
# any_conflict inside exit_clear exclusive to EC instead of double-counting it in AC.
CC = {"t": 0.0, "n": 0}
EC = {"t": 0.0, "n": 0}
_orig_cc = TerminalCapacity.column_clear
_orig_ec = TerminalCapacity.exit_clear


def _timed_cc(self, *a, **k):
    if not SCOPE["plan"]:
        return _orig_cc(self, *a, **k)
    SCOPE["tcap"] += 1
    t0 = perf_counter()
    try:
        return _orig_cc(self, *a, **k)
    finally:
        CC["t"] += perf_counter() - t0
        CC["n"] += 1
        SCOPE["tcap"] -= 1


def _timed_ec(self, *a, **k):
    if not SCOPE["plan"]:
        return _orig_ec(self, *a, **k)
    SCOPE["tcap"] += 1
    t0 = perf_counter()
    try:
        return _orig_ec(self, *a, **k)
    finally:
        EC["t"] += perf_counter() - t0
        EC["n"] += 1
        SCOPE["tcap"] -= 1


# ---- per-window accumulators + the per-flight hook (sim.run progress callback: done,total,intent) ----
W = {"solve": 0.0, "exp": 0, "n": 0, "acc": 0, "den": 0}
prev = {"ac_t": 0.0, "cc_t": 0.0, "ec_t": 0.0}
reasons: Counter = Counter()
windows: list[dict] = []       # first/last snapshots for the closing readout
TOT = {"solve": 0.0, "exp": 0, "n": 0, "acc": 0, "den": 0}


def _reset_window():
    W.update(solve=0.0, exp=0, n=0, acc=0, den=0)


def _emit_window(done, total):
    if not W["n"]:
        return
    p = captured[0]
    n = W["n"]
    total_ms = W["solve"] / n * 1000.0
    ac_ms = (AC["t"] - prev["ac_t"]) / n * 1000.0
    cc_ms = (CC["t"] - prev["cc_t"]) / n * 1000.0     # column_clear (foreign-transit tail-scan)
    ec_ms = (EC["t"] - prev["ec_t"]) / n * 1000.0     # exit_clear
    search_ms = total_ms - ac_ms - cc_ms - ec_ms      # remainder after disjoint direct/gate buckets
    exp = W["exp"] / n
    ns_exp = (search_ms / exp * 1e6) if exp > 0 else float("nan")
    cocc = p._cocc
    corr_nslot, nc = cocc.corr.nslots, cocc.corr.NC
    mx, _mean_len, _frag = _list_stats(cocc.corr)
    row = {"done": done, "total_ms": total_ms, "ac_ms": ac_ms, "ac_pct": ac_ms / total_ms * 100.0,
           "cc_ms": cc_ms, "ec_ms": ec_ms, "search_ms": search_ms, "exp": exp, "ns_exp": ns_exp,
           "corr_nslot": corr_nslot, "frag_cell": (corr_nslot - nc) / nc, "maxlist": mx,
           "n_added": cocc.n_added}
    windows.append(row)
    print(f"[{done:>5}/{total}] tot={total_ms:7.1f}ms ac={ac_ms:5.2f}({row['ac_pct']:4.1f}%) "
          f"colclr={cc_ms:6.2f} exitclr={ec_ms:5.2f} srch={search_ms:7.1f} exp={exp:7.0f} "
          f"ns/exp={ns_exp:8.1f} corr_nslot={corr_nslot:>8} frag/cell={row['frag_cell']:5.2f} "
          f"maxlist={mx:>5} n_add={cocc.n_added:>7} acc={W['acc']} den={W['den']} {dict(reasons)}",
          flush=True)
    prev["ac_t"], prev["cc_t"], prev["ec_t"] = AC["t"], CC["t"], EC["t"]
    _reset_window()


def _hook(done, total, intent):
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one captured planner before the first progress hook, found {len(captured)}"
        )
    p = captured[0]
    W["solve"] += intent.solve_time_s
    W["exp"] += p.last_expansions
    W["n"] += 1
    TOT["solve"] += intent.solve_time_s
    TOT["exp"] += p.last_expansions
    TOT["n"] += 1
    if intent.accepted:
        W["acc"] += 1
        TOT["acc"] += 1
    elif intent.status is IntentStatus.REJECTED:
        W["den"] += 1
        TOT["den"] += 1
        reasons[intent.denial_reason.value] += 1

    if done % WINDOW == 0:
        _emit_window(done, total)


# ---- run serially, then restore every patch ----
ReservationLedger.any_conflict = _timed_ac
TerminalCapacity.column_clear = _timed_cc
TerminalCapacity.exit_clear = _timed_ec
t0 = perf_counter()
try:
    result = run(cfg, demand=demand, progress=_hook, parallel=None)
finally:
    sim_mod.get_planner = _orig_get
    ReservationLedger.any_conflict = _orig_ac
    TerminalCapacity.column_clear = _orig_cc
    TerminalCapacity.exit_clear = _orig_ec

assert len(set(map(id, captured))) == 1, (
    f"expected a single planner (single-USS scenario) but captured {len(set(map(id, captured)))}; "
    f"the [0] read in the hook is only valid for single-USS scenarios")
_emit_window(TOT["n"], TOT["n"])             # retain a final partial window instead of silently dropping it

# ---- closing readout: the decision rule, with numbers ----
wall = perf_counter() - t0
n = TOT["n"]
mean_ms = TOT["solve"] / n * 1000.0
ac_share = AC["t"] / TOT["solve"] * 100.0 if TOT["solve"] else 0.0
cc_share = CC["t"] / TOT["solve"] * 100.0 if TOT["solve"] else 0.0
ec_share = EC["t"] / TOT["solve"] * 100.0 if TOT["solve"] else 0.0
print(f"\nDONE {n} flights in {wall:.1f}s ({mean_ms:.1f} ms/flight mean)  "
      f"acc={TOT['acc']} den={TOT['den']} verified={result.verified}", flush=True)
print(f"time split over ALL flights: any_conflict={ac_share:.1f}%  column_clear={cc_share:.1f}%  "
      f"exit_clear={ec_share:.1f}%  search={100 - ac_share - cc_share - ec_share:.1f}%", flush=True)
if len(windows) >= 2:
    a, b = windows[0], windows[-1]
    def _r(x, y):
        return y / x if x else float("nan")
    print("first→last window (the N-scaling curve):", flush=True)
    print(f"  ns_per_expansion : {a['ns_exp']:.1f} → {b['ns_exp']:.1f}  (×{_r(a['ns_exp'], b['ns_exp']):.2f})"
          f"   [fragmentation term — fixable]", flush=True)
    print(f"  expansions/flight: {a['exp']:.0f} → {b['exp']:.0f}  (×{_r(a['exp'], b['exp']):.2f})"
          f"   [congestion term — inherent]", flush=True)
    print(f"  column_clear ms  : {a['cc_ms']:.1f} → {b['cc_ms']:.1f}  (×{_r(a['cc_ms'], b['cc_ms']):.2f})"
          f"   [terminal-capacity gate]", flush=True)
    print(f"  corr_nslots      : {a['corr_nslot']} → {b['corr_nslot']}  (×{_r(a['corr_nslot'], b['corr_nslot']):.2f})"
          f"   maxlist {a['maxlist']} → {b['maxlist']}", flush=True)
    print(f"  total ms/flight  : {a['total_ms']:.1f} → {b['total_ms']:.1f}  (×{_r(a['total_ms'], b['total_ms']):.2f})",
          flush=True)
