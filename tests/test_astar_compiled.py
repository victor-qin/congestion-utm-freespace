"""Compiled (numba) A* kernel ↔ pure-Python reference EXACT-equivalence tests (issue #8, Track B).

The pure-Python ``AStarPlanner(compiled=False)`` is the oracle (the current production planner). The
compiled kernel is validated by asserting ``compiled == reference`` **exactly** — identical accept/deny,
cost within 1e-9, byte-identical centerline (positions + flight level + times), AND identical
``last_expansions`` (the kernel explores the same search in the same order). Plus occupancy parity, the
terminal replay, and the transparent-fallback safety valve.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.compiled_hex_occupancy import CompiledHexOccupancy
from freespace_sim.planner.occupancy import HexOccupancyService
from freespace_sim.types import FlightRequest, IntentStatus, Terminal, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _req(fid=1, dx=2000.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(dx, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)


def _level_wall(z, x=1000.0, half_y=400.0):
    return Volume4D(
        box_from_segment(vec(x, -half_y, z), vec(x, half_y, z), 40, CFG.corridor_height_m), 0.0, 1e6)


def _clkey(intent):
    return [(round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 3), round(float(t), 6))
            for p, t in (intent.centerline or [])]


def _assert_exact(req, commits, cfg=CFG):
    """Plan ``req`` with reference and compiled against the SAME committed ledger; assert exact equality."""
    ref, com = AStarPlanner(compiled=False), AStarPlanner(compiled=True)
    lr, lc = ReservationLedger(cfg), ReservationLedger(cfg)
    for fid, vols in commits:
        lr.commit(fid, vols); lc.commit(fid, vols)
    a = ref.plan(req, lr, cfg)
    b = com.plan(req, lc, cfg)
    assert com._fb == 0, "unexpected fallback — compiled path did not actually run"
    assert a.status is b.status, f"status {a.status} != {b.status}"
    assert a.denial_reason is b.denial_reason
    if a.accepted:
        assert abs(a.cost - b.cost) < 1e-9, f"cost {a.cost} != {b.cost}"
        assert ref.last_expansions == com.last_expansions, \
            f"expansions {ref.last_expansions} != {com.last_expansions}"
        assert _clkey(a) == _clkey(b), "centerline differs"
    return a, b


# ---------------- A/C: compiled == reference exact (non-terminal + multi-altitude) ----------------

def test_compiled_empty_airspace_exact():
    a, _ = _assert_exact(_req(), [])
    assert a.status is IntentStatus.ACCEPTED


def test_compiled_reroute_wall_exact():
    _assert_exact(_req(), [(99, [_wall()])])


def test_compiled_ground_delay_exact():
    _assert_exact(_req(), [(99, [Volume4D(CylinderSpec(2000, 0, 60, 0, 150), 0.0, 200.0)])])


def test_compiled_climb_over_blocked_low_level_exact():
    _assert_exact(_req(), [(99, [_level_wall(CFG.level_z(0))])])


def test_compiled_midroute_climb_exact():
    _assert_exact(_req(), [(98, [_level_wall(CFG.level_z(1), x=900.0)]),
                           (97, [_level_wall(CFG.level_z(0), x=1500.0)])])


def test_compiled_share_corridor_by_altitude_exact():
    # commit a forward flight (reference), then plan the reverse flight compiled-vs-reference
    fwd = FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0)
    rev = FlightRequest(2, vec(6000, 0, 0), vec(0, 0, 0), 0.0)
    la = ReservationLedger(CFG)
    ia = AStarPlanner(compiled=False).plan(fwd, la, CFG)
    _assert_exact(rev, [(1, ia.volumes)])


def test_compiled_single_level_config_exact():
    cfg = SimConfig(cruise_level_m=150.0, flight_levels_m=(150.0,), airspace_ceiling_m=165.0,
                    z_min_m=150.0, z_max_m=150.0)
    _assert_exact(_req(), [], cfg=cfg)


def test_compiled_denial_budget_exceeded_exact():
    import dataclasses as dc
    cfg = dc.replace(CFG, max_ground_delay_s=20.0)
    _assert_exact(FlightRequest(1, vec(0, 0, 0), vec(400, 0, 0), 0.0),
                  [(99, [Volume4D(CylinderSpec(400, 0, 60, 0, 150), 0.0, 1e5)])], cfg=cfg)


def test_compiled_deterministic():
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    com = AStarPlanner(compiled=True)
    a = com.plan(_req(), led, CFG)
    b = com.plan(_req(), led, CFG)
    assert a.cost == b.cost and _clkey(a) == _clkey(b)


# ---------------- D: occupancy parity (compiled pool == is_blocked over a committed ledger) --------

@pytest.mark.slow
def test_compiled_occupancy_matches_is_blocked():
    from freespace_sim.demand import HubRadiusDemand
    from freespace_sim.uss import USS
    from freespace_sim.dss import DSS
    from freespace_sim.mechanism import FCFSMechanism
    cfg = SimConfig(region_size_m=(20000.0, 15000.0), lam_per_hour=3000.0, horizon_s=300.0,
                    planner="astar", seed=0)
    demand = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 6, "stripmall_uss": 30},
                             radius_m={"walmart_uss": 6000.0, "stripmall_uss": 3000.0},
                             terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                             pads_per_hub=8, return_flights=True)
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar_ref")
    usses = {u: USS(u, dss, cfg, astar) for u in {r.uss_id for r in reqs}}
    for ev in reqs[:400]:
        usses[ev.uss_id].handle_request(ev)
    svc = HexOccupancyService(cfg)
    cocc = CompiledHexOccupancy(cfg)
    for fid, grp in itertools.groupby(led.iter_committed(), key=lambda fv: fv[0]):
        vols = [v for _, v in grp]
        svc.on_commit(fid, vols); cocc.on_commit(fid, vols)
    cells = set()
    for s in list(svc.blocked) + list(svc.term_cells):
        cells |= set(svc.blocked.get(s, ())) | set(svc.term_cells.get(s, {}))
    assert len(cells) > 100
    import random
    random.seed(0)
    checked = blocked = 0
    for (q, r, L) in random.sample(sorted(cells), min(300, len(cells))):
        for s in range(0, cocc.MAXS, 5):
            ref = svc.is_blocked(q, r, L, s, own=())          # own=∅ ⇒ all columns foreign (pool alone)
            got = cocc.blocked_py(q, r, L, s, own_cells=None)
            assert ref == got, f"occ mismatch at (q={q},r={r},L={L},s={s}): is_blocked={ref} pool={got}"
            checked += 1; blocked += ref
    assert checked > 1000 and blocked > 0


def test_compiled_own_foreign_shared_cell_falls_back_exact():
    """Issue #3: when the flight's OWN hub column shares a rasterized cell with a FOREIGN hub's committed
    column, the single-boolean overlay cannot represent it — the host detects it via `col_owners` and falls
    back to the reference (without which the kernel would treat the foreign column as transparent and route
    through it, then file a spurious CONFLICT_FILED). Two hubs 150 m apart share footprint cells; commit a
    foreign-hub landing, then plan an own-hub landing → assert the own∩foreign fallback FIRES (the
    `col_owners` detection works). The outcome-equality check is a determinism sanity check, NOT kernel
    parity: once the fallback fires, the compiled planner IS running the reference for this flight — which
    is exactly the intended behavior. (The exact kernel-vs-reference contract lives in the other tests.)"""
    cfg = SimConfig()
    hub_a, hub_b = Terminal("uss_a#0", 8, 90.0), Terminal("uss_b#0", 8, 90.0)
    Pa, Pb = vec(2000, 2000, 0), vec(2150, 2000, 0)          # 150 m apart → footprints share cells
    foreign = FlightRequest(1, vec(2000, 4000, 0), Pb, 0.0, uss_id="uss_b", dest_terminal=hub_b)
    own = FlightRequest(2, vec(2000, 500, 0), Pa, 20.0, uss_id="uss_a", dest_terminal=hub_a)
    ref, com = AStarPlanner(compiled=False), AStarPlanner(compiled=True)
    lr, lc = ReservationLedger(cfg), ReservationLedger(cfg)
    fa = ref.plan(foreign, lr, cfg)                          # commit the SAME foreign column to both ledgers
    assert fa.accepted
    lr.commit(1, fa.volumes); lc.commit(1, fa.volumes)
    a, b = ref.plan(own, lr, cfg), com.plan(own, lc, cfg)
    assert com._fb_reasons.get("own-foreign-overlap", 0) > 0, "issue-3 own∩foreign fallback did not fire"
    assert a.status is b.status and a.denial_reason is b.denial_reason
    if a.accepted:
        assert abs(a.cost - b.cost) < 1e-9 and _clkey(a) == _clkey(b)


def test_compiled_mask_widen_re_run_exact():
    """A long time-block forces a ground delay PAST the tight mask window, so the kernel returns FB_MASK
    once and re-runs over the full range (com._remask > 0). Regression guard: the per-plan `gen` bump also
    version-resets the kernel's hash, so it must run inside the widen loop — hoisting it made the re-run
    reuse the tight pass's closed nodes and return a spurious BUDGET_EXCEEDED while the reference ACCEPTED.
    Assert compiled == reference exactly THROUGH the widen, and that the widen path was actually taken."""
    wall = Volume4D(box_from_segment(vec(200, -400, 150), vec(200, 400, 150), 200, 400), 0.0, 1000.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0)
    _, b = _assert_exact(req, [(99, [wall])])          # full exact check incl. last_expansions node-parity
    assert b.accepted
    com = AStarPlanner(compiled=True)
    lc = ReservationLedger(CFG); lc.commit(99, [wall])
    com.plan(req, lc, CFG)
    assert com._remask > 0, "mask-widen re-run not exercised — the regression guard would be vacuous"


def test_compiled_always_active_static_terminal_exact():
    """#24 always-active static terminals (permanent foreign column walls) are carried in
    ``CompiledHexOccupancy.static_col`` (the SAME ``hg.terminal_cells`` the reference walls), so the kernel
    deconflicts against them EXACTLY instead of falling back. A foreign flight whose straight path crosses a
    static hub must reroute (cost > 200, not the ~160 straight-through) byte-identically to the reference,
    with node-count parity and NO fallback (the old preventative gate is gone). This is the regression guard
    for the safety bug where the kernel flew straight through a permanent no-fly wall."""
    cfg = SimConfig(terminal_airspace_always_active=True)
    hub = Terminal("foreign_hub#0", 8, 180.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0, uss_id="uss_a")   # foreign to the hub
    ref, com = AStarPlanner(compiled=False), AStarPlanner(compiled=True)
    ref.static_terminals = com.static_terminals = [((1000.0, 0.0), hub)]         # wall dead on the path
    a = ref.plan(req, ReservationLedger(cfg), cfg)
    b = com.plan(req, ReservationLedger(cfg), cfg)
    assert com._ref_dispatch.get("static-terminals", 0) == 0, "preventative gate should be gone"
    assert com._fb == 0, "kernel must handle static terminals, not fall back"
    assert a.status is b.status and a.accepted and abs(a.cost - b.cost) < 1e-9 and _clkey(a) == _clkey(b)
    assert ref.last_expansions == com.last_expansions, "node-count parity through the static wall"
    assert a.cost > 200, "reference should reroute around the wall (straight-through would be ~160)"


def test_compiled_always_active_own_hub_transparent_foreign_walled_exact():
    """Own-hub exemption under always-active: a flight departing its OWN static hub flies THROUGH its own
    terminal (``_build_overlay`` marks the hub's full ``terminal_cells`` own, so the permanent wall is
    transparent to the hub that owns it), while a FOREIGN static hub on its route stays a wall. Kernel ==
    reference exactly (node parity + centerline), no fallback — proving the overlay covers the wider
    ``terminal_cells`` geometry, not merely the hover column."""
    cfg = SimConfig(terminal_airspace_always_active=True)          # fixed_exit_lanes defaults True
    own_hub, foreign_hub = Terminal("own_uss#0", 8, 180.0), Terminal("foreign_uss#0", 8, 180.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(5000, 0, 0), 0.0, uss_id="own_uss", origin_terminal=own_hub)
    ref, com = AStarPlanner(compiled=False), AStarPlanner(compiled=True)
    statics = [((0.0, 0.0), own_hub), ((2500.0, 0.0), foreign_hub)]   # own at origin, foreign on the route
    ref.static_terminals = com.static_terminals = statics
    a = ref.plan(req, ReservationLedger(cfg), cfg)
    b = com.plan(req, ReservationLedger(cfg), cfg)
    assert com._fb == 0 and com._ref_dispatch.get("static-terminals", 0) == 0, "own-hub flight stays compiled"
    assert a.status is b.status and a.accepted
    assert abs(a.cost - b.cost) < 1e-9 and ref.last_expansions == com.last_expansions and _clkey(a) == _clkey(b)


def test_compiled_static_terminal_occupancy_parity():
    """``blocked_py`` (the kernel's oracle) folds ``static_col`` identically to
    ``HexOccupancyService.is_blocked`` folding ``static_term_cells``: register the SAME hub into both and
    assert cell-for-cell agreement over the terminal's cells at every level and step — for own=∅ (all
    foreign → wall) AND own={hub} (transparent). Guards the fold the kernel's ``_blocked`` compiles."""
    import freespace_sim.planner.hexgrid as hg
    cfg = SimConfig(terminal_airspace_always_active=True)
    hub = Terminal("h#0", 8, 180.0)
    center = (3000.0, 3000.0)
    svc = HexOccupancyService(cfg); svc.register_static_terminal(center, hub)
    cocc = CompiledHexOccupancy(cfg); cocc.register_static_terminal(center, hub)
    cells = sorted(hg.terminal_cells(center, hub, cfg))
    assert len(cells) > 10, "expected a non-trivial terminal footprint"
    own_cells = {cocc.cell_id(q, r, L) for (q, r) in cells for L in range(cfg.n_levels)}
    checked = walls = 0
    for (q, r) in cells:
        for L in range(cfg.n_levels):
            for s in (0, 5, 100, cocc.MAXS - 1):
                foreign_ref = svc.is_blocked(q, r, L, s, own=())            # own=∅ ⇒ static cell is a wall
                assert foreign_ref == cocc.blocked_py(q, r, L, s, own_cells=None)
                own_ref = svc.is_blocked(q, r, L, s, own=frozenset({hub.id}))   # own hub ⇒ transparent
                assert own_ref == cocc.blocked_py(q, r, L, s, own_cells=own_cells), \
                    f"own mismatch (q={q},r={r},L={L},s={s})"
                checked += 1; walls += foreign_ref
    assert checked > 40 and walls == checked, "every static cell must be a foreign wall at every step/level"


def test_out_of_box_committed_corridor_skipped_not_crash():
    """Issue #2: a committed corridor cell that maps outside the kernel box is SKIPPED (counted in
    ``oob_corridor_cells``), never a crash. Any later query to that cell gets ``cell_id < 0`` so the kernel
    falls back via FB_OOB and the reference stays the oracle. Pre-fix this raised an uncaught IndexError
    inside ``on_commit`` (fired for every commit, even fallback flights'), crashing the whole run."""
    cfg = SimConfig()
    cocc = CompiledHexOccupancy(cfg, margin=0)          # box == region bbox (no reroute margin)
    # a short corridor 5 km BEYOND the region's origin corner → every rasterized cell maps outside the box
    far = Volume4D(box_from_segment(vec(-5000, -5000, 150), vec(-4400, -5000, 150), 40, 400), 0.0, 5.0)
    cocc.on_commit(7, [far])                            # must NOT raise
    assert cocc.oob_corridor_cells > 0, "expected out-of-box corridor cells for a volume outside the region"
    # a normal in-region commit records cleanly and never trips the counter
    near = Volume4D(box_from_segment(vec(3000, 3000, 150), vec(3600, 3000, 150), 40, 400), 0.0, 5.0)
    ok = CompiledHexOccupancy(cfg)
    ok.on_commit(8, [near])
    assert ok.oob_corridor_cells == 0


# ---------------- E: safety valves / fallback ----------------

def test_compiled_absent_falls_back_to_reference():
    # numba-absent story: compiled=False is byte-identical to the reference.
    ref = AStarPlanner(compiled=False)
    com = AStarPlanner(compiled=False)
    led = ReservationLedger(CFG); led.commit(99, [_wall()])
    a = ref.plan(_req(), ReservationLedger(CFG), CFG)  # empty
    b = com.plan(_req(), ReservationLedger(CFG), CFG)
    assert a.cost == b.cost


def test_compiled_fallback_on_kernel_valve_matches_reference():
    # Force the kernel to report FB_HASH → the plan must transparently fall back to the reference,
    # count the fallback, warn, and return the reference result exactly.
    from freespace_sim.planner import astar_kernel as K

    def led():
        lg = ReservationLedger(CFG); lg.commit(99, [_wall()]); return lg

    com = AStarPlanner(compiled=True)
    ref_intent = AStarPlanner(compiled=False).plan(_req(), led(), CFG)   # same (wall) ledger state
    orig = com._kernel
    com._kernel = lambda *a, **k: (0, 0.0, 0, K.FB_HASH, -1)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = com.plan(_req(), led(), CFG)
        assert com._fb == 1 and com._fb_reasons.get("hash-full") == 1
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
        assert got.status is ref_intent.status
        assert abs(got.cost - ref_intent.cost) < 1e-9
        assert _clkey(got) == _clkey(ref_intent)
    finally:
        com._kernel = orig


# ---------------- A (terminal) / F: terminal replay + end-to-end ----------------

@pytest.mark.slow
def test_compiled_replay_exact_dallas_terminal():
    """Per-plan compiled-vs-reference replay against a reference-committed terminal ledger
    (``fixed_exit_lanes=True``): exact accept/deny, cost, centerline, and expansions across the batch."""
    from freespace_sim.demand import HubRadiusDemand
    cfg = SimConfig(region_size_m=(20000.0, 15000.0), lam_per_hour=2000.0, horizon_s=300.0,
                    planner="astar", seed=1)
    assert cfg.fixed_exit_lanes and cfg.n_levels >= 2
    demand = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 5, "stripmall_uss": 20},
                             radius_m={"walmart_uss": 6000.0, "stripmall_uss": 3000.0},
                             terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                             pads_per_hub=8, return_flights=True)
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    led = ReservationLedger(cfg)
    ref, com = AStarPlanner(compiled=False), AStarPlanner(compiled=True)
    n_term = 0
    for k, rq in enumerate(reqs[:120]):
        a = ref.plan(rq, led, cfg)
        b = com.plan(rq, led, cfg)
        n_term += (rq.origin_terminal is not None or rq.dest_terminal is not None)
        assert a.status is b.status, f"flight {k}: status {a.status} != {b.status}"
        if a.accepted:
            assert abs(a.cost - b.cost) < 1e-9, f"flight {k}: cost {a.cost} != {b.cost}"
            assert ref.last_expansions == com.last_expansions, f"flight {k}: expansions differ"
            assert _clkey(a) == _clkey(b), f"flight {k}: centerline differs"
            led.commit(getattr(rq, "id", k), a.volumes)
    assert com._fb == 0, f"unexpected fallbacks: {dict(com._fb_reasons)}"
    assert n_term > 100


@pytest.mark.slow
def test_compiled_demand_run_is_verified():
    from freespace_sim.sim import run
    cfg = SimConfig(planner="astar", lam_per_hour=40.0, horizon_s=900.0, seed=4,
                    region_size_m=(4000.0, 4000.0))
    assert run(cfg).verified
