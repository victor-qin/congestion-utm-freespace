"""Track A Phase 1 (issue #8): read-envelope instrumentation tests.

The envelope (``parallel.PlanEnvelope``) must be a SUPERSET of everything a plan read — that is the
exact-mode soundness contract — while recording must never perturb the plan itself. Three angles:

  * recording on/off produces byte-identical intents (both compiled + reference paths);
  * an independent class-level probe log (monkeypatched onto ``HexOccupancyService`` itself, so it
    catches call sites that bypass the recorder) is contained in the reported envelope;
  * the accepted corridor's committed volumes sit inside the envelope (the convex-hull lemma the
    shortcut allowlist entries rely on), exercised with knots ACTUALLY removed.
"""
from __future__ import annotations

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.parallel import PlanEnvelope, envelope_intersects
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar.occupancy import HexOccupancyService
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _wall(x=1000.0, t_end=1e6):
    return Volume4D(box_from_segment(vec(x, -200, 150), vec(x, 200, 150), 40, 400), 0.0, t_end)


def _clkey(intent):
    return [(round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 3), round(float(t), 6))
            for p, t in (intent.centerline or [])]


def _plan(planner, req, commits, cfg=CFG):
    led = ReservationLedger(cfg)
    for fid, vols in commits:
        led.commit(fid, vols)
    return planner.plan(req, led, cfg)


def _corners_in_env(vols, env):
    """Every volume's xy-AABB corners inside the envelope (cell box ∪ hub discs)."""
    for v in vols:
        lo, hi = v.aabb()
        for cx, cy in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])):
            ok = env.xy is not None and env.xy[0] <= cx <= env.xy[2] and env.xy[1] <= cy <= env.xy[3]
            if not ok:
                ok = any((cx - hx) ** 2 + (cy - hy) ** 2 <= hr * hr
                         for hx, hy, hr in env.hub_reads)
            if not ok:
                return False
    return True


# ---------------- recording is observer-only ----------------

def test_recording_does_not_change_intents():
    req = FlightRequest(1, vec(0, 0, 0), vec(2500, 400, 0), 0.0)
    commits = [(9, [_wall()])]
    for compiled in (False, True):
        off = AStarPlanner(compiled=compiled)
        on = AStarPlanner(compiled=compiled)
        on.record_envelope = True
        a = _plan(off, req, commits)
        b = _plan(on, req, commits)
        assert a.status is b.status and a.denial_reason is b.denial_reason
        assert off.last_expansions == on.last_expansions
        assert abs(a.cost - b.cost) < 1e-12 and _clkey(a) == _clkey(b)
        assert off.last_envelope is None                     # off → no envelope built
        assert isinstance(on.last_envelope, PlanEnvelope)
        assert not on.last_envelope.unbounded


# ---------------- superset audit: independent probe log ⊆ envelope ----------------

def _audit(monkeypatch):
    """Class-level probe log on the REAL service — sees every read even if a call site bypassed the
    recorder, which is exactly the bug class this audit exists to catch. pad_clear entries are expanded
    to their full documented step×level window (occupancy.pad_clear scans [s0, s0+dwell] × all levels)."""
    log: list[tuple] = []
    orig_ib = HexOccupancyService.is_blocked
    orig_pc = HexOccupancyService.pad_clear

    def ib(self, q, r, L, s, own=()):
        log.append((q, r, L, s))
        return orig_ib(self, q, r, L, s, own)

    def pc(self, q, r, s0, dwell_steps):
        for k in (s0, s0 + dwell_steps):                     # window extremes suffice for a bbox check
            for L in (0, self.cfg.n_levels - 1):
                log.append((q, r, L, k))
        return orig_pc(self, q, r, s0, dwell_steps)

    monkeypatch.setattr(HexOccupancyService, "is_blocked", ib)
    monkeypatch.setattr(HexOccupancyService, "pad_clear", pc)
    return log


def _assert_log_in_bbox(log, env):
    assert log, "audit log empty — the scenario exercised no occupancy read"
    b = env.cell_bbox
    assert b is not None
    for (q, r, L, s) in log:
        assert b[0] <= q <= b[1] and b[2] <= r <= b[3], f"probe ({q},{r}) outside bbox {b}"
        assert b[4] <= L <= b[5] and b[6] <= s <= b[7], f"probe L={L},s={s} outside bbox {b}"
        assert env.t_lo <= s * CFG.dt_s <= env.t_hi


def test_envelope_superset_audit_reference(monkeypatch):
    log = _audit(monkeypatch)
    p = AStarPlanner(compiled=False)
    p.record_envelope = True
    # congested non-terminal flight: wall detour + multi-step pad_clear dwell windows at both ends
    intent = _plan(p, FlightRequest(1, vec(0, 0, 0), vec(2500, 0, 0), 0.0), [(9, [_wall()])])
    assert intent.accepted
    _assert_log_in_bbox(log, p.last_envelope)
    assert _corners_in_env(intent.volumes, p.last_envelope)


def test_envelope_superset_audit_terminal(monkeypatch):
    log = _audit(monkeypatch)
    cfg = SimConfig()
    hub_o, hub_d = Terminal("h#0", 8, 90.0), Terminal("h#1", 8, 90.0)
    p = AStarPlanner(compiled=False)
    p.record_envelope = True
    led = ReservationLedger(cfg)
    a = p.plan(FlightRequest(1, vec(0, 0, 0), vec(3000, 0, 0), 0.0,
                             origin_terminal=hub_o, dest_terminal=hub_d), led, cfg)
    assert a.accepted
    led.commit(1, a.volumes)
    log.clear()
    b = p.plan(FlightRequest(2, vec(0, 0, 0), vec(3000, 0, 0), 4.0,
                             origin_terminal=hub_o, dest_terminal=hub_d), led, cfg)
    assert b.accepted
    _assert_log_in_bbox(log, p.last_envelope)
    assert _corners_in_env(b.volumes, p.last_envelope)
    # both hub discs present, wide enough to contain their terminal columns
    assert len(p.last_envelope.hub_reads) == 2
    for (hx, hy, hr), center in zip(p.last_envelope.hub_reads, (vec(0, 0, 0), vec(3000, 0, 0))):
        assert abs(hx - center[0]) < 1e-9 and abs(hy - center[1]) < 1e-9 and hr > 90.0


def test_envelope_compiled_accumulates_across_mask_widen():
    # long time-block forces the FB_MASK widen re-run (mirrors test_compiled_mask_widen_re_run_exact);
    # the envelope must accumulate across BOTH kernel passes and still cover the accepted corridor.
    wall = Volume4D(box_from_segment(vec(200, -400, 150), vec(200, 400, 150), 200, 400), 0.0, 1000.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0)
    p = AStarPlanner(compiled=True)
    p.record_envelope = True
    intent = _plan(p, req, [(99, [wall])])
    assert intent.accepted and p._remask > 0
    assert _corners_in_env(intent.volumes, p.last_envelope)


def test_envelope_compiled_covers_reference_probes(monkeypatch):
    """Kernel-recorded envelope vs the reference's independently-logged probe set for the SAME plan:
    the searches are trace-identical (node parity), so every reference probe must land inside the
    compiled envelope (cell box in meters ∪ hub discs)."""
    log = _audit(monkeypatch)
    req = FlightRequest(1, vec(0, 0, 0), vec(2500, 400, 0), 0.0)
    commits = [(9, [_wall()])]
    com = AStarPlanner(compiled=True)
    com.record_envelope = True
    b = _plan(com, req, commits)
    ref = AStarPlanner(compiled=False)
    a = _plan(ref, req, commits)
    assert a.status is b.status and ref.last_expansions == com.last_expansions
    env = com.last_envelope
    from freespace_sim.planner import hexgrid as hg
    R = hg.circumradius(CFG)
    for (q, r, L, s) in log:
        x, y = hg.hex_center(q, r, R)
        ok = env.xy is not None and env.xy[0] <= x <= env.xy[2] and env.xy[1] <= y <= env.xy[3]
        ok = ok or any((x - hx) ** 2 + (y - hy) ** 2 <= hr * hr for hx, hy, hr in env.hub_reads)
        assert ok, f"reference probe ({q},{r}) at ({x:.0f},{y:.0f}) outside compiled envelope"
        assert env.t_lo <= s * CFG.dt_s <= env.t_hi


# ---------------- filed corridor ⊆ envelope with the shortcut refiner (hull lemma) ----------------

@pytest.mark.parametrize(
    "planner_name",
    ["astar_shortcut", "astar_heading_shortcut", "astar_batched_shortcut"],
)
def test_envelope_covers_filed_corridor_shortcut(planner_name):
    req = FlightRequest(1, vec(0, 0, 0), vec(2400, 1400, 0), 0.0)   # diagonal → staircase → knots removed
    sc = get_planner(planner_name)
    inner = sc.inner
    assert isinstance(inner, AStarPlanner)
    inner.record_envelope = True
    refined = _plan(sc, req, [])
    bare = _plan(AStarPlanner(), req, [])
    assert refined.accepted and bare.accepted
    assert len(refined.centerline) < len(bare.centerline), \
        "shortcut removed no knots — hull lemma not exercised"
    assert _corners_in_env(refined.volumes, inner.last_envelope)


# ---------------- denial paths ----------------

def test_envelope_denial_paths():
    from freespace_sim.types import DenialReason
    # BUDGET_EXCEEDED: a permanent cylinder over the origin blocks every takeoff; the queue exhausts
    # (ground-wait chain only) → bounded envelope.
    plug = Volume4D(CylinderSpec(cx=0.0, cy=0.0, radius=200.0, z_lo=0.0, z_hi=200.0), 0.0, 1e6)
    for compiled in (False, True):
        p = AStarPlanner(compiled=compiled)
        p.record_envelope = True
        d = _plan(p, FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0), [(9, [plug])])
        assert d.denial_reason is DenialReason.BUDGET_EXCEEDED
        env = p.last_envelope
        assert env is not None and not env.unbounded
        # compiled path: the all-False takeoff mask means the kernel never probes a cell — the read
        # set is the pad windows alone, legitimately carried by the hub discs (cell_bbox may be None).
        # reference path: pad_clear goes through the recorder, so the bbox must be non-empty.
        if not compiled:
            assert env.cell_bbox is not None
        assert any(abs(hx) < 1e-9 and abs(hy) < 1e-9 for hx, hy, _ in env.hub_reads)
    # SEARCH_EXHAUSTED: tiny expansion cap → truncated read set → unbounded envelope.
    for compiled in (False, True):
        p = AStarPlanner(max_expansions=5, compiled=compiled)
        p.record_envelope = True
        d = _plan(p, FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0), [(9, [_wall()])])
        assert d.denial_reason is DenialReason.SEARCH_EXHAUSTED
        assert p.last_envelope is not None and p.last_envelope.unbounded
        assert envelope_intersects(p.last_envelope, [])      # unbounded ⇒ always dirty
        assert envelope_intersects(p.last_envelope, [((0, 0, 0, 1, 1, 1), 0.0, 1.0)])


# ---------------- evict floor: evicting less never changes results ----------------

def test_evict_floor_evicts_less_and_is_noop():
    reqs = [FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
            FlightRequest(2, vec(0, 500, 0), vec(2000, 500, 0), 300.0)]
    base, floored = AStarPlanner(), AStarPlanner()
    lb, lf = ReservationLedger(CFG), ReservationLedger(CFG)
    outs = []
    for pl, led in ((base, lb), (floored, lf)):
        got = []
        for i, rq in enumerate(reqs):
            if pl is floored and i == 1:
                pl.evict_floor = 0.0                        # frontier clock pinned at t=0 → evicts LESS
            intent = pl.plan(rq, led, CFG)
            assert intent.accepted
            led.commit(rq.flight_id, intent.volumes)
            got.append((intent.cost, _clkey(intent), pl.last_expansions))
        outs.append(got)
    assert outs[0] == outs[1]


# ======================================================================================
# SIPP read envelope (context/sipp_lns_plan.md §7)
#
# SIPP's kernel accumulates its read set per CELL, not per (cell, step) probe as A*'s does, because
# it walks a cell's whole free-interval chain rather than answering one blocked-at-step question. The
# chain's shape is derived from every commit that ever touched that cell, so a commit at some other
# step in the same cell can change what the walk finds. Recording the arrival step alone would
# UNDER-report — and an under-reporting envelope is worse than no envelope, because it reads as clean
# and a coordinator merges a stale repair. These tests exist to pin that it does not.
# ======================================================================================

def _sipp(record=True, compiled=True):
    from freespace_sim.planner.sipp import SIPPPlanner

    p = SIPPPlanner(compiled=compiled)
    p.record_envelope = record
    return p


def test_sipp_recording_does_not_change_intents():
    """Observer-only: the accumulator is write-only w.r.t. the search, so kernel parity is untouched."""
    req = FlightRequest(1, vec(0, 0, 0), vec(2500, 400, 0), 0.0)
    commits = [(9, [_wall()])]
    off, on = _sipp(record=False), _sipp(record=True)
    a, b = _plan(off, req, commits), _plan(on, req, commits)
    assert a.status is b.status and a.denial_reason is b.denial_reason
    assert off.last_expansions == on.last_expansions
    assert abs(a.cost - b.cost) < 1e-12 and _clkey(a) == _clkey(b)
    assert off.last_envelope is None                      # off → nothing built
    assert isinstance(on.last_envelope, PlanEnvelope)
    assert not on.last_envelope.unbounded


def test_sipp_envelope_covers_the_filed_corridor():
    """Cheap gross-under-reporting check: the corridor the plan committed must lie inside what it
    read — it was routed through cells the search examined."""
    p = _sipp()
    intent = _plan(p, FlightRequest(1, vec(0, 0, 0), vec(2500, 0, 0), 0.0), [(9, [_wall()])])
    assert intent.accepted
    assert _corners_in_env(intent.volumes, p.last_envelope)


def test_sipp_envelope_covers_the_reference_paths_probes(monkeypatch):
    """Structural cross-check against an INDEPENDENT probe log.

    The compiled kernel reads raw numpy pools, so it cannot be monkeypatched the way
    `HexOccupancyService` can. The pure-Python SIPP reference searches the same lattice for the same
    optimum through `SafeIntervalIndex`, which CAN be — so its probe set is an independent witness of
    what a SIPP search of this flight reads."""
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.planner.sipp import SafeIntervalIndex

    log: list[tuple] = []
    orig_cb, orig_fi = SafeIntervalIndex.cell_blocked, SafeIntervalIndex.free_intervals

    def cb(self, q, r, L, s, own, fixed_lanes):
        log.append((q, r, L))
        return orig_cb(self, q, r, L, s, own, fixed_lanes)

    def fi(self, q, r, L, own, base, max_step, fixed_lanes):
        log.append((q, r, L))
        return orig_fi(self, q, r, L, own, base, max_step, fixed_lanes)

    monkeypatch.setattr(SafeIntervalIndex, "cell_blocked", cb)
    monkeypatch.setattr(SafeIntervalIndex, "free_intervals", fi)

    req = FlightRequest(1, vec(0, 0, 0), vec(2500, 400, 0), 0.0)
    commits = [(9, [_wall()])]
    com = _sipp()
    b = _plan(com, req, commits)
    log.clear()                                    # drop the compiled run's own host-side overlay reads
    ref = _sipp(record=False, compiled=False)
    a = _plan(ref, req, commits)
    assert a.status is b.status and abs(a.cost - b.cost) < 1e-9, "not the same search to compare"
    assert log, "audit log empty — the scenario exercised no occupancy read"

    env = com.last_envelope
    b = env.cell_bbox
    assert b is not None
    # CELL-EXACT, deliberately — against `cell_bbox` rather than the padded `env.xy`. `env_pad_m` is
    # ~190 m here (corridor_width/2 + R), which absorbs a whole ring of cells, so a padded check
    # cannot see a one-ring under-report. This is the assertion with the resolution to catch one.
    for (q, r, L) in log:
        assert b[0] <= q <= b[1] and b[2] <= r <= b[3], \
            f"reference probe cell ({q},{r}) outside the compiled envelope's cell_bbox {b}"
        assert b[4] <= L <= b[5], f"reference probe level {L} outside {b}"
    _ = hg.circumradius(CFG)


@pytest.mark.slow
@pytest.mark.parametrize("planner", ["sipp", "astar"])
def test_envelope_outside_commits_cannot_change_the_plan(planner):
    """THE soundness gate, stated as the property the coordinator actually relies on:

        if a commit does not intersect the envelope, replanning gives the identical answer.

    Brute force rather than by construction — commit each candidate wall alone, replan, compare. A*
    runs the same harness as a control: a harness bug fails both arms, an accumulator bug fails only
    SIPP.

    THE CANDIDATES ARE ANCHORED TO WORLD GEOMETRY, NOT TO `env.xy`, and that is the whole design.
    Two earlier versions sampled positions relative to the envelope under test, which cannot fail:
    shrink the envelope and the sample points move with it, so they always straddle the *reported*
    edge where by construction nothing changes. Both a 2-cell shrink AND deleting the kernel's
    dominant probe site passed that version. Fixing the candidates in world space instead makes the
    test bite the way it must — a wall lying on the flight's corridor MUST be reported dirty, and if
    an under-reporting envelope calls it clean, the replan changes the cost and this fails.
    """
    from freespace_sim.planner.astar import AStarPlanner

    req = FlightRequest(1, vec(0, 0, 0), vec(1200, 0, 0), 0.0)
    p = AStarPlanner() if planner == "astar" else _sipp()
    p.record_envelope = True
    ref = _plan(p, req, [])
    assert ref.accepted
    env = p.last_envelope
    assert env is not None and not env.unbounded and env.xy is not None

    # Short walls stepped out from the corridor centreline at three x stations, plus a sweep past the
    # destination. Fixed in world coordinates, so the sample set does not move when the envelope does.
    candidates = []
    for x in (300.0, 600.0, 900.0):
        for cy in range(0, 1400, 100):
            candidates.append(Volume4D(
                box_from_segment(vec(x, cy - 120, 150), vec(x, cy + 120, 150), 40, 400), 0.0, 1e6))
    for x in range(1200, 2600, 100):
        candidates.append(Volume4D(
            box_from_segment(vec(float(x), -120, 150), vec(float(x), 120, 150), 40, 400), 0.0, 1e6))

    clean, dirty = [], []
    for v in candidates:
        (clean if not envelope_intersects(env, [(v.flat_aabb(), v.t_start, v.t_end)])
         else dirty).append(v)
    assert len(clean) > 10 and len(dirty) > 3, \
        f"vacuous split: {len(clean)} clean / {len(dirty)} dirty"

    key = (round(ref.cost, 9), _clkey(ref))
    for v in clean:
        again = _plan(p, req, [(77, [v])])
        assert again.accepted, f"a commit OUTSIDE the envelope denied the flight: {v.flat_aabb()}"
        assert (round(again.cost, 9), _clkey(again)) == key, \
            f"a commit OUTSIDE the envelope changed the plan: {v.flat_aabb()} — envelope under-reports"

    # The harness is only a gate if SOME candidate is decision-relevant: a wall on the corridor must
    # be reported dirty AND actually change the plan. Without this, an envelope covering the whole
    # region would pass trivially.
    on_path = Volume4D(box_from_segment(vec(600.0, -120, 150), vec(600.0, 120, 150), 40, 400), 0.0, 1e6)
    assert envelope_intersects(env, [(on_path.flat_aabb(), on_path.t_start, on_path.t_end)])
    blocked = _plan(p, req, [(77, [on_path])])
    assert (round(blocked.cost, 9), _clkey(blocked)) != key, \
        "the on-corridor wall changed nothing — the fixture cannot detect under-reporting"


def test_every_accepted_sipp_plan_reports_a_read_set():
    """The coverage number that decides whether the feature does anything.

    A `None` envelope is SAFE — `_read_set_is_clean` reads it as always-dirty — but it merges
    nothing, which is the degenerate behaviour this whole feature exists to remove. So the gate is
    not "None is handled", it is "None does not happen on the path that matters". Over a congested
    replay every ACCEPTED plan must carry one.

    Denials are deliberately not covered: SIPP has host-side early exits (no feasible takeoff step,
    every terminal lane out of box) that return before the kernel runs, and they legitimately leave
    None. Nothing is lost — `try_repair` populates `RepairOutcome.envelopes` only on the accept
    return, so a denied repair carries no envelope for a coordinator to test in the first place."""
    import numpy as np

    from freespace_sim.demand import UniformPoissonDemand
    from freespace_sim.planner.sipp import SIPPPlanner

    cfg = SimConfig(region_size_m=(6000.0, 6000.0), lam_per_hour=600.0, horizon_s=600.0, seed=3)
    reqs = UniformPoissonDemand().generate(cfg, np.random.default_rng(cfg.seed))[:120]
    led = ReservationLedger(cfg)
    p = SIPPPlanner()
    p.record_envelope = True
    n_acc = 0
    for rq in reqs:
        it = p.plan(rq, led, cfg)
        if not it.accepted:
            continue
        n_acc += 1
        assert p.last_envelope is not None, f"accepted flight {rq.flight_id} reported no read set"
        assert p.last_envelope.cell_bbox is not None
        assert _corners_in_env(it.volumes, p.last_envelope)
        led.commit(rq.flight_id, it.volumes)
    assert n_acc > 40, f"only {n_acc} accepted — fixture too thin to be a coverage test"


def test_sipp_shortcut_envelope_covers_the_refined_corridor():
    """The hull lemma with SIPP inside the refiner: the shortcut only REMOVES knots, so the refined
    centreline lies in the convex hull of the searched one, which the envelope already covers. A*
    pins this for its own variants; SIPP is a registry planner (`sipp_shortcut`) and was not."""
    from freespace_sim.planner.sipp import SIPPPlanner

    req = FlightRequest(1, vec(0, 0, 0), vec(2400, 1400, 0), 0.0)   # diagonal ⇒ staircase ⇒ knots
    sc = get_planner("sipp_shortcut")
    inner = sc.inner
    assert isinstance(inner, SIPPPlanner)
    inner.record_envelope = True
    refined = _plan(sc, req, [])
    bare = _plan(_sipp(record=False), req, [])
    assert refined.accepted and bare.accepted
    assert len(refined.centerline) < len(bare.centerline), \
        "shortcut removed no knots — hull lemma not exercised"
    assert _corners_in_env(refined.volumes, inner.last_envelope)
