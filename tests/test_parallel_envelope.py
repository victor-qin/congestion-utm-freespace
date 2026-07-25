"""Track A Phase 1 (issue #8): read-envelope instrumentation tests.

The envelope (``parallel.PlanEnvelope``) must be a SUPERSET of everything a plan read — that is the
exact-mode soundness contract — while recording must never perturb the plan itself. Three angles:

  * recording on/off produces byte-identical intents (both compiled + reference paths);
  * an independent class-level probe log (monkeypatched onto ``HexOccupancyService`` itself, so it
    catches call sites that bypass the recorder) is contained in the reported envelope;
  * the accepted corridor's committed volumes sit inside the envelope (the convex-hull lemma the
    ``astar_shortcut`` allowlist relies on), exercised with knots ACTUALLY removed.
"""
from __future__ import annotations

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.parallel import PlanEnvelope, envelope_intersects
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.occupancy import HexOccupancyService
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

def test_envelope_covers_filed_corridor_shortcut():
    req = FlightRequest(1, vec(0, 0, 0), vec(2400, 1400, 0), 0.0)   # diagonal → staircase → knots removed
    sc = get_planner("astar_shortcut")
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
