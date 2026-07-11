from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.types import Terminal, vec
from freespace_sim.volumes import Volume4D, build_corridor

CFG = SimConfig()


def _corridor(x0=0.0, t0=0.0):
    seg = CFG.corridor_segment_len_m
    cl = [
        (vec(x0 + 0 * seg, 0, CFG.cruise_level_m), t0 + 0 * CFG.dt_s),
        (vec(x0 + 1 * seg, 0, CFG.cruise_level_m), t0 + 1 * CFG.dt_s),
        (vec(x0 + 2 * seg, 0, CFG.cruise_level_m), t0 + 2 * CFG.dt_s),
    ]
    return build_corridor(cl, CFG)


def test_commit_then_identical_query_conflicts():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor())
    incoming = _corridor()
    assert led.any_conflict(incoming)
    assert led.conflicting_flights(incoming) == {1}


def test_time_shifted_query_is_clear():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor(t0=0.0))
    # same path but 1000 s later → no time overlap
    assert not led.any_conflict(_corridor(t0=1000.0))


def test_spatially_far_query_is_clear():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor(x0=0.0))
    assert not led.any_conflict(_corridor(x0=5000.0))


def test_release_clears_conflicts():
    led = ReservationLedger(CFG)
    led.commit(1, _corridor())
    led.commit(2, _corridor(x0=5000.0))
    assert led.any_conflict(_corridor())
    led.release(1)
    assert not led.any_conflict(_corridor())
    assert led.any_conflict(_corridor(x0=5000.0))   # flight 2 still committed


def test_conflicts_reports_committed_volumes():
    led = ReservationLedger(CFG)
    led.commit(7, _corridor())
    hits = led.conflicts(_corridor())
    assert hits and all(fid == 7 for fid, _ in hits)
    assert led.n_volumes == 2


# ---------------- always-active static terminal walls (permanent, on-ledger) ----------------
# `register_static_terminal` files a hub's terminal airspace as a PERMANENT, time-invariant ledger volume so
# any_conflict / verify / refiners see it (previously it lived off-ledger in the A* occupancy only).

_HUB = (3000.0, 3000.0)


def _through_hub(tid=None, t0=0.0, t1=20.0):
    """A corridor box crossing the hub at cruise level; foreign (tid=None) or same-hub (tid set)."""
    x, y = _HUB
    return Volume4D(box_from_segment(vec(x - 200, y, 150), vec(x + 200, y, 150), 40, 300),
                    t0, t1, terminal_id=tid)


def _led_with_static():
    led = ReservationLedger(CFG)
    led.register_static_terminal(_HUB, Terminal("h#0", 8, 180.0))
    return led


def test_static_terminal_walls_foreign_corridor():
    """The core fix: a foreign corridor crossing a registered static terminal now CONFLICTS — any_conflict
    is no longer blind to the wall (it was off-ledger before)."""
    assert _led_with_static().any_conflict([_through_hub()]) is True


def test_static_terminal_exempts_same_hub():
    """A same-hub corridor (same terminal_id) flies through its own column — the conflict.py
    same-tid+cylinder exemption applies to the permanent column, so no self-block."""
    assert _led_with_static().any_conflict([_through_hub(tid="h#0")]) is False


def test_static_terminal_spatially_disjoint_clear():
    """A corridor far from every hub is clear — the AABB prune over _static_aabb must not false-positive."""
    far = Volume4D(box_from_segment(vec(0, 0, 150), vec(400, 0, 150), 40, 300), 0.0, 20.0)
    assert _led_with_static().any_conflict([far]) is False


def test_static_wall_is_time_invariant():
    """The permanent wall conflicts at ANY step (whole horizon), unlike a transient committed volume that
    only conflicts inside its own window."""
    led = _led_with_static()
    assert led.any_conflict([_through_hub(t0=0.0, t1=20.0)])         # early in the horizon
    assert led.any_conflict([_through_hub(t0=1700.0, t1=1720.0)])    # late — still walled (permanent)


def test_static_vols_not_bucketed():
    """Perf trap guard: permanent whole-horizon walls must NOT be registered into the per-step _buckets
    (that would flood every step); they live in _static_vols and are scanned separately."""
    led = ReservationLedger(CFG)
    led.register_static_terminal(_HUB, Terminal("h#0", 8, 180.0))
    assert len(led._buckets) == 0 and len(led._static_vols) == 1


def test_conflicts_static_hit_uses_sentinel_and_excluded_from_reroute_targets():
    """conflicts() reports a static-wall hit with the documented STATIC_WALL_FID sentinel (a wall owns no
    flight id, and its terminal_id is a str that must not leak into the int fid contract); and
    conflicting_flights() (reroute targets) must EXCLUDE that sentinel."""
    led = _led_with_static()
    hits = led.conflicts([_through_hub()])
    assert hits and all(fid == ReservationLedger.STATIC_WALL_FID for fid, _ in hits)
    assert ReservationLedger.STATIC_WALL_FID not in led.conflicting_flights([_through_hub()])


def test_subscribe_static_replays_already_registered():
    """CRIT: subscribe_static must REPLAY every already-registered hub. Occupancy binds lazily on the first
    plan — i.e. AFTER sim.run has registered every hub — so a subscribe-only hook would miss them all and
    leave the A* routing walls empty (a silent no-op)."""
    led = _led_with_static()                            # hub registered BEFORE anyone subscribes
    seen = []
    led.subscribe_static(lambda center, term: seen.append((center, term.id)))
    assert seen == [(_HUB, "h#0")], "a late subscriber must be replayed the already-registered hub"


def _brute_static_any_conflict(led, vols):
    """The pre-F1 behaviour: scan EVERY static wall (no spatial index) — the oracle the grid must match."""
    from freespace_sim.conflict import volumes_conflict
    for v in vols:
        vbb = led._flat_aabb(v)
        for i in range(len(led._static_vols)):
            if led._aabb_miss(vbb, led._static_aabb[i]):
                continue
            if volumes_conflict(v, led._static_vols[i]):
                return True
    return False


def test_static_wall_grid_matches_bruteforce_scan():
    """F1: the xy spatial index over the static walls is only a broadphase prune, so any_conflict must give
    byte-identical answers to a full linear scan across a dallas-like field of hubs — no overlap missed, no
    false positive. Probes span through-hub, between-hub, and far-outside boxes, plus a multi-cell box."""
    import random
    led = ReservationLedger(CFG)
    hubs = [(float(x), float(y)) for x in range(1000, 20000, 1500) for y in range(1000, 15000, 1500)]
    for i, c in enumerate(hubs):
        led.register_static_terminal(c, Terminal(f"h#{i}", 8, 150.0))
    assert len(led._static_vols) == len(hubs) > 100        # a real field, not a toy
    rng = random.Random(0)
    n_hit = n_clear = 0
    for _ in range(500):
        cx, cy = rng.uniform(-2000, 22000), rng.uniform(-2000, 17000)
        q = Volume4D(box_from_segment(vec(cx - 150, cy, 150), vec(cx + 150, cy, 150), 40, 300), 0.0, 50.0)
        grid, brute = led.any_conflict([q]), _brute_static_any_conflict(led, [q])
        assert grid == brute, f"grid {grid} != brute {brute} at ({cx:.0f},{cy:.0f})"
        n_hit += grid
        n_clear += not grid
    assert n_hit > 0 and n_clear > 0, "probes must exercise BOTH walled and clear cells (not vacuous)"
    # a big box spanning many grid cells and several hubs — exercises multi-cell insert AND query. Routed
    # ALONG a hub row (y=2500 ∈ range(1000,15000,1500)) so it genuinely crosses walls, not between rows.
    big = Volume4D(box_from_segment(vec(1000, 2500, 150), vec(9000, 2500, 150), 40, 300), 0.0, 50.0)
    assert led.any_conflict([big]) == _brute_static_any_conflict(led, [big])   # grid == brute over many cells
    assert led.any_conflict([big]) is True                                     # and it really does cross hub walls


def test_static_wall_covers_full_search_reachability_under_tight_detour():
    """G1: the wall's t_end must cover the SEARCH's actual reachability (MAXS*dt = schedulable_horizon_steps),
    NOT a max_detour_factor-scaled seconds budget. Under a tight max_detour_factor the search still explores
    the fixed 3x-hop step budget, and ground-wait/hover can spend it as time, so a committed corridor can
    arrive LATER than any detour-scaled estimate. A crossing in that gap must still be walled; the old
    detour-scaled bound let it escape any_conflict/verify."""
    import math
    from freespace_sim.planner.compiled_hex_occupancy import schedulable_horizon_steps
    cfg = SimConfig(max_detour_factor=1.2)
    hx, hy = 3000.0, 3000.0
    led = ReservationLedger(cfg)
    led.register_static_terminal((hx, hy), Terminal("h#0", 8, 180.0))
    w, h = cfg.region_size_m
    max_climb = max(cfg.climb_time_to(z) for z in cfg.flight_levels_m)
    # the OLD detour-scaled bound (pre-fix, incl. the interim vertical-dwell terms) vs the search reachability
    detour_tend = (cfg.horizon_s + cfg.max_ground_delay_s
                   + cfg.max_detour_factor * math.hypot(w, h) / cfg.nominal_speed_mps
                   + 2.0 * max_climb + cfg.hover_time_s + cfg.time_buffer_s)
    reach_tend = schedulable_horizon_steps(cfg) * cfg.dt_s          # the true bound the wall now uses (the fix)
    assert reach_tend > detour_tend + 1.0, "the fix must extend the wall past the detour-scaled bound"
    t_cross = 0.5 * (detour_tend + reach_tend)                      # in the gap: missed by old, caught by the fix
    box = Volume4D(box_from_segment(vec(hx - 200, hy, 150), vec(hx + 200, hy, 150), 40, 300),
                   t_cross, t_cross + cfg.dt_s)
    assert led.any_conflict([box]) is True, "a crossing within the search's reachability must still be walled"
