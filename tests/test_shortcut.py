import copy
import pickle

import numpy as np
import pytest

import freespace_sim.planner.shortcut as shortcut_mod
from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.shortcut import ShortcutRefiner, shortcut_corners
from freespace_sim.planner.terminal_capacity import TerminalCapacity
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, Terminal, vec
from freespace_sim.volumes import Volume4D, build_reservation_from_corners

CFG = SimConfig()


def _req():
    return FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0)


def _wall_led():
    led = ReservationLedger(CFG)
    led.commit(99, [Volume4D(box_from_segment(vec(1200, -250, 150), vec(1200, 250, 150), 40, 400),
                             0.0, 1e6)])
    return led


def test_get_planner_registers_shortcut_variants():
    legacy = get_planner("astar_shortcut")
    heading = get_planner("astar_heading_shortcut")
    batched = get_planner("astar_batched_shortcut")
    sandwich = get_planner("astar_milp_shortcut")
    assert isinstance(legacy, ShortcutRefiner) and legacy.strategy == "single_knot"
    assert isinstance(heading, ShortcutRefiner) and heading.strategy == "single_knot_heading"
    assert heading.label == legacy.label == "astar_sc"
    assert isinstance(batched, ShortcutRefiner) and batched.strategy == "batched_turns"
    assert batched.label == "astar_batched_sc"
    assert isinstance(sandwich, ShortcutRefiner) and sandwich.strategy == "single_knot"
    assert sandwich.inner.warm_planner.strategy == "single_knot"
    with pytest.raises(ValueError, match="unknown shortcut strategy"):
        ShortcutRefiner(AStarPlanner(), strategy="unknown")


def test_heading_skip_requires_exact_resampling_not_only_collinearity():
    z = CFG.cruise_level_m
    seg = CFG.corridor_segment_len_m
    # Consecutive A* pitch-length legs merge to the exact same two subsegments.
    assert shortcut_mod._merge_preserves_resampling(
        vec(0, 0, z), vec(seg, 0, z), vec(2 * seg, 0, z), seg)
    # These points have the same heading, but independent 1,000 m chords make 9+9 boxes while the
    # merged 2,000 m chord makes 17. A heading-only skip would move reservation boundaries/times.
    assert not shortcut_mod._merge_preserves_resampling(
        vec(0, 0, z), vec(1000, 0, z), vec(2000, 0, z), seg)
    assert not shortcut_mod._merge_preserves_resampling(
        vec(0, 0, z), vec(70, 0, z), vec(160, 0, z), seg)
    near = (vec(0, 0, z), vec(120, 0, z), vec(240, 1e-8, z))
    assert shortcut_mod._same_heading_3d(*near)            # tolerance says "same heading"
    assert not shortcut_mod._merge_preserves_resampling(*near, seg)


def test_heading_only_skip_would_bypass_a_real_temporal_conflict():
    z = CFG.cruise_level_m
    a, b, c = vec(0, 0, z), vec(60, 0, z), vec(120, 0, z)
    assert shortcut_mod._same_heading_3d(a, b, c)
    assert not shortcut_mod._merge_preserves_resampling(
        a, b, c, CFG.corridor_segment_len_m)

    # Split boxes occupy x≈0 early and x≈120 later. Merging creates one coarse box that claims the
    # full x-range for the full traversal window, so this late obstacle conflicts only after merging.
    ledger = ReservationLedger(CFG)
    ledger.commit(99, [Volume4D(CylinderSpec(10, 0, 2, 60, 80), 6.5, 7.5)])
    origin, dest = vec(-500, -500, 0), vec(500, 500, 0)
    assert shortcut_mod._rebuild(
        [a, b, c], origin, dest, 0.0, 0.0, CFG, ledger, 1000.0,
        corridor_t0=0.0,
    ) is not None
    assert shortcut_mod._rebuild(
        [a, c], origin, dest, 0.0, 0.0, CFG, ledger, 1000.0,
        corridor_t0=0.0,
    ) is None


@pytest.mark.parametrize(
    "points",
    [
        (vec(0, 0, 30), vec(120, 0, 30), vec(240, 0, 30)),
        (vec(-0.0, 0, 30), vec(120, 0, 30), vec(240, 0, 30)),
        (vec(0, 0, 0), vec(0, 0, 120), vec(0, 0, 240)),
        (vec(0, 0, 30), vec(60, 60, 30), vec(120, 120, 30)),
    ],
)
def test_heading_skip_predicate_implies_byte_identical_build(points):
    a, b, c = points
    assert shortcut_mod._merge_preserves_resampling(
        a, b, c, CFG.corridor_segment_len_m)
    origin = vec(a[0], a[1], 0)
    dest = vec(c[0], c[1], 0)
    origin_terminal = Terminal("exact-origin", 2, radius=90.0)
    dest_terminal = Terminal("exact-dest", 2, radius=90.0)
    before = build_reservation_from_corners(
        [a, b, c], origin, dest, 0.0, 0.0, CFG,
        origin_term=origin_terminal, dest_term=dest_terminal, corridor_t0=31.25,
    )
    after = build_reservation_from_corners(
        [a, c], origin, dest, 0.0, 0.0, CFG,
        origin_term=origin_terminal, dest_term=dest_terminal, corridor_t0=31.25,
    )
    assert pickle.dumps(after, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
        before, protocol=pickle.HIGHEST_PROTOCOL)


def test_heading_skip_matches_builder_with_nondefault_segment_length():
    cfg = SimConfig(nominal_speed_mps=25.0, dt_s=3.0)
    points = (vec(0, 0, 30), vec(75, 0, 30), vec(150, 0, 30))
    assert shortcut_mod._merge_preserves_resampling(*points, cfg.corridor_segment_len_m)
    before = build_reservation_from_corners(
        list(points), vec(0, 0, 0), vec(150, 0, 0), 0.0, 0.0, cfg,
        corridor_t0=12.5,
    )
    after = build_reservation_from_corners(
        [points[0], points[-1]], vec(0, 0, 0), vec(150, 0, 0), 0.0, 0.0, cfg,
        corridor_t0=12.5,
    )
    assert pickle.dumps(after, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
        before, protocol=pickle.HIGHEST_PROTOCOL)


def test_heading_skip_removes_exact_straight_run_without_candidate_rebuilds(monkeypatch):
    z = CFG.cruise_level_m
    seg = CFG.corridor_segment_len_m
    points = [vec(i * seg, 0, z) for i in range(8)]

    def run_variant(skip_exact_heading):
        calls = 0

        def accept(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return [], [], 0.0, 0.0

        monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
        out = shortcut_corners(
            points, points[0], points[-1], 0.0, 0.0, CFG, ReservationLedger(CFG),
            skip_exact_heading=skip_exact_heading,
        )
        return out, calls

    legacy, legacy_calls = run_variant(False)
    heading, heading_calls = run_variant(True)
    assert _route_signature(heading) == _route_signature(legacy)
    assert len(heading) == 2
    assert legacy_calls == 7                              # baseline + six candidate probes
    assert heading_calls == 1                             # baseline only; all six were byte proofs


def test_heading_skip_falls_back_to_rebuild_when_sampling_would_change(monkeypatch):
    z = CFG.cruise_level_m
    points = [vec(0, 0, z), vec(1000, 0, z), vec(2000, 0, z)]
    calls = 0

    def baseline_only(corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return ([], [], 0.0, 0.0) if len(corners) == 3 else None

    monkeypatch.setattr(shortcut_mod, "_rebuild", baseline_only)
    out = shortcut_corners(
        points, points[0], points[-1], 0.0, 0.0, CFG, ReservationLedger(CFG),
        skip_exact_heading=True,
    )
    assert calls == 2                                    # baseline + the required real candidate check
    assert _route_signature(out) == _route_signature(points)


def test_heading_shortcut_is_byte_identical_to_legacy_across_commits(monkeypatch):
    cfg = SimConfig(region_size_m=(5000.0, 3500.0), max_ground_delay_s=120.0)
    hub = Terminal("byte-hub", 1, radius=90.0)
    requests = [
        FlightRequest(1, vec(0, 0, 0), vec(3600, 0, 0), 0.0),
        FlightRequest(2, vec(3600, 120, 0), vec(0, 120, 0), 2.0),
        FlightRequest(3, vec(0, 900, 0), vec(3600, 2100, 0), 4.0),
        FlightRequest(4, vec(3600, 900, 0), vec(0, 2100, 0), 6.0),
        FlightRequest(
            5, vec(500, 3000, 0), vec(4200, 3000, 0), 180.0,
            origin_terminal=hub,
        ),
    ]
    planners = (get_planner("astar_shortcut"), get_planner("astar_heading_shortcut"))
    ledgers = (ReservationLedger(cfg), ReservationLedger(cfg))
    outputs = [[], []]

    original_merge = shortcut_mod._merge_preserves_resampling
    fast_skips = 0

    def count_fast_skips(*args, **kwargs):
        nonlocal fast_skips
        result = original_merge(*args, **kwargs)
        fast_skips += result
        return result

    monkeypatch.setattr(shortcut_mod, "_merge_preserves_resampling", count_fast_skips)
    for index, (planner, ledger) in enumerate(zip(planners, ledgers, strict=True)):
        for req in copy.deepcopy(requests):
            intent = planner.plan(req, ledger, cfg)
            outputs[index].append(pickle.dumps(intent, protocol=pickle.HIGHEST_PROTOCOL))
            if intent.accepted:
                ledger.commit(req.flight_id, intent.volumes)

    assert outputs[1] == outputs[0]
    assert fast_skips > 0                                # parity exercised the optimization
    assert [ReservationLedger._flat_aabb(v) for v in ledgers[1]._vols] == [
        ReservationLedger._flat_aabb(v) for v in ledgers[0]._vols
    ]


def test_shortcut_empty_airspace_leaves_straight_path_alone():
    a = AStarPlanner().plan(_req(), ReservationLedger(CFG), CFG)
    s = get_planner("astar_shortcut").plan(_req(), ReservationLedger(CFG), CFG)
    assert s.accepted
    assert abs(s.air_detour_m - a.air_detour_m) < 1.0   # already straight; nothing to remove


@pytest.mark.parametrize("planner_name", ["astar_shortcut", "astar_batched_shortcut"])
def test_shortcut_tightens_astar_berth_and_stays_conflict_free(planner_name):
    a = AStarPlanner().plan(_req(), _wall_led(), CFG)
    led = _wall_led()
    s = get_planner(planner_name).plan(_req(), led, CFG)
    assert s.accepted
    assert not led.any_conflict(s.volumes)            # build-then-check contract holds
    assert s.air_detour_m < a.air_detour_m - 50.0     # genuinely tighter, not a nudge
    assert s.cost <= a.cost + 1e-6                     # a post-pass never worsens


def test_shortcut_corners_collapses_a_zigzag_in_open_space():
    led = ReservationLedger(CFG)
    z = CFG.cruise_level_m
    corners = [vec(0, 0, z), vec(400, 120, z), vec(800, 0, z), vec(1200, 120, z), vec(1600, 0, z)]
    out = shortcut_corners(corners, vec(0, 0, 0), vec(1600, 0, 0), 0.0, 0.0, CFG, led)
    assert len(out) < len(corners)                    # redundant knots removed
    vols, _, _, _ = build_reservation_from_corners(out, vec(0, 0, 0), vec(1600, 0, 0), 0.0, 0.0, CFG)
    assert not led.any_conflict(vols)                 # rebuilt path is conflict-free


# --- experimental batched-turn driver -------------------------------------------------------

def _batched_context():
    return shortcut_mod._ShortcutContext(
        vec(0, 0, 0), vec(10, 10, 0), 0.0, 0.0, CFG, ReservationLedger(CFG), 1.0)


def _route_signature(corners):
    return tuple(tuple(float(v) for v in np.asarray(point, float)) for point in corners)


def _turn_fixture(include_outgoing=True):
    points = [vec(i, 0, 0) for i in range(6)]          # A..F; F is the turn
    points.append(vec(5, 1, 0))                        # G
    if include_outgoing:
        points.extend((vec(5, 2, 0), vec(5, 3, 0)))    # H, I
    return points


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ((vec(0, 0, 0), vec(2, 0, 0), vec(7, 0, 0)), True),
        ((vec(0, 0, 0), vec(0, 0, 3), vec(0, 0, 8)), True),
        ((vec(0, 0, 0), vec(2, 1, 1), vec(6, 3, 3)), True),
        ((vec(0, 0, 0), vec(1, 0, 0), vec(1.5, np.sqrt(3) / 2, 0)), False),
        ((vec(0, 0, 0), vec(1, 0, 0), vec(1, 0, 1)), False),
        ((vec(0, 0, 0), vec(1, 0, 0), vec(0, 0, 0)), False),
        ((vec(0, 0, 0), vec(0, 0, 0), vec(1, 0, 0)), False),
    ],
)
def test_same_heading_3d_classifies_hex_and_altitude_segments(points, expected):
    assert shortcut_mod._same_heading_3d(*points) is expected


def test_same_heading_3d_uses_a_relative_roundoff_tolerance():
    assert shortcut_mod._same_heading_3d(
        vec(0, 0, 0), vec(1_000_000, 0, 0), vec(2_000_000, 5e-4, 0))


def test_batched_growth_probe_order_is_local_then_maximal(monkeypatch):
    points = _turn_fixture()
    calls = []

    def accept(corners, *_args, **_kwargs):
        calls.append(_route_signature(corners))
        return [], [], 0.0, 0.0

    monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    assert calls == [
        _route_signature(points[:5] + points[6:]),       # E→G
        _route_signature([points[0], *points[6:]]),      # A→G
        _route_signature([points[0], points[-1]]),       # A→I
    ]
    assert [k.id for k in state.knots] == [0, 8]


def test_batched_growth_seed_failure_does_not_prune_maximum(monkeypatch):
    points = _turn_fixture(include_outgoing=False)
    calls = []

    def oracle(corners, *_args, **_kwargs):
        signature = _route_signature(corners)
        calls.append(signature)
        return None if len(corners) > 2 else ([], [], 0.0, 0.0)

    monkeypatch.setattr(shortcut_mod, "_rebuild", oracle)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    assert len(calls[0]) == 6                              # E→G rejected
    assert len(calls[1]) == 2                              # A→G still attempted and accepted
    assert [k.id for k in state.knots] == [0, 6]


def test_batched_growth_continues_after_failed_left_fallback(monkeypatch):
    points = _turn_fixture(include_outgoing=False)
    accepted_lengths = {6, 4}                             # E→G, then C→G
    calls = []

    def oracle(corners, *_args, **_kwargs):
        calls.append(_route_signature(corners))
        return ([], [], 0.0, 0.0) if len(corners) in accepted_lengths else None

    monkeypatch.setattr(shortcut_mod, "_rebuild", oracle)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    # E→G pass, A→G fail, D→G fail, C→G pass, B→G fail. The D failure did not prune C.
    assert [len(call) for call in calls[:5]] == [6, 2, 5, 4, 3]
    assert [k.id for k in state.knots] == [0, 1, 2, 6]


def test_batched_growth_continues_after_failed_right_fallback(monkeypatch):
    # E-F is the incoming leg; G-H-I-J is the outgoing straight run.
    points = [vec(0, 0, 0), vec(1, 0, 0), vec(1, 1, 0),
              vec(1, 2, 0), vec(1, 3, 0), vec(1, 4, 0)]
    calls = []

    def oracle(corners, *_args, **_kwargs):
        signature = _route_signature(corners)
        calls.append(signature)
        # E→G and E→I pass; maximal E→J and nearer E→H fail.
        chord_right = tuple(corners[1])
        accepted = len(corners) == 5 or chord_right == tuple(points[4])
        return ([], [], 0.0, 0.0) if accepted else None

    monkeypatch.setattr(shortcut_mod, "_rebuild", oracle)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    assert [len(call) for call in calls[:4]] == [5, 2, 4, 3]
    assert [k.id for k in state.knots] == [0, 4, 5]


def test_batched_growth_all_rejections_are_deterministic(monkeypatch):
    points = _turn_fixture(include_outgoing=False)
    runs = []

    for _ in range(2):
        calls = []

        def reject(corners, *_args, **_kwargs):
            calls.append(_route_signature(corners))
            return None

        monkeypatch.setattr(shortcut_mod, "_rebuild", reject)
        state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())
        assert state is not None
        runs.append((calls, [k.id for k in state.knots]))

    assert runs[0] == runs[1]
    assert runs[0][1] == list(range(len(points)))          # endpoints and every rejected knot survive


def test_batched_growth_carries_the_last_verified_build(monkeypatch):
    points = _turn_fixture()
    builds = []

    def accept(_corners, *_args, **_kwargs):
        built = ([], [(vec(len(builds), 0, 0), float(len(builds)))], 0.0, 0.0)
        builds.append(built)
        return built

    monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())
    assert state is not None and len(builds) == 3
    assert state.built is builds[-1]


def test_batched_growth_retains_incoming_build_after_outgoing_conflicts(monkeypatch):
    points = _turn_fixture()
    seed_signature = _route_signature(points[:5] + points[6:])
    incoming_signature = _route_signature([points[0], *points[6:]])
    seed_build = ([], [(vec(1, 0, 0), 1.0)], 1.0, 0.0)
    incoming_build = ([], [(vec(2, 0, 0), 2.0)], 2.0, 0.0)

    def oracle(corners, *_args, **_kwargs):
        signature = _route_signature(corners)
        if signature == seed_signature:
            return seed_build
        if signature == incoming_signature:
            return incoming_build
        return None                                      # maximal/fallback outgoing probes conflict

    monkeypatch.setattr(shortcut_mod, "_rebuild", oracle)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    assert [k.id for k in state.knots] == [0, 6, 7, 8]
    assert state.built is incoming_build                 # exact last verified reservation survives


def test_batched_refiner_reuses_cached_build_without_final_rebuild(monkeypatch):
    z = CFG.cruise_level_m
    points = [vec(0, 0, z), vec(60, 0, z), vec(60, 60, z)]
    req = FlightRequest(1, vec(0, 0, 0), vec(60, 60, 0), 0.0)
    ledger = ReservationLedger(CFG)
    inner_built = build_reservation_from_corners(
        points, req.origin, req.dest, 0.0, 0.0, CFG)
    cached_build = build_reservation_from_corners(
        [points[0], points[-1]], req.origin, req.dest, 0.0, 0.0, CFG)
    inner_intent = OperationalIntent(
        request=req,
        status=IntentStatus.ACCEPTED,
        volumes=inner_built[0],
        centerline=inner_built[1],
        cost=1e9,
        planner="dummy",
    )

    class Inner:
        def plan(self, _req, _ledger, _cfg):
            return inner_intent

    seed_calls = []

    def fake_seed(corners, had_holds, _context):
        seed_calls.append((len(corners), had_holds))
        return shortcut_mod._ShortcutState(
            (
                shortcut_mod._Knot(0, corners[0]),
                shortcut_mod._Knot(len(corners) - 1, corners[-1]),
            ),
            cached_build,
        )

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("plan must consume state.built, not rebuild it again")

    monkeypatch.setattr(shortcut_mod, "_shortcut_turn_seeded", fake_seed)
    monkeypatch.setattr(shortcut_mod, "_rebuild", unexpected_rebuild)
    out = ShortcutRefiner(Inner(), strategy="batched_turns").plan(req, ledger, CFG)

    assert seed_calls == [(3, False)]
    assert out is not inner_intent
    assert out.volumes is cached_build[0]
    assert out.centerline is cached_build[1]


def test_batched_growth_recomputes_adjacent_turns_after_acceptance(monkeypatch):
    # Initially C and D are adjacent turns. Accepting B→D removes C, makes B the new turn,
    # and makes D straight. A stale worklist would visit removed/reclassified turn D instead.
    points = [
        vec(-1, 0, 0),  # A, id 0
        vec(0, 0, 0),   # B, id 1
        vec(1, 0, 0),   # C, id 2
        vec(1, 1, 0),   # D, id 3
        vec(2, 2, 0),   # E, id 4
    ]
    initial_knots = tuple(
        shortcut_mod._Knot(i, point) for i, point in enumerate(points))
    assert shortcut_mod._turn_ids(initial_knots) == [2, 3]
    accepted = _route_signature([points[0], points[1], points[3], points[4]])
    accepted_build = ([], [(vec(9, 9, 9), 9.0)], 9.0, 9.0)
    probes = []

    def oracle(corners, *_args, **_kwargs):
        signature = _route_signature(corners)
        probes.append(signature)
        return accepted_build if signature == accepted else None

    real_grow = shortcut_mod._grow_one_turn
    visited_turns = []

    def recording_grow(state, turn_id, context):
        visited_turns.append(turn_id)
        return real_grow(state, turn_id, context)

    monkeypatch.setattr(shortcut_mod, "_rebuild", oracle)
    monkeypatch.setattr(shortcut_mod, "_grow_one_turn", recording_grow)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())

    assert state is not None
    assert visited_turns == [2, 1]
    assert [k.id for k in state.knots] == [0, 1, 3, 4]
    assert state.built is accepted_build
    assert shortcut_mod._turn_ids(state.knots) == [1]
    assert probes == [
        accepted,
        _route_signature([points[0], points[3], points[4]]),
        _route_signature([points[0], points[3], points[4]]),
    ]


def test_batched_straight_route_performs_no_rebuild(monkeypatch):
    points = [vec(i, 0, 0) for i in range(20)]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a no-turn route must not invoke the feasibility oracle")

    monkeypatch.setattr(shortcut_mod, "_rebuild", unexpected)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())
    assert state is not None and [k.id for k in state.knots] == list(range(20))
    assert state.built is None


def test_batched_load_bearing_hold_rejects_hold_free_baseline(monkeypatch):
    points = _turn_fixture()
    calls = 0

    def reject(_corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(shortcut_mod, "_rebuild", reject)
    assert shortcut_mod._shortcut_turn_seeded(points, True, _batched_context()) is None
    assert calls == 1                                    # no turn probe after the baseline failed


def test_batched_redundant_hold_validates_baseline_then_grows(monkeypatch):
    points = _turn_fixture()
    calls = 0

    def accept(_corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [], [], 0.0, 0.0

    monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
    state = shortcut_mod._shortcut_turn_seeded(points, True, _batched_context())
    assert state is not None and len(state.knots) == 2
    assert calls == 4                                    # baseline + E→G + A→G + A→I


def _held_inner_intent(points, air_hold_s=20.0):
    """Accepted dummy intent whose logical centerline holds at ``points[1]`` once."""
    req = FlightRequest(1, vec(points[0][0], points[0][1], 0),
                        vec(points[-1][0], points[-1][1], 0), 0.0)
    built = build_reservation_from_corners(
        points, req.origin, req.dest, 0.0, 0.0, CFG)
    t0 = float(built[1][0][1])
    centerline = [
        (points[0], t0),
        (points[1], t0 + 20.0),
        (points[1], t0 + 20.0 + air_hold_s),
        (points[2], t0 + 40.0 + air_hold_s),
    ]
    return req, OperationalIntent(
        request=req,
        status=IntentStatus.ACCEPTED,
        volumes=built[0],
        centerline=centerline,
        air_hold_s=air_hold_s,
        cost=1e9,
        planner="dummy",
    )


def test_batched_refiner_keeps_load_bearing_hold_exactly(monkeypatch):
    z = CFG.cruise_level_m
    req, inner_intent = _held_inner_intent(
        [vec(0, 0, z), vec(600, 0, z), vec(600, 600, z)])
    calls = 0

    class Inner:
        def plan(self, _req, _ledger, _cfg):
            return inner_intent

    def reject(_corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(shortcut_mod, "_rebuild", reject)
    out = ShortcutRefiner(Inner(), strategy="batched_turns").plan(
        req, ReservationLedger(CFG), CFG)

    assert calls == 1                                    # hold-free baseline alone was checked
    assert out is inner_intent
    assert out.centerline is inner_intent.centerline
    assert out.volumes is inner_intent.volumes
    assert out.air_hold_s == 20.0


def test_batched_refiner_removes_redundant_hold_with_turn():
    z = CFG.cruise_level_m
    req, inner_intent = _held_inner_intent(
        [vec(0, 0, z), vec(600, 0, z), vec(600, 600, z)])

    class Inner:
        def plan(self, _req, _ledger, _cfg):
            return inner_intent

    out = ShortcutRefiner(Inner(), strategy="batched_turns").plan(
        req, ReservationLedger(CFG), CFG)

    assert out is not inner_intent
    assert out.air_hold_s == 0.0
    assert all(not np.allclose(a, b)
               for (a, _), (b, _) in zip(out.centerline, out.centerline[1:]))


def test_batched_refiner_keeps_straight_hold_without_rebuild(monkeypatch):
    z = CFG.cruise_level_m
    req, inner_intent = _held_inner_intent(
        [vec(0, 0, z), vec(600, 0, z), vec(1200, 0, z)])

    class Inner:
        def plan(self, _req, _ledger, _cfg):
            return inner_intent

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a straight hold-only route must not invoke the rebuild oracle")

    monkeypatch.setattr(shortcut_mod, "_rebuild", unexpected)
    out = ShortcutRefiner(Inner(), strategy="batched_turns").plan(
        req, ReservationLedger(CFG), CFG)

    assert out is inner_intent
    assert out.air_hold_s == 20.0


@pytest.mark.parametrize("run_length", [10, 100, 1000])
def test_batched_fast_path_rebuild_count_is_constant(monkeypatch, run_length):
    points = [vec(i, 0, 0) for i in range(run_length)]
    points.append(vec(run_length, 0, 0))                 # turn F
    points.extend(vec(run_length, j, 0) for j in range(1, run_length + 1))
    calls = 0

    def accept(_corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [], [], 0.0, 0.0

    monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())
    assert state is not None and len(state.knots) == 2
    assert calls == 3                                     # E→G, A→G, A→I for every N


def test_batched_three_point_turn_is_considered(monkeypatch):
    points = [vec(0, 0, 0), vec(1, 0, 0), vec(1, 1, 0)]
    calls = 0

    def accept(_corners, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [], [], 0.0, 0.0

    monkeypatch.setattr(shortcut_mod, "_rebuild", accept)
    state = shortcut_mod._shortcut_turn_seeded(points, False, _batched_context())
    assert state is not None and len(state.knots) == 2 and calls == 1


@pytest.mark.parametrize("strategy", ["single_knot", "single_knot_heading"])
def test_legacy_public_refiners_keep_three_point_guard(strategy, monkeypatch):
    points = [vec(0, 0, 30), vec(120, 0, 30), vec(120, 120, 30)]
    req = FlightRequest(1, vec(0, 0, 0), vec(120, 120, 0), 0.0)
    intent = OperationalIntent(
        request=req,
        status=IntentStatus.ACCEPTED,
        volumes=[],
        centerline=[(point, float(index)) for index, point in enumerate(points)],
        planner="inner",
    )

    class Inner:
        def plan(self, _req, _ledger, _cfg):
            return intent

    def unexpected(*_args, **_kwargs):
        raise AssertionError("legacy strategies must retain their public <=3 early guard")

    monkeypatch.setattr(shortcut_mod, "shortcut_corners", unexpected)
    out = ShortcutRefiner(Inner(), strategy=strategy).plan(req, ReservationLedger(CFG), CFG)
    assert out is intent


def test_batched_growth_handles_nonmonotone_static_feasibility():
    z = CFG.cruise_level_m
    points = [vec(i * 1000, 0, z) for i in range(6)]
    points.extend((vec(5000, 2000, z), vec(6000, 2000, z)))  # G, H
    origin, dest = vec(0, 0, 0), vec(6000, 2000, 0)
    ledger = ReservationLedger(CFG)
    ledger.commit(99, [Volume4D(CylinderSpec(4000, 1000, 60, 0, 300), 0.0, 1e6)])
    straight = float(np.linalg.norm(np.asarray(dest[:2]) - np.asarray(origin[:2])))

    def rebuild(candidate):
        return shortcut_mod._rebuild(
            candidate, origin, dest, 0.0, 0.0, CFG, ledger, straight)

    e_to_g = points[:5] + points[6:]
    d_to_g = points[:4] + points[6:]
    c_to_g = points[:3] + points[6:]
    a_to_g = [points[0], *points[6:]]
    assert rebuild(e_to_g) is not None
    assert rebuild(d_to_g) is None                    # this intermediate bearing hits the cylinder
    assert rebuild(c_to_g) is not None                # farther anchors are not thereby pruned
    assert rebuild(a_to_g) is not None

    context = shortcut_mod._ShortcutContext(
        origin, dest, 0.0, 0.0, CFG, ledger, straight)
    initial = shortcut_mod._ShortcutState(tuple(
        shortcut_mod._Knot(i, np.asarray(point, float)) for i, point in enumerate(points)))
    grown = shortcut_mod._grow_one_turn(initial, turn_id=5, context=context)
    assert [k.id for k in grown.knots] == [0, 6, 7]    # literal A→G logical chord
    assert grown.built is not None
    centerline = grown.built[1]
    assert all(float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
               <= CFG.corridor_segment_len_m + 1e-6
               for (a, _), (b, _) in zip(centerline, centerline[1:]))


def test_batched_growth_handles_nonmonotone_temporal_feasibility():
    cfg = SimConfig(region_size_m=(35000.0, 15000.0), max_detour_factor=3.0)
    z = cfg.cruise_level_m
    g, h, end = vec(25000, 10000, z), vec(25120, 10000, z), vec(32000, 10000, z)
    points = [vec(i * 5000, 0, z) for i in range(6)] + [g, h, end]
    origin, dest = vec(0, 0, 0), vec(32000, 10000, 0)
    straight = float(np.linalg.norm(np.asarray(dest[:2]) - np.asarray(origin[:2])))
    empty = ReservationLedger(cfg)
    candidates = {
        left: points[:left + 1] + points[6:]
        for left in range(5)                              # A→G through E→G
    }
    builds = {
        left: shortcut_mod._rebuild(
            candidate, origin, dest, 0.0, 0.0, cfg, empty, straight)
        for left, candidate in candidates.items()
    }
    assert all(build is not None for build in builds.values())

    def g_to_h(build):
        g_index = next(
            index for index, (point, _) in enumerate(build[1][:-1])
            if np.allclose(point, g)
        )
        return build[0][g_index + 1]                     # volumes[0] is the origin hover column

    downstream = {left: g_to_h(build) for left, build in builds.items()}
    assert all(
        not downstream[left].time_overlaps(downstream[2])
        for left in (0, 1, 3)
    )

    ledger = ReservationLedger(cfg)
    # Reject maximal A→G, then D→G, leave C→G feasible, and reject the later B→G probe too. Three
    # disjoint time-local blockers on the shared G→H segment make the non-monotonicity explicit.
    ledger.commit(99, [downstream[left] for left in (0, 1, 3)])
    assert shortcut_mod._rebuild(
        candidates[3], origin, dest, 0.0, 0.0, cfg, ledger, straight) is None
    assert shortcut_mod._rebuild(
        candidates[2], origin, dest, 0.0, 0.0, cfg, ledger, straight) is not None

    initial = shortcut_mod._ShortcutState(tuple(
        shortcut_mod._Knot(index, np.asarray(point, float))
        for index, point in enumerate(points)
    ))
    context = shortcut_mod._ShortcutContext(
        origin, dest, 0.0, 0.0, cfg, ledger, straight)
    grown = shortcut_mod._grow_one_turn(initial, turn_id=5, context=context)
    assert [k.id for k in grown.knots] == [0, 1, 2, 6, 7, 8]
    assert grown.built is not None


def test_shortcut_rebuild_rejects_terminal_capacity_failure():
    origin, dest = vec(0, 0, 0), vec(1200, 0, 0)
    term = Terminal("H", 1, radius=90.0)
    z = CFG.cruise_level_m
    calls = []

    class RejectingCapacity:
        def reservation_admitted(self, volumes, origin_term=None, dest_term=None):
            calls.append((volumes, origin_term, dest_term))
            return False

    rebuilt = shortcut_mod._rebuild(
        [vec(0, 0, z), vec(1200, 0, z)], origin, dest, 0.0, 0.0, CFG,
        ReservationLedger(CFG), 1200.0, origin_term=term, tcap=RejectingCapacity())
    assert rebuilt is None
    assert len(calls) == 1 and calls[0][1:] == (term, None)


def test_batched_shortcut_returns_inner_when_earlier_landing_exceeds_capacity():
    cfg = SimConfig()
    ledger = ReservationLedger(cfg)
    term = Terminal("H", 1, radius=90.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(1000, 0, 0), 0.0, dest_terminal=term)
    z = cfg.cruise_level_m
    corners = [vec(0, 0, z), vec(0, 2000, z), vec(1000, 0, z)]
    volumes, timed, _, _ = build_reservation_from_corners(
        corners, req.origin, req.dest, 0.0, 0.0, cfg, dest_term=term)
    # Keep exactly three logical points so the only batched probe removes the turn. The committed
    # reservation remains the deliberately later detour and is already capacity-safe.
    centerline = [timed[0], (corners[1], (timed[0][1] + timed[-1][1]) / 2.0), timed[-1]]
    inner_intent = OperationalIntent(
        request=req, status=IntentStatus.ACCEPTED, volumes=volumes, centerline=centerline,
        cost=1e6, planner="dummy",
    )
    tcap = TerminalCapacity(cfg, ledger)
    tcap.dwells[term.id] = [(40.0, 100.0)]

    class Inner:
        _tcap = tcap
        _svc_ledger = ledger

        def plan(self, _req, _ledger, _cfg):
            return inner_intent

    direct = build_reservation_from_corners(
        [corners[0], corners[-1]], req.origin, req.dest, 0.0, 0.0, cfg, dest_term=term)
    assert not ledger.any_conflict(direct[0])             # same-H exemption makes geometry pass
    assert not tcap.reservation_admitted(direct[0], dest_term=term)

    out = ShortcutRefiner(Inner(), strategy="batched_turns").plan(req, ledger, cfg)
    assert out is inner_intent                            # unsafe earlier landing was not substituted


@pytest.mark.parametrize("strategy", ["single_knot", "single_knot_heading", "batched_turns"])
def test_terminal_refinement_without_capacity_authority_is_conservative(strategy):
    cfg = SimConfig()
    ledger = ReservationLedger(cfg)
    term = Terminal("H", 1, radius=90.0)
    req = FlightRequest(1, vec(0, 0, 0), vec(1000, 0, 0), 0.0, dest_terminal=term)
    z = cfg.cruise_level_m
    corners = [vec(0, 0, z), vec(500, 500, z), vec(1000, 0, z)]
    volumes, timed, _, _ = build_reservation_from_corners(
        corners, req.origin, req.dest, 0.0, 0.0, cfg, dest_term=term)
    intent = OperationalIntent(
        request=req, status=IntentStatus.ACCEPTED, volumes=volumes,
        centerline=[timed[0], (corners[1], timed[len(timed) // 2][1]), timed[-1]],
        cost=1e6, planner="dummy",
    )

    class InnerWithoutCapacity:
        def plan(self, _req, _ledger, _cfg):
            return intent

    out = ShortcutRefiner(InnerWithoutCapacity(), strategy=strategy).plan(req, ledger, cfg)
    assert out is intent


@pytest.mark.parametrize(
    "planner_name",
    ["astar_shortcut", "astar_heading_shortcut", "astar_batched_shortcut"],
)
def test_shortcut_demand_run_is_verified(planner_name):
    cfg = SimConfig(planner=planner_name, lam_per_hour=40.0, horizon_s=900.0, seed=4,
                    region_size_m=(4000.0, 4000.0))
    assert run(cfg).verified


def test_astar_shortcut_runs_under_always_active():
    """The payoff + the lifted ban: with the terminal walls now PERMANENT LEDGER VOLUMES, the shortcut
    refiner's ``any_conflict`` recheck respects them, so ``sim.run`` no longer raises for an A*-wrapping
    planner under ``terminal_airspace_always_active`` (it used to be a hard ``ValueError``) — and the result
    stays verified: the refiner does not straighten a corridor through a walled terminal column."""
    from freespace_sim.scenarios import get_scenario, with_overrides
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"), horizon_s=8.0)
    cfg = spec.config()
    assert cfg.terminal_airspace_always_active
    r = run(cfg, demand=spec.demand_model(), planner_name="astar_shortcut")   # must NOT raise
    assert r.verified, "shortcut refiner must respect the ledger walls (verified conflict-free)"


@pytest.mark.slow
def test_milp_shortcut_never_worsens_the_milp_solution():
    base = get_planner("astar_milp").plan(_req(), _wall_led(), CFG)
    led = _wall_led()
    sc = get_planner("astar_milp_shortcut").plan(_req(), led, CFG)
    assert sc.accepted
    assert not led.any_conflict(sc.volumes)
    assert sc.cost <= base.cost + 1e-6                 # post-MILP shortcut is monotone


# --- multi-altitude: the refiner polishes A*'s multi-level output -----------------------------------

def _climb_walls_led():
    """A ledger forcing a mid-route climb: level 1 walled early, level 0 walled late (both mid-route)."""
    led = ReservationLedger(CFG)
    led.commit(98, [Volume4D(box_from_segment(vec(900, -400, CFG.level_z(1)),
                                              vec(900, 400, CFG.level_z(1)), 40, CFG.corridor_height_m),
                             0.0, 1e6)])
    led.commit(97, [Volume4D(box_from_segment(vec(1500, -400, CFG.level_z(0)),
                                              vec(1500, 400, CFG.level_z(0)), 40, CFG.corridor_height_m),
                             0.0, 1e6)])
    return led


@pytest.mark.parametrize("planner_name", ["astar_shortcut", "astar_batched_shortcut"])
def test_astar_shortcut_preserves_multilevel_climb(planner_name):
    led = _climb_walls_led()
    s = get_planner(planner_name).plan(_req(), led, CFG)
    assert s.accepted
    assert not led.any_conflict(s.volumes)                              # build-then-check holds
    levels = sorted({round(float(p[2]), 1) for p, _ in s.centerline})
    assert CFG.level_z(0) in levels and CFG.level_z(1) in levels        # the climb knot survived


@pytest.mark.parametrize("planner_name", ["astar_shortcut", "astar_batched_shortcut"])
def test_astar_shortcut_slants_the_climb_staircase(planner_name):
    a = AStarPlanner().plan(_req(), _climb_walls_led(), CFG)
    led = _climb_walls_led()
    s = get_planner(planner_name).plan(_req(), led, CFG)
    assert s.accepted and not led.any_conflict(s.volumes)
    assert s.cost <= a.cost + 1e-6                                      # a post-pass never worsens
    cl = s.centerline
    slanted = any(abs(float(cl[i + 1][0][0]) - float(cl[i][0][0])) > 1.0
                  and abs(float(cl[i + 1][0][2]) - float(cl[i][0][2])) > 1.0
                  for i in range(len(cl) - 1))
    assert slanted   # A*'s orthogonal cruise→climb→cruise staircase fused into a DIAGONAL climb


def _diagonal_segments(centerline):
    """Count centerline segments that move BOTH horizontally and vertically — a slanted diagonal climb
    box (vs A*'s orthogonal pure-vertical rung / pure-horizontal cruise)."""
    return sum((abs(float(b[0] - a[0])) > 1.0 or abs(float(b[1] - a[1])) > 1.0)
               and abs(float(b[2] - a[2])) > 1.0
               for (a, _), (b, _) in zip(centerline, centerline[1:]))


@pytest.mark.parametrize("planner_name", ["astar_shortcut", "astar_batched_shortcut"])
def test_astar_shortcut_dense_multilevel_run_stays_verified(planner_name):
    # The refiner's diagonal climb-boxes must stay FCL-conflict-free UNDER LOAD — against other traffic at
    # other levels, not just the static walls the single-flight tests use. Dense crossing traffic with
    # capped ground delay makes altitude the deconfliction lever, forcing many climbs the shortcut fuses
    # into diagonals. (test_shortcut_demand_run_is_verified @ λ=40 is far too sparse to force any climb;
    # multilevel_e2e uses plain astar — so nothing else covers this path under load.)
    cfg = SimConfig(planner=planner_name, lam_per_hour=3000.0, horizon_s=300.0,
                    region_size_m=(1200.0, 1200.0), seed=1, max_ground_delay_s=60.0)
    res = run(cfg)
    assert res.verified                                    # FCL replay: no phantom cross-level collision
    # not vacuous: the run actually produced climbs the refiner slanted into diagonal segments
    floor = 2.0 * cfg.flight_levels_m[0]
    assert sum(i.altitude_change_m > floor + 1.0 for i in res.accepted) >= 5    # many climbed the ladder
    assert sum(_diagonal_segments(i.centerline) for i in res.accepted) >= 5     # ... and were slanted


@pytest.mark.parametrize("planner_name", ["astar_shortcut", "astar_batched_shortcut"])
def test_astar_shortcut_deconflicts_from_committed_traffic_laterally(planner_name):
    # Two opposite-direction flights on a shared corridor: the first is committed, the second meets it.
    # With the cost weights normalized to one per-second currency, giving way is cheapest as a short
    # lateral berth (3x/s) rather than a climb over (4x/s applied to the much slower climb) — and the
    # refiner must still produce a conflict-free plan through that berth. Asserts the cost ORDERING the
    # weights imply, not a magic threshold. (The refiner's DIAGONAL climb-boxes are exercised under load
    # — capped ground delay forcing climbs — by test_astar_shortcut_dense_multilevel_run_stays_verified.)
    led = ReservationLedger(CFG)
    i1 = get_planner(planner_name).plan(
        FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0), led, CFG)
    assert i1.accepted
    led.commit(1, i1.volumes)
    i2 = get_planner(planner_name).plan(
        FlightRequest(2, vec(6000, 0, 0), vec(0, 0, 0), 0.0), led, CFG)
    assert i2.accepted
    assert not led.any_conflict(i2.volumes)                # the plan clears the committed flight
    one_rung = CFG.cost_altitude_change_per_m * 2.0 * (CFG.level_z(1) - CFG.level_z(0))
    assert i2.air_detour_m > 0.0                                            # gave way laterally ...
    assert CFG.cost_air_lateral_per_m * i2.air_detour_m < one_rung          # ... undercutting a climb
    assert CFG.cost_ground_delay_per_s * i2.ground_delay_s < one_rung       # and undercutting a hold
    assert max(round(float(p[2]), 1) for p, _ in i2.centerline) < CFG.level_z(1)   # stayed at the floor level


def test_shortcut_reuses_the_inner_capacity_authority():
    """The refiner must find the pad-capacity authority its inner planner already bound to a ledger.

    Regression pin for a SILENT failure mode. ``_terminal_capacity_for`` returning None does not
    raise — it makes ``plan`` hand back the UNREFINED inner intent for every terminal flight, i.e.
    ``astar_shortcut`` quietly degrading to bare ``astar``. Nothing else in this file catches it:
    the ``Inner`` fakes supply their own authority, and
    ``test_terminal_refinement_without_capacity_authority_is_conservative`` deliberately pins the
    DEGRADED branch as correct. This pins the other half — that the lookup actually succeeds.

    It is also the contract test for ``Planner.capacity_authority``: drop that method from
    ``AStarPlanner`` (or rename the private ``_tcap``/``_svc_ledger`` pair it reads) and this fails
    loudly instead of costing a silent 0-refinement run.
    """
    astar = AStarPlanner()
    led = ReservationLedger(CFG)
    astar.plan(_req(), led, CFG)                       # binds the authority to THIS ledger

    found = shortcut_mod._terminal_capacity_for(astar, led)
    assert isinstance(found, TerminalCapacity), "inner authority not found — refinement would no-op"
    assert found is astar._tcap, "found a DIFFERENT authority — the point is to reuse, not rebuild"

    # reachable through the wrapper chain, which is how ShortcutRefiner actually calls it
    assert shortcut_mod._terminal_capacity_for(ShortcutRefiner(astar), led) is found

    # ...and correctly ABSENT for a ledger this planner never planned against
    assert shortcut_mod._terminal_capacity_for(astar, ReservationLedger(CFG)) is None
