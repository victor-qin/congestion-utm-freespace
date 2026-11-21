"""Zeta any-angle geometry, timing and reservation rows match the simulator builders."""

from __future__ import annotations

import math
import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.cost import trajectory_cost
from freespace_sim.geometry import CylinderSpec
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen.network import (
    RowIndex,
    RowKey,
    StaticTerminalCatalog,
    build_flight_graph,
)
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.translate import (
    Column,
    column_to_corners,
    column_to_intent,
)
from freespace_sim.planner.zeta.geometry import (
    anyangle_to_intent,
    reservation_claims,
    volume_claims,
    volume_spatial_cells,
)
from freespace_sim.planner.zeta.timing import edge_end_times
from freespace_sim.planner.colgen.windows import (
    terminal_claim_steps,
)
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import (
    Volume4D,
    build_reservation_from_corners,
    column_dwell_s,
    corridor_segment_volume,
    hover_reservation,
)
from tests._colgen_support import check_canonical  # noqa: F401 -- fixture, used by pytestmark

pytestmark = pytest.mark.usefixtures("check_canonical")


def _cfg(*, time_buffer_s: float = 4.0, planner: str = "colgen",
         hover_radius_m: float = 60.0) -> SimConfig:
    """The v1 single-level world, retaining all default ledger geometry."""

    return SimConfig(
        planner=planner,
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        time_buffer_s=time_buffer_s,
        region_size_m=(20_000.0, 20_000.0),
        terminal_airspace_always_active=True,
        # Pinned, not inherited: these worlds are built from 60 m endpoint discs — the fallback
        # before a delivery pad got its own default (#134). `test_covering_theorem` re-checks the
        # theorem at the 10 m default separately, where the disc no longer spans a cell.
        hover_radius_m=hover_radius_m,
    )


def _ground_point(cell: tuple[int, int], cfg: SimConfig, dx: float = 0.0, dy: float = 0.0):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x + dx, y + dy, cfg.ground_level_m)


@pytest.mark.parametrize("arrival_snap_s", [0.0, 1.0])
def test_anyangle_indexed_wall_checks_preserve_full_scan_seeds_and_contact(
    monkeypatch, arrival_snap_s,
):
    """Indexed seed construction retains full-scan routes, coefficients and wall exceptions."""
    from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions

    cfg = replace(_cfg(), max_ground_delay_s=16.0)
    origin = _ground_point((-5, 0), cfg)
    terminal = Terminal("own-origin", 2, radius=90.0)
    foreign = Terminal("foreign-wall", 1, radius=30.0)
    terms = (
        (origin, terminal),
        (_ground_point((0, 1), cfg), foreign),
        (_ground_point((80, 80), cfg), Terminal("far-wall", 1, radius=90.0)),
    )
    request = FlightRequest(123, origin, _ground_point((5, 0), cfg), 1.3,
                            origin_terminal=terminal)
    params = ColGenParams(max_air_overrun_hops=2)
    catalog = StaticTerminalCatalog(terms, cfg)
    fg = build_flight_graph(request, cfg, catalog, params)
    row_index = RowIndex()
    for _point_, term in terms:
        row_index.register_terminal(term)
    options = ZetaOptions(arrival_snap_s=arrival_snap_s)
    indexed = ZetaGraph(fg, cfg, row_index, options)
    reference = ZetaGraph(fg, cfg, row_index, options)

    def full_scan(volume: Volume4D) -> bool:
        """Retain the former all-wall predicate as an independent broad-phase oracle."""
        bound = volume.flat_aabb()
        return any(
            volume.time_overlaps(wall)
            and all(bound[i] <= wall.flat_aabb()[i + 3]
                    and wall.flat_aabb()[i] <= bound[i + 3] for i in range(3))
            and volumes_conflict(volume, wall)
            for wall in fg.static_walls
        )

    monkeypatch.setattr(reference, "_hits_static_wall", full_scan)
    actual = indexed.seed_columns(params)
    assert actual
    assert actual == reference.seed_columns(params)
    for wall in fg.static_walls[:2]:
        shape = wall.shape
        # Contact and one-sided perturbations exercise closed AABB filtering;
        # time contact stays half-open and own-terminal cylinders stay transparent.
        for offset in (-1e-7, 0.0, 1e-7):
            touching = CylinderSpec(shape.cx + shape.radius + 1.0 + offset,
                                    shape.cy, 1.0, shape.z_lo, shape.z_hi)
            for tag in (None, wall.terminal_id):
                for start, end in ((0.0, 1.0), (wall.t_end, wall.t_end + 1.0)):
                    volume = Volume4D(touching, start, end, tag)
                    assert indexed._hits_static_wall(volume) == full_scan(volume)
        own = Volume4D(shape, 0.0, 1.0, wall.terminal_id)
        foreign_volume = replace(own, terminal_id=None)
        assert not indexed._hits_static_wall(own)
        assert indexed._hits_static_wall(foreign_volume)


@pytest.mark.parametrize("endpoints", ["cc", "tc", "ct", "tt"])
@pytest.mark.parametrize("time_buffer_s", [0.0, 4.0])
@pytest.mark.parametrize("arrival_snap_s", [0.0, 1.0])
def test_anyangle_translation_keeps_builder_terminal_and_budget_contracts(
    endpoints, time_buffer_s, arrival_snap_s,
):
    """Long waypoint links retain exact builder geometry, endpoint rules and pad claims."""
    cfg = replace(_cfg(time_buffer_s=time_buffer_s), max_detour_factor=4.0)
    origin = _ground_point((0, 0), cfg, 7.0, -3.0)
    dest = _ground_point((10, 0), cfg, -4.0, 8.0)
    origin_term = Terminal("zeta-origin", 3, radius=90.0) if endpoints[0] == "t" else None
    dest_term = Terminal("zeta-dest", 2, radius=90.0) if endpoints[1] == "t" else None
    request = FlightRequest(
        800, origin, dest, 1.3, origin_terminal=origin_term, dest_terminal=dest_term,
    )
    origin_lanes = hg.terminal_lanes(origin, origin_term, cfg) if origin_term else []
    dest_lanes = hg.terminal_lanes(dest, dest_term, cfg) if dest_term else []
    origin_idx = (
        max(range(len(origin_lanes)), key=lambda i: origin_lanes[i].cell[0])
        if origin_lanes else None
    )
    dest_idx = (
        min(range(len(dest_lanes)), key=lambda i: dest_lanes[i].cell[0])
        if dest_lanes else None
    )
    start = origin_lanes[origin_idx].cell if origin_idx is not None else (0, 0)
    end = dest_lanes[dest_idx].cell if dest_idx is not None else (10, 0)
    column = Column(800, 3, 0, origin_idx, dest_idx, (start, (5, 2), end), 0.0)
    with pytest.raises(ValueError, match="adjacent lateral hex hops"):
        column_to_intent(column, request, cfg)
    actual = anyangle_to_intent(column, request, cfg, arrival_snap_s=arrival_snap_s)
    corners, build_delay, report_delay, corridor_t0, _, _ = column_to_corners(
        column, request, cfg, _any_angle=True,
    )
    expected, centerline, _, _ = build_reservation_from_corners(
        corners, origin, dest, request.t_departure, build_delay, cfg,
        origin_term=origin_term, dest_term=dest_term, corridor_t0=corridor_t0,
    )
    assert actual.accepted
    if arrival_snap_s == 0.0:
        assert actual.volumes == expected
        assert [stamp for _, stamp in actual.centerline] == [stamp for _, stamp in centerline]
    else:
        assert actual.volumes[0] == expected[0]
        assert [(v.shape, v.terminal_id) for v in actual.volumes] == [
            (v.shape, v.terminal_id) for v in expected
        ]
        assert actual.volumes[-1].t_start == actual.centerline[-1][1]
        assert actual.volumes[-1].t_end == (
            actual.centerline[-1][1] + cfg.hover_time_s
            + column_dwell_s(dest, dest_term, cfg, cfg.flight_levels_m[0])
        )
        for ((a, start_time), (b, end_time)), volume in zip(
            zip(actual.centerline, actual.centerline[1:]), actual.volumes[1:-1],
        ):
            speed_ratio = math.hypot(*(b[:2] - a[:2])) / (
                (end_time - start_time) * cfg.nominal_speed_mps
            )
            assert 0.95 - 1e-12 <= speed_ratio <= 1.05 + 1e-12
            assert volume.t_start == start_time - cfg.time_buffer_s
            assert volume.t_end == end_time + cfg.time_buffer_s
        nominal = anyangle_to_intent(column, request, cfg)
        assert actual.cost == nominal.cost
        assert actual.ground_delay_s == nominal.ground_delay_s
        assert actual.air_hold_s == nominal.air_hold_s
        assert actual.air_detour_m == nominal.air_detour_m
        assert actual.altitude_change_m == nominal.altitude_change_m
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(actual.centerline, centerline))
    assert actual.ground_delay_s == report_delay == 8.0
    assert actual.cost == trajectory_cost(actual, cfg)
    assert len(actual.centerline) > len(column.cell_path)
    assert actual.centerline[-1][1] > corridor_t0 + 2 * cfg.dt_s
    assert not anyangle_to_intent(
        replace(column, departure_step=column.departure_step + 10000), request, cfg,
    ).accepted
    assert not anyangle_to_intent(
        column, request, replace(cfg, max_detour_factor=1.0),
    ).accepted
    with pytest.raises(ValueError, match="distinct waypoints"):
        anyangle_to_intent(replace(column, cell_path=(start, start, end)), request, cfg)
    with pytest.raises(ValueError, match="flight_id"):
        anyangle_to_intent(replace(column, flight_id=999), request, cfg)
    with pytest.raises(ValueError, match="origin_lane_idx"):
        anyangle_to_intent(replace(column, origin_lane_idx=999), request, cfg)

    row_index = RowIndex()
    protected = frozenset(term.id for term in (origin_term, dest_term) if term)
    for term in (origin_term, dest_term):
        if term is not None:
            row_index.register_terminal(term)
    pieces = [
        volume_claims(volume, cfg, row_index=row_index, protected_terminal_ids=protected)
        for volume in actual.volumes
    ]
    union = reservation_claims(
        actual.volumes, cfg, row_index=row_index, protected_terminal_ids=protected,
    )
    assert union == frozenset().union(*pieces)
    assert len(union) < sum(map(len, pieces))
    for volume, rows in zip(actual.volumes, pieces):
        if isinstance(volume.shape, CylinderSpec) and volume.terminal_id is not None:
            assert rows == frozenset(
                RowKey.term(volume.terminal_id, step)
                for step in terminal_claim_steps(volume.t_start, volume.t_end, cfg)
            )
            assert all(row_index.cap(row) > 1 for row in rows)
            with pytest.raises(ValueError, match="permanent-wall protection"):
                volume_claims(volume, cfg, row_index=row_index)
            with pytest.raises(KeyError, match="unknown"):
                volume_claims(
                    volume, cfg, row_index=RowIndex(), protected_terminal_ids=protected,
                )
        else:
            assert rows and all(row.kind == "cell" for row in rows)


@pytest.mark.parametrize("start,durations,snap,expected", [
    (0.0, (5.3, 5.3), 1.0, (5.5, 11.0)),
    (0.0, (5.25, 5.25), 1.0, (5.5, 11.0)),
    (0.0, (5.2, 5.2), 1.0, (5.0, 10.0)),
    (0.49, (10.0,), 1.0, (11.0,)),
    (0.0, (3.2, 3.2), 1.0, (3.2, 6.4)),
    (0.5, (4.0, 4.0), 1.0, (4.5, 8.5)),
    (0.0, (4.75, 4.75), 1.0, (5.0, 10.0)),
    (0.0, (5.25, 5.25), 10.0, (5.0, 10.0)),
    (0.0, (10.500001,), 10.0, (10.500001,)),
    (0.0, (5.75, 5.75), 1.0, (6.0, 12.0)),
    (0.1, (0.1, 0.2, 0.3), 0.0, (0.2, 0.4, 0.7)),
    (0.1, (0.1, 0.2, 0.3), 1.0, (0.2, 0.4, 0.7)),
])
def test_anyangle_arrival_grid_respects_speed_band_and_nominal_fallback(
    start, durations, snap, expected,
):
    """The timing rule chooses nearest feasible arrivals and preserves fallback arithmetic."""
    actual = edge_end_times(start, durations, snap, 0.05)
    assert actual == expected
    for previous, end, nominal in zip((start, *actual), actual, durations):
        assert end > previous
        assert 0.95 - 1e-12 <= nominal / (end - previous) <= 1.05 + 1e-12


@pytest.mark.parametrize("dt_s", [4.0, 4.1])
@pytest.mark.parametrize("arrival_snap_s", [0.0, 1.0])
@pytest.mark.parametrize("connection_mode,path", [
    ("full", ((0, 0), (3, 0), (7, 0), (10, 0))),
    ("full", ((0, 0), (3, 1), (0, 0), (6, -1), (10, 0))),
    ("hex12", ((0, 0), (1, 1), (2, 2), (3, 3))),
    ("hex12_long", ((0, 0), (2, 1), (4, 2), (6, 3))),
])
def test_anyangle_arrival_grid_reconstructs_each_candidate_leg(
    dt_s, arrival_snap_s, connection_mode, path,
):
    """Search and filing clocks agree despite repeated or redundant candidate corners."""
    from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions
    from freespace_sim.planner.colgen.objective import cost_model

    cfg = replace(_cfg(), dt_s=dt_s, max_detour_factor=6.0)
    request = FlightRequest(805, _ground_point(path[0], cfg), _ground_point(path[-1], cfg), 0.13)
    params = ColGenParams(max_air_overrun_hops=20)
    graph = ZetaGraph(
        build_flight_graph(request, cfg, [], params), cfg, RowIndex(),
        ZetaOptions(connection_radius_m=2000.0, connection_mode=connection_mode,
                    arrival_snap_s=arrival_snap_s),
    )
    column = Column(request.flight_id, 3, 0, None, None, path, 0.0)
    intent = anyangle_to_intent(column, request, cfg, arrival_snap_s=arrival_snap_s)
    assert intent.accepted
    expected_ends = []
    stamp = intent.centerline[0][1]
    for index, (a, b) in enumerate(zip(path, path[1:])):
        parts = graph.parts(a, b, index == 0, index == len(path) - 2)
        assert parts is not None
        ends = edge_end_times(stamp, tuple(part.duration for part in parts), arrival_snap_s, 0.05)
        speed_ratio = math.fsum(part.duration for part in parts) / (ends[-1] - stamp)
        assert 0.95 - 1e-12 <= speed_ratio <= 1.05 + 1e-12
        expected_ends.extend(ends)
        stamp = ends[-1]
    assert [stamp for _, stamp in intent.centerline[1:]] == expected_ends
    for previous, end, volume in zip(
        (intent.centerline[0][1], *expected_ends), expected_ends, intent.volumes[1:-1],
    ):
        assert volume.t_start == previous - cfg.time_buffer_s
        assert volume.t_end == end + cfg.time_buffer_s
    assert intent.volumes[-1] == graph.endpoint(False, expected_ends[-1])
    canonical = graph.canonical_column(column, cost_model(cfg, params))
    assert canonical is not None
    assert canonical.claims == reservation_claims(intent.volumes, cfg, row_index=graph.row_index)


@pytest.mark.parametrize("connection_mode,radius,classes", [
    ("hex12", 480.0, 12), ("full", 240.0, 12), ("full", 480.0, None),
])
@pytest.mark.parametrize("forward_cruise", [False, True])
def test_strict_turn_rule_controls_successors_seeds_and_column_admission(
    connection_mode, radius, classes, forward_cruise,
):
    """Interior cruise turns must be strictly below 90 degrees; 90 exactly is rejected.

    Heading classes group incoming links only when their legal next links are identical,
    successor lists follow the incoming heading, and supplied, nominal and canonical routes
    obey the same restricted domain as search. Forward progress remains an independent rule.
    """
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions
    from freespace_sim.planner.colgen.zeta_pool import build_seed_columns
    from freespace_sim.planner.turn_rule import axial_dot, path_turns_valid, turn_allowed

    cfg = replace(_cfg(), dt_s=4.0, nominal_speed_mps=30.0, max_detour_factor=4.0)
    params = ColGenParams(max_air_overrun_hops=4)
    request = FlightRequest(807, _ground_point((0, 0), cfg), _ground_point((4, 0), cfg), 0.0)
    fg = build_flight_graph(request, cfg, [], params)
    fg = replace(fg, corridor_cells=frozenset((q, r) for q in range(-6, 7) for r in range(-6, 7)))
    options = ZetaOptions(connection_radius_m=radius, connection_mode=connection_mode,
                          forward_cruise=forward_cruise, turn_below_90=True)
    graph = ZetaGraph(fg, cfg, RowIndex(), options)
    free = ZetaGraph(fg, cfg, RowIndex(), replace(options, turn_below_90=False))
    assert free.turns is None and free.heading(((0, 0), (1, 0))) == 0
    offsets = graph._offsets
    assert offsets == free._offsets
    centre = {offset: np.asarray(graph.point(offset)[:2]) - graph.point((0, 0))[:2]
              for offset in offsets}
    perpendicular = 0
    for incoming in offsets:
        for outgoing in offsets:
            physical = float(np.dot(centre[incoming], centre[outgoing]))
            scale = float(np.linalg.norm(centre[incoming]) * np.linalg.norm(centre[outgoing]))
            assert turn_allowed(incoming, outgoing) is (physical > 1e-9 * scale)
            if abs(physical) <= 1e-9 * scale:
                perpendicular += 1
                assert axial_dot(incoming, outgoing) == 0
    # Exactly perpendicular pairs exist in every searched lattice and are all rejected.
    assert perpendicular > 0 or forward_cruise
    assert all(turn_allowed(offset, offset) for offset in offsets)
    assert not any(turn_allowed(offset, (-offset[0], -offset[1])) for offset in offsets)
    turns = graph.turns
    if classes is not None and not forward_cruise:
        assert turns.count - 1 == classes
    for offset, heading in turns.classes.items():
        legal = {o for o in offsets if turn_allowed(offset, o)}
        assert turns.allowed[heading] == legal
        assert set(graph.turn_neighbors((0, 0), heading)) == {
            target for target in graph.neighbors((0, 0)) if (target[0], target[1]) in legal}
        assert graph.heading(((0, 0), offset)) == heading
        assert graph.heading(((0, 0), offset), terminal=True) == 0
        # Collinear links of different lengths share one class when futures are identical.
        assert all(turns.classes[o] == heading for o in offsets
                   if o[0] * offset[1] == o[1] * offset[0] and axial_dot(o, offset) > 0)
    assert graph.turn_neighbors((0, 0), 0) == graph.neighbors((0, 0))
    model = cost_model(cfg, params)
    straight = Column(request.flight_id, 0, 0, None, None,
                      ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), 0.0)
    sixty = replace(straight, cell_path=((0, 0), (1, 0), (2, -1), (3, -1), (4, -1), (4, 0)))
    right = replace(straight, cell_path=((0, 0), (2, -1), (2, 0), (3, 0), (4, 0)))
    assert axial_dot((2, -1), (0, 1)) == 0
    assert graph.canonical_column(right, model) is None
    assert free.canonical_column(right, model) is not None
    assert graph.canonical_column(straight, model) is not None
    assert graph.canonical_column(sixty, model) is not None
    seeds = build_seed_columns(
        ZetaGraph(fg, cfg, RowIndex(), replace(options, seed_validation_radius_m=480.0)),
        replace(params, seed_nominal_routes=True), (right, sixty),
    )
    assert seeds.complete
    supplied = {column.cell_path for column in seeds.columns[:1]}
    assert supplied == {sixty.cell_path}
    assert right.cell_path not in {column.cell_path for column in seeds.columns}
    assert all(path_turns_valid(column.cell_path) for column in seeds.columns)
    assert all(path_turns_valid(column.cell_path) for column in graph.seed_columns(params))
    # A legal free shortest seed is kept, so both domains start from the same seed.
    free_seeds = free.seed_columns(params)
    if all(path_turns_valid(column.cell_path) for column in free_seeds):
        assert graph.seed_columns(params) == free_seeds
    # An illegal free shortest path falls back to the exact heading-aware search.
    calls = []
    static = graph._static_shortest

    def illegal_first(start, goal, deadline, *, headings):
        calls.append(headings)
        return static(start, goal, deadline, headings=True) if headings else right.cell_path

    graph._static_shortest = illegal_first
    fallback = graph._shortest_path((0, 0), (4, 0), None)
    assert calls == [False, True] and fallback != right.cell_path
    assert fallback is not None and path_turns_valid(fallback)
    del graph._static_shortest
    # Production DP types still load positional slot pickles written before the flag
    # existed, as the free domain; Zeta types read only captures made by current code.
    restricted = replace(fg, turn_below_90=True)
    assert pickle.loads(pickle.dumps(restricted)) == restricted
    names = [item.name for item in fields(fg)]
    legacy = [value for name, value in zip(names, fg.__getstate__()) if name != "turn_below_90"]
    restored = object.__new__(type(fg))
    restored.__setstate__(legacy)
    assert restored == fg and restored.turn_below_90 is False
    assert type(restored._search_cache) is type(fg._search_cache)
    value = replace(params, turn_below_90=True)
    assert pickle.loads(pickle.dumps(value)) == value
    older = object.__new__(type(value))
    # An archive from before the flag also lacks every field appended after it.
    cut = [item.name for item in fields(value)].index("turn_below_90")
    older.__setstate__(value.__getstate__()[:cut])
    assert older == replace(value, turn_below_90=False)


@pytest.mark.parametrize("connection_mode,radius,count", [
    ("full", 480.0, 60), ("full", 240.0, 18),
    ("hex12", 480.0, 12), ("hex12_long", 480.0, 24),
    ("hex12_long", 240.0, 12),
])
@pytest.mark.parametrize("forward_cruise", [False, True])
@pytest.mark.parametrize("destination", [(4, 0), (0, 4)])
def test_anyangle_connection_set_controls_search_and_column_admission(
    connection_mode, radius, count, forward_cruise, destination,
):
    """Connection and forward-progress rules govern searched and supplied cruise links."""
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions
    from freespace_sim.planner.colgen.zeta_pool import build_seed_columns

    cfg = replace(_cfg(), dt_s=4.0, nominal_speed_mps=30.0, max_detour_factor=4.0)
    params = ColGenParams(max_air_overrun_hops=4)
    request = FlightRequest(806, _ground_point((0, 0), cfg), _ground_point(destination, cfg), 0.0)
    fg = build_flight_graph(request, cfg, [], params)
    fg = replace(fg, corridor_cells=frozenset((q, r) for q in range(-6, 7) for r in range(-6, 7)))
    options = ZetaOptions(connection_radius_m=radius, connection_mode=connection_mode,
                          forward_cruise=forward_cruise)
    graph = ZetaGraph(fg, cfg, RowIndex(), options)
    links = set(graph.neighbors((0, 0)))
    unrestricted = ZetaGraph(fg, cfg, RowIndex(), replace(options, forward_cruise=False))
    all_links = set(unrestricted.neighbors((0, 0)))
    assert len(all_links) == count
    assert all_links == {(-q, -r) for q, r in all_links}
    direction = request.dest[:2] - request.origin[:2]
    direction /= np.linalg.norm(direction)
    progress = {cell: float(np.dot(np.asarray(graph.point(cell)[:2])
                                  - graph.point((0, 0))[:2], direction)) for cell in all_links}
    assert links == ({cell for cell, distance in progress.items() if distance > 1e-8}
                     if forward_cruise else all_links)
    if forward_cruise:
        assert any(abs(distance) <= 1e-8 for distance in progress.values())
        assert not links & {(-q, -r) for q, r in links}
        assert all(distance > 1e-8 for cell, distance in progress.items() if cell in links)
    assert all(math.dist(graph.point((0, 0)), graph.point(cell)) <= radius + 1e-8 for cell in links)
    axial = {(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)}
    between = {(2, -1), (1, -2), (-1, -1), (-2, 1), (-1, 2), (1, 1)}
    long = {(-3, 1), (-3, 2), (-2, -1), (-2, 3), (-1, -2), (-1, 3),
            (1, -3), (1, 2), (2, -3), (2, 1), (3, -2), (3, -1)}
    if connection_mode != "full":
        assert all_links == axial | between | (long if radius > 240 and connection_mode == "hex12_long" else set())
    direct = Column(request.flight_id, 0, 0, None, None, ((0, 0), destination), 0.0)
    assert (graph.canonical_column(direct, cost_model(cfg, params)) is not None) == (
        connection_mode == "full" and radius == 480.0
    )
    dq, dr = destination[0] // 4, destination[1] // 4
    backward_path = ((0, 0), (-dq, -dr), (0, 0), (dq, dr),
                     (2*dq, 2*dr), (3*dq, 3*dr), destination)
    backward = replace(direct, cell_path=backward_path)
    assert (graph.canonical_column(backward, cost_model(cfg, params)) is not None) is not forward_cruise
    wider_seed_graph = ZetaGraph(fg, cfg, RowIndex(), replace(options, seed_validation_radius_m=480.0))
    seeds = build_seed_columns(wider_seed_graph, replace(params, seed_nominal_routes=False),
                               (backward, direct))
    assert seeds.complete
    assert {column.cell_path for column in seeds.columns} == (
        {direct.cell_path} if forward_cruise else {backward_path, direct.cell_path}
    )
    if forward_cruise:
        stationary = replace(fg, request=replace(request, dest=request.origin), dest_cell=fg.origin_cell)
        assert not ZetaGraph(stationary, cfg, RowIndex(), options).neighbors((0, 0))
        if connection_mode == "hex12":
            from freespace_sim.planner.zeta.forward import ForwardTopology

            for heading in (request.dest, request.origin, request.dest + vec(7.0, 11.0, 0.0)):
                directed = ZetaGraph(replace(fg, request=replace(request, dest=heading)),
                                     cfg, RowIndex(), options)
                topology = ForwardTopology.build(directed)
                assert ForwardTopology.build(directed) is topology
                assert set(topology.order) == set(range(len(topology.cells)))
                for node, cell in enumerate(topology.cells):
                    successors = topology.targets[topology.offsets[node]:topology.offsets[node+1]]
                    assert tuple(topology.cells[i] for i in successors) == directed.neighbors(cell)
                    assert all(topology.ranks[node] < topology.ranks[i] for i in successors)
                if np.array_equal(heading, request.origin):
                    assert len(topology.targets) == 0
                elif not np.array_equal(heading, request.dest):
                    assert len(directed._offsets) == 6
                goal = topology.cells.index(fg.dest_cell)
                reachable = topology.reaches(goal)
                assert reachable[goal]
                assert topology.reaches(goal) is reachable
                # Independently grow the predecessor closure to check every retained waypoint.
                expected = {fg.dest_cell}
                while True:
                    enlarged = expected | {cell for cell in topology.cells
                                           if expected.intersection(directed.neighbors(cell))}
                    if enlarged == expected:
                        break
                    expected = enlarged
                assert {cell for i, cell in enumerate(topology.cells) if reachable[i]} == expected
    with pytest.raises(ValueError, match="connection_mode"):
        replace(options, connection_mode="unknown")

    from freespace_sim.planner.zeta.lattice_bound import (
        LatticeDistanceBound, lattice_distance_lower,
    )
    distance_bound = LatticeDistanceBound.build(graph, cost_model(cfg, params), {})
    supported = connection_mode == "hex12" or (connection_mode == "full" and radius == 240.0)
    assert (distance_bound is not None) is supported
    if supported:
        remaining = distance_bound.remaining((0, 0), (3, 1))
        assert remaining > math.dist(graph.point((0, 0)), graph.point((3, 1))) + 1.0
        if not forward_cruise:
            import heapq
            from decimal import Decimal, localcontext

            distances = {(0, 0): 0.0}
            pending = [(0.0, (0, 0))]
            while pending:
                distance, cell = heapq.heappop(pending)
                if distance != distances[cell]:
                    continue
                for dq, dr in all_links:
                    target = cell[0] + dq, cell[1] + dr
                    if max(map(abs, target)) > 20:
                        continue
                    trial = distance + graph.pitch * math.sqrt(dq*dq + dq*dr + dr*dr)
                    if trial < distances.get(target, math.inf):
                        distances[target] = trial
                        heapq.heappush(pending, (trial, target))
            with localcontext() as decimal:
                decimal.prec = 80
                for q, r in [(q, r) for q in range(-12, 13) for r in range(-12, 13)] + [
                    (1 << 27, 1 << 27), (-(1 << 27), 1 << 27), (1 << 27, 1),
                ]:
                    lower = lattice_distance_lower(q, r, graph.pitch)
                    a, b, _c = sorted((abs(q), abs(r), abs(q+r)))
                    exact = Decimal.from_float(graph.pitch) * (
                        Decimal(b) + (Decimal(3).sqrt() - 1) * Decimal(a))
                    assert Decimal.from_float(lower) <= exact
                    if abs(q) <= 12 and abs(r) <= 12:
                        assert lower == pytest.approx(distances[q, r], abs=1e-8)
            with pytest.raises(ValueError, match="bounded axial"):
                lattice_distance_lower((1 << 27) + 1, 0, graph.pitch)


@pytest.mark.parametrize("angle", [0.0, 17.0, 47.0, 90.0])
@pytest.mark.parametrize("time_buffer_s", [0.0, 4.0])
def test_anyangle_full_volume_rows_cover_continuous_conflicts(angle, time_buffer_s):
    """Independent FCL conflicts imply shared full-volume rows across angles and contacts."""
    cfg = _cfg(time_buffer_s=time_buffer_s)
    radians = math.radians(angle)
    along = np.asarray((math.cos(radians), math.sin(radians), 0.0))
    normal = np.asarray((-math.sin(radians), math.cos(radians), 0.0))
    center = np.asarray((720.0, 720.0, 100.0))
    first = corridor_segment_volume(center - 135.0 * along, 100.0,
                                    center + 135.0 * along, 109.0, cfg)
    row_index = RowIndex()
    first_claims = volume_claims(first, cfg, row_index=row_index)
    others = [
        corridor_segment_volume(center - 120.0 * normal, 100.0,
                                center + 120.0 * normal, 108.0, cfg),
        hover_reservation(center, 100.0, cfg),
    ]
    for gap in (0.0, 59.999999, 60.0, 60.000001, 240.0):
        shift = gap * normal
        others.append(corridor_segment_volume(
            center - 135.0 * along + shift, 100.0,
            center + 135.0 * along + shift, 109.0, cfg,
        ))
    conflicts = 0
    for second in others:
        if volumes_conflict(first, second):
            conflicts += 1
            assert first_claims & volume_claims(second, cfg, row_index=row_index)
    assert conflicts >= 4
    # Exact time contact has no conflict and does not acquire a neighbouring period.
    first = replace(first, t_start=100.0, t_end=112.0)
    next_volume = replace(first, t_start=112.0, t_end=124.0)
    assert not volumes_conflict(first, next_volume)
    assert not (
        volume_claims(first, cfg, row_index=row_index)
        & volume_claims(next_volume, cfg, row_index=row_index)
    )


def test_anyangle_spatial_claims_include_customer_cylinder_level_and_tile_contact():
    """Customer cylinders reserve every touched cruise band and closed hex boundary."""
    cfg = replace(_cfg(), flight_levels_m=(100.0, 150.0), airspace_ceiling_m=175.0)
    shape = CylinderSpec(0.0, 0.0, 60.0, 0.0, 175.0)
    cells = volume_spatial_cells(shape, cfg)
    assert {(0, 0, 0), (0, 0, 1)} <= cells
    assert all((q, r, level) in cells for q, r in hg.AXIAL_NEIGHBORS for level in (0, 1))
    low = replace(shape, z_hi=120.0)
    assert {level for _, _, level in volume_spatial_cells(low, cfg)} == {0}
    tagged_box = replace(
        corridor_segment_volume((0.0, 0.0, 100.0), 100.0,
                                (120.0, 0.0, 100.0), 104.0, cfg),
        terminal_id="same-hub",
    )
    assert volumes_conflict(tagged_box, tagged_box)
    rows = volume_claims(tagged_box, cfg, row_index=RowIndex())
    assert rows and all(row.kind == "cell" for row in rows)
