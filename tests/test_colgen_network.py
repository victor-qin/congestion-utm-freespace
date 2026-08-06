"""Contracts for colgen's space-time network and ledger translation.

These tests deliberately use the ledger's continuous FCL predicate as an independent
referee.  The important implication is one-way: whenever two filed geometry templates
conflict, the corresponding columns must share a capacity row.  Conservative extra rows
are permitted by the formulation; a missed row is not.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import replace

import numpy as np
import pytest

from analysis.ab_column_clear import _value_sig
from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.cost import trajectory_cost
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.metrics import flight_row, total_delay_s
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
import freespace_sim.planner.colgen.network as network_module
from freespace_sim.planner.colgen.network import (
    RowIndex,
    RowKey,
    StaticTerminalCatalog,
    build_flight_graph,
    column_claims,
)
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import seed_column
from freespace_sim.planner.colgen.translate import (
    Column,
    column_to_corners,
    column_to_intent,
)
from freespace_sim.planner.colgen.windows import (
    derive_cell_window,
    endpoint_claim_cells,
    endpoint_claim_steps,
    terminal_claim_steps,
    visit_rows,
)
from freespace_sim.types import DenialReason, FlightRequest, Terminal, vec
from freespace_sim.volumes import (
    column_dwell_s,
    corridor_segment_volume,
    enroute_detour_m,
    enroute_reference_m,
    exit_radius,
    hover_reservation,
)


def _cfg(*, time_buffer_s: float = 4.0, planner: str = "colgen") -> SimConfig:
    """The v1 single-level world, retaining all default ledger geometry."""

    return SimConfig(
        planner=planner,
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        time_buffer_s=time_buffer_s,
        region_size_m=(20_000.0, 20_000.0),
        terminal_airspace_always_active=True,
    )


def _point(cell: tuple[int, int], cfg: SimConfig, *, z: float | None = None):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.flight_levels_m[0] if z is None else z)


def _ground_point(cell: tuple[int, int], cfg: SimConfig, dx: float = 0.0, dy: float = 0.0):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x + dx, y + dy, cfg.ground_level_m)


def _walk(start: tuple[int, int], directions) -> tuple[tuple[int, int], ...]:
    out = [start]
    for dq, dr in directions:
        q, r = out[-1]
        out.append((q + dq, r + dr))
    return tuple(out)


def _shortest_path(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return one deterministic axial geodesic, including both endpoints."""

    out = [a]
    while out[-1] != b:
        distance = hg.hex_distance(out[-1], b)
        candidates = sorted(
            neighbour
            for neighbour in hg.hex_neighbors(*out[-1])
            if hg.hex_distance(neighbour, b) == distance - 1
        )
        assert candidates
        out.append(candidates[0])
    return tuple(out)


def _lane_index(lanes, cell: tuple[int, int]) -> int:
    return next(index for index, lane in enumerate(lanes) if lane.cell == cell)


def _priced_delay_s(
    req: FlightRequest,
    path: tuple[tuple[int, int], ...],
    departure_step: int,
    cfg: SimConfig,
    *,
    origin_lane=None,
    dest_lane=None,
) -> float:
    """Independent arc-price form from the plan: hold + hops/folds/snaps - reference."""

    radius = hg.circumradius(cfg)
    centres = [np.asarray(hg.hex_center(*cell, radius), float) for cell in path]
    flown = sum(float(np.linalg.norm(b - a)) for a, b in zip(centres, centres[1:]))

    origin_term = req.origin_terminal
    if origin_term is None:
        flown += float(np.linalg.norm(np.asarray(req.origin[:2], float) - centres[0]))
    else:
        assert origin_lane is not None
        flown += max(0.0, origin_lane.dist - exit_radius(origin_term, cfg))

    dest_term = req.dest_terminal
    if dest_term is None:
        flown += float(np.linalg.norm(np.asarray(req.dest[:2], float) - centres[-1]))
    else:
        assert dest_lane is not None
        flown += max(0.0, dest_lane.dist - exit_radius(dest_term, cfg))

    reference = enroute_reference_m(
        req.origin, req.dest, req.origin_terminal, req.dest_terminal, cfg
    )
    base = math.ceil(req.t_departure / cfg.dt_s)
    ground = (departure_step - base) * cfg.dt_s
    return ground + enroute_detour_m(flown, reference) / cfg.nominal_speed_mps


def _column_for(
    req: FlightRequest,
    path: tuple[tuple[int, int], ...],
    cfg: SimConfig,
    *,
    departure_step: int | None = None,
    slack: int = 12,
) -> tuple[Column, object]:
    """Build a graph-valid column and populate claims through the production owner."""

    fg = build_flight_graph(req, cfg, [], ColGenParams(detour_slack_hops=slack))
    departure_step = fg.base_step if departure_step is None else departure_step
    origin_lane_idx = None
    dest_lane_idx = None
    origin_lane = None
    dest_lane = None
    if fg.origin_terminal is not None:
        origin_lane_idx = _lane_index(fg.origin_lanes, path[0])
        origin_lane = fg.origin_lanes[origin_lane_idx]
    if fg.dest_terminal is not None:
        dest_lane_idx = _lane_index(fg.dest_lanes, path[-1])
        dest_lane = fg.dest_lanes[dest_lane_idx]
    delay_s = _priced_delay_s(
        req,
        path,
        departure_step,
        cfg,
        origin_lane=origin_lane,
        dest_lane=dest_lane,
    )
    raw = Column(
        req.flight_id,
        departure_step,
        0,
        origin_lane_idx,
        dest_lane_idx,
        path,
        delay_s,
    )
    return replace(raw, claims=column_claims(raw, fg, cfg)), fg


def test_derive_cell_window_value():
    default = SimConfig()
    stacked = replace(
        default,
        flight_levels_m=(80.0, 95.0, 110.0),
        corridor_height_m=14.0,
    )

    assert derive_cell_window(default) == (-2, 1)
    assert derive_cell_window(stacked) == (-2, 1)
    assert derive_cell_window(replace(default, time_buffer_s=0.0)) == (-1, 0)


def test_wide_corridor_rejected_when_nonincident_edges_can_conflict():
    cfg = replace(_cfg(), corridor_width_m=100.0)
    req = FlightRequest(
        9,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    with pytest.raises(NotImplementedError, match="nonincident lattice edges.*issue #72"):
        build_flight_graph(req, cfg, [], ColGenParams())


def test_endpoint_claim_reach_covers_custom_large_hover_radius():
    cfg = replace(_cfg(), hover_radius_m=100.0)
    first = vec(0.0, 0.0, 0.0)
    second = vec(
        195.0, 0.0, 0.0
    )  # 100 m cylinders overlap, but their snapped centres are 240 m apart
    second_cell = hg.enu_to_axial(float(second[0]), float(second[1]), hg.circumradius(cfg))
    assert second_cell in endpoint_claim_cells(first, cfg.effective_hover_radius_m, cfg)


def test_visit_rows_uses_inclusive_derived_offsets():
    assert list(visit_rows(10, (-2, 1))) == [8, 9, 10, 11]
    assert list(visit_rows(10, (-1, 0))) == [9, 10]
    with pytest.raises(ValueError, match="lo <= hi"):
        visit_rows(10, (1, -1))


def test_row_index_interns_stably_and_uses_terminal_pad_capacity():
    rows = RowIndex({"hub": 3})
    cell = RowKey.cell((2, -4), 0, 17)
    terminal = RowKey.term("hub", 17)

    cell_id = rows.intern(cell)
    assert cell_id == 0
    assert rows.intern(("cell", 2, -4, 0, 17)) == cell_id
    terminal_id = rows[terminal]
    assert terminal_id == 1
    assert rows.key(cell_id) == ("cell", 2, -4, 0, 17)
    assert rows.key(terminal_id) == terminal
    assert rows.cap(cell) == rows.cap(cell_id) == 1
    assert rows.cap(np.int64(cell_id)) == 1
    assert rows.cap(terminal) == rows.cap(terminal_id) == 3
    assert rows.items() == ((cell, 0), (terminal, 1))

    rows.register_terminal(Terminal("hub", 3))  # idempotent metadata
    with pytest.raises(ValueError, match="inconsistent capacities"):
        rows.register_terminal("hub", 2)
    with pytest.raises(KeyError, match="capacity.*unknown"):
        RowIndex().cap(RowKey.term("unregistered", 0))
    with pytest.raises(IndexError, match="row index -1"):
        rows.cap(-1)


def test_params_validate_integral_nonnegative_slack():
    assert ColGenParams(np.int64(3)).detour_slack_hops == 3
    with pytest.raises(TypeError, match="must be an integer"):
        ColGenParams(1.5)
    with pytest.raises(ValueError, match="non-negative"):
        ColGenParams(-1)


def test_corridor_prune_contains_a_shortest_path_and_obeys_ellipse():
    cfg = _cfg()
    origin_cell, dest_cell = (-5, 2), (6, -3)
    req = FlightRequest(
        10,
        _ground_point(origin_cell, cfg, 7.0, -4.0),
        _ground_point(dest_cell, cfg, -5.0, 3.0),
        4.1,
        9.2,
    )
    slack = 4
    fg = build_flight_graph(req, cfg, [], ColGenParams(detour_slack_hops=slack))

    assert fg.origin_cell == origin_cell and fg.dest_cell == dest_cell
    assert set(_shortest_path(origin_cell, dest_cell)) <= fg.corridor_cells
    assert all(
        hg.hex_distance(origin_cell, cell) + hg.hex_distance(cell, dest_cell)
        <= fg.shortest_hops + slack
        for cell in fg.corridor_cells
    )
    assert fg.index_to_cell == tuple(sorted(fg.corridor_cells))
    assert all(fg.cell_to_index[cell] == i for i, cell in enumerate(fg.index_to_cell))
    assert fg.base_step == math.ceil(req.t_departure / cfg.dt_s)
    assert fg.latest_departure_step == fg.base_step + math.floor(cfg.max_ground_delay_s / cfg.dt_s)
    rebuilt = pickle.loads(pickle.dumps(fg))
    assert rebuilt == fg
    assert rebuilt.cell_to_index == fg.cell_to_index
    assert not hasattr(fg.cell_to_index, "clear")
    with pytest.raises(TypeError):
        fg.cell_to_index[origin_cell] = 999
    with pytest.raises(AttributeError, match="immutable"):
        fg.cell_to_index._data = {origin_cell: 999}
    with pytest.raises(AttributeError, match="reinitialized"):
        fg.cell_to_index.__init__(((origin_cell, 999),))
    assert fg.cell_to_index[origin_cell] == rebuilt.cell_to_index[origin_cell]

    request_origin = fg.request.origin.copy()
    req.origin[0] += 1234.0
    assert np.array_equal(fg.request.origin, request_origin)
    with pytest.raises(ValueError, match="read-only"):
        fg.request.origin[0] += 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        fg.request.origin.setflags(write=True)
    with pytest.raises(AttributeError, match="snapshot is immutable"):
        fg.request.origin = np.zeros(3)
    with pytest.raises(AttributeError, match="snapshot is immutable"):
        fg.request.origin_terminal = Terminal("mutated", 1)
    with pytest.raises(AttributeError, match="snapshot is immutable"):
        rebuilt.request.dest_terminal = Terminal("mutated", 1)


def test_graph_build_keeps_corridor_and_static_arcs_lazy(monkeypatch):
    """Construction must not enumerate the ellipse or classify any directed hop."""

    cfg = _cfg()
    req = FlightRequest(
        109,
        _ground_point((-40, 0), cfg),
        _ground_point((40, 0), cfg),
        0.0,
    )
    wall = Terminal("far-wall", 1, radius=30.0)

    def fail_if_checked(*_args, **_kwargs):
        pytest.fail("build_flight_graph eagerly classified a static hop")

    # `_static_hop_allowed_roles`, NOT the `_static_hop_forbidden` wrapper: the lazy oracle
    # calls the former directly, so a sentinel on the wrapper can never fire and the guard
    # would pass no matter how eager construction became.
    monkeypatch.setattr(network_module, "_static_hop_allowed_roles", fail_if_checked)
    graph = build_flight_graph(
        req,
        cfg,
        [(_ground_point((0, 20), cfg), wall)],
        ColGenParams(detour_slack_hops=12),
    )

    assert not graph.corridor_cells.is_materialized
    assert dict(graph.arc_cache_stats) == {
        "expanded_nodes": 0,
        "arc_checks": 0,
        "cache_hits": 0,
        "allowed_arcs": 0,
        "blocked_arcs": 0,
    }


def test_static_catalog_is_shared_without_rebuilding_terminal_cells(monkeypatch):
    cfg = _cfg()
    terminals = tuple(
        (_ground_point(cell, cfg), Terminal(f"wall-{index}", 1, radius=90.0))
        for index, cell in enumerate(((40, 40), (50, 50), (60, 60)))
    )
    calls = 0
    original_terminal_cells = hg.terminal_cells

    def count_terminal_cells(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_terminal_cells(*args, **kwargs)

    monkeypatch.setattr(hg, "terminal_cells", count_terminal_cells)
    catalog = StaticTerminalCatalog(terminals, cfg)
    catalog_calls = calls
    requests = (
        FlightRequest(120, _ground_point((-5, 0), cfg), _ground_point((5, 0), cfg), 0.0),
        FlightRequest(121, _ground_point((-5, 5), cfg), _ground_point((5, 5), cfg), 0.0),
    )
    graphs = tuple(
        build_flight_graph(request, cfg, catalog, ColGenParams(detour_slack_hops=2))
        for request in requests
    )

    assert catalog_calls == len(terminals)
    assert calls == catalog_calls
    assert all(graph.static_walls is catalog.walls for graph in graphs)
    assert all(graph._wall_index is catalog.wall_index for graph in graphs)
    assert all(not graph.corridor_cells.is_materialized for graph in graphs)


def test_many_far_walls_never_reach_narrow_phase(monkeypatch):
    cfg = _cfg()
    terminals = tuple(
        (
            _ground_point((80 + index, 80), cfg),
            Terminal(f"far-{index}", 1, radius=90.0),
        )
        for index in range(100)
    )
    catalog = StaticTerminalCatalog(terminals, cfg)
    request = FlightRequest(
        122,
        _ground_point((-5, 0), cfg),
        _ground_point((5, 0), cfg),
        0.0,
    )
    graph = build_flight_graph(
        request,
        cfg,
        catalog,
        ColGenParams(detour_slack_hops=2),
    )
    calls = 0
    original_conflict = network_module.volumes_conflict

    def count_conflicts(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_conflict(*args, **kwargs)

    monkeypatch.setattr(network_module, "volumes_conflict", count_conflicts)

    seed = seed_column(graph, cfg)

    assert seed.claims
    assert calls == 0
    assert catalog.wall_index.stats["queries"] > 0
    assert catalog.wall_index.stats["candidates"] == 0


def test_lazy_outgoing_arcs_expand_each_source_once():
    cfg = _cfg()
    req = FlightRequest(
        110,
        _ground_point((-4, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    graph = build_flight_graph(req, cfg, (), ColGenParams(detour_slack_hops=2))

    first = graph.outgoing_neighbors(graph.origin_cell)
    after_first = dict(graph.arc_cache_stats)
    second = graph.outgoing_neighbors(graph.origin_cell)
    after_second = dict(graph.arc_cache_stats)

    assert first == second
    assert first
    assert after_first["expanded_nodes"] == 1
    assert after_first["arc_checks"] == len(first)
    assert after_second["expanded_nodes"] == 1
    assert after_second["arc_checks"] == after_first["arc_checks"]
    assert after_second["cache_hits"] == after_first["cache_hits"] + 1
    assert not graph.corridor_cells.is_materialized


def test_lazy_static_arc_verdicts_match_eager_reference():
    """Laziness changes evaluation order, not the all-tag-variants predicate."""

    cfg = _cfg()
    wall_center = vec(-61.4786398, 33.5115978, 0.0)
    wall = Terminal("small-wall", 1, radius=30.0)
    req = FlightRequest(
        1110,
        _ground_point((-3, 0), cfg),
        _ground_point((3, 0), cfg),
        0.0,
    )
    graph = build_flight_graph(
        req,
        cfg,
        ((wall_center, wall),),
        ColGenParams(detour_slack_hops=2),
    )
    cells = frozenset(graph.corridor_cells)
    eager = network_module._forbidden_static_hops(
        cells,
        graph.static_walls,
        graph.request,
        graph.origin_terminal,
        graph.dest_terminal,
        graph.origin_lanes,
        graph.dest_lanes,
        cfg,
    )

    for source in cells:
        for target in hg.hex_neighbors(*source):
            if target in cells:
                assert graph.hop_is_forbidden(source, target) == ((source, target) in eager)


def test_warmed_lazy_graph_pickles_with_cold_answer_equivalent_cache():
    cfg = _cfg()
    req = FlightRequest(
        1111,
        _ground_point((-4, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    graph = build_flight_graph(req, cfg, (), ColGenParams(detour_slack_hops=2))
    expected = graph.outgoing_neighbors(graph.origin_cell)
    assert graph.arc_cache_stats["expanded_nodes"] == 1

    rebuilt = pickle.loads(pickle.dumps(graph))

    assert rebuilt == graph
    assert rebuilt.arc_cache_stats["expanded_nodes"] == 0
    assert rebuilt.outgoing_neighbors(rebuilt.origin_cell) == expected


def test_corridor_prune_removes_foreign_terminal_cells():
    cfg = _cfg()
    req = FlightRequest(
        11,
        _ground_point((-8, 0), cfg),
        _ground_point((8, 0), cfg),
        0.0,
    )
    foreign_center = _ground_point((0, 0), cfg)
    foreign = Terminal("foreign", 4, radius=90.0)
    fg = build_flight_graph(
        req,
        cfg,
        [(foreign_center, foreign)],
        ColGenParams(detour_slack_hops=6),
    )
    wall = hg.terminal_cells(foreign_center, foreign, cfg)

    assert wall
    assert fg.foreign_exclusions == wall
    assert fg.corridor_cells.isdisjoint(wall)


def test_customer_endpoint_disk_cannot_overlap_foreign_static_terminal():
    cfg = _cfg()
    foreign = Terminal("foreign", 4, radius=90.0)
    # Snaps outside terminal_cells, but 149.82 m < 90 m wall + 60 m customer cylinder.
    origin = vec(-137.5, -59.5, 0.0)
    assert hg.enu_to_axial(*origin[:2], hg.circumradius(cfg)) not in hg.terminal_cells(
        vec(0.0, 0.0, 0.0), foreign, cfg
    )
    req = FlightRequest(111, origin, _ground_point((-8, -8), cfg), 0.0)
    with pytest.raises(ValueError, match="origin cylinder overlaps foreign static terminal"):
        build_flight_graph(
            req,
            cfg,
            [(vec(0.0, 0.0, 0.0), foreign)],
            ColGenParams(),
        )


def test_hop_overlapping_both_endpoint_walls_is_rejected():
    """A one-box hub route cannot carry both endpoint terminal tags."""

    cfg = _cfg()
    origin = vec(5.0, -55.0, 0.0)
    dest = vec(345.0, 60.0, 0.0)
    origin_terminal = Terminal("A", 1, radius=90.0)
    dest_terminal = Terminal("B", 1, radius=90.0)
    assert hg.terminal_cells(origin, origin_terminal, cfg).isdisjoint(
        hg.terminal_cells(dest, dest_terminal, cfg)
    )

    req = FlightRequest(
        113,
        origin,
        dest,
        0.0,
        origin_terminal=origin_terminal,
        dest_terminal=dest_terminal,
    )
    fg = build_flight_graph(
        req,
        cfg,
        [(origin, origin_terminal), (dest, dest_terminal)],
        ColGenParams(),
    )
    path = ((1, 0), (2, 0))
    raw = Column(
        req.flight_id,
        fg.base_step,
        0,
        _lane_index(fg.origin_lanes, path[0]),
        _lane_index(fg.dest_lanes, path[-1]),
        path,
        0.0,
    )

    assert (path[0], path[1]) in fg.forbidden_hops
    with pytest.raises(ValueError, match="permanent static terminal airspace"):
        column_claims(raw, fg, cfg)

    # Exercise the authoritative translated-volume backstop independently of
    # the pricing-usable forbidden-hop layer.  The builder gives the one box
    # the origin tag, so the distinct destination wall must still reject it.
    exact_only_fg = replace(fg, forbidden_hops=frozenset())
    with pytest.raises(ValueError, match="overlaps permanent static terminal 'B'"):
        column_claims(raw, exact_only_fg, cfg)


def test_resampled_endpoint_hop_with_separate_tags_is_not_falsely_forbidden():
    """A float-split edge may safely carry A then B instead of one ambiguous tag."""

    cfg = _cfg()
    origin = _ground_point((1, 39), cfg)
    dest = _ground_point((-2, 42), cfg)
    origin_terminal = Terminal("A", 1, radius=90.0)
    dest_terminal = Terminal("B", 1, radius=90.0)
    req = FlightRequest(
        116,
        origin,
        dest,
        0.0,
        origin_terminal=origin_terminal,
        dest_terminal=dest_terminal,
    )
    fg = build_flight_graph(
        req,
        cfg,
        [(origin, origin_terminal), (dest, dest_terminal)],
        ColGenParams(),
    )
    path = ((0, 40), (-1, 41))
    raw = Column(
        req.flight_id,
        fg.base_step,
        0,
        _lane_index(fg.origin_lanes, path[0]),
        _lane_index(fg.dest_lanes, path[-1]),
        path,
        0.0,
    )

    intent = column_to_intent(raw, req, cfg)
    assert intent.accepted
    assert [volume.terminal_id for volume in intent.volumes] == ["A", "A", "B", "B"]
    assert not any(
        volumes_conflict(volume, wall) for volume in intent.volumes for wall in fg.static_walls
    )
    assert (path[0], path[1]) not in fg.forbidden_hops
    assert column_claims(raw, fg, cfg)


def test_exact_static_hops_cover_empty_terminal_cell_projection():
    """A small off-grid wall remains real ledger geometry even with no raster cells."""

    cfg = _cfg()
    wall_center = vec(-61.4786398, 33.5115978, 0.0)
    wall = Terminal("small-wall", 1, radius=30.0)
    assert hg.terminal_cells(wall_center, wall, cfg) == set()

    path = tuple((q, 0) for q in range(-3, 3))
    req = FlightRequest(
        114,
        _ground_point(path[0], cfg),
        _ground_point(path[-1], cfg),
        0.0,
    )
    fg = build_flight_graph(
        req,
        cfg,
        [(wall_center, wall)],
        ColGenParams(detour_slack_hops=2),
    )
    raw = Column(req.flight_id, fg.base_step, 0, None, None, path, 0.0)

    assert fg.foreign_exclusions == frozenset()
    assert any((a, b) in fg.forbidden_hops for a, b in zip(path, path[1:]))
    with pytest.raises(ValueError, match="permanent static terminal airspace"):
        column_claims(raw, fg, cfg)

    safe_path = ((-3, 0), (-3, 1), (-2, 1), (-1, 1), (0, 1), (1, 0), (2, 0))
    safe = Column(req.flight_id, fg.base_step, 0, None, None, safe_path, 0.0)
    assert column_claims(safe, fg, cfg)
    safe_intent = column_to_intent(safe, req, cfg)
    assert safe_intent.accepted
    ledger = ReservationLedger(cfg)
    ledger.register_static_terminal(wall_center, wall)
    assert not ledger.any_conflict(safe_intent.volumes)


def test_graph_max_step_preserves_takeoff_and_origin_lane_time_budget():
    cfg = replace(_cfg(), max_ground_delay_s=20.0)
    terminal = Terminal("origin", 2, radius=180.0)
    req = FlightRequest(
        112,
        _ground_point((0, 0), cfg),
        _ground_point((10, 0), cfg),
        0.0,
        origin_terminal=terminal,
    )
    params = ColGenParams(detour_slack_hops=4)
    fg = build_flight_graph(req, cfg, [], params)
    assert fg.max_step == (
        fg.latest_departure_step
        + max(fg.takeoff_steps)
        + max(lane.steps for lane in fg.origin_lanes)
        + fg.shortest_hops
        + params.detour_slack_hops
    )


def test_flight_graph_one_hop_and_same_cell_degenerate_guard():
    cfg = _cfg()
    one_hop = FlightRequest(
        12,
        _ground_point((0, 0), cfg),
        _ground_point((1, 0), cfg),
        0.0,
    )
    fg = build_flight_graph(one_hop, cfg, [], ColGenParams(detour_slack_hops=0))
    assert fg.shortest_hops == 1
    assert fg.corridor_cells == frozenset({(0, 0), (1, 0)})

    same_cell = FlightRequest(
        13,
        _ground_point((2, -1), cfg, -5.0, 3.0),
        _ground_point((2, -1), cfg, 4.0, -2.0),
        0.0,
    )
    with pytest.raises(ValueError, match="same hex.*at least one lateral hop"):
        build_flight_graph(same_cell, cfg, [], ColGenParams())


def test_hub_graph_rejects_legacy_nonfixed_exit_lane_geometry():
    cfg = replace(_cfg(), fixed_exit_lanes=False)
    hub = Terminal("legacy", 1, radius=90.0)
    hub_req = FlightRequest(
        117,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
        origin_terminal=hub,
    )
    with pytest.raises(NotImplementedError, match="requires fixed_exit_lanes=True"):
        build_flight_graph(hub_req, cfg, [], ColGenParams())

    lanes = hg.terminal_lanes(hub_req.origin, hub, cfg)
    lane_idx = min(
        range(len(lanes)),
        key=lambda index: hg.hex_distance(lanes[index].cell, (4, 0)),
    )
    path = _shortest_path(lanes[lane_idx].cell, (4, 0))
    raw = Column(hub_req.flight_id, 0, 0, lane_idx, None, path, 0.0)
    with pytest.raises(NotImplementedError, match="requires fixed_exit_lanes=True"):
        column_to_intent(raw, hub_req, cfg)

    customer_req = FlightRequest(
        118,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    assert build_flight_graph(customer_req, cfg, [], ColGenParams()).corridor_cells
    customer_col = Column(
        customer_req.flight_id,
        0,
        0,
        None,
        None,
        _shortest_path((0, 0), (4, 0)),
        0.0,
    )
    assert column_to_intent(customer_col, customer_req, cfg).accepted


def test_hub_boundaries_require_always_active_terminal_airspace():
    cfg = replace(_cfg(), terminal_airspace_always_active=False)
    hub = Terminal("dynamic", 1, radius=90.0)
    req = FlightRequest(
        122,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
        origin_terminal=hub,
    )
    with pytest.raises(NotImplementedError, match="terminal_airspace_always_active=True"):
        build_flight_graph(req, cfg, [], ColGenParams())

    lanes = hg.terminal_lanes(req.origin, hub, cfg)
    lane_idx = min(
        range(len(lanes)),
        key=lambda index: hg.hex_distance(lanes[index].cell, (4, 0)),
    )
    path = _shortest_path(lanes[lane_idx].cell, (4, 0))
    raw = Column(req.flight_id, 0, 0, lane_idx, None, path, 0.0)
    with pytest.raises(NotImplementedError, match="terminal_airspace_always_active=True"):
        column_to_intent(raw, req, cfg)

    customer = FlightRequest(
        123,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    assert build_flight_graph(customer, cfg, [], ColGenParams()).corridor_cells
    customer_col = Column(
        customer.flight_id,
        0,
        0,
        None,
        None,
        _shortest_path((0, 0), (4, 0)),
        0.0,
    )
    assert column_to_intent(customer_col, customer, cfg).accepted


def test_terminal_lane_to_same_customer_cell_cannot_form_zero_hop_column():
    cfg = _cfg()
    hub = vec(0.0, 0.0, 0.0)
    terminal = Terminal("hub", 1, radius=90.0)
    customer = vec(170.0, 0.0, 0.0)
    req = FlightRequest(115, hub, customer, 0.0, origin_terminal=terminal)
    fg = build_flight_graph(
        req,
        cfg,
        [(hub, terminal)],
        ColGenParams(),
    )
    shared_cell = hg.enu_to_axial(float(customer[0]), float(customer[1]), hg.circumradius(cfg))
    lane_idx = _lane_index(fg.origin_lanes, shared_cell)

    assert fg.origin_cell != fg.dest_cell
    with pytest.raises(ValueError, match="at least two cells"):
        Column(req.flight_id, fg.base_step, 0, lane_idx, None, (shared_cell,), 0.0)

    # The claims and translation boundaries retain their own checks for a
    # deserialized/adversarial value that bypasses the dataclass constructor.
    neighbour = hg.hex_neighbors(*shared_cell)[0]
    bypassed = Column(
        req.flight_id,
        fg.base_step,
        0,
        lane_idx,
        None,
        (shared_cell, neighbour),
        0.0,
    )
    object.__setattr__(bypassed, "cell_path", (shared_cell,))
    with pytest.raises(ValueError, match="at least two cells"):
        column_claims(bypassed, fg, cfg)
    with pytest.raises(ValueError, match="at least two cells"):
        column_to_corners(bypassed, req, cfg)
    with pytest.raises(ValueError, match="at least two cells"):
        column_to_intent(bypassed, req, cfg)


def test_column_requires_integral_network_clock_and_coordinates():
    cfg = _cfg()
    req = FlightRequest(
        119,
        _ground_point((0, 0), cfg),
        _ground_point((1, 0), cfg),
        0.0,
    )
    with pytest.raises(TypeError, match="departure_step must be an integer"):
        Column(req.flight_id, 0.5, 0, None, None, ((0, 0), (1, 0)), 0.0)
    with pytest.raises(TypeError, match="level must be an integer"):
        Column(req.flight_id, 0, 0.5, None, None, ((0, 0), (1, 0)), 0.0)
    with pytest.raises(TypeError, match="cells must be integer"):
        Column(req.flight_id, 0, 0, None, None, ((0.5, 0), (1, 0)), 0.0)

    valid = Column(
        np.int64(req.flight_id),
        np.int64(0),
        np.int64(0),
        None,
        None,
        ((np.int64(0), np.int64(0)), (np.int64(1), np.int64(0))),
        0.0,
    )
    assert valid.departure_step == 0 and valid.cell_path == ((0, 0), (1, 0))

    fg = build_flight_graph(req, cfg, [], ColGenParams(detour_slack_hops=0))
    object.__setattr__(valid, "departure_step", 0.5)
    with pytest.raises(TypeError, match="departure_step must be an integer"):
        column_to_intent(valid, req, cfg)
    with pytest.raises(TypeError, match="departure_step must be an integer"):
        column_claims(valid, fg, cfg)


def _translation_fixture(kind: str):
    cfg = _cfg()
    terminal = Terminal(f"hub-{kind}", 2, radius=90.0)
    if kind == "clean":
        hub_cell = (0, 0)
        path = ((1, 0), (2, 0), (3, 0), (4, 0))
    else:
        # This path contains the floating-point resampling tripwire named in the plan:
        # the interior hop (0, 40) -> (-1, 41) rebuilds into two sub-boxes.
        hub_cell = (0, 36)
        path = ((0, 37), (0, 38), (0, 39), (0, 40), (-1, 41), (-2, 42), (-3, 43))
    origin = _ground_point(hub_cell, cfg)
    dest = _ground_point(path[-1], cfg, 13.0, -7.0)
    req = FlightRequest(
        100 if kind == "clean" else 101,
        origin,
        dest,
        100.0,
        101.3,
        origin_terminal=terminal,
    )
    lanes = hg.terminal_lanes(origin, terminal, cfg)
    lane_idx = _lane_index(lanes, path[0])
    departure_step = math.ceil(req.t_departure / cfg.dt_s) + 3
    delay_s = _priced_delay_s(
        req,
        path,
        departure_step,
        cfg,
        origin_lane=lanes[lane_idx],
    )
    return (
        cfg,
        req,
        Column(
            req.flight_id,
            departure_step,
            0,
            lane_idx,
            None,
            path,
            delay_s,
        ),
    )


@pytest.mark.parametrize("kind", ["clean", "resampled"])
def test_translate_containment_parity_vs_astar(kind):
    """Rebuilding may split a float-long hop, but every filed box stays inside A*'s box."""

    cfg, req, col = _translation_fixture(kind)
    intent = column_to_intent(col, req, cfg)
    corners, _build_delay, _report_delay, corridor_t0, origin_term, dest_term = column_to_corners(
        col, req, cfg
    )
    cruise_wps = [(point, corridor_t0 + i * cfg.dt_s) for i, point in enumerate(corners)]
    base = math.ceil(req.t_departure / cfg.dt_s)
    astar_volumes, *_ = AStarPlanner()._build(
        cruise_wps,
        req.origin,
        req.dest,
        base,
        col.departure_step - base,
        cfg,
        origin_term=origin_term,
        dest_term=dest_term,
    )

    # The takeoff clock is integer-anchored on both paths, so the whole origin cylinder
    # is exact, not merely close.
    assert _value_sig(intent.volumes[0]) == _value_sig(astar_volumes[0])

    rebuilt_dest, astar_dest = intent.volumes[-1], astar_volumes[-1]
    assert isinstance(rebuilt_dest.shape, CylinderSpec)
    assert _value_sig(rebuilt_dest.shape) == _value_sig(astar_dest.shape)
    assert rebuilt_dest.terminal_id == astar_dest.terminal_id
    assert rebuilt_dest.t_start == astar_dest.t_start
    assert rebuilt_dest.t_end == astar_dest.t_end

    rebuilt_boxes = intent.volumes[1:-1]
    astar_boxes = astar_volumes[1:-1]
    assert len(astar_boxes) == len(col.cell_path) - 1
    if kind == "resampled":
        assert ((0, 40), (-1, 41)) in set(zip(col.cell_path, col.cell_path[1:]))
        assert len(rebuilt_boxes) > len(astar_boxes)
    else:
        assert len(rebuilt_boxes) == len(astar_boxes)

    for rebuilt in rebuilt_boxes:
        matches = []
        rebuilt_lo, rebuilt_hi = rebuilt.aabb()
        for parent in astar_boxes:
            parent_lo, parent_hi = parent.aabb()
            spatially_contained = bool(
                np.all(rebuilt_lo >= parent_lo) and np.all(rebuilt_hi <= parent_hi)
            )
            temporally_contained = (
                rebuilt.t_start >= parent.t_start - 1e-9 and rebuilt.t_end <= parent.t_end + 1e-9
            )
            if spatially_contained and temporally_contained:
                matches.append(parent)
        assert matches, f"no A* hop contains rebuilt box {_value_sig(rebuilt)!r}"
        assert any(rebuilt.terminal_id == parent.terminal_id for parent in matches)
        assert (
            min(
                max(parent.t_start - rebuilt.t_start, rebuilt.t_end - parent.t_end, 0.0)
                for parent in matches
            )
            < 1e-9
        )


@pytest.mark.parametrize(
    ("time_buffer_s", "second_departure_step"),
    [(4.0, 4), (0.0, 2)],
    ids=["default", "zero-buffer"],
)
def test_translated_hops_stay_on_integer_clock_and_preserve_row_coverage(
    time_buffer_s, second_departure_step
):
    """Float-long lattice hops must not cross the first row-disjoint boundary."""

    cfg = _cfg(time_buffer_s=time_buffer_s)
    first_path = tuple((6 - i, 34 + i) for i in range(13))
    second_path = tuple((-6 + i, 40) for i in range(13))
    assert first_path[6] == second_path[6] == (0, 40)
    first_req = FlightRequest(
        120,
        _ground_point(first_path[0], cfg),
        _ground_point(first_path[-1], cfg),
        0.0,
    )
    second_req = FlightRequest(
        121,
        _ground_point(second_path[0], cfg),
        _ground_point(second_path[-1], cfg),
        0.0,
    )
    first, first_fg = _column_for(
        first_req,
        first_path,
        cfg,
        departure_step=0,
        slack=0,
    )
    second, _second_fg = _column_for(
        second_req,
        second_path,
        cfg,
        departure_step=second_departure_step,
        slack=0,
    )

    assert first.claims.isdisjoint(second.claims)
    first_intent = column_to_intent(first, first_req, cfg)
    second_intent = column_to_intent(second, second_req, cfg)
    assert first_intent.accepted and second_intent.accepted
    # The known (0,40)->(-1,41) floating tripwire still subdivides spatially.
    assert len(first_intent.volumes[1:-1]) > len(first_path) - 1
    expected_arrival = (
        first.departure_step + first_fg.takeoff_steps[first.level] + len(first_path) - 1
    ) * cfg.dt_s
    assert first_intent.centerline[-1][1] == expected_arrival
    assert first_intent.volumes[-1].t_start == expected_arrival
    assert not any(
        volumes_conflict(first_volume, second_volume)
        for first_volume in first_intent.volumes
        for second_volume in second_intent.volumes
    )


@pytest.mark.parametrize("endpoint_kinds", ["hub-hub", "hub-customer", "customer-customer"])
def test_column_delay_equals_metrics_total_delay(endpoint_kinds):
    cfg = _cfg()
    term_a = Terminal(f"A-{endpoint_kinds}", 3, radius=90.0)
    term_b = Terminal(f"B-{endpoint_kinds}", 2, radius=90.0)

    if endpoint_kinds == "hub-hub":
        origin = _ground_point((0, 0), cfg)
        dest = _ground_point((8, 0), cfg)
        origin_term, dest_term = term_a, term_b
        origin_lanes = hg.terminal_lanes(origin, term_a, cfg)
        dest_lanes = hg.terminal_lanes(dest, term_b, cfg)
        start = max(origin_lanes, key=lambda lane: lane.cell[0]).cell
        end = min(dest_lanes, key=lambda lane: lane.cell[0]).cell
    elif endpoint_kinds == "hub-customer":
        origin = _ground_point((0, 0), cfg)
        dest = _ground_point((7, 2), cfg, 19.0, -11.0)
        origin_term, dest_term = term_a, None
        origin_lanes = hg.terminal_lanes(origin, term_a, cfg)
        start = max(origin_lanes, key=lambda lane: lane.cell[0]).cell
        end = hg.enu_to_axial(float(dest[0]), float(dest[1]), hg.circumradius(cfg))
    else:
        origin = _ground_point((-1, -1), cfg, -17.0, 9.0)
        dest = _ground_point((7, 2), cfg, 19.0, -11.0)
        origin_term = dest_term = None
        start = hg.enu_to_axial(float(origin[0]), float(origin[1]), hg.circumradius(cfg))
        end = hg.enu_to_axial(float(dest[0]), float(dest[1]), hg.circumradius(cfg))

    req = FlightRequest(
        200 + ["hub-hub", "hub-customer", "customer-customer"].index(endpoint_kinds),
        origin,
        dest,
        40.0,
        41.3,
        origin_terminal=origin_term,
        dest_terminal=dest_term,
    )
    path = _shortest_path(start, end)
    departure_step = math.ceil(req.t_departure / cfg.dt_s) + 2
    col, _fg = _column_for(
        req,
        path,
        cfg,
        departure_step=departure_step,
        slack=12,
    )
    intent = column_to_intent(col, req, cfg)

    assert col.delay_s == pytest.approx(total_delay_s(intent, cfg), abs=1e-9)
    assert intent.cost == pytest.approx(trajectory_cost(intent, cfg), abs=1e-12)


def test_reported_cost_and_lattice_overhead_exclude_empty_world_deconfliction():
    cfg = _cfg()
    # A 30-degree-ish customer route needs a staircase even in empty airspace.
    origin = _ground_point((0, 0), cfg, 11.0, -8.0)
    dest = _ground_point((8, 4), cfg, -17.0, 12.0)
    req = FlightRequest(300, origin, dest, 0.0)
    start = hg.enu_to_axial(float(origin[0]), float(origin[1]), hg.circumradius(cfg))
    end = hg.enu_to_axial(float(dest[0]), float(dest[1]), hg.circumradius(cfg))
    col, _fg = _column_for(req, _shortest_path(start, end), cfg, slack=0)
    intent = column_to_intent(col, req, cfg)
    row = flight_row(intent, cfg)

    assert intent.cost == pytest.approx(trajectory_cost(intent, cfg), abs=1e-12)
    assert intent.lattice_overhead_m > 0.0
    assert intent.lattice_overhead_m == pytest.approx(intent.air_detour_m, abs=1e-9)
    assert row["deconfliction_detour_m"] == pytest.approx(0.0, abs=1e-9)


def test_max_detour_factor_rejects_graph_valid_column():
    cfg = replace(_cfg(), max_detour_factor=1.1)
    path = ((0, 0), (0, 1), (1, 1), (2, 0), (3, 0))
    req = FlightRequest(
        301,
        _ground_point(path[0], cfg),
        _ground_point(path[-1], cfg),
        0.0,
    )
    fg = build_flight_graph(req, cfg, [], ColGenParams(detour_slack_hops=2))
    raw = Column(req.flight_id, fg.base_step, 0, None, None, path, 0.0)

    with pytest.raises(ValueError, match="max_detour_factor"):
        column_claims(raw, fg, cfg)
    denied = column_to_intent(raw, req, cfg)
    assert not denied.accepted
    assert denied.denial_reason is DenialReason.BUDGET_EXCEEDED
    assert denied.planner == "colgen"

    shortest = _shortest_path(path[0], path[-1])
    control = Column(req.flight_id, fg.base_step, 0, None, None, shortest, 0.0)
    assert column_claims(control, fg, cfg)
    assert column_to_intent(control, req, cfg).accepted


def test_claims_use_translated_detour_gate_at_ulp_boundary():
    path = (
        (404, -380),
        (404, -381),
        (405, -382),
        (404, -382),
        (405, -383),
        (404, -383),
        (403, -382),
        (402, -382),
        (403, -382),
        (402, -381),
        (403, -381),
        (404, -382),
        (405, -383),
        (406, -384),
        (406, -385),
        (405, -384),
        (405, -385),
        (406, -386),
        (406, -387),
        (406, -386),
        (405, -385),
        (404, -385),
        (403, -385),
        (402, -385),
        (403, -385),
        (403, -384),
        (402, -383),
        (402, -384),
        (401, -383),
        (402, -384),
        (403, -385),
        (402, -385),
        (403, -386),
        (404, -386),
        (405, -386),
        (404, -386),
        (404, -387),
        (403, -386),
        (403, -385),
        (404, -385),
        (403, -384),
        (404, -384),
        (405, -385),
        (404, -385),
        (403, -384),
    )
    cfg = replace(
        _cfg(),
        region_size_m=(1_000_000.0, 1_000_000.0),
        max_detour_factor=9.60158717038366,
    )
    req = FlightRequest(
        302,
        _ground_point(path[0], cfg),
        _ground_point(path[-1], cfg),
        0.0,
    )
    fg = build_flight_graph(req, cfg, [], ColGenParams(detour_slack_hops=100))
    raw = Column(req.flight_id, fg.base_step, 0, None, None, path, 0.0)

    denied = column_to_intent(raw, req, cfg)
    assert denied.denial_reason is DenialReason.BUDGET_EXCEEDED
    with pytest.raises(ValueError, match="max_detour_factor"):
        column_claims(raw, fg, cfg)


def test_direct_translation_enforces_ground_delay_budget():
    cfg = replace(_cfg(), max_ground_delay_s=4.0)
    path = ((0, 0), (1, 0))
    req = FlightRequest(
        303,
        _ground_point(path[0], cfg),
        _ground_point(path[-1], cfg),
        0.0,
    )
    raw = Column(req.flight_id, 2, 0, None, None, path, 0.0)

    denied = column_to_intent(raw, req, cfg)
    assert denied.denial_reason is DenialReason.BUDGET_EXCEEDED
    assert denied.ground_delay_s == 0.0


def _cells_within(origin: tuple[int, int], rings: int) -> tuple[tuple[int, int], ...]:
    oq, or_ = origin
    return tuple(
        (q, r)
        for q in range(oq - rings, oq + rings + 1)
        for r in range(or_ - rings, or_ + rings + 1)
        if hg.hex_distance(origin, (q, r)) <= rings
    )


def _hop_volume(
    source: tuple[int, int],
    direction: tuple[int, int],
    step: int,
    cfg: SimConfig,
):
    target = source[0] + direction[0], source[1] + direction[1]
    return corridor_segment_volume(
        _point(source, cfg),
        step * cfg.dt_s,
        _point(target, cfg),
        (step + 1) * cfg.dt_s,
        cfg,
    )


def _hop_claim_keys(
    source: tuple[int, int],
    direction: tuple[int, int],
    step: int,
    cfg: SimConfig,
) -> frozenset[RowKey]:
    target = source[0] + direction[0], source[1] + direction[1]
    offsets = derive_cell_window(cfg)
    return frozenset(
        RowKey.cell(cell, 0, row_step)
        for cell, visit_step in ((source, step), (target, step + 1))
        for row_step in visit_rows(visit_step, offsets)
    )


def _endpoint_claim_keys(point, volume, cfg: SimConfig) -> frozenset[RowKey]:
    assert isinstance(volume.shape, CylinderSpec)
    return frozenset(
        RowKey.cell(cell, level, step)
        for cell in endpoint_claim_cells(point, volume.shape.radius, cfg)
        for level in range(cfg.n_levels)
        for step in endpoint_claim_steps(volume.t_start, volume.t_end, cfg)
    )


@pytest.mark.slow
@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0], ids=["default", "zero-buffer"])
def test_covering_theorem(time_buffer_s):
    """FCL conflict implies a shared cap-1 row for every v1 geometry pair class.

    The hop sweep is exhaustive over all directed edges rooted at one cell, all
    directed second edges with sources through three rings, and relative clock
    offsets -8..8.  Endpoint coverage then checks 200 deterministic off-centre
    customer locations against every nearby directed edge and the same clock
    offsets.  Finally, customer cylinder pairs exercise the spatial shared-cell
    argument independently of corridor boxes.
    """

    cfg = _cfg(time_buffer_s=time_buffer_s)
    dt = cfg.dt_s

    # Hop x hop: same direction, crossing, trailing, and head-on/swap instances
    # are all present in this finite lattice-template enumeration.
    hop_conflicts = 0
    relative_sources = _cells_within((0, 0), 3)
    for first_direction in hg.AXIAL_NEIGHBORS:
        first_volume = _hop_volume((0, 0), first_direction, 0, cfg)
        first_claims = _hop_claim_keys((0, 0), first_direction, 0, cfg)
        for second_source in relative_sources:
            for second_direction in hg.AXIAL_NEIGHBORS:
                for shift in range(-8, 9):
                    second_volume = _hop_volume(second_source, second_direction, shift, cfg)
                    if not volumes_conflict(first_volume, second_volume):
                        continue
                    hop_conflicts += 1
                    second_claims = _hop_claim_keys(second_source, second_direction, shift, cfg)
                    assert not first_claims.isdisjoint(second_claims), (
                        first_direction,
                        second_source,
                        second_direction,
                        shift,
                    )
    assert hop_conflicts >= (600 if time_buffer_s else 100)

    # Hop x customer endpoint: sample offsets continuously instead of placing
    # every endpoint on a convenient cell centre.  Nearby edges include spillover
    # cases in which neither hop endpoint is cell(P).
    rng = random.Random(20260801)
    endpoint_conflicts = 0
    radius = hg.circumradius(cfg)
    z = cfg.flight_levels_m[0]
    for _ in range(200):
        point = vec(
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius),
            cfg.ground_level_m,
        )
        cylinder = hover_reservation(
            point,
            0.0,
            cfg,
            climb_time_s=cfg.climb_time_to(z),
        )
        endpoint_claims = _endpoint_claim_keys(point, cylinder, cfg)
        point_cell = hg.enu_to_axial(float(point[0]), float(point[1]), radius)
        for source in _cells_within(point_cell, 2):
            for direction in hg.AXIAL_NEIGHBORS:
                for shift in range(-8, 9):
                    hop = _hop_volume(source, direction, shift, cfg)
                    if not volumes_conflict(hop, cylinder):
                        continue
                    endpoint_conflicts += 1
                    assert not endpoint_claims.isdisjoint(
                        _hop_claim_keys(source, direction, shift, cfg)
                    ), (point, source, direction, shift)
    assert endpoint_conflicts > 40_000

    # Customer cylinder x cylinder: random position and time offsets exercise
    # the proof's cell(P2) witness without assuming aligned centres or clocks.
    cylinder_conflicts = 0
    for _ in range(200):
        first_point = vec(
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius),
            cfg.ground_level_m,
        )
        second_point = vec(
            first_point[0] + rng.uniform(-2.5 * radius, 2.5 * radius),
            first_point[1] + rng.uniform(-2.5 * radius, 2.5 * radius),
            cfg.ground_level_m,
        )
        first = hover_reservation(first_point, 0.0, cfg, climb_time_s=cfg.climb_time_to(z))
        first_claims = _endpoint_claim_keys(first_point, first, cfg)
        for shift in range(-8, 9):
            second = hover_reservation(
                second_point,
                shift * dt,
                cfg,
                climb_time_s=cfg.climb_time_to(z),
            )
            if not volumes_conflict(first, second):
                continue
            cylinder_conflicts += 1
            assert not first_claims.isdisjoint(_endpoint_claim_keys(second_point, second, cfg)), (
                first_point,
                second_point,
                shift,
            )
    assert cylinder_conflicts > 500


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0], ids=["default", "zero-buffer"])
def test_endpoint_claim_steps_bruteforce(time_buffer_s):
    """Every incident hop that FCL-conflicts with an endpoint has a shared time row."""

    cfg = _cfg(time_buffer_s=time_buffer_s)
    point = _ground_point((0, 0), cfg, 21.0, -13.0)
    conflicts = 0
    intervals = [(-3.7, 0.25), (0.0, 4.0), (1.3, 34.6), (8.0, 54.2)]
    offsets = derive_cell_window(cfg)
    endpoint_cell_keys = set(endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg))

    for t0, t1 in intervals:
        cylinder = replace(
            hover_reservation(point, t0, cfg),
            t_start=t0,
            t_end=t1,
        )
        claimed_steps = set(endpoint_claim_steps(t0, t1, cfg))
        for visit_step in range(-16, 21):
            for direction in hg.AXIAL_NEIGHBORS:
                # Exercise both boxes incident to a visit: inbound ends at v,
                # outbound starts at v.  Their union defines visit_rows(v).
                neighbour = direction
                incident = (
                    corridor_segment_volume(
                        _point(neighbour, cfg),
                        (visit_step - 1) * cfg.dt_s,
                        _point((0, 0), cfg),
                        visit_step * cfg.dt_s,
                        cfg,
                    ),
                    corridor_segment_volume(
                        _point((0, 0), cfg),
                        visit_step * cfg.dt_s,
                        _point(neighbour, cfg),
                        (visit_step + 1) * cfg.dt_s,
                        cfg,
                    ),
                )
                if not any(volumes_conflict(box, cylinder) for box in incident):
                    continue
                conflicts += 1
                assert (0, 0) in endpoint_cell_keys
                assert not claimed_steps.isdisjoint(visit_rows(visit_step, offsets)), (
                    (t0, t1),
                    visit_step,
                    direction,
                )
    assert conflicts > 100


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0], ids=["default", "zero-buffer"])
@pytest.mark.parametrize("hub_endpoint", ["origin", "destination"])
def test_hub_terminal_rows_cover_dwell_overlap(time_buffer_s, hub_endpoint):
    """Production column claims mirror pad dwell rows for takeoff and landing hubs."""

    cfg = _cfg(time_buffer_s=time_buffer_s)
    terminal = Terminal(f"terminal-{hub_endpoint}", 3, radius=90.0)
    hub = _ground_point((0, 0), cfg)
    customer = _ground_point((7, 0), cfg, 9.0, -4.0)
    if hub_endpoint == "origin":
        req = FlightRequest(400, hub, customer, 0.0, origin_terminal=terminal)
        lanes = hg.terminal_lanes(hub, terminal, cfg)
        start = max(lanes, key=lambda lane: lane.cell[0]).cell
        end = hg.enu_to_axial(float(customer[0]), float(customer[1]), hg.circumradius(cfg))
    else:
        req = FlightRequest(401, customer, hub, 0.0, dest_terminal=terminal)
        lanes = hg.terminal_lanes(hub, terminal, cfg)
        start = hg.enu_to_axial(float(customer[0]), float(customer[1]), hg.circumradius(cfg))
        end = max(lanes, key=lambda lane: lane.cell[0]).cell
    path = _shortest_path(start, end)

    columns = []
    for hold_steps in range(0, 18, 2):
        col, fg = _column_for(
            req,
            path,
            cfg,
            departure_step=math.ceil(req.t_departure / cfg.dt_s) + hold_steps,
            slack=12,
        )
        intent = column_to_intent(col, req, cfg)
        cylinder = intent.volumes[0 if hub_endpoint == "origin" else -1]
        expected = frozenset(
            RowKey.term(terminal.id, step)
            for step in terminal_claim_steps(cylinder.t_start, cylinder.t_end, cfg)
        )
        actual = frozenset(key for key in col.claims if key.kind == "term")
        assert actual == expected
        row_index = RowIndex(fg.terminal_capacities)
        assert all(row_index.cap(key) == terminal.capacity for key in actual)
        columns.append((cylinder, actual))

    overlapping = separated = 0
    for index, (first_cylinder, first_claims) in enumerate(columns):
        for second_cylinder, second_claims in columns[index + 1 :]:
            if first_cylinder.time_overlaps(second_cylinder):
                overlapping += 1
                assert not first_claims.isdisjoint(second_claims)
            else:
                separated += 1
    assert overlapping > 0
    assert separated > 0


def test_terminal_claim_steps_are_exact_half_open_periods():
    cfg = _cfg()
    first_t0 = 0.0
    first_t1 = 50.66666666666667
    second_t0 = 52.0
    second_t1 = second_t0 + (first_t1 - first_t0)

    first = set(terminal_claim_steps(first_t0, first_t1, cfg))
    second = set(terminal_claim_steps(second_t0, second_t1, cfg))
    assert first == set(range(0, 13))
    assert second == set(range(13, 26))
    assert first.isdisjoint(second)

    # Customer cylinder claims intentionally retain outward floating-point cover.
    padded_first = set(endpoint_claim_steps(first_t0, first_t1, cfg))
    padded_second = set(endpoint_claim_steps(second_t0, second_t1, cfg))
    assert 12 in padded_first & padded_second

    # Non-binary step sizes must classify timestamps built as ``k * dt`` by
    # the grid boundary itself, not by a quotient that can round below/above k.
    nonbinary = replace(cfg, dt_s=0.7)
    assert list(terminal_claim_steps(3 * nonbinary.dt_s, 6 * nonbinary.dt_s, nonbinary)) == [
        3,
        4,
        5,
    ]


def test_origin_terminal_cylinder_uses_exact_departure_clock():
    dt = 7.50045227268982
    cfg = replace(_cfg(), dt_s=dt, max_ground_delay_s=6000.0)
    terminal = Terminal("H", 1, radius=90.0)
    req = FlightRequest(
        504,
        _ground_point((0, 0), cfg),
        _ground_point((8, 0), cfg),
        0.0,
        1314.5403973767247,
        origin_terminal=terminal,
    )
    fg = build_flight_graph(req, cfg, [], ColGenParams())
    path = ((1, -1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0))
    raw = Column(
        req.flight_id,
        867,
        0,
        _lane_index(fg.origin_lanes, path[0]),
        None,
        path,
        0.0,
    )

    claims = column_claims(raw, fg, cfg)
    intent = column_to_intent(raw, req, cfg)
    cylinder = intent.volumes[0]
    assert cylinder.t_start == raw.departure_step * dt
    expected = {
        RowKey.term(terminal.id, step)
        for step in terminal_claim_steps(cylinder.t_start, cylinder.t_end, cfg)
    }
    assert {key for key in claims if key.kind == "term"} == expected


def test_customer_endpoint_rows_are_owned_by_column_claims():
    cfg = _cfg()
    origin = _ground_point((0, 0), cfg, 18.0, -7.0)
    dest = _ground_point((5, 1), cfg, -13.0, 12.0)
    req = FlightRequest(500, origin, dest, 0.0)
    start = hg.enu_to_axial(float(origin[0]), float(origin[1]), hg.circumradius(cfg))
    end = hg.enu_to_axial(float(dest[0]), float(dest[1]), hg.circumradius(cfg))
    col, fg = _column_for(req, _shortest_path(start, end), cfg, slack=0)
    corridor_start = col.departure_step + fg.takeoff_steps[col.level]
    arrival_step = corridor_start + len(col.cell_path) - 1

    for point, t0 in (
        (origin, col.departure_step * cfg.dt_s),
        (dest, arrival_step * cfg.dt_s),
    ):
        t1 = (
            t0 + cfg.hover_time_s + column_dwell_s(point, None, cfg, cfg.flight_levels_m[col.level])
        )
        expected = {
            RowKey.cell(cell, level, step)
            for cell in endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
            for level in range(cfg.n_levels)
            for step in endpoint_claim_steps(t0, t1, cfg)
        }
        assert expected <= col.claims


def test_claims_deduped_constructed():
    cfg = _cfg()
    path = ((0, 0), (1, 0), (1, -1), (0, 0), (1, 0), (2, 0), (3, 0))
    req = FlightRequest(
        600,
        _ground_point(path[0], cfg),
        _ground_point(path[-1], cfg),
        0.0,
    )
    col, fg = _column_for(req, path, cfg, slack=4)
    corridor_start = col.departure_step + fg.takeoff_steps[col.level]
    offsets = derive_cell_window(cfg)
    raw_visit_rows = [
        RowKey.cell(cell, col.level, row_step)
        for visit_step, cell in enumerate(path, start=corridor_start)
        for row_step in visit_rows(visit_step, offsets)
    ]

    # The triangle returns to (0, 0) after three steps, so its two visit
    # windows share the row at first_visit + 1.  The production result must
    # retain one coefficient, never COO-style coefficient 2.
    duplicate = RowKey.cell((0, 0), col.level, corridor_start + 1)
    assert raw_visit_rows.count(duplicate) == 2
    assert len(raw_visit_rows) > len(set(raw_visit_rows))
    assert duplicate in col.claims
    assert isinstance(col.claims, frozenset)
    assert len(col.claims) == len(set(col.claims))
    assert set(raw_visit_rows) <= col.claims
