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
    _graph_max_step,
    build_flight_graph,
    column_claims,
)
from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import seed_column
from freespace_sim.planner.colgen.translate import (
    Column,
    column_to_corners,
    column_to_intent,
)
from freespace_sim.planner.colgen.zeta_geometry import (
    anyangle_to_intent,
    reservation_claims,
    volume_claims,
    volume_spatial_cells,
)
from freespace_sim.planner.colgen.zeta_timing import edge_end_times
from freespace_sim.planner.colgen.windows import (
    derive_cell_window,
    endpoint_claim_cells,
    endpoint_claim_steps,
    hop_box_stays_in_its_cells,
    terminal_claim_steps,
    visit_rows,
)
from freespace_sim.types import DenialReason, FlightRequest, Terminal, vec
from freespace_sim.volumes import (
    Volume4D,
    build_reservation_from_corners,
    column_dwell_s,
    corridor_segment_volume,
    enroute_detour_m,
    enroute_reference_m,
    exit_radius,
    hover_reservation,
    permanent_terminal_reservation,
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


def _rewalled(fg, cfg: SimConfig, walls):
    """Return ``fg`` with ``walls`` and a hop oracle built for exactly that graph.

    The graph builder refuses some wall layouts outright, so tests assemble them by hand.

    Parameters
    ------------
    - fg (FlightGraph): the graph to copy.
    - cfg (SimConfig): its config.
    - walls (tuple[Volume4D, ...]): the replacement static walls.

    Return
    --------
    - output (FlightGraph): a fresh-cache copy whose wall index and lazy oracle use ``walls``.
    """
    index = network_module._WallSpatialIndex(walls, max(256.0, 4.0 * cfg.corridor_width_m))
    walled = replace(fg, static_walls=walls, _wall_index=index)
    return replace(walled, forbidden_hops=network_module._LazyForbiddenHops(
        walled.corridor_cells, walls, walled.request, walled.origin_terminal,
        walled.dest_terminal, walled.origin_lanes, walled.dest_lanes, cfg, index))


def _precheck_canonical(fg, cfg: SimConfig, raw: Column):
    """Canonicalize ``raw`` as pricing does, with its pre-check claims and dual attached.

    Parameters
    ------------
    - fg (FlightGraph): the graph to certify on.
    - cfg (SimConfig): the graph's config.
    - raw (Column): the route; only its departure, lanes and cells are read.

    Return
    --------
    - output (tuple): ``(fast, result)`` -- the volume-free attempt (``pricing._DEFER`` when it
      cannot vouch for the route) and the full answer, which check mode (on for this module)
      compares bit for bit with the reference certification.
    """
    label = pricing._Label(0.0, raw.departure_step, raw.origin_lane_idx, raw.cell_path,
                           frozenset())
    claims = pricing._path_claims(fg, cfg, label, raw.dest_lane_idx)
    view = pricing.DualView({}, cfg)
    candidate = pricing._Candidate(0.0, 0.0, label, raw.dest_lane_idx, claims,
                                   view.claim_cost(claims))
    model = pricing.DELAY_MODEL
    fast = pricing._fast_canonical(candidate, fg, 0.0, cfg, 1000.0, frozenset(), model)
    return fast, pricing._canonical_candidate(candidate, fg, view, 0.0, cfg, 1000.0,
                                              frozenset(), model)


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
    overrun: int = 12,
) -> tuple[Column, object]:
    """Build a graph-valid column and populate claims through the production owner."""

    fg = build_flight_graph(req, cfg, [], ColGenParams(max_air_overrun_hops=overrun))
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
    """Two overlapping cylinders share a claimed cell, at any hover radius.

    What the cover needs is a shared cell, not that either endpoint claims the
    *other's* snapped cell: the two disks overlap at some point, and the hex holding
    that point is within one circumradius of it, hence inside both claim discs.  The
    witness here is a cell between the two, which neither endpoint sits in — pinning
    the stronger "claims the other's cell" property instead would demand a reach that
    over-claims.
    """

    cfg = replace(_cfg(), hover_radius_m=100.0)
    first = vec(0.0, 0.0, 0.0)
    second = vec(
        195.0, 0.0, 0.0
    )  # 100 m cylinders overlap, but their snapped centres are 240 m apart
    first_cell = hg.enu_to_axial(float(first[0]), float(first[1]), hg.circumradius(cfg))
    second_cell = hg.enu_to_axial(float(second[0]), float(second[1]), hg.circumradius(cfg))
    assert first_cell != second_cell

    first_claims = set(endpoint_claim_cells(first, cfg.effective_hover_radius_m, cfg))
    second_claims = set(endpoint_claim_cells(second, cfg.effective_hover_radius_m, cfg))
    shared = first_claims & second_claims
    assert shared, (sorted(first_claims), sorted(second_claims))
    assert shared.isdisjoint({first_cell, second_cell})


def test_endpoint_claim_ring_covers_diagonal_large_radius_overlap():
    """Finite axial enumeration cannot truncate a large endpoint disc.

    Along diagonal lattice directions, axial rings are closer together than the
    centre-to-centre pitch.  This near-tangent pair used to reach common cells just
    outside the pitch-derived ring bound, leaving two FCL-conflicting cylinders with
    disjoint master claims.
    """

    cfg = replace(_cfg(), hover_radius_m=1_600.0)
    radius = cfg.effective_hover_radius_m
    first = vec(0.0, 0.0, 0.0)
    bearing = math.radians(22.0)
    separation = 2.0 * radius - 1.0
    second = vec(
        separation * math.cos(bearing),
        separation * math.sin(bearing),
        0.0,
    )

    first_volume = hover_reservation(first, 0.0, cfg)
    second_volume = hover_reservation(second, 0.0, cfg)
    assert volumes_conflict(first_volume, second_volume)

    first_claims = set(endpoint_claim_cells(first, radius, cfg))
    second_claims = set(endpoint_claim_cells(second, radius, cfg))
    assert first_claims & second_claims


def test_a_covering_endpoint_claim_must_forbid_a_pair_the_ledger_accepts():
    """The false positive is structural, not a tuning error.

    The module contract is one-way: a real conflict must share a row, extra rows are
    allowed.  This pins what that permission costs, using the ledger as the referee.

    The witness is the extremal configuration, so the argument does not depend on *how*
    the claim is tested.  Straight up from a pointy-top hex centre is a VERTEX
    direction, so a cylinder centred one ``radius + circumradius`` away still touches
    the hex -- exactly at that vertex.  Any rule that covers cylinder/cylinder
    conflicts must therefore claim the hex for it.  Place a second cylinder the same
    distance beyond the opposite vertex and both claim the hex, while the two are
    ``2 * (radius + circumradius)`` apart and could not touch if they tried.  Capacity
    one then refuses a pair ``volumes_conflict`` passes.

    This is not hypothetical at the shipped reach, which tests centre distance and so
    fires well inside the extreme.  An exact disk/hexagon test removes those cases but
    not this structural one --
    :func:`test_endpoint_claims_are_disjoint_beyond_twice_the_reach` is what bounds it.
    """

    cfg = _cfg()
    radius = cfg.effective_hover_radius_m
    hex_radius = hg.circumradius(cfg)
    centre = np.asarray(hg.hex_center(0, 0, hex_radius), dtype=float)
    up = np.array([0.0, 1.0])
    reach = radius + hex_radius
    # A metre inside the extreme, not an ulp: the disk then reaches a whole metre PAST
    # the vertex, so the witness does not turn on how a predicate rounds its boundary.
    offset = reach - 1.0

    below = vec(*(centre - offset * up), 0.0)
    above = vec(*(centre + offset * up), 0.0)

    # Each disk genuinely reaches into hex (0, 0), touching it at the near vertex --
    # so claiming that hex is forced by the cover, not by the choice of predicate.
    for point, vertex in ((below, centre - hex_radius * up), (above, centre + hex_radius * up)):
        assert math.dist((float(point[0]), float(point[1])), tuple(vertex)) <= radius

    below_claims = set(endpoint_claim_cells(below, radius, cfg))
    above_claims = set(endpoint_claim_cells(above, radius, cfg))
    assert (0, 0) in below_claims & above_claims

    # ... and yet the cylinders provably cannot meet: two radii is the whole reach.
    separation = 2.0 * offset
    assert separation > 2.0 * radius
    first = hover_reservation(below, 0.0, cfg)
    second = hover_reservation(above, 0.0, cfg)
    assert first.t_start < second.t_end and second.t_start < first.t_end
    assert not volumes_conflict(first, second)


@pytest.mark.parametrize("hover_radius_m", [None, 100.0])
def test_endpoint_claims_are_disjoint_beyond_twice_the_reach(hover_radius_m):
    """How wrong the endpoint rows can be is bounded by twice their reach.

    Two endpoints share a claimed cell only if some hex centre is inside both claim
    discs, so the coupling can never exceed ``2 * (radius + circumradius)``.  That bound
    is what makes the false positive in
    :func:`test_a_covering_endpoint_claim_must_forbid_a_pair_the_ledger_accepts` a
    known, bounded modelling cost rather than an open-ended one, and it is the first
    thing a re-widened endpoint reach breaks.

    The bound is also checked tight, so it cannot pass by being vacuously generous.
    """

    cfg = replace(_cfg(), hover_radius_m=hover_radius_m)
    assert hop_box_stays_in_its_cells(cfg)  # the tightened-reach branch, not the fallback
    radius = cfg.effective_hover_radius_m
    hex_radius = hg.circumradius(cfg)
    reach = radius + hex_radius

    rng = random.Random(20260819)
    widest_coupled = 0.0
    for _ in range(3000):
        first = vec(rng.uniform(-500.0, 500.0), rng.uniform(-500.0, 500.0), 0.0)
        second = vec(rng.uniform(-500.0, 500.0), rng.uniform(-500.0, 500.0), 0.0)
        shared = set(endpoint_claim_cells(first, radius, cfg)) & set(
            endpoint_claim_cells(second, radius, cfg)
        )
        if not shared:
            continue
        separation = math.hypot(
            float(first[0]) - float(second[0]), float(first[1]) - float(second[1])
        )
        assert separation < 2.0 * reach, (separation, sorted(shared))
        widest_coupled = max(widest_coupled, separation)

    # Tight: a pair just inside the bound really is coupled, on the vertex axis where
    # the reach is attained.  Without this the assertion above could hold trivially.
    centre = np.asarray(hg.hex_center(0, 0, hex_radius), dtype=float)
    up = np.array([0.0, 1.0])
    edge = reach - 1e-6
    assert (0, 0) in set(endpoint_claim_cells(vec(*(centre - edge * up), 0.0), radius, cfg))
    assert (0, 0) in set(endpoint_claim_cells(vec(*(centre + edge * up), 0.0), radius, cfg))
    # ... and the random sample actually reached into the false-positive band.
    assert widest_coupled > 2.0 * radius


def test_hop_box_stays_inside_the_two_cells_it_runs_between():
    """The containment lemma ``endpoint_claim_cells`` drops its corridor term for.

    Checked against the built geometry rather than the predicate's arithmetic: every
    corner of every directed hop box must lie inside the union of the two hexes it
    runs between.  If this ever stops holding at defaults, the endpoint claim reach is
    no longer a cover and :func:`hop_box_stays_in_its_cells` must say so.
    """

    cfg = _cfg()
    assert hop_box_stays_in_its_cells(cfg)
    radius = hg.circumradius(cfg)

    def hex_contains(cell, x, y):
        cx, cy = hg.hex_center(*cell, radius)
        # A hexagon is the intersection of three slabs, each bounded by the inradius.
        px, py = x - float(cx), y - float(cy)
        inradius = float(cfg.corridor_segment_len_m) / 2.0
        for angle in (0.0, 60.0, 120.0):
            axis = math.radians(angle)
            if abs(px * math.cos(axis) + py * math.sin(axis)) > inradius + 1e-9:
                return False
        return True

    for direction in hg.AXIAL_NEIGHBORS:
        source = (0, 0)
        target = (direction[0], direction[1])
        volume = _hop_volume(source, direction, 0, cfg)
        lo, hi = volume.shape.aabb()
        rot = np.asarray(volume.shape.rot, dtype=float).reshape(3, 3)
        centre = np.asarray(volume.shape.center, dtype=float)
        half = np.asarray(volume.shape.extents, dtype=float) / 2.0
        del lo, hi
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = centre + rot @ (half * np.array([sx, sy, sz], dtype=float))
                    assert hex_contains(source, corner[0], corner[1]) or hex_contains(
                        target, corner[0], corner[1]
                    ), (direction, corner)


def test_wide_corridor_keeps_the_conservative_endpoint_reach():
    """A corridor too wide to stay in its cells falls back to the old wider disc.

    The tightened reach is only sound while the containment lemma holds, so the
    predicate has to be load-bearing rather than decorative.
    """

    narrow = _cfg()
    wide = replace(narrow, corridor_width_m=100.0)
    assert hop_box_stays_in_its_cells(narrow)
    assert not hop_box_stays_in_its_cells(wide)

    point = vec(0.0, 0.0, 0.0)
    narrow_cells = endpoint_claim_cells(point, narrow.effective_hover_radius_m, narrow)
    wide_cells = endpoint_claim_cells(point, wide.effective_hover_radius_m, wide)
    assert len(wide_cells) > len(narrow_cells)


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


def test_params_validate_integral_nonnegative_overrun():
    assert ColGenParams().max_air_overrun_hops == 6
    # `max_air_overrun_hops` is the first field, so these positional forms reach it.
    assert ColGenParams(np.int64(3)).max_air_overrun_hops == 3
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
        turnaround_s=19.75,
    )
    overrun = 4
    fg = build_flight_graph(req, cfg, [], ColGenParams(max_air_overrun_hops=overrun))

    assert fg.origin_cell == origin_cell and fg.dest_cell == dest_cell
    assert set(_shortest_path(origin_cell, dest_cell)) <= fg.corridor_cells
    assert all(
        hg.hex_distance(origin_cell, cell) + hg.hex_distance(cell, dest_cell)
        <= fg.shortest_hops + overrun
        for cell in fg.corridor_cells
    )
    # One knob sizes both, so the corridor radius is recoverable from the graph: there is no
    # separate field that could disagree with it.
    assert fg.max_air_hops - fg.shortest_hops == overrun
    assert fg.base_step == math.ceil(req.t_departure / cfg.dt_s)
    assert fg.latest_departure_step == fg.base_step + math.floor(cfg.max_ground_delay_s / cfg.dt_s)
    rebuilt = pickle.loads(pickle.dumps(fg))
    assert rebuilt == fg
    assert rebuilt.corridor_cells == fg.corridor_cells
    assert rebuilt.request.return_to_origin == fg.request.return_to_origin == req.return_to_origin
    assert rebuilt.request.turnaround_s == fg.request.turnaround_s == req.turnaround_s

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


@pytest.mark.parametrize("origin_is_terminal", [False, True])
@pytest.mark.parametrize("dest_is_terminal", [False, True])
def test_corridor_cruise_endpoint_bound_and_materialized_parity(
    origin_is_terminal, dest_is_terminal
):
    """Every endpoint combination uses the cruise-hop bound and preserves static exclusions."""
    cfg = _cfg()
    origin, dest = (-3, 0), (5, 0)
    req = FlightRequest(
        108,
        _ground_point(origin, cfg),
        _ground_point(dest, cfg),
        0.0,
        origin_terminal=Terminal("origin", 1, radius=90.0) if origin_is_terminal else None,
        dest_terminal=Terminal("dest", 1, radius=90.0) if dest_is_terminal else None,
    )
    wall = Terminal("foreign", 1, radius=90.0)
    graph = build_flight_graph(
        req, cfg, [(_ground_point((1, 1), cfg), wall)], ColGenParams(max_air_overrun_hops=2)
    )
    starts = tuple(lane.cell for lane in graph.origin_lanes) or (origin,)
    ends = tuple(lane.cell for lane in graph.dest_lanes) or (dest,)
    explicit = {lane.cell for lane in (*graph.origin_lanes, *graph.dest_lanes)}
    probes = {(q, r) for q in range(-16, 18) for r in range(-14, 15)}
    expected = {
        cell
        for cell in probes
        if cell not in graph.foreign_exclusions
        and (
            cell in explicit
            or (
                cell not in graph.own_terminal_interiors
                and any(
                    hg.hex_distance(start, cell) + hg.hex_distance(cell, end) <= graph.max_air_hops
                    for start in starts
                    for end in ends
                )
            )
        )
    }
    assert not graph.corridor_cells.is_materialized
    assert {cell for cell in probes if cell in graph.corridor_cells} == expected
    assert not graph.corridor_cells.is_materialized
    assert frozenset(graph.corridor_cells) == expected
    assert graph.corridor_cells.is_materialized
    assert {cell for cell in probes if cell in graph.corridor_cells} == expected
    restored = pickle.loads(pickle.dumps(graph.corridor_cells))
    assert not restored.is_materialized
    assert {cell for cell in probes if cell in restored} == expected


def test_terminal_exit_corridor_keeps_real_127_hop_route_below_128_cap():
    """Flight 1600's valid detour must survive pruning with its original hop/time caps."""
    cfg = replace(
        _cfg(),
        nominal_speed_mps=30.0,
        max_ground_delay_s=3600.0,
        max_detour_factor=100.0,
    )
    req = FlightRequest(
        1600,
        vec(36835.9224761, 27246.97435197, 0.0),
        vec(29286.78386012, 14239.54583087, 0.0),
        319.7304140898543,
        832.1099729256033,
        origin_terminal=Terminal("wing_zipline_uss#124", 40, radius=180.0),
    )
    # Exact path from analysis/colgen_pricing_profile_20260909/flight_1600_wider_column.json.
    path = _walk(
        (176, 260),
        [(0, -1)] * 22
        + [(1, -1)] * 3
        + [(0, -1)] * 11
        + [(1, -1)]
        + [(0, -1)] * 2
        + [(-1, 0)]
        + [(0, -1)] * 51
        + [(-1, 0)] * 3
        + [(0, -1)] * 33,
    )
    col, graph = _column_for(req, path, cfg, departure_step=209, overrun=3)
    assert (graph.origin_cell, graph.dest_cell) == ((176, 262), (176, 137))
    assert (graph.shortest_hops, graph.max_air_hops, graph.max_step) == (125, 128, 1245)
    assert len(path) - 1 == 127
    excluded_before = {
        cell
        for cell in path
        if hg.hex_distance(graph.origin_cell, cell) + hg.hex_distance(cell, graph.dest_cell)
        > graph.max_air_hops
    }
    assert excluded_before == {(180, 223), (180, 222), (180, 221)}
    assert set(path) <= graph.corridor_cells
    assert col.claims
    assert column_to_intent(col, req, cfg) is not None


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
    from freespace_sim.planner import route_domain
    monkeypatch.setattr(route_domain, "_static_hop_allowed_roles", fail_if_checked)
    graph = build_flight_graph(
        req,
        cfg,
        [(_ground_point((0, 20), cfg), wall)],
        ColGenParams(max_air_overrun_hops=12),
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
        build_flight_graph(request, cfg, catalog, ColGenParams(max_air_overrun_hops=2))
        for request in requests
    )

    assert catalog_calls == len(terminals)
    assert calls == catalog_calls
    assert all(graph.static_walls is catalog.walls for graph in graphs)
    assert all(graph._wall_index is catalog.wall_index for graph in graphs)
    assert all(not graph.corridor_cells.is_materialized for graph in graphs)


@pytest.mark.parametrize("planner", ["dp", "zeta"])
def test_many_far_walls_never_reach_narrow_phase(monkeypatch, planner):
    """Initial routes must avoid scanning physical conflicts against irrelevant walls."""
    from freespace_sim.planner.colgen import zeta

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
        ColGenParams(max_air_overrun_hops=2),
    )
    calls = 0
    import importlib
    module = importlib.import_module(
        "freespace_sim.planner.route_domain" if planner == "dp"
        else "freespace_sim.planner.zeta.search")
    original_conflict = module.volumes_conflict

    def count_conflicts(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_conflict(*args, **kwargs)

    monkeypatch.setattr(module, "volumes_conflict", count_conflicts)

    if planner == "dp":
        seeds = (seed_column(graph, cfg),)
    else:
        seeds = zeta.ZetaGraph(graph, cfg, RowIndex()).seed_columns(ColGenParams())

    assert seeds and all(seed.claims for seed in seeds)
    assert calls == 0
    assert catalog.wall_index.stats["queries"] > 0
    assert catalog.wall_index.stats["candidates"] == 0


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


def test_lazy_outgoing_arcs_expand_each_source_once():
    cfg = _cfg()
    req = FlightRequest(
        110,
        _ground_point((-4, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
    )
    graph = build_flight_graph(req, cfg, (), ColGenParams(max_air_overrun_hops=2))

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
        ColGenParams(max_air_overrun_hops=2),
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
    graph = build_flight_graph(req, cfg, (), ColGenParams(max_air_overrun_hops=2))
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
        ColGenParams(max_air_overrun_hops=6),
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

    # Pricing's volume-free certification reads the per-role hop verdict and defers; without
    # the lazy oracle (union over roles only) it cannot vouch for any walled route.  Either
    # way the reference rejects.
    for graph in (fg, exact_only_fg):
        fast, result = _precheck_canonical(graph, cfg, raw)
        assert fast is pricing._DEFER and result is None


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
    # The per-role hop verdict sees the same two differently tagged boxes, so pricing certifies
    # this route without building its volumes, and identically to the reference.
    fast, result = _precheck_canonical(fg, cfg, raw)
    assert fast is not pricing._DEFER and result is not None

    # A foreign wall over the origin pad touches no corridor box, only the origin cylinder:
    # the per-graph endpoint verdict must catch it.
    pad_wall = permanent_terminal_reservation(origin, Terminal("C", 1, radius=30.0), cfg)
    walled = _rewalled(fg, cfg, fg.static_walls + (pad_wall,))
    assert walled.hop_allowed_for_role(*path, first=True, last=True)
    assert not pricing._route_geometry(walled, cfg).endpoints_clear
    fast, result = _precheck_canonical(walled, cfg, raw)
    assert fast is pricing._DEFER and result is None

    # Hop verdicts that cannot vouch for this route make pricing defer; the reference still
    # accepts it (check mode compares the two).
    late = _rewalled(fg, cfg, tuple(replace(wall, t_start=1.0) for wall in fg.static_walls))
    late.outgoing_neighbors(path[0])  # expanded, so only the time guard can defer
    assert late.forbidden_hops.allows_known(*path, first=True, last=True)
    assert not pricing._route_geometry(late, cfg).hop_verdicts_exact
    foreign = replace(fg, corridor_cells=frozenset(fg.corridor_cells))  # not the oracle's own
    excluded = _rewalled(fg, cfg, fg.static_walls)
    excluded.outgoing_neighbors(path[0])
    # This one-hop route uses its arc as first AND last; keep every other role allowed.
    excluded.forbidden_hops._roles[path] = (
        network_module._ALL_ARC_ROLES & ~network_module._ARC_FIRST_LAST)
    for graph in (late, foreign, excluded):
        fast, result = _precheck_canonical(graph, cfg, raw)
        assert fast is pricing._DEFER and result is not None


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
        ColGenParams(max_air_overrun_hops=2),
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


@pytest.mark.parametrize("overrun", [1, 4, 9])
def test_graph_max_step_preserves_takeoff_and_origin_lane_time_budget(overrun):
    """The clock must reach the ceiling from the LAST legal departure.

    `max_step` is a single scalar serving every departure, so it is sized for the latest one
    and an earlier departure inherits the surplus. That makes the term it carries for route
    length load-bearing: anything smaller than `max_air_hops` in that slot caps the last
    departures by the clock and earlier ones by the ceiling -- the departure-dependent cap the
    ceiling exists to remove.
    """

    cfg = replace(_cfg(), max_ground_delay_s=20.0)
    terminal = Terminal("origin", 2, radius=180.0)
    req = FlightRequest(
        112,
        _ground_point((0, 0), cfg),
        _ground_point((10, 0), cfg),
        0.0,
        origin_terminal=terminal,
    )
    params = ColGenParams(max_air_overrun_hops=overrun)
    fg = build_flight_graph(req, cfg, [], params)
    preamble = max(fg.takeoff_steps) + max(lane.steps for lane in fg.origin_lanes)
    assert fg.max_step == fg.latest_departure_step + preamble + fg.max_air_hops
    # The invariant the formula exists to serve, stated independently of it.
    assert fg.max_step - (fg.latest_departure_step + preamble) >= fg.max_air_hops, (
        "the clock binds before the ceiling at the last departure"
    )


def test_graph_max_step_route_term_is_the_ceiling_not_a_re_derivation():
    """The route term must be `max_air_hops`, not a re-derivation from `shortest_hops`.

    The bug was `_graph_max_step` carrying `shortest_hops + detour_slack_hops` where it needed
    `max_air_hops`, so the advertised horizon exceeded what was reachable at the last
    departure. One knob can no longer express that disagreement through `ColGenParams`, so the
    regression is pinned on the function itself -- with a budget deliberately unrelated to any
    ellipse radius, so a future re-derivation from `shortest_hops` plus anything cannot pass.
    """

    lanes = (hg.Lane(cell=(1, 0), bearing=0.0, dist=0.0, steps=5),)
    assert _graph_max_step(100, (7,), lanes, 26) == 100 + 7 + 5 + 26
    # The whole route budget survives the preamble, whatever it is.
    for max_air_hops in (0, 1, 26, 1000):
        horizon = _graph_max_step(100, (7,), lanes, max_air_hops)
        assert horizon - (100 + 7 + 5) == max_air_hops


def test_flight_graph_one_hop_and_same_cell_degenerate_guard():
    cfg = _cfg()
    one_hop = FlightRequest(
        12,
        _ground_point((0, 0), cfg),
        _ground_point((1, 0), cfg),
        0.0,
    )
    fg = build_flight_graph(one_hop, cfg, [], ColGenParams(max_air_overrun_hops=0))
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


@pytest.mark.parametrize(
    ("make_cfg", "raises_match"),
    [
        (
            lambda: replace(_cfg(), fixed_exit_lanes=False),
            "requires fixed_exit_lanes=True",
        ),
        (
            lambda: replace(_cfg(), terminal_airspace_always_active=False),
            "terminal_airspace_always_active=True",
        ),
    ],
    ids=["fixed_exit_lanes", "terminal_airspace"],
)
def test_hub_geometry_requires_its_enabling_config(make_cfg, raises_match):
    """A hub request needs the config that makes its fixed geometry expressible: with the knob
    off, both the graph build and the column->intent translation must refuse, while an ordinary
    customer request on the same config still builds a corridor and files cleanly.
    """

    cfg = make_cfg()
    hub = Terminal("hub", 1, radius=90.0)
    hub_req = FlightRequest(
        117,
        _ground_point((0, 0), cfg),
        _ground_point((4, 0), cfg),
        0.0,
        origin_terminal=hub,
    )
    with pytest.raises(NotImplementedError, match=raises_match):
        build_flight_graph(hub_req, cfg, [], ColGenParams())

    lanes = hg.terminal_lanes(hub_req.origin, hub, cfg)
    lane_idx = min(
        range(len(lanes)),
        key=lambda index: hg.hex_distance(lanes[index].cell, (4, 0)),
    )
    path = _shortest_path(lanes[lane_idx].cell, (4, 0))
    raw = Column(hub_req.flight_id, 0, 0, lane_idx, None, path, 0.0)
    with pytest.raises(NotImplementedError, match=raises_match):
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

    fg = build_flight_graph(req, cfg, [], ColGenParams(max_air_overrun_hops=0))
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
        overrun=0,
    )
    second, _second_fg = _column_for(
        second_req,
        second_path,
        cfg,
        departure_step=second_departure_step,
        overrun=0,
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
        overrun=12,
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
    col, _fg = _column_for(req, _shortest_path(start, end), cfg, overrun=0)
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
    fg = build_flight_graph(req, cfg, [], ColGenParams(max_air_overrun_hops=2))
    raw = Column(req.flight_id, fg.base_step, 0, None, None, path, 0.0)

    with pytest.raises(ValueError, match="max_detour_factor"):
        column_claims(raw, fg, cfg)
    unchecked = column_to_intent(raw, req, cfg, _validate=False)
    assert unchecked.accepted and unchecked.volumes and unchecked.centerline
    assert column_claims(raw, fg, cfg, _intent=unchecked, _validate=False)
    assert not fg._search_cache.certified_claims
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
    fg = build_flight_graph(req, cfg, [], ColGenParams(max_air_overrun_hops=100))
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
@pytest.mark.parametrize("hover_radius_m", [60.0, 10.0], ids=["60m-pad", "10m-pad"])
@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0], ids=["default", "zero-buffer"])
def test_covering_theorem(time_buffer_s, hover_radius_m):
    """FCL conflict implies a shared cap-1 row for every v1 geometry pair class.

    The hop sweep is exhaustive over all directed edges rooted at one cell, all
    directed second edges with sources through three rings, and relative clock
    offsets -8..8.  Endpoint coverage then checks 200 deterministic off-centre
    customer locations against every nearby directed edge and the same clock
    offsets.  Finally, customer cylinder pairs exercise the spatial shared-cell
    argument independently of corridor boxes.
    """

    cfg = _cfg(time_buffer_s=time_buffer_s, hover_radius_m=hover_radius_m)
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
    # Floors against a vacuous sweep. A 10 m pad disc reaches a third as far as a 60 m one, so it
    # meets far fewer hops; what must not change is that every conflict it DOES have is covered.
    assert endpoint_conflicts > (40_000 if hover_radius_m >= 60.0 else 10_000)

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
    # Two 10 m discs sampled within one hex rarely overlap at all (17 pairs here, vs >500 at 60 m),
    # so this floor only guards against a sweep that finds nothing to check.
    assert cylinder_conflicts > (500 if hover_radius_m >= 60.0 else 10)


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
            overrun=12,
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
    col, fg = _column_for(req, _shortest_path(start, end), cfg, overrun=0)
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
    col, fg = _column_for(req, path, cfg, overrun=4)
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


@pytest.mark.parametrize("endpoints", ["cc", "tc", "ct", "tt"])
@pytest.mark.parametrize(
    "dt, clock", [(4.0, 0.0), (0.7, 0.0), (7.50045227268982, 1314.5403973767247), (0.7, 1e12)]
)
@pytest.mark.parametrize("time_buffer_s", [0.0, 4.0])
def test_shifted_spatial_certificate_matches_cold_claims(
    monkeypatch, endpoints, dt, clock, time_buffer_s
):
    """Clock shifts preserve cold-graph claims, including rounding at dwell boundaries."""
    import freespace_sim.planner.colgen.translate as translate_module

    cfg = replace(
        _cfg(time_buffer_s=time_buffer_s),
        dt_s=dt,
        nominal_speed_mps=120.0 / dt,
        max_ground_delay_s=64 * dt,
        max_detour_factor=100.0,
    )
    req = FlightRequest(
        601,
        _ground_point((0, 0), cfg),
        _ground_point((8, 0), cfg),
        clock,
        clock,
        origin_terminal=Terminal("origin", 1, radius=90.0) if endpoints[0] == "t" else None,
        dest_terminal=Terminal("dest", 1, radius=90.0) if endpoints[1] == "t" else None,
    )
    dwell = column_dwell_s(req.origin, req.origin_terminal, cfg, cfg.flight_levels_m[0])
    cfg = replace(cfg, hover_time_s=(math.ceil(dwell / dt) + 2) * dt - dwell)
    params = ColGenParams(max_air_overrun_hops=3)
    terms = [(_ground_point((4, 5), cfg), Terminal("foreign", 1, radius=90.0))]
    graph = build_flight_graph(req, cfg, terms, params)
    starts = tuple(lane.cell for lane in graph.origin_lanes) or (graph.origin_cell,)
    ends = tuple(lane.cell for lane in graph.dest_lanes) or (graph.dest_cell,)
    start, end = min(
        ((s, e) for s in starts for e in ends), key=lambda pair: hg.hex_distance(*pair)
    )
    path = _shortest_path(start, end)
    raw = Column(
        req.flight_id,
        graph.base_step + 16,
        0,
        _lane_index(graph.origin_lanes, start) if graph.origin_lanes else None,
        _lane_index(graph.dest_lanes, end) if graph.dest_lanes else None,
        path,
        0.0,
    )
    shifts = (-16, -5, -1, 0, 1, 5, 16)
    columns = [replace(raw, departure_step=raw.departure_step + delta) for delta in shifts]
    expected = [
        column_claims(column, build_flight_graph(req, cfg, terms, params), cfg)
        for column in columns
    ]
    before = dict(graph._search_cache.certified_claims)
    assert [column_claims(column, graph, cfg, _validate=False) for column in columns] == expected
    assert dict(graph._search_cache.certified_claims) == before
    if endpoints == "tt" and clock == 1e12:
        from freespace_sim.planner.colgen.solver import _shift_claims

        baseline = expected[shifts.index(0)]
        assert any(
            claims != _shift_claims(baseline, delta)
            for claims, delta in zip(expected, shifts, strict=True)
        )
    calls = []
    original = translate_module.column_to_intent

    def observed(*args, **kwargs):
        calls.append(args[0].departure_step)
        return original(*args, **kwargs)

    monkeypatch.setattr(translate_module, "column_to_intent", observed)
    column_claims(raw, graph, cfg)
    for column, cold_claims in zip(columns, expected, strict=True):
        assert column_claims(column, graph, cfg) == cold_claims
    assert calls == [raw.departure_step]
    for field, lanes in (
        ("origin_lane_idx", graph.origin_lanes),
        ("dest_lane_idx", graph.dest_lanes),
    ):
        if lanes:
            with pytest.raises(ValueError, match="invalid .* terminal lane index"):
                column_claims(replace(raw, **{field: len(lanes)}), graph, cfg)
            alternate = (getattr(raw, field) + 1) % len(lanes)
            with pytest.raises(ValueError, match="does not match selected lane"):
                column_claims(replace(raw, **{field: alternate}), graph, cfg)


def test_shifted_spatial_certificate_does_not_trust_changed_columns_or_claims():
    """Cached validation applies only to its route/configuration and legal departure bounds."""
    cfg = _cfg()
    req = FlightRequest(602, _ground_point((0, 0), cfg), _ground_point((5, 0), cfg), 0.0)
    params = ColGenParams(max_air_overrun_hops=3)
    graph = build_flight_graph(req, cfg, [], params)
    raw = Column(req.flight_id, 4, 0, None, None, _shortest_path((0, 0), (5, 0)), 0.0)
    claims = column_claims(raw, graph, cfg)
    forged = frozenset({RowKey.cell((1000, 1000), 0, 0)})
    assert column_claims(replace(raw, claims=forged), graph, cfg) == claims
    assert (
        column_claims(replace(raw, claims=forged), build_flight_graph(req, cfg, [], params), cfg)
        == claims
    )
    shifted = replace(raw, departure_step=5, claims=forged)
    assert column_claims(shifted, graph, cfg) == column_claims(
        shifted, build_flight_graph(req, cfg, [], params), cfg
    )
    invalid = (
        (replace(raw, flight_id=999), "flight_id"),
        (replace(raw, level=1), "outside graph levels"),
        (replace(raw, origin_lane_idx=0), "origin_lane_idx"),
        (replace(raw, dest_lane_idx=0), "dest_lane_idx"),
        (replace(raw, cell_path=((0, 0), (2, 0), (3, 0), (4, 0), (5, 0))), "non-neighbour"),
        (replace(raw, departure_step=-1), "outside legal range"),
        (replace(raw, departure_step=graph.latest_departure_step + 1), "outside legal range"),
    )
    for column, message in invalid:
        with pytest.raises(ValueError, match=message):
            column_claims(column, graph, cfg)
    with pytest.raises(ValueError, match="SimConfig"):
        column_claims(raw, graph, replace(cfg, max_detour_factor=cfg.max_detour_factor + 1.0))
    with pytest.raises(ValueError, match="cell-window"):
        column_claims(raw, graph, cfg, W=999)
    tight = replace(
        graph, max_step=raw.departure_step + graph.takeoff_steps[0] + len(raw.cell_path) - 1
    )
    column_claims(raw, tight, cfg)
    with pytest.raises(ValueError, match="beyond graph maximum"):
        column_claims(shifted, tight, cfg)


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
            from freespace_sim.planner.colgen.zeta_forward import ForwardTopology

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

    from freespace_sim.planner.colgen.zeta_lattice_bound import (
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
