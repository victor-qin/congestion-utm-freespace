"""Flight-specific space-time networks and their capacity-row membership.

The master problem never reasons about reservation geometry directly.  A column instead
claims immutable row keys produced here: one family for buffered cell visits and one for
terminal-pad dwell.  Keeping row construction in this module is important -- every master,
rounding, repair, and filing check must see exactly the same coefficients.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from ...config import SimConfig
from ...types import FlightRequest, IntentStatus, Terminal, as_terminal
from ...conflict import volumes_conflict
from ...volumes import (
    Volume4D,
    column_dwell_s,
    corridor_segment_volume,
    permanent_terminal_reservation,
    segment_overlaps_column,
    terminal_radius,
)
from .. import hexgrid as hg
from .windows import (
    derive_cell_window,
    endpoint_claim_cells,
    endpoint_claim_steps,
    terminal_claim_steps,
    validate_edge_locality,
    visit_rows,
)

if TYPE_CHECKING:
    from .params import ColGenParams
    from .translate import Column

Cell = tuple[int, int]


class _ImmutableCellIndex(Mapping[Cell, int]):
    """Read-only, pickle-safe mapping from axial cells to dense node indices.

    ``MappingProxyType`` provides the desired mutation guard but is not itself
    picklable.  This wrapper keeps the live storage behind a proxy and teaches
    pickle to reconstruct it from immutable item pairs.  Consequently both the
    public mapping and its exposed backing view reject mutation while process-
    pool graph serialization remains supported.
    """

    __slots__ = ("_data",)

    def __init__(self, items: Mapping[Cell, int] | tuple[tuple[Cell, int], ...]) -> None:
        if hasattr(self, "_data"):
            raise AttributeError("cell index is immutable and cannot be reinitialized")
        data = dict(items.items()) if isinstance(items, Mapping) else dict(items)
        object.__setattr__(self, "_data", MappingProxyType(data))

    def __setattr__(self, _name, _value) -> None:
        raise AttributeError("cell index is immutable")

    def __getitem__(self, cell: Cell) -> int:
        return self._data[cell]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self):
        return type(self), (tuple(self._data.items()),)


class _ImmutableFlightRequest(FlightRequest):
    """A detached request whose geometry/terminal signature cannot drift."""

    def __post_init__(self) -> None:
        super().__post_init__()
        origin_source = np.asarray(self.origin, dtype=float)
        dest_source = np.asarray(self.dest, dtype=float)
        # Arrays that own their allocation can re-enable WRITEABLE.  Back these
        # views with immutable bytes so even ``setflags(write=True)`` fails.
        origin = np.frombuffer(origin_source.tobytes(), dtype=float).reshape(origin_source.shape)
        dest = np.frombuffer(dest_source.tobytes(), dtype=float).reshape(dest_source.shape)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "_colgen_frozen", True)

    def __setattr__(self, name, value) -> None:
        if getattr(self, "_colgen_frozen", False):
            raise AttributeError("flight-graph request snapshot is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
        if getattr(self, "_colgen_frozen", False):
            raise AttributeError("flight-graph request snapshot is immutable")
        super().__delattr__(name)

    def __reduce__(self):
        return type(self), (
            self.flight_id,
            np.array(self.origin, copy=True),
            np.array(self.dest, copy=True),
            self.t_request,
            self.t_departure,
            self.uss_id,
            self.origin_terminal,
            self.dest_terminal,
        )


def _snapshot_request(
    req: FlightRequest,
    origin_terminal: Terminal | None,
    dest_terminal: Terminal | None,
) -> FlightRequest:
    """Detach graph geometry from caller-owned mutable endpoint arrays."""

    return _ImmutableFlightRequest(
        flight_id=req.flight_id,
        origin=req.origin,
        dest=req.dest,
        t_request=req.t_request,
        t_departure=req.t_departure,
        uss_id=req.uss_id,
        origin_terminal=origin_terminal,
        dest_terminal=dest_terminal,
    )


class RowKey(tuple):
    """Tuple-compatible key for one capacity row.

    Cell rows are ``("cell", q, r, level, step)`` and have capacity one.  Terminal
    rows are ``("term", terminal_id, step)`` and take their capacity from a
    :class:`RowIndex` terminal registration.  Subclassing :class:`tuple` deliberately
    preserves equality, hashing, unpacking, and serialization compatibility with the
    literal representation used in the formulation and tests.
    """

    __slots__ = ()

    def __new__(cls, *parts: Any) -> "RowKey":
        if len(parts) == 1 and isinstance(parts[0], (tuple, list, RowKey)):
            parts = tuple(parts[0])
        if not parts:
            raise ValueError("a row key cannot be empty")

        kind = parts[0]
        if kind == "cell":
            if len(parts) != 5:
                raise ValueError("cell row keys must be ('cell', q, r, level, step)")
            try:
                q, r, level, step = (operator.index(value) for value in parts[1:])
            except TypeError as exc:
                raise TypeError("cell row coordinates, level, and step must be integers") from exc
            normalized = ("cell", q, r, level, step)
        elif kind == "term":
            if len(parts) != 3:
                raise ValueError("terminal row keys must be ('term', terminal_id, step)")
            terminal_id = parts[1]
            try:
                hash(terminal_id)
            except TypeError as exc:
                raise TypeError("terminal row ids must be hashable") from exc
            try:
                step = operator.index(parts[2])
            except TypeError as exc:
                raise TypeError("terminal row steps must be integers") from exc
            normalized = ("term", terminal_id, step)
        else:
            raise ValueError(f"unknown capacity-row kind {kind!r}")
        return tuple.__new__(cls, normalized)

    @classmethod
    def cell(cls, *args: Any) -> "RowKey":
        """Build a cell key from ``(q, r, level, step)`` or ``((q, r), level, step)``."""
        if len(args) == 3:
            cell, level, step = args
            try:
                q, r = cell
            except (TypeError, ValueError) as exc:
                raise TypeError("cell must be a (q, r) pair") from exc
            return cls("cell", q, r, level, step)
        if len(args) == 4:
            return cls("cell", *args)
        raise TypeError("RowKey.cell expects (q, r, level, step) or ((q, r), level, step)")

    @classmethod
    def term(cls, terminal_id: Hashable, step: int) -> "RowKey":
        """Build a terminal-capacity key."""
        return cls("term", terminal_id, step)

    terminal = term

    @property
    def kind(self) -> str:
        return tuple.__getitem__(self, 0)

    @property
    def step(self) -> int:
        return tuple.__getitem__(self, 4 if self.kind == "cell" else 2)

    @property
    def cell_coord(self) -> Cell:
        if self.kind != "cell":
            raise AttributeError("terminal rows do not have a cell")
        return tuple.__getitem__(self, 1), tuple.__getitem__(self, 2)

    @property
    def level(self) -> int:
        if self.kind != "cell":
            raise AttributeError("terminal rows do not have a flight level")
        return tuple.__getitem__(self, 3)

    @property
    def terminal_id(self) -> Hashable:
        if self.kind != "term":
            raise AttributeError("cell rows do not have a terminal id")
        return tuple.__getitem__(self, 1)

    def __getnewargs__(self) -> tuple[tuple[Any, ...]]:
        return (tuple(self),)


class RowIndex:
    """Intern row keys to dense integer ids and own terminal row capacities.

    Unknown terminal capacity is an error rather than an implicit capacity-one fallback: the
    ledger's pad authority uses ``Terminal.capacity``, and silently substituting one would change
    the feasible set.  Call :meth:`register_terminal` (or pass a mapping to the constructor) before
    asking for such a row's capacity.
    """

    def __init__(
        self,
        terminal_capacities: Mapping[Hashable, int] | None = None,
    ) -> None:
        self._key_to_index: dict[RowKey, int] = {}
        self._index_to_key: list[RowKey] = []
        self._terminal_capacities: dict[Hashable, int] = {}
        for terminal_id, capacity in (terminal_capacities or {}).items():
            self.register_terminal(terminal_id, capacity)

    def register_terminal(
        self,
        terminal_or_id: Terminal | Hashable,
        capacity: int | None = None,
    ) -> None:
        """Register a terminal id/capacity, rejecting inconsistent duplicate metadata."""
        if isinstance(terminal_or_id, Terminal):
            terminal_id = terminal_or_id.id
            supplied_capacity = terminal_or_id.capacity
            if capacity is not None and operator.index(capacity) != supplied_capacity:
                raise ValueError(
                    f"terminal {terminal_id!r} supplied two capacities: "
                    f"{supplied_capacity} and {capacity}"
                )
            capacity = supplied_capacity
        else:
            terminal_id = terminal_or_id
        if capacity is None:
            raise TypeError("capacity is required when registering a terminal id")
        try:
            hash(terminal_id)
        except TypeError as exc:
            raise TypeError("terminal ids must be hashable") from exc
        try:
            cap = operator.index(capacity)
        except TypeError as exc:
            raise TypeError("terminal capacity must be an integer") from exc
        if cap < 1:
            raise ValueError(f"terminal {terminal_id!r} capacity must be positive, got {cap}")
        previous = self._terminal_capacities.get(terminal_id)
        if previous is not None and previous != cap:
            raise ValueError(
                f"terminal {terminal_id!r} has inconsistent capacities {previous} and {cap}"
            )
        self._terminal_capacities[terminal_id] = cap

    def intern(self, key: RowKey | tuple[Any, ...]) -> int:
        """Return the stable dense id for ``key``, creating it on first use."""
        normalized = key if isinstance(key, RowKey) else RowKey(key)
        existing = self._key_to_index.get(normalized)
        if existing is not None:
            return existing
        index = len(self._index_to_key)
        self._key_to_index[normalized] = index
        self._index_to_key.append(normalized)
        return index

    def get(self, key: RowKey | tuple[Any, ...], default: Any = None) -> int | Any:
        """Return an already-interned id without mutating the index."""
        normalized = key if isinstance(key, RowKey) else RowKey(key)
        return self._key_to_index.get(normalized, default)

    def key(self, index: int) -> RowKey:
        """Resolve a dense id back to its tuple-compatible key."""
        dense_id = operator.index(index)
        if not 0 <= dense_id < len(self._index_to_key):
            raise IndexError(f"row index {dense_id} is outside [0, {len(self._index_to_key)})")
        return self._index_to_key[dense_id]

    def cap(self, key_or_index: RowKey | tuple[Any, ...] | int) -> int:
        """Return one for cell rows or the registered pad count for terminal rows."""
        if isinstance(key_or_index, (RowKey, tuple, list)):
            key = key_or_index if isinstance(key_or_index, RowKey) else RowKey(key_or_index)
        else:
            try:
                dense_id = operator.index(key_or_index)
            except TypeError as exc:
                raise TypeError("capacity expects a row key or integral row index") from exc
            key = self.key(dense_id)
        if key.kind == "cell":
            return 1
        terminal_id = key.terminal_id
        try:
            return self._terminal_capacities[terminal_id]
        except KeyError as exc:
            raise KeyError(
                f"capacity for terminal {terminal_id!r} is unknown; "
                "register the Terminal before materializing its rows"
            ) from exc

    capacity = cap

    def __getitem__(self, key: RowKey | tuple[Any, ...]) -> int:
        return self.intern(key)

    def __contains__(self, key: object) -> bool:
        try:
            normalized = key if isinstance(key, RowKey) else RowKey(key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return normalized in self._key_to_index

    def __iter__(self):
        return iter(self._index_to_key)

    def __len__(self) -> int:
        return len(self._index_to_key)

    def keys(self) -> tuple[RowKey, ...]:
        return tuple(self._index_to_key)

    def items(self) -> tuple[tuple[RowKey, int], ...]:
        return tuple((key, index) for index, key in enumerate(self._index_to_key))


@dataclass(frozen=True, slots=True)
class FlightGraph:
    """Immutable, flight-specific network domain consumed by pricing.

    ``origin_lanes`` and ``dest_lanes`` preserve the full deterministic order returned by
    :func:`hexgrid.terminal_lanes`; a lane blocked by overlapping foreign terminal airspace stays
    addressable by its original index but its cell is absent from ``corridor_cells``.
    """

    request: FlightRequest = field(repr=False, compare=False)
    origin_cell: Cell
    dest_cell: Cell
    corridor_cells: frozenset[Cell]
    index_to_cell: tuple[Cell, ...]
    cell_to_index: Mapping[Cell, int] = field(repr=False, compare=False)
    levels: tuple[float, ...]
    takeoff_steps: tuple[int, ...]
    origin_terminal: Terminal | None
    dest_terminal: Terminal | None
    origin_lanes: tuple[hg.Lane, ...]
    dest_lanes: tuple[hg.Lane, ...]
    base_step: int
    latest_departure_step: int
    min_step: int
    max_step: int
    shortest_hops: int
    detour_slack_hops: int
    static_exclusions: frozenset[Cell]
    foreign_exclusions: frozenset[Cell]
    own_terminal_interiors: frozenset[Cell]
    static_walls: tuple[Volume4D, ...] = field(repr=False, compare=False)
    forbidden_hops: frozenset[tuple[Cell, Cell]]

    @property
    def cells(self) -> tuple[Cell, ...]:
        return self.index_to_cell

    @property
    def cell_index(self) -> Mapping[Cell, int]:
        return self.cell_to_index

    @property
    def o_cell(self) -> Cell:
        return self.origin_cell

    @property
    def d_cell(self) -> Cell:
        return self.dest_cell

    @property
    def o_lanes(self) -> tuple[hg.Lane, ...]:
        return self.origin_lanes

    @property
    def d_lanes(self) -> tuple[hg.Lane, ...]:
        return self.dest_lanes

    @property
    def t_min(self) -> int:
        return self.min_step

    @property
    def t_max(self) -> int:
        return self.max_step

    @property
    def terminal_capacities(self) -> Mapping[Hashable, int]:
        """Endpoint terminal metadata suitable for :class:`RowIndex` construction."""
        capacities: dict[Hashable, int] = {}
        for terminal in (self.origin_terminal, self.dest_terminal):
            if terminal is None:
                continue
            previous = capacities.setdefault(terminal.id, terminal.capacity)
            if previous != terminal.capacity:
                raise ValueError(
                    f"terminal {terminal.id!r} has inconsistent endpoint capacities "
                    f"{previous} and {terminal.capacity}"
                )
        return MappingProxyType(capacities)


def _ellipse_cells(origin: Cell, dest: Cell, slack: int) -> set[Cell]:
    """Enumerate the finite axial O-D ellipse used by the pricing network."""
    shortest = hg.hex_distance(origin, dest)
    radius = shortest + slack
    oq, orr = origin
    cells: set[Cell] = set()
    # Enumerate the exact axial disk around the origin, then retain the O-D ellipse.
    for dq in range(-radius, radius + 1):
        dr_lo = max(-radius, -dq - radius)
        dr_hi = min(radius, -dq + radius)
        for dr in range(dr_lo, dr_hi + 1):
            cell = (oq + dq, orr + dr)
            if hg.hex_distance(origin, cell) + hg.hex_distance(cell, dest) <= shortest + slack:
                cells.add(cell)
    return cells


def _graph_max_step(
    latest_departure_step: int,
    takeoff_steps: tuple[int, ...],
    origin_lanes: tuple[hg.Lane, ...],
    shortest_hops: int,
    detour_slack_hops: int,
) -> int:
    """Budget-preserving final air-state bound.

    The Phase-1 plan's abbreviated expression omitted the climb and origin-lane traverse even
    though those advance the same integer clock before the first cell visit.  Include both so
    neither consumes the ground-delay or route budget (user-approved Phase-1 clarification).
    """
    return (
        latest_departure_step
        + max(takeoff_steps)
        + max((lane.steps for lane in origin_lanes), default=0)
        + shortest_hops
        + detour_slack_hops
    )


def _segment_terminal_id(
    p0,
    p1,
    req: FlightRequest,
    origin_terminal: Terminal | None,
    dest_terminal: Terminal | None,
    cfg: SimConfig,
) -> Hashable | None:
    """Apply the reservation builder's geometry-dependent terminal tag priority."""

    if origin_terminal is not None and segment_overlaps_column(
        p0,
        p1,
        req.origin,
        terminal_radius(origin_terminal, cfg),
        cfg,
    ):
        return origin_terminal.id
    if dest_terminal is not None and segment_overlaps_column(
        p0,
        p1,
        req.dest,
        terminal_radius(dest_terminal, cfg),
        cfg,
    ):
        return dest_terminal.id
    return None


def _forbidden_static_hops(
    cells: frozenset[Cell],
    walls: tuple[Volume4D, ...],
    req: FlightRequest,
    origin_terminal: Terminal | None,
    dest_terminal: Terminal | None,
    origin_lanes: tuple[hg.Lane, ...],
    dest_lanes: tuple[hg.Lane, ...],
    cfg: SimConfig,
) -> frozenset[tuple[Cell, Cell]]:
    """Find directed centre-to-centre lattice hops rejected by permanent walls.

    This is the pricing-usable constraint corresponding to the exact full-column
    check in :func:`column_claims`.  The latter remains authoritative because
    fixed exit-lane tags depend on a hop's position within the whole path.  A
    hop is globally forbidden only when every tag assignment possible for that
    arc conflicts; ambiguous endpoint arcs are left to the exact column guard.
    """

    if not walls:
        return frozenset()

    radius = hg.circumradius(cfg)
    z = cfg.flight_levels_m[0]
    segment_len = cfg.corridor_segment_len_m
    origin_lane_cells = {lane.cell for lane in origin_lanes}
    dest_lane_cells = {lane.cell for lane in dest_lanes}
    wall_bounds = tuple((wall, wall.flat_aabb()) for wall in walls)
    forbidden: set[tuple[Cell, Cell]] = set()
    for source in cells:
        source_xy = hg.hex_center(*source, radius)
        p0 = (float(source_xy[0]), float(source_xy[1]), float(z))
        for dq, dr in hg.AXIAL_NEIGHBORS:
            target = source[0] + dq, source[1] + dr
            if target not in cells:
                continue
            target_xy = hg.hex_center(*target, radius)
            p1 = (float(target_xy[0]), float(target_xy[1]), float(z))

            # Reproduce ``build_reservation_from_corners`` scalar interpolation.
            # A nominal lattice edge can be a few ulps longer than ``segment_len``
            # and split into two boxes that legitimately carry different tags.
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            nsub = max(1, math.ceil(length / segment_len))
            subvolumes: list[Volume4D] = []
            base_tags: list[Hashable | None] = []
            for k in range(1, nsub + 1):
                f0, f1 = (k - 1) / nsub, k / nsub
                sa = (p0[0] + f0 * dx, p0[1] + f0 * dy, p0[2] + f0 * dz)
                sb = (p0[0] + f1 * dx, p0[1] + f1 * dy, p0[2] + f1 * dz)
                tag = _segment_terminal_id(sa, sb, req, origin_terminal, dest_terminal, cfg)
                base_tags.append(tag)
                subvolumes.append(
                    corridor_segment_volume(
                        sa,
                        (k - 1) * cfg.dt_s / nsub,
                        sb,
                        k * cfg.dt_s / nsub,
                        cfg,
                        terminal_id=tag,
                    )
                )

            # The same arc may occur internally in a wide loop or as the
            # first/last arc.  Enumerate every builder tag outcome possible for
            # those roles.  If even one is wall-safe, do not globally prune the
            # arc; column_claims checks the candidate's actual final tags.
            tag_variants = {tuple(base_tags)}
            can_be_first = (
                cfg.fixed_exit_lanes and origin_terminal is not None and source in origin_lane_cells
            )
            can_be_last = (
                cfg.fixed_exit_lanes and dest_terminal is not None and target in dest_lane_cells
            )
            if can_be_first:
                tags = list(base_tags)
                tags[0] = origin_terminal.id
                tag_variants.add(tuple(tags))
            if can_be_last:
                tags = list(base_tags)
                tags[-1] = dest_terminal.id
                tag_variants.add(tuple(tags))
            if can_be_first and can_be_last:
                tags = list(base_tags)
                tags[0] = origin_terminal.id
                if len(tags) > 1:
                    tags[-1] = dest_terminal.id
                tag_variants.add(tuple(tags))

            def variant_conflicts(tags: tuple[Hashable | None, ...]) -> bool:
                for base_volume, tag in zip(subvolumes, tags):
                    hop = (
                        base_volume
                        if tag == base_volume.terminal_id
                        else replace(base_volume, terminal_id=tag)
                    )
                    hxmin, hymin, hzmin, hxmax, hymax, hzmax = hop.flat_aabb()
                    for wall, wall_bound in wall_bounds:
                        wxmin, wymin, wzmin, wxmax, wymax, wzmax = wall_bound
                        if (
                            hxmax < wxmin
                            or wxmax < hxmin
                            or hymax < wymin
                            or wymax < hymin
                            or hzmax < wzmin
                            or wzmax < hzmin
                        ):
                            continue
                        if volumes_conflict(hop, wall):
                            return True
                return False

            if all(variant_conflicts(tags) for tags in tag_variants):
                forbidden.add((source, target))
    return frozenset(forbidden)


def build_flight_graph(
    req: FlightRequest,
    cfg: SimConfig,
    static_terms,
    params: ColGenParams,
) -> FlightGraph:
    """Build the frozen corridor and timing bounds for one flight.

    The domain is the hex-distance O-D ellipse, with every foreign terminal's full
    ``terminal_cells`` wall removed.  A flight's own terminal interior is also removed; only the
    canonical boundary lanes are added back, preserving A*'s fixed-lane geometry.
    """
    if cfg.n_levels != 1:
        raise NotImplementedError(
            "colgen v1 supports a single flight level; multi-level level choice is a planned extension"
        )
    validate_edge_locality(cfg)
    try:
        slack = operator.index(params.detour_slack_hops)
    except (AttributeError, TypeError) as exc:
        raise TypeError("params.detour_slack_hops must be an integer") from exc
    if slack < 0:
        raise ValueError(f"detour_slack_hops must be non-negative, got {slack}")

    dt = cfg.dt_s
    radius = hg.circumradius(cfg)
    origin_cell = hg.enu_to_axial(float(req.origin[0]), float(req.origin[1]), radius)
    dest_cell = hg.enu_to_axial(float(req.dest[0]), float(req.dest[1]), radius)
    if origin_cell == dest_cell:
        raise ValueError(
            f"flight {req.flight_id} origin and destination map to the same hex {origin_cell}; "
            "colgen requires at least one lateral hop"
        )

    origin_terminal = as_terminal(req.origin_terminal)
    dest_terminal = as_terminal(req.dest_terminal)
    if not cfg.terminal_airspace_always_active and (
        origin_terminal is not None or dest_terminal is not None
    ):
        raise NotImplementedError(
            "colgen v1 requires terminal_airspace_always_active=True for terminal endpoints"
        )
    if not cfg.fixed_exit_lanes and (origin_terminal is not None or dest_terminal is not None):
        raise NotImplementedError("colgen v1 requires fixed_exit_lanes=True for terminal endpoints")
    origin_lanes = tuple(
        hg.terminal_lanes(req.origin, origin_terminal, cfg) if origin_terminal is not None else ()
    )
    dest_lanes = tuple(
        hg.terminal_lanes(req.dest, dest_terminal, cfg) if dest_terminal is not None else ()
    )
    own_ids = frozenset(
        terminal.id for terminal in (origin_terminal, dest_terminal) if terminal is not None
    )

    normalized_static_terms: list[tuple[object, Terminal]] = []
    static_walls: list[Volume4D] = []
    foreign_exclusions: set[Cell] = set()
    for center, raw_terminal in static_terms:
        terminal = as_terminal(raw_terminal)
        if terminal is None:
            raise ValueError("static terminal entries must include terminal metadata")
        normalized_static_terms.append((center, terminal))
        static_walls.append(permanent_terminal_reservation(center, terminal, cfg))
        if terminal.id not in own_ids:
            foreign_exclusions.update(hg.terminal_cells(center, terminal, cfg))

    # Static terminal walls are not master rows.  Reject an endpoint whose actual filed cylinder
    # would hit a differently tagged wall even when its snapped cell lies outside terminal_cells.
    for label, point, endpoint_terminal in (
        ("origin", req.origin, origin_terminal),
        ("destination", req.dest, dest_terminal),
    ):
        endpoint_radius = (
            terminal_radius(endpoint_terminal, cfg)
            if endpoint_terminal is not None
            else cfg.effective_hover_radius_m
        )
        px, py = float(point[0]), float(point[1])
        for center, static_terminal in normalized_static_terms:
            if endpoint_terminal is not None and endpoint_terminal.id == static_terminal.id:
                continue
            distance = math.hypot(px - float(center[0]), py - float(center[1]))
            reach = endpoint_radius + terminal_radius(static_terminal, cfg)
            if distance <= reach + 1e-9:
                raise ValueError(
                    f"flight {req.flight_id} {label} cylinder overlaps foreign static terminal "
                    f"{static_terminal.id!r} (distance={distance:g} m, combined radius={reach:g} m)"
                )

    own_interiors: set[Cell] = set()
    for center, terminal, lanes in (
        (req.origin, origin_terminal, origin_lanes),
        (req.dest, dest_terminal, dest_lanes),
    ):
        if terminal is None:
            continue
        lane_cells = {lane.cell for lane in lanes}
        own_interiors.update(hg.terminal_cells(center, terminal, cfg) - lane_cells)

    shortest_hops = hg.hex_distance(origin_cell, dest_cell)
    corridor = _ellipse_cells(origin_cell, dest_cell, slack)
    static_exclusions = foreign_exclusions | own_interiors
    corridor.difference_update(static_exclusions)

    # Own lanes are explicit takeoff/arrival nodes, even when they fall just outside the centre-based
    # O-D ellipse.  Foreign walls still win in the unlikely event that two terminal footprints overlap.
    for lane in (*origin_lanes, *dest_lanes):
        if lane.cell not in foreign_exclusions:
            corridor.add(lane.cell)

    if origin_terminal is None:
        if origin_cell not in corridor:
            raise ValueError(
                f"flight {req.flight_id} customer origin {origin_cell} is excluded by "
                "terminal airspace"
            )
    elif not any(lane.cell in corridor for lane in origin_lanes):
        raise ValueError(f"flight {req.flight_id} has no available origin terminal lane")
    if dest_terminal is None:
        if dest_cell not in corridor:
            raise ValueError(
                f"flight {req.flight_id} customer destination {dest_cell} is excluded by "
                "terminal airspace"
            )
    elif not any(lane.cell in corridor for lane in dest_lanes):
        raise ValueError(f"flight {req.flight_id} has no available destination terminal lane")

    ordered_cells = tuple(sorted(corridor))
    frozen_static_walls = tuple(static_walls)
    forbidden_hops = _forbidden_static_hops(
        frozenset(corridor),
        frozen_static_walls,
        req,
        origin_terminal,
        dest_terminal,
        origin_lanes,
        dest_lanes,
        cfg,
    )
    cell_to_index = _ImmutableCellIndex(tuple((cell, i) for i, cell in enumerate(ordered_cells)))
    base_step = int(math.ceil(req.t_departure / dt))
    # A departure is legal only when its reported hold, measured from ``base_step``, remains within
    # max_ground_delay_s.  Floor is therefore the correct outward network bound for this budget.
    max_ground_steps = int(math.floor(cfg.max_ground_delay_s / dt + 1e-12))
    latest_departure_step = base_step + max_ground_steps
    levels = tuple(cfg.flight_levels_m)
    takeoff_steps = tuple(cfg.climb_steps_to(z) for z in levels)
    max_step = _graph_max_step(
        latest_departure_step,
        takeoff_steps,
        origin_lanes,
        shortest_hops,
        slack,
    )

    return FlightGraph(
        request=_snapshot_request(req, origin_terminal, dest_terminal),
        origin_cell=origin_cell,
        dest_cell=dest_cell,
        corridor_cells=frozenset(corridor),
        index_to_cell=ordered_cells,
        cell_to_index=cell_to_index,
        levels=levels,
        takeoff_steps=takeoff_steps,
        origin_terminal=origin_terminal,
        dest_terminal=dest_terminal,
        origin_lanes=origin_lanes,
        dest_lanes=dest_lanes,
        base_step=base_step,
        latest_departure_step=latest_departure_step,
        min_step=base_step,
        max_step=max_step,
        shortest_hops=shortest_hops,
        detour_slack_hops=slack,
        static_exclusions=frozenset(static_exclusions),
        foreign_exclusions=frozenset(foreign_exclusions),
        own_terminal_interiors=frozenset(own_interiors),
        static_walls=frozen_static_walls,
        forbidden_hops=forbidden_hops,
    )


def _selected_lane(lanes: tuple[hg.Lane, ...], index: int | None, endpoint: str) -> hg.Lane:
    if index is None:
        raise ValueError(f"{endpoint}_lane_idx is required for a terminal endpoint")
    try:
        lane = lanes[operator.index(index)]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}") from exc
    if index < 0:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}")
    return lane


def column_claims(
    column: Column,
    fg: FlightGraph,
    cfg: SimConfig,
    W: int | tuple[int, int] | None = None,
) -> frozenset[RowKey]:
    """Return the de-duplicated capacity rows claimed by ``column``.

    ``W`` is retained as a compatibility/checking argument for the formulation's window-width
    notation.  The actual offsets always come from :func:`derive_cell_window`; a scalar width alone
    cannot describe the shifted ``(-1, 0)`` footprint used when ``time_buffer_s == 0``.
    """
    if column.flight_id != fg.request.flight_id:
        raise ValueError(
            f"column flight_id {column.flight_id} does not match graph flight "
            f"{fg.request.flight_id}"
        )
    if not 0 <= column.level < len(fg.levels):
        raise ValueError(
            f"column level {column.level} is outside graph levels [0, {len(fg.levels)})"
        )
    if not fg.base_step <= column.departure_step <= fg.latest_departure_step:
        raise ValueError(
            f"column departure step {column.departure_step} is outside legal range "
            f"[{fg.base_step}, {fg.latest_departure_step}]"
        )

    offsets = derive_cell_window(cfg)
    width = offsets[1] - offsets[0] + 1
    if W is not None:
        if isinstance(W, tuple):
            if W != offsets:
                raise ValueError(f"supplied cell-window offsets {W} != derived offsets {offsets}")
        else:
            try:
                supplied_width = operator.index(W)
            except TypeError as exc:
                raise TypeError("W must be an integer width or an (lo, hi) offset tuple") from exc
            if supplied_width != width:
                raise ValueError(
                    f"supplied cell-window width {supplied_width} != derived width {width}"
                )

    path = tuple(tuple(cell) for cell in column.cell_path)
    if len(path) < 2:
        raise ValueError("a column path must contain at least two cells and one lateral hop")
    if any(cell not in fg.corridor_cells for cell in path):
        invalid = next(cell for cell in path if cell not in fg.corridor_cells)
        raise ValueError(f"column cell {invalid} is outside the flight graph corridor")
    for a, b in zip(path, path[1:]):
        if hg.hex_distance(a, b) != 1:
            raise ValueError(f"column contains a non-neighbour lateral hop {a} -> {b}")
        if (a, b) in fg.forbidden_hops:
            raise ValueError(f"column hop {a} -> {b} overlaps permanent static terminal airspace")

    origin_lane_steps = 0
    if fg.origin_terminal is not None:
        origin_lane = _selected_lane(fg.origin_lanes, column.origin_lane_idx, "origin")
        if path[0] != origin_lane.cell:
            raise ValueError(
                f"column origin cell {path[0]} does not match selected lane {origin_lane.cell}"
            )
        origin_lane_steps = origin_lane.steps
    else:
        if column.origin_lane_idx is not None:
            raise ValueError("origin_lane_idx must be None for a customer origin")
        if path[0] != fg.origin_cell:
            raise ValueError(
                f"column origin cell {path[0]} does not match request cell {fg.origin_cell}"
            )

    if fg.dest_terminal is not None:
        dest_lane = _selected_lane(fg.dest_lanes, column.dest_lane_idx, "dest")
        if path[-1] != dest_lane.cell:
            raise ValueError(
                f"column destination cell {path[-1]} does not match selected lane {dest_lane.cell}"
            )
    else:
        if column.dest_lane_idx is not None:
            raise ValueError("dest_lane_idx must be None for a customer destination")
        if path[-1] != fg.dest_cell:
            raise ValueError(
                f"column destination cell {path[-1]} does not match request cell {fg.dest_cell}"
            )

    corridor_start_step = column.departure_step + fg.takeoff_steps[column.level] + origin_lane_steps
    arrival_step = corridor_start_step + len(path) - 1
    if arrival_step > fg.max_step:
        raise ValueError(
            f"column arrives at step {arrival_step}, beyond graph maximum {fg.max_step}"
        )

    # The graph's additive hop slack is an enumeration bound, not permission
    # to violate the simulation's independent budgets.  Translation is the
    # canonical gate: using its actual resampled centerline here prevents even
    # an ulp-level difference from making a claimed column fail at filing.
    z = fg.levels[column.level]
    from .translate import column_to_intent

    intent = column_to_intent(column, fg.request, cfg)
    if intent.status is not IntentStatus.ACCEPTED:
        raise ValueError(
            f"column violates max_detour_factor or another translation budget "
            f"({intent.denial_reason})"
        )

    # Static walls are deliberately not master rows.  The discrete
    # ``terminal_cells`` pruning is useful for pricing, but it is not an exact
    # representation of the permanent cylinders (and a single box can touch
    # both endpoint hubs while carrying only one tag).  Certify every canonical
    # column against the exact ledger geometry before exposing its row claims.
    # Reusing ``volumes_conflict`` is essential: its same-terminal cylinder
    # exemption is precisely the one applied again when the intent is filed.
    if fg.static_walls:
        for volume in intent.volumes:
            for wall in fg.static_walls:
                if volumes_conflict(volume, wall):
                    raise ValueError(
                        f"column for flight {column.flight_id} overlaps permanent static "
                        f"terminal {wall.terminal_id!r}"
                    )

    claims: set[RowKey] = set()
    for visit_step, (q, r) in enumerate(path, start=corridor_start_step):
        claims.update(
            RowKey.cell(q, r, column.level, row_step)
            for row_step in visit_rows(visit_step, offsets)
        )

    dt = cfg.dt_s
    origin_t0 = column.departure_step * dt
    arrival_t0 = arrival_step * dt
    endpoints = (
        (fg.request.origin, fg.origin_terminal, origin_t0, 0),
        (fg.request.dest, fg.dest_terminal, arrival_t0, len(path) - 1),
    )
    for point, terminal, t0, timing_steps in endpoints:
        dwell_t1 = t0 + cfg.hover_time_s + column_dwell_s(point, terminal, cfg, z)
        if terminal is not None:
            claims.update(
                RowKey.term(terminal.id, row_step)
                for row_step in terminal_claim_steps(t0, dwell_t1, cfg)
            )
            continue

        # Customer cylinders are untagged ledger geometry.  They span the regulated tube, hence
        # claim every flight level in every nearby cell and time row that could meet a transit.
        endpoint_cells = endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
        endpoint_steps = endpoint_claim_steps(t0, dwell_t1, cfg, timing_steps=timing_steps)
        claims.update(
            RowKey.cell(q, r, level, row_step)
            for q, r in endpoint_cells
            for level in range(len(fg.levels))
            for row_step in endpoint_steps
        )

    return frozenset(claims)


__all__ = [
    "Cell",
    "FlightGraph",
    "RowIndex",
    "RowKey",
    "build_flight_graph",
    "column_claims",
]
