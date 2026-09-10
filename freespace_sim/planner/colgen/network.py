"""Flight-specific space-time networks and their capacity-row membership.

The master problem never reasons about reservation geometry directly.  A column instead
claims immutable row keys produced here: one family for buffered cell visits and one for
terminal-pad dwell.  Keeping row construction in this module is important -- every master,
rounding, repair, and filing check must see exactly the same coefficients.
"""

from __future__ import annotations

import math
import operator
import threading
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Set as AbstractSet
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
FlatAabb = tuple[float, float, float, float, float, float]
WallBound = tuple[Volume4D, FlatAabb]

_ARC_INTERNAL = 1 << 0
_ARC_FIRST = 1 << 1
_ARC_LAST = 1 << 2
_ARC_FIRST_LAST = 1 << 3
_ALL_ARC_ROLES = _ARC_INTERNAL | _ARC_FIRST | _ARC_LAST | _ARC_FIRST_LAST
_MAX_CERTIFIED_COLUMNS = 2
# Distinct `(origin, step, timing_steps)` endpoint claim sets kept per graph.  The reachable
# key space is bounded by `2 * n_steps * (max_air_hops + 1)` (grows with the horizon), each
# entry a frozenset of freshly built `RowKey`s.  A memory bound, not a correctness one: an
# eviction only costs the recompute it was avoiding, so this is answer-neutral and free to
# tune.  Enforced per graph (on one graph's `_search_cache`) and not yet calibrated against
# that per-graph key count -- smaller than any whole-sweep total, and larger on density.
_MAX_ENDPOINT_CLAIMS = 2048


def _aabbs_overlap(first: FlatAabb, second: FlatAabb) -> bool:
    """True if two flat AABBs overlap on all three axes (separating-axis test)."""
    axmin, aymin, azmin, axmax, aymax, azmax = first
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = second
    return not (
        axmax < bxmin
        or bxmax < axmin
        or aymax < bymin
        or bymax < aymin
        or azmax < bzmin
        or bzmax < azmin
    )


class _WallSpatialIndex:
    """Exact broad phase for immutable permanent terminal cylinders.

    Each wall is inserted into every uniform XY bucket its AABB touches, so a query AABB cannot
    miss a possible intersection; the final full 3-D AABB test and ``volumes_conflict`` remain
    authoritative.  A solve-scoped terminal catalog shares one instance across flight graphs;
    standalone graph construction still creates a self-contained instance.
    """

    __slots__ = (
        "_bounds",
        "_bucket_size",
        "_buckets",
        "_candidate_count",
        "_lock",
        "_query_count",
        "_walls",
    )

    def __init__(self, walls: tuple[Volume4D, ...], bucket_size: float) -> None:
        """Bucket every wall's AABB into a uniform XY grid for later broad-phase queries.

        Parameters
        ------------
        - walls (tuple[Volume4D, ...]): the permanent terminal cylinders to index.
        - bucket_size (float): uniform XY bucket edge (m); must be finite and positive.

        Return
        --------
        - output (None): populates the bucket grid; raises ``ValueError`` on a non-positive size.
        """
        if not math.isfinite(bucket_size) or bucket_size <= 0.0:
            raise ValueError("wall-index bucket size must be finite and positive")
        self._walls = walls
        self._bounds: tuple[WallBound, ...] = tuple(
            (wall, wall.flat_aabb()) for wall in walls
        )
        self._bucket_size = float(bucket_size)
        mutable: dict[tuple[int, int], list[int]] = {}
        for wall_index, (_wall, bound) in enumerate(self._bounds):
            xmin, ymin, _zmin, xmax, ymax, _zmax = bound
            for x_bucket in range(
                math.floor(xmin / self._bucket_size),
                math.floor(xmax / self._bucket_size) + 1,
            ):
                for y_bucket in range(
                    math.floor(ymin / self._bucket_size),
                    math.floor(ymax / self._bucket_size) + 1,
                ):
                    mutable.setdefault((x_bucket, y_bucket), []).append(wall_index)
        self._buckets = {
            bucket: tuple(indices) for bucket, indices in mutable.items()
        }
        self._query_count = 0
        self._candidate_count = 0
        self._lock = threading.RLock()

    def candidates(self, bound: FlatAabb) -> tuple[WallBound, ...]:
        """Return indexed walls whose AABB overlaps ``bound`` (exact filter over bucket hits)."""
        xmin, ymin, _zmin, xmax, ymax, _zmax = bound
        indices: set[int] = set()
        for x_bucket in range(
            math.floor(xmin / self._bucket_size),
            math.floor(xmax / self._bucket_size) + 1,
        ):
            for y_bucket in range(
                math.floor(ymin / self._bucket_size),
                math.floor(ymax / self._bucket_size) + 1,
            ):
                indices.update(self._buckets.get((x_bucket, y_bucket), ()))
        result = tuple(
            self._bounds[index]
            for index in sorted(indices)
            if _aabbs_overlap(bound, self._bounds[index][1])
        )
        with self._lock:
            self._query_count += 1
            self._candidate_count += len(result)
        return result

    @property
    def all_bounds(self) -> tuple[WallBound, ...]:
        return self._bounds

    @property
    def stats(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(
                {
                    "queries": self._query_count,
                    "candidates": self._candidate_count,
                    "walls": len(self._walls),
                }
            )

    def __reduce__(self):
        # Query counters are diagnostics, not semantic state.
        return type(self), (self._walls, self._bucket_size)


class StaticTerminalCatalog:
    """Solve-scoped read-only static geometry shared by every flight graph."""

    __slots__ = (
        "_cell_terminal_ids",
        "_cfg",
        "_entries",
        "_wall_index",
        "_walls",
    )

    def __init__(self, static_terms, cfg: SimConfig) -> None:
        """Freeze the solve-wide static terminals: walls, keep-out cells, and broad-phase index.

        Parameters
        ------------
        - static_terms (Iterable[tuple[Sequence[float], Terminal-like]]): ``(center, terminal)``
          pairs; each center needs two or three coordinates and terminal metadata.
        - cfg (SimConfig): geometry used to build the walls, keep-out cells, and bucket size.

        Return
        --------
        - output (None): populates the catalog; raises ``ValueError`` on a missing terminal or a
          malformed center.
        """
        entries: list[tuple[tuple[float, ...], Terminal]] = []
        walls: list[Volume4D] = []
        cell_terminal_ids: dict[Cell, set[Hashable]] = {}
        for center, raw_terminal in static_terms:
            terminal = as_terminal(raw_terminal)
            if terminal is None:
                raise ValueError("static terminal entries must include terminal metadata")
            frozen_center = tuple(float(coordinate) for coordinate in center)
            if len(frozen_center) not in {2, 3}:
                raise ValueError("static terminal centers must have two or three coordinates")
            entries.append((frozen_center, terminal))
            walls.append(permanent_terminal_reservation(frozen_center, terminal, cfg))
            for cell in hg.terminal_cells(frozen_center, terminal, cfg):
                cell_terminal_ids.setdefault(cell, set()).add(terminal.id)
        self._cfg = cfg
        self._entries = tuple(entries)
        self._walls = tuple(walls)
        self._cell_terminal_ids = {
            cell: frozenset(terminal_ids)
            for cell, terminal_ids in cell_terminal_ids.items()
        }
        self._wall_index = _WallSpatialIndex(
            self._walls,
            max(256.0, 4.0 * cfg.corridor_width_m),
        )

    @property
    def entries(self) -> tuple[tuple[tuple[float, ...], Terminal], ...]:
        return self._entries

    @property
    def walls(self) -> tuple[Volume4D, ...]:
        return self._walls

    @property
    def wall_index(self) -> _WallSpatialIndex:
        return self._wall_index

    def terminal_ids_at(self, cell: Cell) -> frozenset[Hashable]:
        """Terminal ids whose keep-out flood-fill covers ``cell`` (empty if none)."""
        return self._cell_terminal_ids.get(cell, frozenset())

    @property
    def excluded_cell_count(self) -> int:
        return len(self._cell_terminal_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StaticTerminalCatalog):
            return NotImplemented
        return self._cfg == other._cfg and self.entries == other.entries

    def __reduce__(self):
        return type(self), (self._entries, self._cfg)


class _ForeignTerminalCells(AbstractSet[Cell]):
    """Per-flight view over shared terminal-cell membership by terminal id."""

    __slots__ = ("_catalog", "_own_ids")

    def __init__(
        self,
        catalog: StaticTerminalCatalog,
        own_ids: frozenset[Hashable],
    ) -> None:
        self._catalog = catalog
        self._own_ids = own_ids

    def __contains__(self, raw_cell: object) -> bool:
        if not isinstance(raw_cell, tuple) or len(raw_cell) != 2:
            return False
        try:
            cell = operator.index(raw_cell[0]), operator.index(raw_cell[1])
        except TypeError:
            return False
        return not self._catalog.terminal_ids_at(cell).issubset(self._own_ids)

    def __iter__(self):
        return (
            cell
            for cell in self._catalog._cell_terminal_ids
            if cell in self
        )

    def __len__(self) -> int:
        return sum(cell in self for cell in self._catalog._cell_terminal_ids)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ForeignTerminalCells):
            if self._catalog == other._catalog and self._own_ids == other._own_ids:
                return True
            return frozenset(self) == frozenset(other)
        if isinstance(other, AbstractSet):
            return frozenset(self) == other
        return NotImplemented


class _CombinedCellSet(AbstractSet[Cell]):
    """Lazy union used by the compatibility ``static_exclusions`` field."""

    __slots__ = ("_base", "_extra")

    def __init__(self, base: AbstractSet[Cell], extra: frozenset[Cell]) -> None:
        self._base = base
        self._extra = extra

    def __contains__(self, cell: object) -> bool:
        return cell in self._extra or cell in self._base

    def __iter__(self):
        return iter(frozenset(self._base) | self._extra)

    def __len__(self) -> int:
        return len(frozenset(self._base) | self._extra)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CombinedCellSet):
            if self._base == other._base and self._extra == other._extra:
                return True
            return frozenset(self) == frozenset(other)
        if isinstance(other, AbstractSet):
            return frozenset(self) == other
        return NotImplemented


class _ImmutableFlightRequest(FlightRequest):
    """A detached request whose geometry/terminal signature cannot drift."""

    def __post_init__(self) -> None:
        """Freeze the snapshot's endpoint arrays onto immutable bytes after construction.

        Parameters
        ------------
        - none: reads ``self.origin``/``self.dest`` and rebinds them to read-only buffers.

        Return
        --------
        - output (None): rebinds the endpoint arrays and sets ``_colgen_frozen`` so that even
          ``setflags(write=True)`` cannot make them writable.
        """
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
    """Detach graph geometry from caller-owned mutable endpoint arrays.

    Parameters
    ------------
    - req (FlightRequest): the source request whose scalar fields are copied.
    - origin_terminal (Terminal | None): origin terminal to attach to the snapshot.
    - dest_terminal (Terminal | None): destination terminal to attach to the snapshot.

    Return
    --------
    - output (FlightRequest): an immutable copy with detached endpoint arrays.
    """

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
        """Normalize and validate parts into a canonical cell/term key, raising on bad input."""
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
        """The row kind: ``"cell"`` or ``"term"``."""
        return tuple.__getitem__(self, 0)

    @property
    def step(self) -> int:
        """The time step (tuple slot 4 for a cell row, slot 2 for a terminal row)."""
        return tuple.__getitem__(self, 4 if self.kind == "cell" else 2)

    @property
    def cell_coord(self) -> Cell:
        """The ``(q, r)`` cell of a cell row; raises ``AttributeError`` on a terminal row."""
        if self.kind != "cell":
            raise AttributeError("terminal rows do not have a cell")
        return tuple.__getitem__(self, 1), tuple.__getitem__(self, 2)

    @property
    def level(self) -> int:
        """The flight level of a cell row; raises ``AttributeError`` on a terminal row."""
        if self.kind != "cell":
            raise AttributeError("terminal rows do not have a flight level")
        return tuple.__getitem__(self, 3)

    @property
    def terminal_id(self) -> Hashable:
        """The terminal id of a terminal row; raises ``AttributeError`` on a cell row."""
        if self.kind != "term":
            raise AttributeError("cell rows do not have a terminal id")
        return tuple.__getitem__(self, 1)

    def __getnewargs__(self) -> tuple[tuple[Any, ...]]:
        """Round-trip through pickle by re-passing the flat tuple to :meth:`__new__`."""
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
        """Create an empty row-key intern table, pre-registering any terminal capacities.

        Parameters
        ------------
        - terminal_capacities (Mapping[Hashable, int] | None): terminal id -> pad count to
          register up front, or ``None`` for none.

        Return
        --------
        - output (None): initializes the intern maps and the terminal-capacity registry.
        """
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
        """Register a terminal id and its pad capacity, rejecting inconsistent duplicates.

        Accepts either a :class:`Terminal` (capacity read from it) or a bare id plus ``capacity``.

        Parameters
        ------------
        - terminal_or_id (Terminal | Hashable): the terminal, or its hashable id.
        - capacity (int | None): pad count; required with a bare id, and must match the Terminal's
          own capacity when both are supplied.

        Return
        --------
        - output (None): records the capacity; raises ``TypeError``/``ValueError`` on bad or
          conflicting input.
        """
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
        """Return the stable dense id for ``key``, creating it on first use.

        Parameters
        ------------
        - key (RowKey | tuple): the row key, normalized to :class:`RowKey` if needed.

        Return
        --------
        - output (int): the dense id, newly assigned if ``key`` was not yet interned.
        """
        normalized = key if isinstance(key, RowKey) else RowKey(key)
        existing = self._key_to_index.get(normalized)
        if existing is not None:
            return existing
        index = len(self._index_to_key)
        self._key_to_index[normalized] = index
        self._index_to_key.append(normalized)
        return index

    def get(self, key: RowKey | tuple[Any, ...], default: Any = None) -> int | Any:
        """Return an already-interned dense id without mutating the index.

        Parameters
        ------------
        - key (RowKey | tuple): the row key to look up (normalized to :class:`RowKey`).
        - default (Any): value returned when ``key`` has not been interned.

        Return
        --------
        - output (int | Any): the dense id, or ``default`` if absent.
        """
        normalized = key if isinstance(key, RowKey) else RowKey(key)
        return self._key_to_index.get(normalized, default)

    def key(self, index: int) -> RowKey:
        """Resolve a dense id back to its tuple-compatible :class:`RowKey`.

        Parameters
        ------------
        - index (int): a dense id previously returned by :meth:`intern`.

        Return
        --------
        - output (RowKey): the key for ``index``; raises ``IndexError`` if it is out of range.
        """
        dense_id = operator.index(index)
        if not 0 <= dense_id < len(self._index_to_key):
            raise IndexError(f"row index {dense_id} is outside [0, {len(self._index_to_key)})")
        return self._index_to_key[dense_id]

    def cap(self, key_or_index: RowKey | tuple[Any, ...] | int) -> int:
        """Return the row's capacity: one for a cell row, the registered pad count for a terminal.

        Parameters
        ------------
        - key_or_index (RowKey | tuple | int): a row key or an interned dense id.

        Return
        --------
        - output (int): the capacity; raises ``KeyError`` if a terminal row's capacity was never
          registered.
        """
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
        """Intern ``key`` and return its dense id (subscript sugar for :meth:`intern`)."""
        return self.intern(key)

    def __contains__(self, key: object) -> bool:
        """True if ``key`` has been interned; a malformed key returns False rather than raising."""
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
        """All interned keys, in dense-id order."""
        return tuple(self._index_to_key)

    def items(self) -> tuple[tuple[RowKey, int], ...]:
        """All ``(key, dense_id)`` pairs, in dense-id order."""
        return tuple((key, index) for index, key in enumerate(self._index_to_key))


class _LazyCorridorCells(AbstractSet[Cell]):
    """Implicit finite corridor bounded by actual cruise endpoints and the hop cap.

    Pricing only needs membership checks while it explores a route.  Materializing
    every cell (and then duplicating it into a tuple and index mapping) was a large
    all-flight startup barrier at density scale.  Iteration remains available for
    diagnostics/tests and materializes the same membership set once on demand.
    """

    __slots__ = (
        "_dest",
        "_dest_cells",
        "_explicit_lanes",
        "_foreign_exclusions",
        "_lock",
        "_materialized",
        "_origin",
        "_origin_cells",
        "_overrun",
        "_own_interiors",
        "_shortest",
    )

    def __init__(
        self,
        origin: Cell,
        dest: Cell,
        overrun: int,
        foreign_exclusions: AbstractSet[Cell],
        own_interiors: frozenset[Cell],
        explicit_lanes: frozenset[Cell],
        origin_cells: tuple[Cell, ...] | None = None,
        dest_cells: tuple[Cell, ...] | None = None,
    ) -> None:
        """Store the O-D ellipse parameters; membership stays lazy until first materialized."""
        self._origin = origin
        self._dest = dest
        self._origin_cells = origin_cells if origin_cells is not None else (origin,)
        self._dest_cells = dest_cells if dest_cells is not None else (dest,)
        self._overrun = overrun
        self._shortest = hg.hex_distance(origin, dest)
        self._foreign_exclusions = foreign_exclusions
        self._own_interiors = own_interiors
        self._explicit_lanes = explicit_lanes
        self._materialized: frozenset[Cell] | None = None
        self._lock = threading.RLock()

    def __contains__(self, raw_cell: object) -> bool:
        """True if ``raw_cell`` is in the O-D ellipse: within ``shortest + overrun`` summed hops
        of both endpoints, minus foreign exclusions and own interiors, plus explicit lanes
        (see context/figures/od_hop_ellipse.png)."""
        if not isinstance(raw_cell, tuple) or len(raw_cell) != 2:
            return False
        try:
            cell = operator.index(raw_cell[0]), operator.index(raw_cell[1])
        except TypeError:
            return False
        if cell in self._foreign_exclusions:
            return False
        if cell in self._explicit_lanes:
            return True
        if cell in self._own_interiors:
            return False
        return self._within_budget(cell)

    def _within_budget(self, cell: Cell) -> bool:
        """Apply a lower bound on cruise hops through this cell."""
        return (
            min(hg.hex_distance(start, cell) for start in self._origin_cells)
            + min(hg.hex_distance(cell, end) for end in self._dest_cells)
            <= self._shortest + self._overrun
        )

    def _cells(self) -> frozenset[Cell]:
        """Materialize (once, under lock) the frozenset the lazy membership test implies."""
        materialized = self._materialized
        if materialized is not None:
            return materialized
        with self._lock:
            materialized = self._materialized
            if materialized is None:
                # See context/figures/colgen_lane_aware_corridor.png.
                padding = max(
                    hg.hex_distance(self._origin, start) for start in self._origin_cells
                ) + max(hg.hex_distance(end, self._dest) for end in self._dest_cells)
                cells = {
                    cell
                    for cell in _ellipse_cells(self._origin, self._dest, self._overrun + padding)
                    if self._within_budget(cell)
                }
                cells.difference_update(self._foreign_exclusions)
                cells.difference_update(self._own_interiors)
                cells.update(
                    cell for cell in self._explicit_lanes if cell not in self._foreign_exclusions
                )
                materialized = frozenset(cells)
                self._materialized = materialized
            return materialized

    def __iter__(self):
        return iter(self._cells())

    def __len__(self) -> int:
        return len(self._cells())

    def isdisjoint(self, other) -> bool:
        """True if no cell of ``other`` lies in the ellipse (``other`` is normally far smaller)."""
        return all(cell not in self for cell in other)

    @property
    def is_materialized(self) -> bool:
        return self._materialized is not None

    def _signature(self) -> tuple[Any, ...]:
        """The identity tuple used for cheap equality and pickling."""
        return (
            self._origin,
            self._dest,
            self._overrun,
            self._foreign_exclusions,
            self._own_interiors,
            self._explicit_lanes,
            self._origin_cells,
            self._dest_cells,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _LazyCorridorCells):
            if self._signature() == other._signature():
                return True
            return self._cells() == other._cells()
        if isinstance(other, AbstractSet):
            return self._cells() == other
        return NotImplemented

    def __reduce__(self):
        # The materialized diagnostic view is deliberately not serialized.
        return type(self), self._signature()


class _FlightSearchCache:
    """Mutable, answer-neutral cache kept outside frozen graph semantics."""

    __slots__ = (
        "lock",
        "certified_claims",
        "dag_budget",
        "endpoint_claims",
        "prepared",
        "seed_columns",
        "seed_delay_certified",
        "seed_model",
        "seed_search_complete",
    )

    def __init__(self) -> None:
        """Initialize an empty, answer-neutral per-flight search cache."""
        self.lock = threading.RLock()
        # `(PreparedTopology, PreparedRows)` for the compiled pricing path, built on first
        # use.  Both are pure functions of the graph and its `SimConfig`, so this is a
        # memo rather than state -- but it is not an optional one: the same flight is
        # priced on EVERY colgen iteration, and rebuilding was measured at up to 80% of the
        # compiled search's own time on a cheap density flight.  See `dp_prepare.prepared_for`.
        self.prepared: Any | None = None
        # `(label_capacity, log2cap, candidate_capacity)` the last completed compiled search
        # actually needed.  A density flight can far exceed the 65,536 label default, and
        # re-discovering that by restarting costs ~1.7x on every iteration after the first.
        # Answer-neutral: a budget only bounds work, never the search.
        self.dag_budget: tuple[int, int, int] | None = None
        self.certified_claims: OrderedDict[
            tuple[Any, ...], tuple[int, frozenset[RowKey], tuple[tuple[int, int], ...]]
        ] = OrderedDict()
        # Endpoint dwell rows, keyed `(origin, step, timing_steps)`.  See
        # `pricing._endpoint_claims`, which owns the purity argument that makes this
        # answer-neutral, and `_MAX_ENDPOINT_CLAIMS` for why it is bounded.
        self.endpoint_claims: OrderedDict[
            tuple[bool, int, int], frozenset[RowKey]
        ] = OrderedDict()
        self.seed_columns: tuple[Any, ...] | None = None
        self.seed_delay_certified = False
        # The objective the cached seed was costed under.  Everything else in this cache
        # is answer-neutral -- it restates what the graph already determines -- but a seed
        # is NOT: its ``delay_s`` is the cost model's verdict, and the cheapest seed under
        # one weighting is not the cheapest under another.  Keying on the model turns a
        # documented "one model per graph" assumption into one that cannot be violated
        # silently, which matters because the failure mode is a wrong number rather than
        # an exception.
        self.seed_model: Any | None = None
        self.seed_search_complete = False

    def __reduce__(self):
        # A transported graph starts with a cold cache: the lock cannot be pickled and the
        # payload is answer-neutral, so rebuilding it is cheaper than shipping it.
        return type(self), ()


@dataclass(frozen=True, slots=True)
class FlightGraph:
    """Logically immutable flight domain with lazy, answer-neutral search caches.

    ``origin_lanes`` and ``dest_lanes`` preserve the full deterministic order returned by
    :func:`hexgrid.terminal_lanes`; a lane blocked by overlapping foreign terminal airspace stays
    addressable by its original index but its cell is absent from ``corridor_cells``.
    """

    request: FlightRequest = field(repr=False, compare=False)
    _cfg: SimConfig = field(repr=False, compare=False)
    origin_cell: Cell
    dest_cell: Cell
    corridor_cells: AbstractSet[Cell]
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
    # Hard ceiling on a priced route's hop count, from `params.max_air_overrun_hops`.
    # Resolved here rather than in pricing so both searches over this domain agree on it
    # and so it participates in graph identity.
    #
    # The same budget determines admissible corridor membership; terminal lane access time
    # remains separate from cruise hops. See context/figures/colgen_lane_aware_corridor.png.
    # Hand-built graphs may use another corridor; this ceiling remains the route-length bound.
    max_air_hops: int
    static_exclusions: AbstractSet[Cell]
    foreign_exclusions: AbstractSet[Cell]
    own_terminal_interiors: frozenset[Cell]
    static_walls: tuple[Volume4D, ...] = field(repr=False, compare=False)
    _wall_index: _WallSpatialIndex = field(repr=False, compare=False)
    forbidden_hops: frozenset[tuple[Cell, Cell]] | _LazyForbiddenHops
    _search_cache: _FlightSearchCache = field(
        default_factory=_FlightSearchCache,
        init=False,
        repr=False,
        compare=False,
    )

    def outgoing_neighbors(self, source: Cell) -> tuple[Cell, ...]:
        """Generate/cache all admissible directed arcs leaving ``source``.

        Parameters
        ------------
        - source (Cell): the axial ``(q, r)`` cell to expand.

        Return
        --------
        - output (tuple[Cell, ...]): the in-corridor neighbours reachable by a non-forbidden hop
          (empty if ``source`` lies outside the corridor).
        """

        lazy = self.forbidden_hops
        if isinstance(lazy, _LazyForbiddenHops):
            return lazy.outgoing(source)
        if source not in self.corridor_cells:
            return ()
        sq, sr = source
        return tuple(
            target
            for dq, dr in hg.AXIAL_NEIGHBORS
            if (target := (sq + dq, sr + dr)) in self.corridor_cells
            and (source, target) not in lazy
        )

    def hop_is_forbidden(self, source: Cell, target: Cell) -> bool:
        """Return the path-independent permanent-wall verdict for one arc."""

        return (source, target) in self.forbidden_hops

    def hop_allowed_for_role(
        self,
        source: Cell,
        target: Cell,
        *,
        first: bool,
        last: bool,
    ) -> bool:
        """Return whether the arc is safe with its actual path-position tags.

        Parameters
        ------------
        - source (Cell): the arc's start cell.
        - target (Cell): the arc's end cell.
        - first (bool): whether the arc is the path's first hop (origin-lane tag).
        - last (bool): whether the arc is the path's last hop (dest-lane tag).

        Return
        --------
        - output (bool): True if the arc, tagged for that role, clears every permanent wall with
          both cells in the corridor.
        """

        lazy = self.forbidden_hops
        if isinstance(lazy, _LazyForbiddenHops):
            return lazy.allows(source, target, first=first, last=last)
        return (
            source in self.corridor_cells
            and target in self.corridor_cells
            and (source, target) not in lazy
        )

    def __hash__(self) -> int:
        """Hash stable graph identity without materializing lazy set fields."""

        return hash(
            (
                self.origin_cell,
                self.dest_cell,
                self.levels,
                self.base_step,
                self.latest_departure_step,
                self.max_step,
                self.shortest_hops,
                self.max_air_hops,
            )
        )

    @property
    def arc_cache_stats(self) -> Mapping[str, int]:
        """Arc-expansion diagnostics from the lazy hop cache (eager hops report zeros)."""
        lazy = self.forbidden_hops
        if isinstance(lazy, _LazyForbiddenHops):
            return lazy.stats
        return MappingProxyType(
            {
                "expanded_nodes": 0,
                "arc_checks": 0,
                "cache_hits": 0,
                "allowed_arcs": 0,
                "blocked_arcs": len(lazy),
            }
        )

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


def _ellipse_cells(origin: Cell, dest: Cell, overrun: int) -> set[Cell]:
    """Enumerate the finite axial O-D ellipse used by the pricing network.

    A path within ``shortest + overrun`` hops cannot leave this endpoint ellipse.
    Terminal-lane corridors supply their own endpoints and remaining hop allowance;
    ``overrun`` is therefore not necessarily ``params.max_air_overrun_hops``.

    Parameters
    ------------
    - origin (Cell): the O-D ellipse's origin axial cell.
    - dest (Cell): the O-D ellipse's destination axial cell.
    - overrun (int): extra hops allowed beyond the shortest O-D hop distance; sets the radius.

    Return
    --------
    - output (set[Cell]): every axial cell ``c`` with ``d(origin, c) + d(c, dest) <= shortest +
      overrun``.
    """
    shortest = hg.hex_distance(origin, dest)
    radius = shortest + overrun
    oq, orr = origin
    cells: set[Cell] = set()
    # Enumerate the exact axial disk around the origin, then retain the O-D ellipse.
    for dq in range(-radius, radius + 1):
        dr_lo = max(-radius, -dq - radius)
        dr_hi = min(radius, -dq + radius)
        for dr in range(dr_lo, dr_hi + 1):
            cell = (oq + dq, orr + dr)
            if hg.hex_distance(origin, cell) + hg.hex_distance(cell, dest) <= radius:
                cells.add(cell)
    return cells


def _graph_max_step(
    latest_departure_step: int,
    takeoff_steps: tuple[int, ...],
    origin_lanes: tuple[hg.Lane, ...],
    max_air_hops: int,
) -> int:
    """Budget-preserving final air-state bound.

    Includes the climb and origin-lane traverse, not just the route: both advance the same
    integer clock before the first cell visit, so omitting either would let it silently consume
    the ground-delay or route budget.

    The route term must be ``max_air_hops`` -- the ceiling itself, not a re-derivation of it.
    The clock has to reach the LATEST legal departure plus that departure's longest legal route;
    anything smaller here becomes the binding constraint for late departures only, reinstating
    exactly the departure-dependent cap the ceiling exists to remove.

    Parameters
    ------------
    - latest_departure_step (int): the latest legal departure step.
    - takeoff_steps (tuple[int, ...]): per-level climb durations in steps (the max is used).
    - origin_lanes (tuple[hg.Lane, ...]): origin exit lanes; their ``.steps`` traverse cost.
    - max_air_hops (int): the route-length ceiling in hops.

    Return
    --------
    - output (int): the final air-state step bound (latest departure + climb + longest origin
      lane + max_air_hops).
    """
    return (
        latest_departure_step
        + max(takeoff_steps)
        + max((lane.steps for lane in origin_lanes), default=0)
        + max_air_hops
    )


def _segment_terminal_id(
    p0,
    p1,
    req: FlightRequest,
    origin_terminal: Terminal | None,
    dest_terminal: Terminal | None,
    cfg: SimConfig,
) -> Hashable | None:
    """Apply the reservation builder's geometry-dependent terminal tag priority.

    Parameters
    ------------
    - p0 (tuple[float, float, float]): segment start point (ENU x, y, altitude z).
    - p1 (tuple[float, float, float]): segment end point (ENU x, y, altitude z).
    - req (FlightRequest): supplies the origin/dest column centres tested for overlap.
    - origin_terminal (Terminal | None): origin terminal whose column takes tag priority.
    - dest_terminal (Terminal | None): destination terminal, checked after the origin.
    - cfg (SimConfig): supplies terminal radii and geometry.

    Return
    --------
    - output (Hashable | None): the origin terminal id if the segment overlaps its column, else
      the dest terminal id if it overlaps that column, else ``None``.
    """

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

    Parameters
    ------------
    - cells (frozenset[Cell]): the corridor cells whose outgoing lattice hops are tested.
    - walls (tuple[Volume4D, ...]): permanent wall volumes; an empty tuple forbids nothing.
    - req (FlightRequest): the flight, forwarded for terminal-tag geometry.
    - origin_terminal (Terminal | None): origin terminal, or ``None`` for a customer origin.
    - dest_terminal (Terminal | None): destination terminal, or ``None`` for a customer dest.
    - origin_lanes (tuple[hg.Lane, ...]): origin exit lanes, whose cells may take a first-arc tag.
    - dest_lanes (tuple[hg.Lane, ...]): destination exit lanes, whose cells may take a last-arc tag.
    - cfg (SimConfig): supplies lattice geometry, flight level, and segment length.

    Return
    --------
    - output (frozenset[tuple[Cell, Cell]]): directed ``(source, target)`` hops for which every
      possible builder tag assignment conflicts with a wall; empty when there are no walls.
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


def _static_hop_allowed_roles(
    source: Cell,
    target: Cell,
    walls_with_bounds: tuple[WallBound, ...] | _WallSpatialIndex,
    req: FlightRequest,
    origin_terminal: Terminal | None,
    dest_terminal: Terminal | None,
    origin_lane_cells: frozenset[Cell],
    dest_lane_cells: frozenset[Cell],
    cfg: SimConfig,
) -> int:
    """Return a bit mask of wall-safe internal/endpoint roles for one hop.

    Parameters
    ------------
    - source (Cell): the hop's source axial cell.
    - target (Cell): the hop's target axial cell.
    - walls_with_bounds (tuple[WallBound, ...] | _WallSpatialIndex): the permanent walls, either
      as ``(volume, aabb)`` pairs or a spatial index that narrows candidates per sub-box.
    - req (FlightRequest): the flight, forwarded for terminal-tag geometry.
    - origin_terminal (Terminal | None): origin terminal, or ``None`` for a customer origin.
    - dest_terminal (Terminal | None): destination terminal, or ``None`` for a customer dest.
    - origin_lane_cells (frozenset[Cell]): cells eligible for the origin's first-arc tag.
    - dest_lane_cells (frozenset[Cell]): cells eligible for the destination's last-arc tag.
    - cfg (SimConfig): supplies lattice geometry, flight level, and segment length.

    Return
    --------
    - output (int): the OR of the ``_ARC_*`` role bits whose builder tag assignment clears every
      wall; ``_ALL_ARC_ROLES`` when there are no walls.
    """

    if isinstance(walls_with_bounds, _WallSpatialIndex):
        all_wall_bounds = walls_with_bounds.all_bounds
    else:
        all_wall_bounds = walls_with_bounds
    if not all_wall_bounds:
        return _ALL_ARC_ROLES
    radius = hg.circumradius(cfg)
    z = cfg.flight_levels_m[0]
    source_xy = hg.hex_center(*source, radius)
    target_xy = hg.hex_center(*target, radius)
    p0 = (float(source_xy[0]), float(source_xy[1]), float(z))
    p1 = (float(target_xy[0]), float(target_xy[1]), float(z))

    # Reproduce ``build_reservation_from_corners`` scalar interpolation.  A
    # nominal edge can be a few ulps long and split into differently tagged boxes.
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    nsub = max(1, math.ceil(length / cfg.corridor_segment_len_m))
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

    # Builder endpoint overrides are path-position dependent.  Cache all four
    # exact outcomes: the union is useful to discover a possible arc, while
    # pricing consumes the bit matching its actual first/last role before any
    # label dominance can discard an otherwise feasible path.
    can_be_first = (
        cfg.fixed_exit_lanes and origin_terminal is not None and source in origin_lane_cells
    )
    can_be_last = cfg.fixed_exit_lanes and dest_terminal is not None and target in dest_lane_cells

    def role_tags(*, first: bool, last: bool) -> tuple[Hashable | None, ...]:
        tags = list(base_tags)
        if first and can_be_first:
            assert origin_terminal is not None
            tags[0] = origin_terminal.id
        if last and can_be_last:
            assert dest_terminal is not None
            # On a one-subvolume first+last hop the builder's origin override
            # wins; this mirrors ``build_reservation_from_corners`` exactly.
            if not (first and can_be_first and len(tags) == 1):
                tags[-1] = dest_terminal.id
        return tuple(tags)

    variants = (
        (_ARC_INTERNAL, role_tags(first=False, last=False)),
        (_ARC_FIRST, role_tags(first=True, last=False)),
        (_ARC_LAST, role_tags(first=False, last=True)),
        (_ARC_FIRST_LAST, role_tags(first=True, last=True)),
    )

    wall_candidates = tuple(
        (
            walls_with_bounds.candidates(base_volume.flat_aabb())
            if isinstance(walls_with_bounds, _WallSpatialIndex)
            else all_wall_bounds
        )
        for base_volume in subvolumes
    )

    def variant_conflicts(tags: tuple[Hashable | None, ...]) -> bool:
        for base_volume, tag, candidates in zip(subvolumes, tags, wall_candidates):
            hop = (
                base_volume
                if tag == base_volume.terminal_id
                else replace(base_volume, terminal_id=tag)
            )
            hop_bound = hop.flat_aabb()
            for wall, wall_bound in candidates:
                if _aabbs_overlap(hop_bound, wall_bound) and volumes_conflict(hop, wall):
                    return True
        return False

    conflict_by_tags: dict[tuple[Hashable | None, ...], bool] = {}
    allowed_roles = 0
    for role, tags in variants:
        conflicts = conflict_by_tags.get(tags)
        if conflicts is None:
            conflicts = variant_conflicts(tags)
            conflict_by_tags[tags] = conflicts
        if not conflicts:
            allowed_roles |= role
    return allowed_roles


class _LazyForbiddenHops(AbstractSet[tuple[Cell, Cell]]):
    """Generate and cache invariant outgoing spatial arcs one source at a time."""

    __slots__ = (
        "_allowed",
        "_arc_checks",
        "_blocked",
        "_cache_hits",
        "_cfg",
        "_corridor",
        "_dest_lane_cells",
        "_dest_lanes",
        "_dest_terminal",
        "_lock",
        "_origin_lane_cells",
        "_origin_lanes",
        "_origin_terminal",
        "_request",
        "_roles",
        "_walls",
        "_wall_index",
    )

    def __init__(
        self,
        corridor: AbstractSet[Cell],
        walls: tuple[Volume4D, ...],
        request: FlightRequest,
        origin_terminal: Terminal | None,
        dest_terminal: Terminal | None,
        origin_lanes: tuple[hg.Lane, ...],
        dest_lanes: tuple[hg.Lane, ...],
        cfg: SimConfig,
        wall_index: _WallSpatialIndex | None = None,
    ) -> None:
        """Store the corridor, walls, and endpoint geometry; arc verdicts stay lazy per source."""
        self._corridor = corridor
        self._walls = walls
        self._wall_index = (
            wall_index
            if wall_index is not None
            else _WallSpatialIndex(walls, max(256.0, 4.0 * cfg.corridor_width_m))
        )
        self._request = request
        self._origin_terminal = origin_terminal
        self._dest_terminal = dest_terminal
        self._origin_lanes = origin_lanes
        self._dest_lanes = dest_lanes
        self._origin_lane_cells = frozenset(lane.cell for lane in origin_lanes)
        self._dest_lane_cells = frozenset(lane.cell for lane in dest_lanes)
        self._cfg = cfg
        self._allowed: dict[Cell, tuple[Cell, ...]] = {}
        self._blocked: set[tuple[Cell, Cell]] = set()
        self._roles: dict[tuple[Cell, Cell], int] = {}
        self._arc_checks = 0
        self._cache_hits = 0
        self._lock = threading.RLock()

    def outgoing(self, source: Cell) -> tuple[Cell, ...]:
        cached = self._allowed.get(source)
        if cached is not None:
            # Deliberately unlocked.  A pricing run logged ~16M hits against 40k
            # misses, so taking the RLock here spent real wall time purely to make
            # a diagnostic counter exact.  ``_cache_hits`` is reported by
            # ``arc_cache_stats`` and asserted only for growth, never for an exact
            # value, so a lost increment under concurrent search is acceptable;
            # every counter that feeds a decision stays inside the lock below.
            self._cache_hits += 1
            return cached
        with self._lock:
            cached = self._allowed.get(source)
            if cached is not None:
                self._cache_hits += 1
                return cached
            if source not in self._corridor:
                self._allowed[source] = ()
                return ()
            sq, sr = source
            allowed: list[Cell] = []
            for dq, dr in hg.AXIAL_NEIGHBORS:
                target = sq + dq, sr + dr
                if target not in self._corridor:
                    continue
                self._arc_checks += 1
                roles = _static_hop_allowed_roles(
                    source,
                    target,
                    self._wall_index,
                    self._request,
                    self._origin_terminal,
                    self._dest_terminal,
                    self._origin_lane_cells,
                    self._dest_lane_cells,
                    self._cfg,
                )
                self._roles[(source, target)] = roles
                if not roles:
                    self._blocked.add((source, target))
                else:
                    allowed.append(target)
            cached = tuple(allowed)
            self._allowed[source] = cached
            return cached

    def role_mask(self, source: Cell, target: Cell) -> int:
        """Return the cached role bitmask for one directed arc, expanding on demand.

        A missing entry means the arc was never admissible in any role: the source
        or target sits outside the corridor, the pair is not adjacent, or expansion
        already found every role blocked.  All four cases share the ``0`` answer, so
        callers need no separate adjacency test.
        """

        self.outgoing(source)
        return self._roles.get((source, target), 0)

    def allows(
        self,
        source: Cell,
        target: Cell,
        *,
        first: bool,
        last: bool,
    ) -> bool:
        role = (
            _ARC_FIRST_LAST
            if first and last
            else _ARC_FIRST
            if first
            else _ARC_LAST
            if last
            else _ARC_INTERNAL
        )
        # ``role_mask`` subsumes a ``target not in self.outgoing(source)`` membership test that
        # would rescan a <=6-tuple on each of the two calls the pricing DP makes per arc.
        return bool(self.role_mask(source, target) & role)

    def __contains__(self, raw_hop: object) -> bool:
        if not isinstance(raw_hop, tuple) or len(raw_hop) != 2:
            return False
        source, target = raw_hop
        if not isinstance(source, tuple) or not isinstance(target, tuple):
            return False
        if hg.hex_distance(source, target) != 1:
            return False
        if source not in self._corridor or target not in self._corridor:
            return False
        return target not in self.outgoing(source)

    def _materialized_blocked(self) -> frozenset[tuple[Cell, Cell]]:
        for source in self._corridor:
            self.outgoing(source)
        with self._lock:
            return frozenset(self._blocked)

    def __iter__(self):
        return iter(self._materialized_blocked())

    def __len__(self) -> int:
        return len(self._materialized_blocked())

    @property
    def stats(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(
                {
                    "expanded_nodes": len(self._allowed),
                    "arc_checks": self._arc_checks,
                    "cache_hits": self._cache_hits,
                    "allowed_arcs": sum(len(neighbours) for neighbours in self._allowed.values()),
                    "blocked_arcs": len(self._blocked),
                }
            )

    def _wall_signature(self) -> tuple[Any, ...]:
        return tuple(
            (bounds, wall.t_start, wall.t_end, wall.terminal_id, type(wall.shape).__name__)
            for wall, bounds in self._wall_index.all_bounds
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _LazyForbiddenHops):
            same_semantics = (
                self._corridor == other._corridor
                and self._wall_signature() == other._wall_signature()
                and self._origin_terminal == other._origin_terminal
                and self._dest_terminal == other._dest_terminal
                and self._origin_lanes == other._origin_lanes
                and self._dest_lanes == other._dest_lanes
                and self._cfg == other._cfg
                and self._request.flight_id == other._request.flight_id
                and np.array_equal(self._request.origin, other._request.origin)
                and np.array_equal(self._request.dest, other._request.dest)
            )
            if same_semantics:
                return True
            return self._materialized_blocked() == other._materialized_blocked()
        if isinstance(other, AbstractSet):
            return self._materialized_blocked() == other
        return NotImplemented

    def __reduce__(self):
        # Geometry caches are deterministic, so a transported copy rebuilds them cold
        # rather than shipping them.
        return (
            type(self),
            (
                self._corridor,
                self._walls,
                self._request,
                self._origin_terminal,
                self._dest_terminal,
                self._origin_lanes,
                self._dest_lanes,
                self._cfg,
                self._wall_index,
            ),
        )


class FlightGraphInfeasible(ValueError):
    """A request cannot be represented by the configured column-generation graph."""


def build_flight_graph(
    req: FlightRequest,
    cfg: SimConfig,
    static_terms: object,
    params: ColGenParams,
) -> FlightGraph:
    """Build the frozen corridor and timing bounds for one flight.

    The domain is the hex-distance O-D ellipse, with every foreign terminal's full
    ``terminal_cells`` wall removed.  A flight's own terminal interior is also removed; only the
    canonical boundary lanes are added back, preserving A*'s fixed-lane geometry.

    Parameters
    ------------
    - req (FlightRequest): the flight to build a graph for (origin/dest and terminals).
    - cfg (SimConfig): lattice, timing, and terminal geometry; must match the static catalog.
    - static_terms (StaticTerminalCatalog | Iterable): the solve-wide static terminals, either a
      prebuilt catalog or ``(center, terminal)`` pairs to build one from.
    - params (ColGenParams): supplies ``max_air_overrun_hops``, which sizes the corridor and caps
      route length.

    Return
    --------
    - output (FlightGraph): the frozen flight domain (corridor cells, lanes, timing bounds, and
      lazy answer-neutral caches); raises on a degenerate/excluded endpoint or a config mismatch.
    """
    if cfg.n_levels != 1:
        raise NotImplementedError(
            "colgen v1 supports a single flight level; multi-level level choice is a planned extension"
        )
    validate_edge_locality(cfg)
    # A flat hop budget over the geodesic, identical for every flight (see ColGenParams, which
    # owns the default).  Read once and threaded down: it sizes the corridor below AND caps
    # route length AND denominates the horizon, and a second copy of that number would silently
    # disagree with this one the moment either moved.
    try:
        overrun = operator.index(params.max_air_overrun_hops)
    except (AttributeError, TypeError) as exc:
        raise TypeError("params.max_air_overrun_hops must be an integer") from exc
    if overrun < 0:
        raise ValueError(f"max_air_overrun_hops must be non-negative, got {overrun}")

    dt = cfg.dt_s
    radius = hg.circumradius(cfg)
    origin_cell = hg.enu_to_axial(float(req.origin[0]), float(req.origin[1]), radius)
    dest_cell = hg.enu_to_axial(float(req.dest[0]), float(req.dest[1]), radius)

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
    if origin_cell == dest_cell:
        raise FlightGraphInfeasible(
            f"flight {req.flight_id} origin and destination map to the same hex {origin_cell}; "
            "colgen requires at least one lateral hop"
        )
    origin_lanes = tuple(
        hg.terminal_lanes(req.origin, origin_terminal, cfg) if origin_terminal is not None else ()
    )
    dest_lanes = tuple(
        hg.terminal_lanes(req.dest, dest_terminal, cfg) if dest_terminal is not None else ()
    )
    own_ids = frozenset(
        terminal.id for terminal in (origin_terminal, dest_terminal) if terminal is not None
    )

    if isinstance(static_terms, StaticTerminalCatalog):
        catalog = static_terms
        if catalog._cfg != cfg:
            raise ValueError("static terminal catalog was built for a different SimConfig")
    else:
        catalog = StaticTerminalCatalog(static_terms, cfg)
    normalized_static_terms = catalog.entries
    foreign_exclusions = _ForeignTerminalCells(catalog, own_ids)

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
                raise FlightGraphInfeasible(
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
    max_air_hops = shortest_hops + overrun
    frozen_own_interiors = frozenset(own_interiors)
    static_exclusions = _CombinedCellSet(foreign_exclusions, frozen_own_interiors)
    explicit_lanes = frozenset(lane.cell for lane in (*origin_lanes, *dest_lanes))
    # See context/figures/colgen_lane_aware_corridor.png for the admissible cruise-hop bound.
    corridor = _LazyCorridorCells(
        origin_cell,
        dest_cell,
        overrun,
        foreign_exclusions,
        frozen_own_interiors,
        explicit_lanes,
        tuple(lane.cell for lane in origin_lanes)
        if origin_terminal is not None
        else (origin_cell,),
        tuple(lane.cell for lane in dest_lanes) if dest_terminal is not None else (dest_cell,),
    )

    if origin_terminal is None:
        if origin_cell not in corridor:
            raise FlightGraphInfeasible(
                f"flight {req.flight_id} customer origin {origin_cell} is excluded by "
                "terminal airspace"
            )
    elif not any(lane.cell in corridor for lane in origin_lanes):
        raise FlightGraphInfeasible(f"flight {req.flight_id} has no available origin terminal lane")
    if dest_terminal is None:
        if dest_cell not in corridor:
            raise FlightGraphInfeasible(
                f"flight {req.flight_id} customer destination {dest_cell} is excluded by "
                "terminal airspace"
            )
    elif not any(lane.cell in corridor for lane in dest_lanes):
        raise FlightGraphInfeasible(f"flight {req.flight_id} has no available destination terminal lane")

    frozen_static_walls = catalog.walls
    frozen_request = _snapshot_request(req, origin_terminal, dest_terminal)
    wall_index = catalog.wall_index
    forbidden_hops = _LazyForbiddenHops(
        corridor,
        frozen_static_walls,
        frozen_request,
        origin_terminal,
        dest_terminal,
        origin_lanes,
        dest_lanes,
        cfg,
        wall_index,
    )
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
        max_air_hops,
    )

    return FlightGraph(
        request=frozen_request,
        _cfg=cfg,
        origin_cell=origin_cell,
        dest_cell=dest_cell,
        corridor_cells=corridor,
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
        max_air_hops=max_air_hops,
        static_exclusions=static_exclusions,
        foreign_exclusions=foreign_exclusions,
        own_terminal_interiors=frozen_own_interiors,
        static_walls=frozen_static_walls,
        _wall_index=wall_index,
        forbidden_hops=forbidden_hops,
    )


def _selected_lane(lanes: tuple[hg.Lane, ...], index: int | None, endpoint: str) -> hg.Lane:
    """Return ``lanes[index]``, raising ``ValueError`` if ``index`` is missing or out of range.

    Parameters
    ------------
    - lanes (tuple[hg.Lane, ...]): the terminal's exit lanes to index into.
    - index (int | None): the chosen lane index; ``None`` or a negative/out-of-range value raises.
    - endpoint (str): endpoint label ("origin"/"dest") used in error messages.

    Return
    --------
    - output (hg.Lane): ``lanes[index]``. Raises ``ValueError`` when ``index`` is ``None``,
      negative, or beyond the lane tuple.
    """
    if index is None:
        raise ValueError(f"{endpoint}_lane_idx is required for a terminal endpoint")
    try:
        lane = lanes[operator.index(index)]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}") from exc
    if index < 0:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}")
    return lane


def _column_endpoint_steps(column, fg, cfg, arrival_step):
    """Compute endpoint windows at the actual clock, including float boundary rules."""

    windows = []
    for point, terminal, step, hops in (
        (fg.request.origin, fg.origin_terminal, column.departure_step, 0),
        (fg.request.dest, fg.dest_terminal, arrival_step, len(column.cell_path) - 1),
    ):
        t0 = step * cfg.dt_s
        t1 = t0 + cfg.hover_time_s + column_dwell_s(point, terminal, cfg, fg.levels[column.level])
        windows.append(
            terminal_claim_steps(t0, t1, cfg) if terminal is not None else
            endpoint_claim_steps(t0, t1, cfg, timing_steps=hops)
        )
    return tuple(windows)


def column_claims(
    column: Column,
    fg: FlightGraph,
    cfg: SimConfig,
    W: int | tuple[int, int] | None = None,
    *,
    _intent=None,
) -> frozenset[RowKey]:
    """Return the de-duplicated capacity rows claimed by ``column``.

    Validates the column against the graph, certifies it against the exact permanent-wall
    geometry, then collects its cell-visit and terminal/endpoint dwell rows.  The result is
    memoized on the graph's search cache.

    ``W`` is retained as a compatibility/checking argument for the formulation's window-width
    notation.  The actual offsets always come from :func:`derive_cell_window`; a scalar width alone
    cannot describe the shifted ``(-1, 0)`` footprint used when ``time_buffer_s == 0``.

    Parameters
    ------------
    - column (Column): the candidate schedule (cell path, departure step, level, lane indices).
    - fg (FlightGraph): the graph the column was priced over; supplies geometry and caches.
    - cfg (SimConfig): must equal the config the graph was built with.
    - W (int | tuple[int, int] | None): optional cell-window width or ``(lo, hi)`` offsets, checked
      against :func:`derive_cell_window`; ``None`` skips the check.
    - _intent: optional pre-built intent to reuse instead of re-translating the column.

    Return
    --------
    - output (frozenset[RowKey]): the capacity rows the column claims; raises ``ValueError`` if the
      column is invalid, out of budget, or overlaps permanent static terminal airspace.
    """
    if cfg != fg._cfg:
        raise ValueError("column claims require the SimConfig used to build the flight graph")
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
    certificate_key = (
        column.flight_id,
        column.level,
        column.origin_lane_idx,
        column.dest_lane_idx,
        path,
    )
    with fg._search_cache.lock:
        certificate = fg._search_cache.certified_claims.get(certificate_key)
        if certificate is not None:
            fg._search_cache.certified_claims.move_to_end(certificate_key)
    if certificate is not None:
        old_departure, cached_claims, old_windows = certificate
        lane_steps = (0 if column.origin_lane_idx is None else
                      fg.origin_lanes[column.origin_lane_idx].steps)
        arrival = column.departure_step + fg.takeoff_steps[column.level] + lane_steps + len(path) - 1
        if arrival > fg.max_step:
            raise ValueError(f"column arrives at step {arrival}, beyond graph maximum {fg.max_step}")
        delta = column.departure_step - old_departure
        if delta == 0:
            return cached_claims
        windows = _column_endpoint_steps(column, fg, cfg, arrival)
        relative_windows = tuple((w.start - column.departure_step, w.stop - column.departure_step)
                                 for w in windows)
        # Endpoint rounding depends on absolute floating-point timestamps. Only translate
        # the full union when its windows translate exactly; otherwise rebuild rows below.
        if relative_windows == old_windows:
            return frozenset(
                RowKey.cell(*row.cell_coord, row.level, row.step + delta)
                if row.kind == "cell" else RowKey.term(row.terminal_id, row.step + delta)
                for row in cached_claims
            )
    if len(path) < 2:
        raise ValueError("a column path must contain at least two cells and one lateral hop")
    if certificate is None and any(cell not in fg.corridor_cells for cell in path):
        invalid = next(cell for cell in path if cell not in fg.corridor_cells)
        raise ValueError(f"column cell {invalid} is outside the flight graph corridor")
    for a, b in (() if certificate is not None else zip(path, path[1:])):
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
    from .translate import column_to_intent

    intent = (_intent if _intent is not None else column_to_intent(column, fg.request, cfg)) if certificate is None else None
    if intent is not None and intent.status is not IntentStatus.ACCEPTED:
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
    if fg.static_walls and intent is not None:
        for volume in intent.volumes:
            for wall, _wall_bound in fg._wall_index.candidates(volume.flat_aabb()):
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

    windows = _column_endpoint_steps(column, fg, cfg, arrival_step)
    endpoints = (
        (fg.request.origin, fg.origin_terminal),
        (fg.request.dest, fg.dest_terminal),
    )
    for (point, terminal), endpoint_steps in zip(endpoints, windows, strict=True):
        if terminal is not None:
            claims.update(
                RowKey.term(terminal.id, row_step)
                for row_step in endpoint_steps
            )
            continue

        # Customer cylinders are untagged ledger geometry.  They span the regulated tube, hence
        # claim every flight level in every nearby cell and time row that could meet a transit.
        endpoint_cells = endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
        claims.update(
            RowKey.cell(q, r, level, row_step)
            for q, r in endpoint_cells
            for level in range(len(fg.levels))
            for row_step in endpoint_steps
        )

    result = frozenset(claims)
    with fg._search_cache.lock:
        fg._search_cache.certified_claims[certificate_key] = (
            column.departure_step, result,
            tuple((w.start - column.departure_step, w.stop - column.departure_step) for w in windows),
        )
        fg._search_cache.certified_claims.move_to_end(certificate_key)
        while len(fg._search_cache.certified_claims) > _MAX_CERTIFIED_COLUMNS:
            fg._search_cache.certified_claims.popitem(last=False)
    return result


__all__ = [
    "Cell",
    "FlightGraph",
    "FlightGraphInfeasible",
    "RowIndex",
    "RowKey",
    "StaticTerminalCatalog",
    "build_flight_graph",
    "column_claims",
]
