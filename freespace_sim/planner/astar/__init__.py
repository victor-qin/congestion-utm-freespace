"""Space-time A* on the hex lattice — the planner, its compiled twin, and their occupancy images.

The family is five modules that only make sense together:

  ``planner``                 :class:`AStarPlanner` — the search, the four cost levers, the commit
                              geometry. Also the reference oracle the kernel is pinned byte-for-byte
                              against, so it must stay importable without numba.
  ``kernel``                  the ``@njit`` hot loop (``while pq`` + ``_edges`` + ``is_blocked``).
  ``occupancy``               :class:`HexOccupancyService` — incremental hex rasterization of the ledger.
  ``compiled_hex_occupancy``  :class:`CompiledHexOccupancy` — the same image as flat interval pools,
                              the only form the kernel can read in O(1).
  ``_packed``                 the array-of-structs record layout the kernel and the compiled occupancy
                              share.

What deliberately stayed OUT, at ``planner/`` level: ``hexgrid`` and ``terminal_capacity`` (lattice and
pad-capacity primitives that colgen, the MILP, and ``demand`` also consume — the same reason ``colgen/``
does not own them) and ``shortcut`` (a planner-agnostic refiner; it wraps the MILP too, in the
``astar_milp_shortcut`` sandwich).

Unlike ``colgen``, nothing here is re-exported lazily: ``planner`` already imports both occupancy
modules at module level, so a ``__getattr__`` would defer nothing. The one genuinely lazy import is
numba, and that guard lives where it belongs — inside ``AStarPlanner``, which falls back to the
reference search when the kernel will not compile.

``AStarPlanner`` is re-exported so ``from freespace_sim.planner.astar import AStarPlanner`` reads
exactly as it did when this package was a single module.
"""

from __future__ import annotations

from .compiled_hex_occupancy import CompiledHexOccupancy
from .occupancy import HexOccupancyService
from .planner import AStarPlanner

__all__ = ["AStarPlanner", "CompiledHexOccupancy", "HexOccupancyService"]
