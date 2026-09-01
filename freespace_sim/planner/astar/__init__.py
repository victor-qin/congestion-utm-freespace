"""Space-time A* on the hex lattice — the reference search, its compiled twin, and the occupancy.

``planner`` holds :class:`AStarPlanner`; ``kernel`` is the ``@njit`` hot loop that must reproduce
it byte-for-byte; ``occupancy`` and ``compiled_hex_occupancy`` are the same ledger image in the two
shapes the two searches need; ``window`` is the per-plan dense bitmap ``kernel`` answers occupancy
probes from when the host supplies one. Each module's own docstring is authoritative on its
contents — deliberately not restated here, so this one cannot drift out of date.

Three couplings are easy to get wrong from the file names alone:

  * ``_packed`` is NOT compiled-only. ``planner`` imports seven names from it for the pure-Python
    reference's own g-hash, so its layout constants are pinned by the parity suite on BOTH sides —
    editing them without re-pinning the oracle is the trap the array-of-structs work left behind.
  * ``compiled_hex_occupancy`` is NOT compiled-only either. Alongside :class:`CompiledHexOccupancy`
    it owns ``search_horizon`` / ``hover_tail_steps`` / ``schedulable_horizon_steps``, plain
    cfg-to-step arithmetic and the ONE definition (issue #5) shared by the reference path, the
    compiled path, and the ledger. Gating this module on numba would silently un-bound all three.
  * ``window`` is ``@njit`` too, so ``kernel`` is no longer this family's only compiled module — and
    unlike ``kernel`` it is imported at MODULE level by ``planner`` (for ``empty_wbox`` /
    ``window_bounds`` / ``disable``, which are plain Python). ``AStarPlanner.__init__``'s numba
    fallback is an ImportError guard around ``.kernel`` only, so it cannot catch that import;
    ``window`` therefore carries its own guard, without which a numba-less install stops importing
    the package instead of degrading to the reference search. Pinned by
    ``test_window_module_imports_without_numba``.

What stayed at ``planner/`` level, owned by neither family: ``hexgrid`` (the lattice primitives —
colgen, ``milp``, ``demand``, ``volumes``, ``viz_html`` and ``parallel`` all consume it, and it
also owns ``lattice_overhead_m``, the geometry-vs-traffic split both planners report),
``terminal_capacity`` (``milp``, ``shortcut``, ``parallel``), and ``shortcut`` — a refiner that is
planner-agnostic in its imports and wraps the MILP, not an A*, in the ``astar_milp_shortcut`` arm.
With ``lattice_overhead_m`` hoisted, colgen imports nothing from this package.

Nothing is re-exported lazily. ``AStarPlanner`` is the surface every consumer wants and it pulls
both occupancy modules at import anyway; the cost is that importing a leaf here (``occupancy``,
``_packed``) also builds the planner — about ten extra modules. numba is the one import genuinely
deferred, and its guard lives in ``AStarPlanner.__init__``, which falls back to the reference.

``AStarPlanner`` is re-exported so ``from freespace_sim.planner.astar import AStarPlanner`` reads
exactly as it did when this package was a single module.
"""

from __future__ import annotations

from .compiled_hex_occupancy import CompiledHexOccupancy
from .occupancy import HexOccupancyService
from .planner import AStarPlanner

__all__ = ["AStarPlanner", "CompiledHexOccupancy", "HexOccupancyService"]
