"""Cost-aware Safe Interval Path Planning on the shared hex lattice.

``planner`` contains :class:`SIPPPlanner` and its pure-Python reference machinery; ``kernel`` is
the compiled search loop; ``window`` derives per-plan safe intervals from A*'s claim arena.

The planner and reference index are re-exported so existing consumers can continue importing them
from :mod:`freespace_sim.planner.sipp` while the implementation remains split by responsibility.
"""

from __future__ import annotations

from .planner import SIPPPlanner, SafeIntervalIndex

__all__ = ["SIPPPlanner", "SafeIntervalIndex"]
