"""Configuration for the column-generation planner.

Phase 1 only consumes the spatial corridor slack.  Solver and batch-integration
parameters are deliberately added in their respective reviewed phases.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ColGenParams:
    """Phase-1 parameters for each flight's pruned hex-lattice network."""

    detour_slack_hops: int = 12

    def __post_init__(self) -> None:
        try:
            slack = operator.index(self.detour_slack_hops)
        except TypeError as exc:
            raise TypeError("detour_slack_hops must be an integer") from exc
        if slack < 0:
            raise ValueError("detour_slack_hops must be non-negative")
        object.__setattr__(self, "detour_slack_hops", slack)
