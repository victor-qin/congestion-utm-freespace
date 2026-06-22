"""Scenario registry — named worlds an experiment runs on (mirrors the ``planner`` package layout).

A :class:`ScenarioSpec` builds the ``(SimConfig, DemandModel)`` pair the simulator needs; ``SCENARIOS``
names canonical worlds, assembled from one module per family (``metro``, ``dallas``). ``get_scenario``
resolves a name to its spec — the scenario analogue of ``planner.get_planner``. Add a new family by
dropping in a module with its own ``SCENARIOS`` dict and merging it below.
"""

from __future__ import annotations

from . import dallas, metro
from .spec import DemandSpec, ScenarioSpec, with_overrides

SCENARIOS: dict[str, ScenarioSpec] = {**metro.SCENARIOS, **dallas.SCENARIOS}


def get_scenario(name: str) -> ScenarioSpec:
    """Resolve a scenario by name (mirrors :func:`planner.get_planner`)."""
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario: {name!r} (have {sorted(SCENARIOS)})") from None


__all__ = ["ScenarioSpec", "DemandSpec", "with_overrides", "SCENARIOS", "get_scenario"]
