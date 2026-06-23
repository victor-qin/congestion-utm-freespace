"""Metro scenarios — uniform-demand baselines (no geographic structure)."""

from __future__ import annotations

from .spec import DemandSpec, ScenarioSpec

SCENARIOS: dict[str, ScenarioSpec] = {
    # single-operator uniform baseline
    "metro_uniform": ScenarioSpec("metro_uniform", region_m=(5000.0, 5000.0)),
    # two operators, uniform O/D — exercises cross-USS deconfliction without geographic structure
    "metro_2uss": ScenarioSpec(
        "metro_2uss", region_m=(5000.0, 5000.0),
        demand=DemandSpec(pattern="uniform", uss=("uss_a", "uss_b")),
    ),
}
