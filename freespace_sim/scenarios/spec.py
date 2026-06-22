"""ScenarioSpec / DemandSpec — the frozen recipes a named world is built from.

A ``ScenarioSpec`` knows how to build the two things the simulator needs: a :class:`SimConfig`
(geometry / kinematics / horizon / planner) and a :class:`~freespace_sim.demand.DemandModel` (who
flies, from where, in what pattern). It is the config recipe; :class:`freespace_sim.scenario.Scenario`
(the time-ordered event list) is a different, lower-level thing the sim builds internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..config import SimConfig
from ..demand import DemandModel, HubRadiusDemand, HubVoronoiDemand, UniformPoissonDemand

# default Walmart/strip-mall split for the hub patterns when counts aren't given explicitly
_DEFAULT_HUB_LABELS = ("walmart_uss", "stripmall_uss")
_DEFAULT_HUB_COUNTS = (6, 20)


@dataclass(frozen=True)
class DemandSpec:
    """How demand is generated — pattern + operator (USS) structure. Builds a concrete DemandModel."""

    pattern: str = "uniform"               # "uniform" | "hub" | "hub_radius"
    uss: tuple[str, ...] = ()              # () → single "default" USS; else multi-operator labels
    hubs: tuple[int, ...] = ()            # per-USS hub counts (hub patterns; defaults if empty)
    direction: str = "delivery"            # hub pattern: "delivery" (hub→customer) | "pickup"
    # --- hub_radius extras (multi-pad hubs, radius service areas, return flights) ---
    radius_m: "float | dict[str, float]" = 3000.0   # customer radius (scalar, or per-USS dict)
    pads_per_hub: int = 1                  # terminal capacity N per hub
    return_flights: bool = True            # each delivery → a return to its origin hub
    turnaround_s: float = 120.0            # delay before the return is filed

    def _hub_labels_counts(self) -> tuple[list[str], list[int]]:
        labels = self.uss or _DEFAULT_HUB_LABELS
        counts = self.hubs or (
            _DEFAULT_HUB_COUNTS if labels == _DEFAULT_HUB_LABELS else (10,) * len(labels)
        )
        if len(counts) != len(labels):
            raise ValueError(
                f"hub counts ({len(counts)}) must match the number of USS labels ({len(labels)})")
        return list(labels), [int(c) for c in counts]

    def build(self) -> DemandModel | None:
        """Construct the DemandModel, or ``None`` to use the simulator's bare single-USS default."""
        if self.pattern == "hub":
            labels, counts = self._hub_labels_counts()
            return HubVoronoiDemand(n_hubs_per_uss=dict(zip(labels, counts)), direction=self.direction)
        if self.pattern == "hub_radius":
            labels, counts = self._hub_labels_counts()
            return HubRadiusDemand(
                n_hubs_per_uss=dict(zip(labels, counts)),
                radius_m=self.radius_m, pads_per_hub=self.pads_per_hub,
                return_flights=self.return_flights, turnaround_s=self.turnaround_s,
            )
        if self.pattern != "uniform":
            raise ValueError(
                f"unknown demand pattern {self.pattern!r} (want 'uniform' | 'hub' | 'hub_radius')")
        if self.uss:
            return UniformPoissonDemand(uss_ids=tuple(self.uss))
        return None   # bare default: single "default" USS, uniform O/D


@dataclass(frozen=True)
class ScenarioSpec:
    """A named world: region + horizon + demand rate + planner + demand pattern.

    ``config()`` and ``demand_model()`` are the two builders the execute step calls. Override any
    field with :func:`with_overrides` (a thin ``dataclasses.replace``) — that's how CLI flags layer
    on top of a registry entry without mutating it.
    """

    name: str
    region_m: tuple[float, float] = (8000.0, 8000.0)
    horizon_s: float = 3600.0
    lam_per_hour: float = 600.0
    seed: int = 0
    planner: str | None = None             # None → SimConfig's default planner
    demand: DemandSpec = field(default_factory=DemandSpec)

    def config(self) -> SimConfig:
        """The override layer over SimConfig defaults (never edits config.py)."""
        return SimConfig(
            region_size_m=(float(self.region_m[0]), float(self.region_m[1])),
            lam_per_hour=self.lam_per_hour,
            horizon_s=self.horizon_s,
            seed=self.seed,
            **({"planner": self.planner} if self.planner else {}),
        )

    def demand_model(self) -> DemandModel | None:
        return self.demand.build()


def with_overrides(spec: ScenarioSpec, *, demand_overrides: dict | None = None, **overrides) -> ScenarioSpec:
    """Return a copy of ``spec`` with top-level fields and/or DemandSpec fields replaced."""
    if demand_overrides:
        overrides["demand"] = replace(spec.demand, **demand_overrides)
    return replace(spec, **overrides)
