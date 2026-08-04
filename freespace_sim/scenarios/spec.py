"""ScenarioSpec / DemandSpec — the frozen recipes a named world is built from.

A ``ScenarioSpec`` knows how to build the two things the simulator needs: a :class:`SimConfig`
(geometry / kinematics / horizon / planner) and a :class:`~freespace_sim.demand.DemandModel` (who
flies, from where, in what pattern). It is the config recipe; :class:`freespace_sim.scenario.Scenario`
(the time-ordered event list) is a different, lower-level thing the sim builds internally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from ..config import SimConfig
from ..demand import DemandModel, HubRadiusDemand, HubVoronoiDemand, UniformPoissonDemand
from .demand_dfw import DfwGeoDemand

# default Walmart/strip-mall split for the hub patterns when counts aren't given explicitly
_DEFAULT_HUB_LABELS = ("walmart_uss", "stripmall_uss")
_DEFAULT_HUB_COUNTS = (6, 20)

# Bumped when the persisted scenario_spec.json layout changes incompatibly. Stamped by
# ScenarioSpec.to_json_dict and checked by from_json_dict so an archived run cannot be silently
# reinterpreted under a schema it was not written with. v2 adds the dfw_geo demand fields (geo_dataset,
# sampled/fixed hub operators, category/type filters) — a pre-v2 reader would silently drop them.
# v3 adds ScenarioSpec.keepout_zones (permanent no-fly cylinders) — a pre-v3 reader would silently drop the
# zones, replaying a run that routed AROUND an airport as if the airspace were open.
_SPEC_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class DemandSpec:
    """How demand is generated — pattern + operator (USS) structure. Builds a concrete DemandModel."""

    pattern: str = "uniform"               # "uniform" | "hub" | "hub_radius"
    uss: tuple[str, ...] = ()              # () → single "default" USS; else multi-operator labels
    hubs: tuple[int, ...] = ()            # per-USS hub counts (hub patterns; defaults if empty)
    direction: str = "delivery"            # hub pattern: "delivery" (hub→customer) | "pickup"
    # --- hub_radius extras (multi-pad hubs, radius service areas, return flights) ---
    radius_m: "float | dict[str, float]" = 3000.0   # customer demand radius (scalar, or per-USS dict)
    pads_per_hub: "int | dict[str, int]" = 1   # terminal capacity N per hub (scalar, or per-USS dict)
    terminal_radius_m: "float | dict[str, float] | None" = None   # column size; None → hover footprint
    corridor_overlap_m: "float | None" = None        # exit-lane overlap into column; None/0 → flush at edge
    return_flights: bool = True            # each delivery → a return to its origin hub
    turnaround_s: float = 0.0              # delay before the return is filed (0 ⇒ on est. arrival)
    uss_share: "dict[str, float] | None" = None      # demand split across USSs (None ⇒ equal weight)
    # hub_radius: per-USS delivery Poisson rate (/hr). When set, REPLACES cfg.lam_per_hour × uss_share —
    # each USS is its own independent stream. None ⇒ the global-λ path.
    lam_per_uss: "dict[str, float] | None" = None
    # hub_radius: per-USS desired-departure lead as ``(mean_s, std_s)``. Absent USSs (or None) depart on
    # filing. Legacy returns draw a second lead; paired returns are filed with the outbound and do not.
    departure_offset_s: "dict[str, tuple[float, float]] | None" = None
    # "request" samples filings then adds the lead; "departure" samples outbound desired departures
    # over the common demand window, subtracts the lead, then shifts the full clock nonnegative.
    timing_mode: str = "request"
    # Strategic round-trip filing: return shares the outbound filing time and requests departure after
    # the outbound's nominal arrival. False preserves the legacy independently-filed return behavior.
    paired_return_request: bool = False
    min_hub_gap_m: float = 100.0           # hub_radius: clearance between terminal-airspace edges (no overlap)
    # --- dfw_geo extras: real-geography hub placement + census-density destinations. () / None keep the
    # DfwGeoDemand model defaults; the dfw_* scenarios set sampled/fixed operators explicitly. ---
    geo_dataset: "str | None" = None       # artifact dir under freespace_sim/data/ (None → "dfw")
    sampled_hub_uss: tuple[str, ...] = ()  # operators whose hubs are density-sampled from retail POIs
    fixed_hub_uss: tuple[str, ...] = ()    # operators whose hubs are fixed real facilities
    fixed_hub_types: tuple[str, ...] = ()  # facility types eligible as fixed hubs ((): model default)
    hub_categories: tuple[str, ...] = ()   # retail categories eligible as sampled hubs ((): model default)
    use_all_fixed_hubs: bool = False       # True → use every listed fixed facility (ignore the hub count)

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
            return HubVoronoiDemand(n_hubs_per_uss=dict(zip(labels, counts)), direction=self.direction,
                                    uss_share=self.uss_share)
        if self.pattern in ("hub_radius", "dfw_geo"):
            labels, counts = self._hub_labels_counts()
            hub_kw = dict(
                n_hubs_per_uss=dict(zip(labels, counts)),
                radius_m=self.radius_m, pads_per_hub=self.pads_per_hub,
                terminal_radius_m=self.terminal_radius_m, corridor_overlap_m=self.corridor_overlap_m,
                return_flights=self.return_flights, turnaround_s=self.turnaround_s, uss_share=self.uss_share,
                lam_per_uss=self.lam_per_uss, departure_offset_s=self.departure_offset_s,
                timing_mode=self.timing_mode, paired_return_request=self.paired_return_request,
                min_hub_gap_m=self.min_hub_gap_m,
            )
            if self.pattern == "hub_radius":
                return HubRadiusDemand(**hub_kw)
            # dfw_geo: same hub_radius knobs + real-geography placement ((): keep DfwGeoDemand defaults)
            geo_kw = {k: v for k, v in (("fixed_hub_types", self.fixed_hub_types),
                                        ("hub_categories", self.hub_categories)) if v}
            return DfwGeoDemand(**hub_kw, dataset=self.geo_dataset or "dfw",
                                sampled_hub_uss=self.sampled_hub_uss, fixed_hub_uss=self.fixed_hub_uss,
                                use_all_fixed_hubs=self.use_all_fixed_hubs, **geo_kw)
        if self.pattern != "uniform":
            raise ValueError(
                f"unknown demand pattern {self.pattern!r} "
                "(want 'uniform' | 'hub' | 'hub_radius' | 'dfw_geo')")
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
    description: str = ""
    region_m: tuple[float, float] = (8000.0, 8000.0)
    # ENU projection anchor (lat, lon) → region centre. None keeps SimConfig's DFW default; the real-
    # geography dfw_* scenarios set it to their frame centre so hubs/tracts project into the region box.
    region_center_latlon: "tuple[float, float] | None" = None
    horizon_s: float = 3600.0
    demand_duration_s: float | None = None
    lam_per_hour: float = 600.0
    seed: int = 0
    planner: str | None = None             # None → SimConfig's default planner
    fixed_exit_lanes: bool | None = None    # None → SimConfig's default (issue #18: on); set to override
    terminal_airspace_always_active: bool | None = None   # None → SimConfig default (off)
    # flight-level ladder override (None → SimConfig default (30,70,110) multi-level). Pin a scenario to
    # one A* plane with flight_levels_m=(z,); widen by listing more levels. This is the single altitude
    # knob: SimConfig derives the single-plane samplers' cruise + z-band from it, so a scenario needn't
    # (and can't) set cruise_level_m / z_min_m / z_max_m separately.
    flight_levels_m: "tuple[float, ...] | None" = None
    # Vertical protection-box height override (None → SimConfig default, 30 m). Each flight level is the
    # centre of a corridor_height_m-tall box, so SimConfig requires adjacent levels to be spaced strictly
    # MORE than this apart (else the boxes overlap in z). Lower it below the level gap to stack levels
    # closer than 30 m — e.g. 14 m tubes let the (80, 95, 110) ladder sit 15 m apart and stay FCL-disjoint.
    corridor_height_m: "float | None" = None
    # Permanent no-fly cylinders every flight routes around (SimConfig.keepout_zones): ``(cx, cy, radius_m)``
    # ENU triples. () → none. The dfw_* twins set a zone over DFW airport; see scenarios/dfw.py.
    keepout_zones: tuple = ()
    demand: DemandSpec = field(default_factory=DemandSpec)

    def config(self) -> SimConfig:
        """The override layer over SimConfig defaults (never edits config.py).

        ``demand_duration_s`` is forwarded UNCLAMPED, so overriding ``horizon_s`` below it raises
        from :meth:`SimConfig.__post_init__`. That error is deliberate: clamping the demand window
        to a shrunken horizon looks like it makes ``--horizon 600`` work, but the departure lead
        (``departure_offset_s``, up to N(1800, 300) for amazon_uss) is unaffected by the clamp, so
        every generated departure lands past the horizon and the "quick smoke test" silently becomes
        a meaningless run on the box-guard fallback path. Shrink a density_* scenario by overriding
        BOTH knobs — ``with_overrides(spec, horizon_s=..., demand_duration_s=...)``.
        """
        return SimConfig(
            region_size_m=(float(self.region_m[0]), float(self.region_m[1])),
            lam_per_hour=self.lam_per_hour,
            horizon_s=self.horizon_s,
            demand_duration_s=self.demand_duration_s,
            seed=self.seed,
            **({"planner": self.planner} if self.planner else {}),
            **({"fixed_exit_lanes": self.fixed_exit_lanes} if self.fixed_exit_lanes is not None else {}),
            **({"terminal_airspace_always_active": self.terminal_airspace_always_active}
               if self.terminal_airspace_always_active is not None else {}),
            **({"flight_levels_m": self.flight_levels_m} if self.flight_levels_m is not None else {}),
            **({"corridor_height_m": self.corridor_height_m} if self.corridor_height_m is not None else {}),
            **({"region_center_latlon": (float(self.region_center_latlon[0]),
                                         float(self.region_center_latlon[1]))}
               if self.region_center_latlon is not None else {}),
            keepout_zones=self.keepout_zones,
        )

    def demand_model(self) -> DemandModel | None:
        return self.demand.build()

    def to_json_dict(self) -> dict:
        """A JSON-safe dict that :meth:`from_json_dict` can turn back into an equal ``ScenarioSpec``.

        ``dataclasses.asdict`` alone does NOT round-trip: JSON has no tuple, so ``region_m`` /
        ``flight_levels_m`` / ``uss`` / ``hubs`` / the ``departure_offset_s`` pairs all come back as
        lists, and the nested ``demand`` comes back a plain dict whose ``.build()`` raises
        ``AttributeError``. A run folder that cannot rebuild its own recipe is not self-contained.
        """
        payload = asdict(self)
        payload["schema_version"] = _SPEC_SCHEMA_VERSION
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict) -> "ScenarioSpec":
        """Rebuild a ``ScenarioSpec`` from :meth:`to_json_dict` output (or a bare ``asdict``).

        Unknown keys are dropped rather than raising, matching ``runs.load_run``'s whitelist so an
        archived run stays loadable after a field is renamed — including pre-round-trip folders whose
        JSON was a raw ``asdict`` with no ``schema_version``. A future or non-numeric schema version is
        a hard ``ValueError`` (not a ``TypeError``): silently reinterpreting an old recipe is how you
        replay under the wrong world.
        """
        payload = dict(payload)
        version = payload.pop("schema_version", _SPEC_SCHEMA_VERSION)
        if not isinstance(version, (int, float)) or isinstance(version, bool) or version > _SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"scenario_spec schema_version {version!r} is not readable by this code "
                f"(understands integer versions <= {_SPEC_SCHEMA_VERSION}) — upgrade freespace_sim")

        demand_payload = dict(payload.pop("demand", None) or {})
        demand_fields = DemandSpec.__dataclass_fields__
        demand_kw = {k: v for k, v in demand_payload.items() if k in demand_fields}
        for name in ("uss", "hubs", "sampled_hub_uss", "fixed_hub_uss",
                     "fixed_hub_types", "hub_categories"):
            if demand_kw.get(name) is not None:
                demand_kw[name] = tuple(demand_kw[name])
        if demand_kw.get("departure_offset_s"):
            demand_kw["departure_offset_s"] = {
                k: (float(v[0]), float(v[1])) for k, v in demand_kw["departure_offset_s"].items()}

        spec_fields = cls.__dataclass_fields__
        kw = {k: v for k, v in payload.items() if k in spec_fields}
        if kw.get("region_m") is not None:
            kw["region_m"] = (float(kw["region_m"][0]), float(kw["region_m"][1]))
        if kw.get("region_center_latlon") is not None:
            kw["region_center_latlon"] = (
                float(kw["region_center_latlon"][0]), float(kw["region_center_latlon"][1]))
        if kw.get("flight_levels_m") is not None:
            kw["flight_levels_m"] = tuple(float(z) for z in kw["flight_levels_m"])
        if kw.get("keepout_zones") is not None:          # JSON lists → tuple of (cx, cy, radius) float triples;
            kw["keepout_zones"] = tuple(                  # `is not None` so an empty [] coerces back to () too —
                (float(cx), float(cy), float(r)) for cx, cy, r in kw["keepout_zones"])  # else [] != () breaks EVERY scenario
        return cls(**kw, demand=DemandSpec(**demand_kw))


def with_overrides(spec: ScenarioSpec, *, demand_overrides: dict | None = None, **overrides) -> ScenarioSpec:
    """Return a copy of ``spec`` with top-level fields and/or DemandSpec fields replaced."""
    if demand_overrides:
        overrides["demand"] = replace(spec.demand, **demand_overrides)
    return replace(spec, **overrides)
