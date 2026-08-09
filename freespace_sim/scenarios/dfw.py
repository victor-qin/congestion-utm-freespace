"""DFW real-geography scenarios — geo twins of the ``density_*`` family.

Each ``dfw_*`` scenario is its ``density_*`` parent with **identical** numbers (region intent, hub
counts, hub sizes, per-USS demand rates, timing) but **real** hub locations and destinations: wing/
zipline hubs sampled from Overture retail POIs weighted by Census tract population density, Amazon hubs
at real facility coordinates, and customers drawn by tract population density (see
:class:`.demand_dfw.DfwGeoDemand`). The one deliberate difference is the **region**: the full ~192×147 km
metroplex frame instead of the compact density 60×30 km window, so the real metro-wide geography (incl.
all Amazon last-mile sites) fits.

The twins are DERIVED from :data:`density.SCENARIOS` — one geo transform per parent — so the two
families can never drift, and the artifacts under ``freespace_sim/data/dfw/`` (regenerate with
``analysis/prep_dfw.py``) supply the geography.
"""

from __future__ import annotations

from dataclasses import replace

from ..geo import project_lonlat_to_enu, region_size_for_frame
from . import density
from .demand_dfw import DEFAULT_FIXED_TYPES, DEFAULT_HUB_CATEGORIES
from .density import AMAZON_USS, WING_ZIPLINE_USS
from .spec import ScenarioSpec

# Full DFW metroplex frame (minlon, maxlon, minlat, maxlat) — the SINGLE definition of the frame;
# analysis/prep_dfw.py imports it to clip the baked artifacts to exactly this box. The region IS this
# box: DfwGeoDemand projects real lon/lat into it about the frame centre, and region_size_for_frame
# inverts geo.project_lonlat_to_enu so the four corners land exactly on [0, w] × [0, h].
DFW_FRAME = (-97.9767, -95.9240, 32.1788, 33.5030)
DFW_REGION_CENTER_LATLON = ((DFW_FRAME[2] + DFW_FRAME[3]) / 2.0, (DFW_FRAME[0] + DFW_FRAME[1]) / 2.0)
DFW_REGION_M = region_size_for_frame(*DFW_FRAME)

# DFW International Airport is a permanent NO-FLY ZONE — drones do not overfly a major Class B airport, and
# it sits near the centre of the metroplex frame. Its reference point (lon, lat) is projected into the
# region like every other real coordinate; a 4 km radius covers the runway/terminal complex (tiny against
# the ~192 km-wide region, but enough to force every cruise path AROUND the field instead of across it). The
# zone rides SimConfig.keepout_zones → the static-wall rail (see sim._keepout_terminals): the A* occupancy
# routes around it, the ledger + verify wall it off, and DfwGeoDemand keeps hubs/customers out of it.
DFW_AIRPORT_LONLAT = (-97.0380, 32.8968)
DFW_AIRPORT_KEEPOUT_RADIUS_M = 4000.0
_apt = project_lonlat_to_enu(DFW_AIRPORT_LONLAT[0], DFW_AIRPORT_LONLAT[1],
                             DFW_REGION_CENTER_LATLON[0], DFW_REGION_CENTER_LATLON[1],
                             DFW_REGION_M[0], DFW_REGION_M[1])
DFW_KEEPOUT_ZONES = ((float(_apt[0]), float(_apt[1]), DFW_AIRPORT_KEEPOUT_RADIUS_M),)


def _geo_twin(spec: ScenarioSpec) -> ScenarioSpec:
    """A ``density_*`` spec → its ``dfw_*`` twin: same numbers, wide frame, real-geography demand."""
    # The siting rules are recorded EXPLICITLY rather than left () for DemandSpec.build() to fill from
    # DfwGeoDemand's defaults. A spec that leans on those defaults does not pin its own world: editing
    # DEFAULT_HUB_CATEGORIES would silently replay every archived dfw_* run against a different hub
    # pool, with no schema_version bump to catch it — the reinterpretation that guard exists to stop.
    demand = replace(
        spec.demand, pattern="dfw_geo", geo_dataset="dfw",
        sampled_hub_uss=(WING_ZIPLINE_USS,),
        fixed_hub_uss=(AMAZON_USS,) if AMAZON_USS in spec.demand.uss else (),
        hub_categories=DEFAULT_HUB_CATEGORIES, fixed_hub_types=DEFAULT_FIXED_TYPES,
    )
    return replace(
        spec, name="dfw_" + spec.name.removeprefix("density_"),
        description="Real-geography (Overture retail + Census density) twin of " + spec.name + ".",
        region_m=DFW_REGION_M, region_center_latlon=DFW_REGION_CENTER_LATLON,
        keepout_zones=DFW_KEEPOUT_ZONES, demand=demand,
    )


SCENARIOS: dict[str, ScenarioSpec] = {s.name: s for s in map(_geo_twin, density.SCENARIOS.values())}
