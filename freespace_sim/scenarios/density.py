"""Explicit FAA-filing and far-future density-study scenarios."""

from __future__ import annotations

from .spec import DemandSpec, ScenarioSpec

WING_ZIPLINE_USS = "wing_zipline_uss"
AMAZON_USS = "amazon_uss"

REGION_M = (60_000.0, 30_000.0)
SIM_HORIZON_S = 2.0 * 60.0 * 60.0
DEMAND_DURATION_S = 30.0 * 60.0
CRUISE_ALTITUDE_M = 100.0

PADS_PER_HUB = 40
TERMINAL_RADIUS_M = 180.0
WING_ZIPLINE_RADIUS_M = 16_000.0
AMAZON_RADIUS_M = 12_000.0
WING_ZIPLINE_LEAD_S = (8.0 * 60.0, 1.5 * 60.0)
AMAZON_LEAD_S = (30.0 * 60.0, 5.0 * 60.0)

# Vertically-stacked variant: three cruise levels 15 m apart. The 15 m gap is below the default 30 m
# corridor_height_m, so the ladder ships its own 14 m box (±7 m tubes, a 1 m dead-band between adjacent
# levels) — the tightest tidy height that keeps SimConfig's "gap > corridor_height_m" invariant (a 15 m
# box would exactly touch and be rejected). Top box 110 + 7 = 117 m stays under the 125 m ceiling.
STACKED_LEVELS_M = (80.0, 95.0, 110.0)
STACKED_CORRIDOR_HEIGHT_M = 14.0


def _density_scenario(
    name: str,
    description: str,
    *,
    wing_hubs: int,
    wing_rate_per_hub: float,
    amazon_hubs: int | None = None,
    amazon_rate_per_hub: float | None = None,
    flight_levels_m: tuple[float, ...] = (CRUISE_ALTITUDE_M,),
    corridor_height_m: float | None = None,
) -> ScenarioSpec:
    """Build one density recipe, preserving Wing/Zipline-first USS and hub ordering.

    ``flight_levels_m`` defaults to the single 100 m cruise plane; pass a wider ladder (plus a matching
    ``corridor_height_m`` if the levels sit closer than 30 m apart) to study vertically-stacked traffic.
    ``corridor_height_m=None`` keeps SimConfig's 30 m default.
    """
    if (amazon_hubs is None) != (amazon_rate_per_hub is None):
        raise ValueError("amazon_hubs and amazon_rate_per_hub must be supplied together")

    uss = [WING_ZIPLINE_USS]
    hubs = [wing_hubs]
    radius_m = {WING_ZIPLINE_USS: WING_ZIPLINE_RADIUS_M}
    pads_per_hub = {WING_ZIPLINE_USS: PADS_PER_HUB}
    terminal_radius_m = {WING_ZIPLINE_USS: TERMINAL_RADIUS_M}
    lam_per_uss = {WING_ZIPLINE_USS: round(wing_hubs * wing_rate_per_hub, 2)}
    departure_offset_s = {WING_ZIPLINE_USS: WING_ZIPLINE_LEAD_S}

    if amazon_hubs is not None and amazon_rate_per_hub is not None:
        uss.append(AMAZON_USS)
        hubs.append(amazon_hubs)
        radius_m[AMAZON_USS] = AMAZON_RADIUS_M
        pads_per_hub[AMAZON_USS] = PADS_PER_HUB
        terminal_radius_m[AMAZON_USS] = TERMINAL_RADIUS_M
        lam_per_uss[AMAZON_USS] = round(amazon_hubs * amazon_rate_per_hub, 2)
        departure_offset_s[AMAZON_USS] = AMAZON_LEAD_S

    return ScenarioSpec(
        name=name,
        description=description,
        region_m=REGION_M,
        horizon_s=SIM_HORIZON_S,
        demand_duration_s=DEMAND_DURATION_S,
        lam_per_hour=round(sum(lam_per_uss.values()), 2),
        fixed_exit_lanes=True,
        terminal_airspace_always_active=True,
        flight_levels_m=flight_levels_m,
        corridor_height_m=corridor_height_m,
        demand=DemandSpec(
            pattern="hub_radius",
            uss=tuple(uss),
            hubs=tuple(hubs),
            radius_m=radius_m,
            pads_per_hub=pads_per_hub,
            terminal_radius_m=terminal_radius_m,
            return_flights=True,
            turnaround_s=0.0,
            lam_per_uss=lam_per_uss,
            departure_offset_s=departure_offset_s,
            timing_mode="departure",
            paired_return_request=True,
        ),
    )


SCENARIOS: dict[str, ScenarioSpec] = {
    "density_faa_wing_zipline": _density_scenario(
        "density_faa_wing_zipline",
        "FAA-filing density: 182 Wing/Zipline-type hubs and paired delivery returns.",
        wing_hubs=182,
        wing_rate_per_hub=26.67,
    ),
    "density_future_wing_zipline": _density_scenario(
        "density_future_wing_zipline",
        "Far-future density: 476 Wing/Zipline-type hubs and paired delivery returns.",
        wing_hubs=476,
        wing_rate_per_hub=57.4,
    ),
    "density_faa_wing_zipline_amazon": _density_scenario(
        "density_faa_wing_zipline_amazon",
        "FAA-filing density with separate Wing/Zipline and Amazon USS traffic.",
        wing_hubs=182,
        wing_rate_per_hub=26.67,
        amazon_hubs=7,
        amazon_rate_per_hub=66.67,
    ),
    "density_future_wing_zipline_amazon": _density_scenario(
        "density_future_wing_zipline_amazon",
        "Far-future density with separate Wing/Zipline and Amazon USS traffic.",
        wing_hubs=476,
        wing_rate_per_hub=57.4,
        amazon_hubs=14,
        amazon_rate_per_hub=157.0,
    ),
    "density_faa_wing_zipline_3lvl": _density_scenario(
        "density_faa_wing_zipline_3lvl",
        "FAA-filing density on three cruise levels 80/95/110 m (15 m separation, ±7 m tubes).",
        wing_hubs=182,
        wing_rate_per_hub=26.67,
        flight_levels_m=STACKED_LEVELS_M,
        corridor_height_m=STACKED_CORRIDOR_HEIGHT_M,
    ),
    "density_faa_wing_zipline_amazon_3lvl": _density_scenario(
        "density_faa_wing_zipline_amazon_3lvl",
        "FAA-filing density with separate Wing/Zipline and Amazon USS traffic on three cruise levels 80/95/110 m (15 m separation, ±7 m tubes).",
        wing_hubs=182,
        wing_rate_per_hub=26.67,
        amazon_hubs=7,
        amazon_rate_per_hub=66.67,
        flight_levels_m=STACKED_LEVELS_M,
        corridor_height_m=STACKED_CORRIDOR_HEIGHT_M,
    ),
    "density_future_wing_zipline_3lvl": _density_scenario(
        "density_future_wing_zipline_3lvl",
        "Far-future density on three cruise levels 80/95/110 m (15 m separation, ±7 m tubes).",
        wing_hubs=476,
        wing_rate_per_hub=57.4,
        flight_levels_m=STACKED_LEVELS_M,
        corridor_height_m=STACKED_CORRIDOR_HEIGHT_M,
    ),
    "density_future_wing_zipline_amazon_3lvl": _density_scenario(
        "density_future_wing_zipline_amazon_3lvl",
        "Far-future density with separate Wing/Zipline and Amazon USS traffic on three cruise levels 80/95/110 m (15 m separation, ±7 m tubes).",
        wing_hubs=476,
        wing_rate_per_hub=57.4,
        amazon_hubs=14,
        amazon_rate_per_hub=157.0,
        flight_levels_m=STACKED_LEVELS_M,
        corridor_height_m=STACKED_CORRIDOR_HEIGHT_M,
    ),
}
