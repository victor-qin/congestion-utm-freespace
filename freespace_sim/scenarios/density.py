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

# --- scheduling-lead arms: does filing further ahead buy FCFS priority? ---------------------------
# FCFS order is (t_request, flight_id) and timing_mode="departure" derives t_request = t_departure -
# lead, so a longer lead moves a flight EARLIER in the queue at the same desired departure. Ground
# delay is measured against t_departure, so the lead is not itself charged as delay — the whole effect
# is queue position. An arm re-cuts a mixed world with ONE operator's lead replaced; the other keeps
# its default, so the arm isolates that operator's queue position against an unchanged competitor.
#
# The ladder is operator-agnostic: 8 minutes is Wing/Zipline's own lead and 30 minutes is Amazon's, so
# the same three rungs express "file like Wing/Zipline", an intermediate, and "file like Amazon" for
# EITHER operator. The standard deviation scales with the mean (mean/6, the ratio both defaults
# already use) rather than being held fixed: N(480, 300) would push ~5.5% of draws through the
# max(0, ·) floor in HubRadiusDemand._lead_for, so the arm's realized mean would drift off its label.
LEAD_ARMS: dict[str, tuple[float, float]] = {
    "08m": WING_ZIPLINE_LEAD_S,
    "15m": (15.0 * 60.0, 2.5 * 60.0),
    "30m": AMAZON_LEAD_S,
}

# Which operator an arm varies, as (name token, _density_scenario keyword, USS label). The token
# prefixes the arm suffix — density_..._azlead15m varies Amazon, ..._wzlead15m varies Wing/Zipline.
LEAD_ARM_OPERATORS: dict[str, tuple[str, str]] = {
    "az": ("amazon_lead_s", AMAZON_USS),
    "wz": ("wing_lead_s", WING_ZIPLINE_USS),
}

# Fixed preroll shared by every lead arm (see HubRadiusDemand.request_clock_offset_s). Pinning it is
# what makes the arms comparable flight-by-flight: without it, _shift_request_clock's data-dependent
# shift translates the whole world by a different amount per arm, moving every t_departure.
#
# It must exceed the largest realized lead draw — the binding case is the lead30m arm, whose preroll
# measures 2653 s (FAA) / 2341 s (far-future) at the default seed, leaving ~950-1260 s of margin. That
# margin is not load-bearing: _shift_request_clock raises rather than clipping if a seed ever exceeds
# the offset. The latest return departure is then offset + DEMAND_DURATION_S + the longest nominal trip
# (16 km / 30 m/s + two climbs + a hover ≈ 597 s) ≈ 5990 s, comfortably inside SIM_HORIZON_S — so the
# arms need no horizon bump, and the compiled A* occupancy box (sized from horizon_s) is unchanged.
LEAD_ARM_CLOCK_OFFSET_S = 60.0 * 60.0

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
    wing_lead_s: tuple[float, float] = WING_ZIPLINE_LEAD_S,
    amazon_lead_s: tuple[float, float] = AMAZON_LEAD_S,
    request_clock_offset_s: float | None = None,
    flight_levels_m: tuple[float, ...] = (CRUISE_ALTITUDE_M,),
    corridor_height_m: float | None = None,
) -> ScenarioSpec:
    """Build one density recipe, preserving Wing/Zipline-first USS and hub ordering.

    ``flight_levels_m`` defaults to the single 100 m cruise plane; pass a wider ladder (plus a matching
    ``corridor_height_m`` if the levels sit closer than 30 m apart) to study vertically-stacked traffic.
    ``corridor_height_m=None`` keeps SimConfig's 30 m default.

    ``wing_lead_s`` / ``amazon_lead_s`` / ``request_clock_offset_s`` exist for the scheduling-lead arms
    (:data:`LEAD_ARMS`); all three default to today's values, so every recipe that omits them is
    unchanged. ``wing_lead_s`` applies to single-operator worlds too, but note that shifting the MEAN
    lead of the only operator present just translates every filing equally and leaves FCFS order
    untouched — the contrast is only meaningful when a second operator holds still.
    """
    if (amazon_hubs is None) != (amazon_rate_per_hub is None):
        raise ValueError("amazon_hubs and amazon_rate_per_hub must be supplied together")

    uss = [WING_ZIPLINE_USS]
    hubs = [wing_hubs]
    radius_m = {WING_ZIPLINE_USS: WING_ZIPLINE_RADIUS_M}
    pads_per_hub = {WING_ZIPLINE_USS: PADS_PER_HUB}
    terminal_radius_m = {WING_ZIPLINE_USS: TERMINAL_RADIUS_M}
    lam_per_uss = {WING_ZIPLINE_USS: round(wing_hubs * wing_rate_per_hub, 2)}
    departure_offset_s = {WING_ZIPLINE_USS: wing_lead_s}

    if amazon_hubs is not None and amazon_rate_per_hub is not None:
        uss.append(AMAZON_USS)
        hubs.append(amazon_hubs)
        radius_m[AMAZON_USS] = AMAZON_RADIUS_M
        pads_per_hub[AMAZON_USS] = PADS_PER_HUB
        terminal_radius_m[AMAZON_USS] = TERMINAL_RADIUS_M
        lam_per_uss[AMAZON_USS] = round(amazon_hubs * amazon_rate_per_hub, 2)
        departure_offset_s[AMAZON_USS] = amazon_lead_s

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
            request_clock_offset_s=request_clock_offset_s,
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

# The two mixed-operator worlds the lead arms are cut from — same hubs and rates as the base entries
# above, so an arm differs from its base only in Amazon's lead and the pinned clock.
_LEAD_ARM_WORLDS: dict[str, dict] = {
    "density_faa_wing_zipline_amazon": dict(
        wing_hubs=182, wing_rate_per_hub=26.67, amazon_hubs=7, amazon_rate_per_hub=66.67),
    "density_future_wing_zipline_amazon": dict(
        wing_hubs=476, wing_rate_per_hub=57.4, amazon_hubs=14, amazon_rate_per_hub=157.0),
}


_OPERATOR_LABELS = {AMAZON_USS: "Amazon", WING_ZIPLINE_USS: "Wing/Zipline"}


def _lead_arm_scenarios() -> dict[str, ScenarioSpec]:
    """Two mixed worlds × two operators × three leads, all on :data:`LEAD_ARM_CLOCK_OFFSET_S`.

    Within a world and seed every arm is the SAME world: identical hubs, customers, and desired
    departure times for every flight — only the varied operator's filing times (hence FCFS queue
    position) move, and the other operator's filings do not move at all. That holds because each
    operator draws from its own child RNG stream and ``rng.normal`` consumes the same entropy whatever
    its mean/std, and because the pinned clock offset removes the one remaining coupling. So arms can
    be differenced flight-by-flight, not just in aggregate.

    Both sweeps share a pivot: ``azlead30m`` and ``wzlead08m`` are both operators at their defaults, so
    they are the SAME recipe under two names — the status-quo point each sweep rotates about. Run one
    of them, not both. Arm names zero-pad the minutes so ``sorted(SCENARIOS)`` — the ``--scenario``
    choice list — groups them by operator and orders them by lead.
    """
    out: dict[str, ScenarioSpec] = {}
    for base, world in _LEAD_ARM_WORLDS.items():
        for token, (kwarg, uss) in LEAD_ARM_OPERATORS.items():
            for arm, lead_s in LEAD_ARMS.items():
                name = f"{base}_{token}lead{arm}"
                out[name] = _density_scenario(
                    name,
                    f"{SCENARIOS[base].description.rstrip('.')}; {_OPERATOR_LABELS[uss]} files "
                    f"N({lead_s[0]:.0f}, {lead_s[1]:.0f}) s before departure on a fixed "
                    f"{LEAD_ARM_CLOCK_OFFSET_S:.0f} s clock (scheduling-lead arm).",
                    request_clock_offset_s=LEAD_ARM_CLOCK_OFFSET_S,
                    **{kwarg: lead_s},
                    **world,
                )
    return out


SCENARIOS.update(_lead_arm_scenarios())
