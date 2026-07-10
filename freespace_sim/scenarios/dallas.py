"""Dallas scenarios — geographic hub-and-spoke demand (Voronoi cells / radius vertiports)."""

from __future__ import annotations

from .spec import DemandSpec, ScenarioSpec

SCENARIOS: dict[str, ScenarioSpec] = {
    # the headline: two operators, geographic hub-and-spoke demand (few Walmarts, many strip malls)
    "dallas_hub_2uss": ScenarioSpec(
        "dallas_hub_2uss", region_m=(10000.0, 10000.0),
        demand=DemandSpec(pattern="hub", uss=("walmart_uss", "stripmall_uss"), hubs=(6, 20), pads_per_hub=2,),
    ),
    # hub-funnel: a few concentrated multi-pad vertiports (6 Walmarts, 20 strip malls) in a 10×10 km
    # region with radius service areas + return flights — funnels demand onto few hubs to exercise pad
    # capacity under load. λ counts deliveries (returns ~2×).
    "dallas_hub_2uss_large": ScenarioSpec(
        "dallas_hub_2uss_large", region_m=(10000.0, 10000.0), lam_per_hour=8000.0, horizon_s=1800.0,
        flight_levels_m=(30.0, 70.0, 110.0),
        terminal_airspace_always_active=True,
        demand=DemandSpec(
            pattern="hub_radius", uss=("walmart_uss", "stripmall_uss"), hubs=(6, 20),
            # fewer Walmarts ⇒ each reaches farther; many strip malls ⇒ tighter local delivery
            radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
            terminal_radius_m={"walmart_uss": 180.0, "stripmall_uss": 105.0},
            pads_per_hub={"walmart_uss": 40, "stripmall_uss": 16},
            return_flights=True,

        ),
    ),
    # the full metro world (issue #9): 60×45 km, 20 Walmarts + 240 strip malls, λ=34.5k deliveries
    # (~2× with returns). Demand splits 1:2 Walmart:strip-mall (uss_share). Pads AND boundary-hex exit
    # lanes (issue #18/19) provisioned ABOVE the measured peak per-hub demand at λ=34.5k so neither pad
    # capacity nor exit lanes constrain takeoffs — leaving path-planning (air congestion) as the only
    # delay source. Two time-scales: a column dwell holds a PAD for 55 s (hover+climb) → Walmart peak
    # demand ~37 → 40 pads; an exit lane holds a boundary hex for only ~12 s (corridor transit) →
    # Walmart lane demand ~14 → 180 m column = 18 lanes. Strip-mall: 13 dwell / 6 lane demand → 16 pads,
    # 105 m = 12 lanes. (Earlier 24/8 pads + 125 m were pad-bound: peak 28-37 dwells ≫ 24 → saturation.)
    "dallas_full": ScenarioSpec(
        "dallas_full", region_m=(60000.0, 45000.0), lam_per_hour=34500.0, horizon_s=1800.0,
        # Three A* flight levels (30/70/110 m) + always-active terminal airspace: foreign transit routes
        # AROUND terminals (air detour) instead of ground-blocking same-hub takeoffs, so the congestion
        # measured here is airspace-density AIR delay. Pad dwell = hover(30) + climb-to-level (30/70/110 m
        # ⇒ 5/12/18 s), so the 40/16 pads stay over-provisioned even at the 110 m top level.
        flight_levels_m=(30.0, 70.0, 110.0),
        terminal_airspace_always_active=True,
        demand=DemandSpec(
            pattern="hub_radius", uss=("walmart_uss", "stripmall_uss"), hubs=(20, 240),
            radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
            terminal_radius_m={"walmart_uss": 180.0, "stripmall_uss": 105.0},
            pads_per_hub={"walmart_uss": 40, "stripmall_uss": 16},
            uss_share={"walmart_uss": 1.0, "stripmall_uss": 2.0},
            return_flights=True,
        ),
    ),
}
