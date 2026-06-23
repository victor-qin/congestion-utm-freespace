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
        "dallas_hub_2uss_large", region_m=(10000.0, 10000.0), lam_per_hour=34500.0, horizon_s=1800.0,
        demand=DemandSpec(
            pattern="hub_radius", uss=("walmart_uss", "stripmall_uss"), hubs=(6, 20),
            # fewer Walmarts ⇒ each reaches farther; many strip malls ⇒ tighter local delivery
            radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0}, pads_per_hub=4,
            return_flights=True,
        ),
    ),
}
