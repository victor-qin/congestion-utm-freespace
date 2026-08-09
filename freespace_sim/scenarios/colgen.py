"""Small, congested hub world used to exercise whole-schedule column generation."""

from __future__ import annotations

from .spec import DemandSpec, ScenarioSpec

COLGEN_USS = "colgen_uss"


SCENARIOS: dict[str, ScenarioSpec] = {
    "colgen_test": ScenarioSpec(
        name="colgen_test",
        description=(
            "Column-generation acceptance miniature with eight single-level hub terminals "
            "and paired delivery returns."
        ),
        region_m=(8_000.0, 8_000.0),
        horizon_s=1_800.0,
        demand_duration_s=300.0,
        lam_per_hour=600.0,
        seed=0,
        fixed_exit_lanes=True,
        terminal_airspace_always_active=True,
        flight_levels_m=(100.0,),
        demand=DemandSpec(
            pattern="hub_radius",
            uss=(COLGEN_USS,),
            hubs=(8,),
            radius_m={COLGEN_USS: 2_500.0},
            pads_per_hub={COLGEN_USS: 8},
            terminal_radius_m={COLGEN_USS: 180.0},
            return_flights=True,
            turnaround_s=0.0,
            lam_per_uss={COLGEN_USS: 600.0},
            departure_offset_s={COLGEN_USS: (120.0, 30.0)},
            timing_mode="departure",
            paired_return_request=True,
        ),
    ),
}
