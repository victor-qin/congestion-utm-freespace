"""Pure CLI-to-ScenarioSpec override tests for the execute entry point."""

from types import SimpleNamespace

import pytest

from experiments.run import spec_from_args
from freespace_sim.scenarios import SCENARIOS


def _args(scenario: str, *, lam: float | None = None):
    return SimpleNamespace(
        scenario=scenario,
        region=None,
        lam=lam,
        horizon=None,
        seed=None,
        planner=None,
        terminal_airspace_always_active=None,
        demand=None,
        uss=None,
        hubs=None,
        direction=None,
        radius=None,
        pads_per_hub=None,
        terminal_radius=None,
        corridor_overlap=None,
        return_flights=None,
        turnaround=None,
    )


def test_lam_override_scales_explicit_per_uss_rates():
    spec = spec_from_args(
        _args("density_faa_wing_zipline_amazon", lam=1000.0)
    )
    assert spec.lam_per_hour == 1000.0
    assert sum(spec.demand.lam_per_uss.values()) == pytest.approx(1000.0)


def test_lam_override_preserves_per_uss_proportions():
    base = SCENARIOS["density_future_wing_zipline_amazon"]
    scaled = spec_from_args(_args(base.name, lam=5000.0))
    factors = {
        uss_id: scaled.demand.lam_per_uss[uss_id] / rate
        for uss_id, rate in base.demand.lam_per_uss.items()
    }
    first = next(iter(factors.values()))
    assert all(factor == pytest.approx(first) for factor in factors.values())


def test_lam_override_keeps_legacy_global_lambda_behavior():
    spec = spec_from_args(_args("metro_2uss", lam=321.0))
    assert spec.lam_per_hour == 321.0
    assert spec.demand.lam_per_uss is None
