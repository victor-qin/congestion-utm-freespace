"""Pure CLI-to-ScenarioSpec override tests for the execute entry point."""

import pytest

from experiments.run import build_parser, spec_from_args
from freespace_sim.scenarios import SCENARIOS


def _args(scenario: str, *extra: str):
    """Parse a real argv through the real parser — never hand-mirror the flag set.

    A ``SimpleNamespace`` fixture has to list every attribute ``spec_from_args`` reads, so adding
    a flag breaks these tests with an ``AttributeError`` that says nothing about the actual change.
    """
    return build_parser().parse_args(["--scenario", scenario, *extra])


def test_lam_override_scales_explicit_per_uss_rates():
    spec = spec_from_args(
        _args("density_faa_wing_zipline_amazon", "--lam", "1000")
    )
    assert spec.lam_per_hour == 1000.0
    assert sum(spec.demand.lam_per_uss.values()) == pytest.approx(1000.0)


def test_lam_override_preserves_per_uss_proportions():
    base = SCENARIOS["density_future_wing_zipline_amazon"]
    scaled = spec_from_args(_args(base.name, "--lam", "5000"))
    factors = {
        uss_id: scaled.demand.lam_per_uss[uss_id] / rate
        for uss_id, rate in base.demand.lam_per_uss.items()
    }
    first = next(iter(factors.values()))
    assert all(factor == pytest.approx(first) for factor in factors.values())


def test_lam_override_keeps_legacy_global_lambda_behavior():
    spec = spec_from_args(_args("metro_2uss", "--lam", "321"))
    assert spec.lam_per_hour == 321.0
    assert spec.demand.lam_per_uss is None


def test_horizon_alone_cannot_shrink_a_density_scenario():
    """The demand window is NOT clamped to a shrunken horizon — that failure must stay loud.

    Clamping would make ``--horizon 600`` appear to work while the (unclamped) departure lead put
    every departure past the horizon, silently demoting the run to the box-guard fallback path.
    """
    spec = spec_from_args(_args("density_faa_wing_zipline", "--horizon", "600"))
    with pytest.raises(ValueError, match="exceeds horizon_s"):
        spec.config()


def test_demand_duration_flag_shrinks_a_density_scenario_for_a_smoke_run():
    """Both knobs together are the supported way to get a short density run from the CLI."""
    cfg = spec_from_args(
        _args("density_faa_wing_zipline", "--horizon", "900", "--demand-duration", "60")
    ).config()
    assert (cfg.horizon_s, cfg.effective_demand_duration_s) == (900.0, 60.0)
