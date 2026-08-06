"""Pure CLI-to-ScenarioSpec override tests for the execute entry point."""

import pytest

from experiments.run import colgen_params_from_args, parse_args, spec_from_args
from freespace_sim.scenarios import SCENARIOS


def _args(scenario: str, *extra: str):
    """Parse a real argv through the real parser — never hand-mirror the flag set.

    A ``SimpleNamespace`` fixture has to list every attribute ``spec_from_args`` reads, so adding
    a flag breaks these tests with an ``AttributeError`` that says nothing about the actual change.
    """
    return parse_args(["--scenario", scenario, *extra])


@pytest.mark.parametrize("planner_args", [(), ("--planner", "astar"), ("--planner", "milp")])
def test_execution_mode_defaults_to_sequential_for_every_planner(planner_args):
    assert _args("metro_uniform", *planner_args).mode == "sequential"


@pytest.mark.parametrize(
    "parallel_args",
    [
        ("--workers", "2"),
        ("--parallel-window", "8"),
        ("--workers", "2", "--parallel-window", "8"),
    ],
)
def test_sequential_mode_rejects_parallel_tuning_flags(parallel_args):
    for mode_args in ((), ("--mode", "sequential")):
        with pytest.raises(SystemExit) as exc:
            _args("metro_uniform", *mode_args, *parallel_args)
        assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["exact", "relaxed"])
def test_parallel_modes_are_explicit_opt_ins(mode):
    args = _args(
        "metro_uniform", "--mode", mode, "--workers", "2", "--parallel-window", "8"
    )
    assert (args.mode, args.workers, args.parallel_window) == (mode, 2, 8)


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


def test_colgen_flags_reach_the_planner_params():
    """The solver budget has to survive the CLI, or a sweep silently runs at the default."""

    args = _args(
        "colgen_test", "--planner", "colgen",
        "--colgen-time-limit", "900", "--colgen-max-iterations", "50",
        "--colgen-objective", "total_cost", "--colgen-solver", "highs",
        "--colgen-gap-metric", "cost",
    )
    params = colgen_params_from_args(args)

    assert params.time_limit_s == 900.0
    assert params.max_iterations == 50
    assert params.objective == "total_cost"
    assert params.solver == "highs"
    assert params.gap_metric == "cost"


def test_colgen_params_are_none_for_every_other_planner():
    """``None`` keeps non-colgen planners on ``get_planner``'s no-params path."""

    assert colgen_params_from_args(_args("metro_uniform", "--planner", "astar")) is None


def test_unset_colgen_flags_leave_the_defaults_alone():
    """An unset flag must not be forwarded as ``None`` and overwrite a real default."""

    defaults = colgen_params_from_args(_args("colgen_test", "--planner", "colgen"))

    assert defaults.time_limit_s == 120.0
    assert defaults.objective == "total_delay"


@pytest.mark.parametrize(
    "colgen_flag",
    [
        ("--colgen-time-limit", "900"),
        ("--colgen-max-iterations", "50"),
        ("--colgen-objective", "total_cost"),
        ("--colgen-solver", "highs"),
        ("--colgen-gap-metric", "cost"),
    ],
)
def test_colgen_flags_require_the_colgen_planner(colgen_flag):
    """Accepting them for another planner would drop the budget without saying so."""

    with pytest.raises(SystemExit) as exc:
        _args("metro_uniform", "--planner", "astar", *colgen_flag)
    assert exc.value.code == 2
