"""CLI-to-ScenarioSpec override tests for the execute entry point, plus its stderr capture."""

import multiprocessing as mp
import os
import sys
import time

import pytest

import experiments.run as run_module
from experiments.run import colgen_params_from_args, parse_args, spec_from_args
from freespace_sim.scenarios import SCENARIOS, with_overrides


def _shout(message: str) -> None:
    """Module level, because a `spawn` child has to import the target rather than inherit it."""

    print(message, file=sys.stderr)


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
        "--colgen-max-air-overrun", "3",
        "--colgen-workers", "4",
        "--colgen-seed-ladder", "30",
        "--colgen-greedy-budget-rate", "1.5",
        "--colgen-ip-time-limit", "300",
        "--colgen-max-eager-rows", "1000",
        "--colgen-warm-start", "astar",
    )
    params = colgen_params_from_args(args, "colgen")

    assert params.ip_time_limit_s == 300.0
    assert params.max_eager_ip_rows == 1000
    assert params.warm_start_planner == "astar"
    assert params.time_limit_s == 900.0
    assert params.max_iterations == 50
    assert params.objective == "total_cost"
    assert params.solver == "highs"
    assert params.gap_metric == "cost"
    assert params.max_air_overrun_hops == 3
    assert params.n_pricing_workers == 4
    assert params.seed_ladder_steps == 30
    assert params.greedy_budget_s_per_flight == 1.5


def test_colgen_params_are_none_for_every_other_planner():
    """``None`` keeps non-colgen planners on ``get_planner``'s no-params path."""

    assert colgen_params_from_args(_args("metro_uniform", "--planner", "astar"), "astar") is None


def test_unset_colgen_flags_leave_the_defaults_alone():
    """An unset flag must not be forwarded as ``None`` and overwrite a real default."""

    defaults = colgen_params_from_args(_args("colgen_test", "--planner", "colgen"), "colgen")

    assert defaults.time_limit_s == 1200.0
    assert defaults.objective == "total_cost"
    # The pricing-path knobs specifically.  These pin the SHIPPED defaults so that changing
    # one has to be deliberate -- which is the whole point of the test, and both of the
    # values below moved for measured reasons rather than drifting:
    #
    #   `n_pricing_workers` STAYS 0, and the attempt to default it to 4 is worth knowing
    #   about: the pool is fast (3.50x at 4 workers on density x50) but its memory is
    #   LINEAR -- 3.9 GB sequential, 12.5 GB at 4 workers, 22.7 GB at 8, sampled across the
    #   process tree.  The evidence that briefly said otherwise was `rss_children`, which
    #   is the largest single child rather than the sum and therefore reads flat no matter
    #   how many workers run.
    #
    #   `greedy_budget_s_per_flight` 0.7 -> 0.0, which DISABLES the stage.  At convergence
    #   it buys 0.129% of objective for +57% of wall, and iteration 1 is bit-identical
    #   without it because `round_heuristic` sets `best_heuristic` anyway.
    #
    # `seed_ladder_steps` still defaults ON, so a `None` leaking through would silently
    # disable the ladder rather than merely resetting a budget -- the original point here.
    assert defaults.n_pricing_workers == 0
    assert defaults.seed_ladder_steps == 20
    assert defaults.greedy_budget_s_per_flight == 0.0
    #   The bootstrap is ON at K=1, and the two fields move together: `bootstrap_roots=1`
    #   works only because `bootstrap_ranking="bound"` orders roots by `g+h`.  At "score"
    #   K=1 provably fails (`entry_rc` stays exactly 0.0000) and 2 is the floor.  Both are
    #   ANSWER-AFFECTING -- they change which equally-optimal column returns -- so a change
    #   here needs an `ab_colgen_parity.py` re-baseline, not just a green suite.
    assert defaults.bootstrap_roots == 1
    assert defaults.bootstrap_ranking == "bound"
    #   The four that moved (or arrived) with the objective-scale change, pinned here for
    #   the same reason as everything above -- these are the ones an archived run cannot be
    #   compared across, so drifting one silently is the expensive failure:
    #
    #   `solver` "auto" -> "gurobi".  Not a speed knob: the two backends return different
    #   optimal dual vertices on a degenerate master, so "auto" quietly falling back to
    #   HiGHS changes the answer.  Failing loudly is the point.
    #
    #   `M` 1e6 -> 1e4 and `gap_metric` "revenue" -> "cost" SHIP AS A PAIR.  The revenue
    #   gate reduces to `tau*M` (n cancels), so moving M alone retunes the stopping rule by
    #   100x and turns ordinary runs into time-limit runs.
    #
    #   `ip_time_limit_s` is new: without it the final MILP inherits whatever the CG loop
    #   did not spend, which is unbounded exactly when the loop went well.
    assert defaults.solver == "gurobi"
    assert defaults.M == 10_000.0
    assert defaults.gap_metric == "cost"
    assert defaults.ip_time_limit_s == 120.0
    # None is "no ceiling", which is what every measurement in this PR ran under.
    assert defaults.max_eager_ip_rows is None
    assert defaults.warm_start_planner is None


def test_zero_disables_the_ladder_and_the_greedy_rather_than_erroring():
    """``0`` is the documented disable path for both, so it must not be rejected.

    Neither is a plain budget: ``seed_ladder_steps=0`` means seed no retimes at all and
    ``greedy_budget_s_per_flight=0`` means skip the stage.  A validator that treated them
    like ``n_pricing_workers`` and demanded positivity would make the documented way to
    turn either off an argparse error instead.
    """

    off = colgen_params_from_args(
        _args("colgen_test", "--planner", "colgen",
              "--colgen-seed-ladder", "0", "--colgen-greedy-budget-rate", "0"),
        "colgen",
    )
    assert (off.seed_ladder_steps, off.greedy_budget_s_per_flight) == (0, 0.0)


@pytest.mark.parametrize(
    "colgen_flag",
    [
        ("--colgen-time-limit", "900"),
        ("--colgen-max-iterations", "50"),
        ("--colgen-objective", "total_cost"),
        ("--colgen-solver", "highs"),
        ("--colgen-gap-metric", "cost"),
        ("--colgen-max-air-overrun", "3"),
        ("--colgen-workers", "4"),
        ("--colgen-seed-ladder", "30"),
        ("--colgen-greedy-budget-rate", "1.5"),
        ("--colgen-ip-time-limit", "300"),
        ("--colgen-max-eager-rows", "1000"),
        ("--colgen-warm-start", "astar"),
    ],
)
def test_colgen_flags_require_the_colgen_planner(colgen_flag):
    """Accepting them for another planner would drop the budget without saying so."""

    with pytest.raises(SystemExit) as exc:
        _args("metro_uniform", "--planner", "astar", *colgen_flag)
    assert exc.value.code == 2


def test_colgen_flags_follow_the_scenario_when_it_selects_the_planner(monkeypatch):
    """``--planner`` is an override, not the question "which planner runs".

    A ``ScenarioSpec`` carries its own planner. Gating the colgen flags on the override
    alone meant such a scenario could not be given a solver budget at all -- and, worse,
    that a run of it silently used ``ColGenParams()`` defaults, which buys one iteration
    on a real world. ``colgen_test`` does not set ``planner`` today, so this pins the
    behaviour against a spec that does, which is one scenario edit away.
    """

    colgen_spec = with_overrides(SCENARIOS["colgen_test"], planner="colgen")
    monkeypatch.setattr(run_module, "get_scenario", lambda _name: colgen_spec)

    # `spec_from_args(...).config().planner` is verbatim what `main` passes, so this
    # exercises the real derivation rather than re-deriving the answer the test wants.
    def _params_as_main_would(argv):
        parsed = parse_args(argv)
        return colgen_params_from_args(parsed, spec_from_args(parsed).config().planner)

    # No --planner anywhere: the budget must be accepted, not rejected...
    assert _params_as_main_would(
        ["--scenario", "colgen_test", "--colgen-time-limit", "600"]
    ).time_limit_s == 600.0

    # ...and a run with no colgen flags at all must still be recognised as a colgen run,
    # rather than falling through to the shipped defaults by accident.
    assert _params_as_main_would(["--scenario", "colgen_test"]) is not None


def test_colgen_flags_are_still_refused_when_the_scenario_picks_another_planner(monkeypatch):
    """The guard has to survive the fix, or it stops catching the typo it exists for."""

    astar_spec = with_overrides(SCENARIOS["metro_uniform"], planner="astar")
    monkeypatch.setattr(run_module, "get_scenario", lambda _name: astar_spec)

    with pytest.raises(SystemExit) as exc:
        parse_args(["--scenario", "metro_uniform", "--colgen-time-limit", "600"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- run.log capture


def test_the_stderr_tee_copies_to_both_the_terminal_and_the_file(tmp_path, capfd):
    """Copies, not moves: losing the terminal would break every interactive run.

    Writes to the DESCRIPTOR rather than through ``sys.stderr``, because pytest replaces the
    latter with an object of its own and the message would then never reach fd 2 — which
    would test pytest's capture instead of this one. Under the real entry point the two are
    the same file.
    """

    tee = run_module._StderrTee(tmp_path / "run.log")
    try:
        os.write(2, b"visible and captured\n")
    finally:
        tee.close()

    assert "visible and captured" in (tmp_path / "run.log").read_text()
    assert "visible and captured" in capfd.readouterr().err


def test_the_stderr_tee_captures_a_spawned_child(tmp_path):
    """The half that matters: pricing warnings come from SPAWNED workers, not the parent.

    A `sys.stderr` swap would pass the parent test above and fail this one, which is why the
    capture is at the descriptor level.
    """

    tee = run_module._StderrTee(tmp_path / "run.log")
    try:
        ctx = mp.get_context("spawn")
        process = ctx.Process(target=_shout, args=("from a spawned child",))
        process.start()
        process.join(timeout=60)
    finally:
        tee.close()

    assert "from a spawned child" in (tmp_path / "run.log").read_text()


def test_the_stderr_tee_closes_after_a_spawn_pool_has_run(tmp_path):
    """`close()` must not wait for EOF on the pipe, because EOF never comes.

    `multiprocessing` launches its resource tracker with `sys.stderr.fileno()` in
    `fds_to_pass`, and that child outlives the parent — so from the first pool onward,
    something we do not control holds a duplicate of the tee's write end for the rest of the
    run. A pump that drained until `read()` returned `b''` would block here forever, *after*
    the solve finished, on exactly the `--colgen-workers N` runs the capture exists for.
    Verified directly: after this teardown a `select` on the read end still reports nothing
    ready and no EOF.

    A plain child-process test cannot reach this — an ordinary child exits and releases the
    descriptor — which is why it is a separate case from the one above.

    No external timeout marker: ``close()`` joins with a 5 s cap and the pump is a daemon, so
    the regression is a five-second stall rather than a hang. The elapsed assertion below is
    what distinguishes "stopped because it was told to" from "stopped because it gave up".
    """

    tee = run_module._StderrTee(tmp_path / "run.log")
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            assert pool.map(int, ["7"]) == [7]
    finally:
        started = time.monotonic()
        tee.close()
        elapsed = time.monotonic() - started

    # The pump polls at 0.2 s and `close()` joins with a 5 s timeout, so anything near the
    # timeout means it stopped because it gave up rather than because it was told to.
    assert elapsed < 3.0, f"close() took {elapsed:.1f}s — the pump is waiting on EOF"
