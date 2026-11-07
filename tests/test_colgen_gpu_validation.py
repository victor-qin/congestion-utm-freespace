"""Requirements for reproducible long-flight GPU validation reports."""

import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from analysis.colgen.gpu_validation import failure_reasons, select_requests, serializable


def test_selection_preserves_requests_and_required_regression() -> None:
    """Selection retains original coordinates, IDs, and clocks without shortening flights."""
    requests = [SimpleNamespace(flight_id=i, origin=np.zeros(3), dest=np.array([i, 0., 0.]),
                                t_departure=100. + i) for i in range(12)]
    selected = select_requests(requests, count=10, required_id=0)
    assert [r.flight_id for r in selected] == list(range(11, 2, -1)) + [0]
    assert all(r is requests[r.flight_id] for r in selected)
    assert serializable(requests[11].dest) == [11., 0., 0.]


@pytest.mark.parametrize("changes,stats_changes,selected_ids,passes", [
    ({}, {}, {1752}, True),
    ({"search_complete": False, "certified": False, "status": "label_limit"}, {}, {1752}, False),
    ({"certified": False}, {}, {1752}, False),
    ({"stats": {"gpu_dispatches": 0}}, {}, {1752}, False),
    ({}, {}, {1752, 1420}, False),
    ({"upper_bound": None}, {}, {1752}, False),
    ({"upper_bound": float("inf")}, {}, {1752}, False),
    ({"upper_bound": 0.5}, {}, {1752}, False),
    ({}, {"pricing_search_complete": False}, {1752}, False),
    ({}, {"pricing_bound_certified": False}, {1752}, False),
    ({}, {"pricing_failure_status": "deadline"}, {1752}, False),
])
def test_success_requires_gpu_and_complete_certified_pricing(
    changes: dict, stats_changes: dict, selected_ids: set[int], passes: bool,
) -> None:
    """All accepted/IP optimal cannot hide missing flights, invalid bounds or incomplete pricing."""
    record = dict(flight_id=1752, search_complete=True, certified=True,
                  backend="metal", status="complete", upper_bound=1., reduced_cost=1.,
                  stats={"gpu_dispatches": 1})
    record.update(changes)
    stats = dict(ip_status="optimal", kernel_fell_back=0, pricing_search_complete=True,
                 pricing_bound_certified=True, pricing_failure_status=None)
    stats.update(stats_changes)
    simulation = SimpleNamespace(verified=True, denied=[], planner_stats=stats)
    assert (not failure_reasons([record], selected_ids, simulation)) is passes


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("selection", [[], ["--all-flights"], ["--flight-ids", "2", "0"]])
def test_prepare_serializes_original_and_legacy_capture_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, selection: list[str],
) -> None:
    """Legacy missing slots gain recorded defaults without rewriting captured parameter values."""
    from analysis.colgen import gpu_validation
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.types import FlightRequest

    params = ColGenParams(bootstrap_max_labels=12345)
    missing = ["pricing_device", "pricing_gpu_backend"] if legacy else []
    for name in missing:
        object.__delattr__(params, name)
    requests = [FlightRequest(i, np.zeros(3), np.array([16000., i, 0.]), 0., 10.)
                for i in [1752, *range(12)]]
    captured = dict(params=params, requests=requests, cfg={}, static_terminals=[])
    monkeypatch.setattr(gpu_validation, "load_capture", lambda *args: (captured, "capturehash"))
    out = tmp_path / "prepared"
    assert gpu_validation.main(["prepare", "--capture", str(tmp_path / "legacy.pkl"),
                                "--out", str(out), *selection]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["capture_sha256"] == "capturehash"
    assert manifest["original_params_defaulted_fields"] == missing
    assert manifest["original_params"]["bootstrap_max_labels"] == 12345
    assert manifest["original_params"]["pricing_device"] == "cpu"
    assert manifest["run_params"]["pricing_device"] == "gpu"
    assert manifest["run_params"]["pricing_gpu_backend"] == "metal"
    assert all(not hasattr(params, name) for name in missing)
    expected = ([requests[3], requests[1]] if selection[:1] == ["--flight-ids"] else
                requests if selection else select_requests(requests))
    assert manifest["flight_ids"] == [request.flight_id for request in expected]


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("explicit_ids", [None, [10, 3, 6]])
@pytest.mark.parametrize("mode", ["replay", "solve"])
def test_capture_selection_preserves_order_and_uses_the_requested_price_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int,
    explicit_ids: list[int] | None, mode: str,
) -> None:
    """Selection reaches either fixed-price replay or a fresh subset simulation unchanged.

    Parameters
    ------------
    - tmp_path (Path): trusted fixture capture and output directory.
    - monkeypatch (pytest.MonkeyPatch): replaces pricing with a deterministic adapter stub.
    - batch_size (int): dispatch group size, including a shorter final group.
    - explicit_ids (list[int] | None): caller-ordered subset, or all captured flights.
    - mode (str): fixed-price replay or an explicit subset passed to the simulation.

    Return
    --------
    - output (None): checks selection, price propagation and reports without GPU or LP work.
    """
    from analysis.colgen import gpu_validation
    from freespace_sim import sim
    from freespace_sim.planner.colgen import pricing, pricing_platform
    from freespace_sim.planner.colgen.network import RowKey, StaticTerminalCatalog, build_flight_graph
    from tests.test_colgen_solver import _cfg, _params, _request

    cfg = _cfg(flight_levels_m=(30.0,))
    if mode == "solve" and explicit_ids is None:
        explicit_ids = [3, 8, 1, 6, 10]
    ids = [3, 8, 1, 6, 10]
    requests = [_request(i, (0, 0), (3, 0), cfg, departure=i * 4.0) for i in ids]
    row = RowKey.cell((1, 0), 0, 4)
    capture = tmp_path / "all.pkl"
    capture.write_bytes(pickle.dumps(dict(
        capture_schema=1, params=_params(max_air_overrun_hops=6), requests=requests,
        cfg=cfg, static_terminals=[], duals={row: 7.0},
        flight_duals={i: float(i) for i in ids}, known_columns={},
    )))
    batches, views = [], []

    def price_once(self, problems, deadline=None):
        """Observe orchestration while keeping numerical pricing outside this CLI test."""
        batches.append([p.graph.request.flight_id for p in problems])
        for problem in problems:
            views.append(problem.duals)
            assert isinstance(problem.duals, pricing.DualView)
            assert problem.duals.row_cost(row) == (7.0 if mode == "replay" else 0.0)
            assert problem.pi_f == (float(problem.graph.request.flight_id) if mode == "replay" else 0.0)
        return tuple(pricing_platform.PricingResult(
            flight_id=p.graph.request.flight_id, reduced_cost=0.0, column=None,
            upper_bound=0.0, search_complete=True, certified=True, backend="metal",
            cpu_recompute=True, status="complete", stats={"gpu_dispatches": 1},
        ) for p in problems)

    def forbid_solver(*args, **kwargs):
        """A fixed-price replay must never start the master/IP simulation."""
        pytest.fail("Replay unexpectedly started the full solver")

    def fresh_solve(
        config: Any, *, requests: list[Any], planner_params: Any,
        static_terminals: list[Any], **kwargs: Any,
    ) -> SimpleNamespace:
        """Construct fresh subset prices from unchanged selected requests.

        Parameters
        ------------
        - config (Any): original simulation configuration.
        - requests (list[Any]): caller-selected original requests in order.
        - planner_params (Any): requested GPU resource limits and solver settings.
        - static_terminals (list[Any]): original catalog used to construct graphs.
        - kwargs (Any): simulation interface controls outside this selection test.

        Return
        --------
        - output (SimpleNamespace): completed stub simulation with separately priced flights.
        """
        assert [r.flight_id for r in requests] == explicit_ids
        assert config == cfg
        catalog = StaticTerminalCatalog(static_terminals, config)
        view = pricing.DualView({}, config)
        problems = [pricing_platform.PricingProblem(
            build_flight_graph(r, config, catalog, planner_params), config, planner_params, view, 0.0,
        ) for r in requests]
        platform = pricing_platform.PricingPlatform()
        for start in range(0, len(problems), batch_size):
            platform.price_batch(problems[start:start + batch_size])
        return SimpleNamespace(verified=True, accepted=requests, denied=[], planner_stats=dict(
            pricing_search_complete=True, pricing_bound_certified=True, pricing_failure_status=None,
        ))

    monkeypatch.setattr(pricing_platform.PricingPlatform, "price_batch", price_once)
    monkeypatch.setattr(sim, "run", forbid_solver if mode == "replay" else fresh_solve)
    out = tmp_path / "replay"
    selection = (["--flight-ids", *map(str, explicit_ids)] if explicit_ids is not None
                 else ["--all-flights"])
    assert gpu_validation.main([
        mode, "--capture", str(capture), "--out", str(out),
        *selection, "--batch-size", str(batch_size),
    ]) == 0
    ids = explicit_ids if explicit_ids is not None else ids
    assert batches == [ids[start:start + batch_size] for start in range(0, len(ids), batch_size)]
    assert all(view is views[0] for view in views)
    manifest = json.loads((out / "manifest.json").read_text())
    report = json.loads((out / "result.json").read_text())
    results = json.loads((out / "pricing.json").read_text())["results"]
    assert manifest["flight_ids"] == ids
    assert manifest["selection"] == (
        "Explicit flight IDs in caller-supplied order" if explicit_ids is not None else
        "All captured flights in captured request order"
    )
    assert [r["flight_id"] for r in results] == ids
    assert report["passed"] and report["pricing_results"] == len(ids)


@pytest.mark.parametrize("mode,selection,error,match", [
    ("prepare", ["--flight-ids", "3", "3"], ValueError, "distinct flight IDs"),
    ("replay", ["--flight-ids", "99"], ValueError, "absent from capture"),
    ("prepare", ["--all-flights", "--flight-ids", "3"], SystemExit, "2"),
    ("solve", ["--flight-ids", "99"], ValueError, "absent from capture"),
])
def test_explicit_selection_rejects_invalid_requests_before_output(
    tmp_path: Path, mode: str, selection: list[str], error: type[Exception], match: str,
) -> None:
    """Invalid subsets cannot start pricing or create a misleading prepared manifest."""
    from analysis.colgen import gpu_validation

    capture = tmp_path / "capture.pkl"
    capture.write_bytes(pickle.dumps(dict(
        capture_schema=1, requests=[SimpleNamespace(flight_id=3)], cfg={}, params={},
        static_terminals=[], duals={}, flight_duals={}, known_columns={},
    )))
    out = tmp_path / "rejected"
    with pytest.raises(error, match=match):
        gpu_validation.main([mode, "--capture", str(capture), "--out", str(out), *selection])
    assert not out.exists()


@pytest.mark.parametrize("problem", [None, "missing_result", "changed_inputs", "infeasible"])
def test_comparison_preserves_modes_limits_incomplete_results_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None,
) -> None:
    """All modes run with matched limits; missing/invalid output cannot masquerade as a fast solve.

    Parameters
    ------------
    - tmp_path (Path): isolated capture, lock and output artifacts.
    - monkeypatch (pytest.MonkeyPatch): replaces expensive solver subprocesses with recorded outcomes.
    - problem (str | None): one failed run, or expected incomplete/uncertified pricing only.

    Return
    --------
    - output (None): validates mode flags, fresh runs, honest summary and nonzero failure reporting.
    """
    from analysis.colgen import compare_gpu_checks

    capture = tmp_path / "capture.pkl"
    capture.touch()
    output = tmp_path / "comparison"
    commands = []

    def simulate(command, *, cwd, env, stdout, stderr, check):
        """Write one stub solve while checking the user-facing command's exact semantics."""
        commands.append(command)
        target = Path(command[command.index("--out") + 1])
        mode = target.name.split("-", 1)[1]
        assert command[command.index("--flight-ids") + 1:] == ["3", "2"]
        assert command[command.index("--budget") + 1] == "120.0"
        assert command[command.index("--max-labels") + 1] == "12345"
        assert command[command.index("--batch-size") + 1] == "1"
        assert env["PYTHONPATH"] == str(cwd)
        if mode == "full":
            assert "analysis.colgen.gpu_validation" in command
            assert "--epsilon" not in command
        else:
            assert "analysis.colgen.gpu_epsilon_experiment" in command
            epsilon = command[command.index("--epsilon") + 1]
            assert epsilon == ("0.002" if mode == "epsilon_border" else "0")
            assert ("--verify-border" in command) == (mode == "epsilon_border")
            assert ("--skip-completion-validation" in command) == (mode == "no_checks")
        target.mkdir()
        changed = problem == "changed_inputs" and len(commands) == 2
        feasible = not (problem == "infeasible" and len(commands) == 2)
        manifests = dict(
            capture_sha256="capture", requests_sha256="requests", config_sha256="config",
            flight_ids=[3, 2], static_terminals_sha256="terminals", run_params={"M": 100},
            harness_sha256="harness", source_sha256={"a.py": "changed" if changed else "same"},
        )
        result = dict(elapsed_s=4.0, accepted=2, verified=feasible, passed=False,
                      cpu_full_pricing_guard=True, failure_reasons=["label_limit"])
        stats = dict(objective=10.0 if mode == "full" else 12.0, objective_name="total_cost",
                     pricing_search_complete=False, pricing_bound_certified=mode == "full",
                     termination_reason="pricing_label_limit", pricing_failure_status="label_limit")
        records = [dict(flight_id=3, search_complete=False, certified=False,
                        stats={"candidates": 10, "cpu_recomputations": 2})]
        for name, data in (("manifest", manifests), ("result", result), ("planner_stats", stats),
                           ("pricing", {"results": records})):
            if name == "result" and problem == "missing_result" and len(commands) == 2:
                continue
            (target / f"{name}.json").write_text(json.dumps(data))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(compare_gpu_checks.subprocess, "run", simulate)
    status = compare_gpu_checks.main([
        "--capture", str(capture), "--out", str(output), "--metal-lock", str(tmp_path / "lock"),
        "--flight-ids", "3", "2", "--budget", "120", "--max-labels", "12345",
        "--epsilon", "0.002", "--repeats", "2",
    ])
    report = json.loads((output / "comparison.json").read_text())
    rows = report["runs"]
    assert len(commands) == len(rows) == 6
    assert [r["mode"] for r in rows] == ["full", "epsilon_border", "no_checks",
                                         "epsilon_border", "no_checks", "full"]
    assert status == int(problem is not None)
    assert rows[0]["pricing_bound_certified"] and not rows[0]["pricing_complete"]
    assert rows[0]["first_incomplete_flight"] == 3
    assert rows[0]["cost_minus_full"] == 0.0
    assert rows[2]["cost_minus_full"] == 2.0 and not rows[2]["pricing_bound_certified"]
    assert rows[3]["cost_minus_full"] == 2.0
    assert bool(rows[1].get("error")) == (problem is not None)
    assert rows[1]["cost_minus_full"] == (2.0 if problem is None else None)
    assert (output / "comparison.csv").exists()
