"""Configuration and CLI contracts for full-solver GPU pricing selection."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from analysis.colgen import benchmark
from freespace_sim.planner.colgen.params import ColGenParams


@pytest.mark.parametrize("before", ["gpu", "endpoint"])
def test_legacy_capture_parameters_preserve_values_and_fill_gpu_defaults(before: str) -> None:
    """Old positional slot states remain readable without shifting existing solver settings."""
    original = ColGenParams(
        n_pricing_workers=2,
        bootstrap_roots=3,
        bootstrap_method="dp",
        columns_per_flight=2,
        objective="total_delay",
        ip_time_limit_s=42.0,
    )
    legacy_names = [
        field.name
        for field in fields(ColGenParams)
        if (field.name not in benchmark._PRICING_DEFAULTS if before == "gpu"
            else not field.name.startswith("pricing_gpu_endpoint_"))
    ]
    legacy = object.__new__(ColGenParams)
    legacy.__setstate__([getattr(original, name) for name in legacy_names])
    restored = ColGenParams(
        **{
            field.name: getattr(legacy, field.name)
            for field in fields(ColGenParams)
            if hasattr(legacy, field.name)
        }
    )
    assert all(getattr(restored, name) == getattr(original, name) for name in legacy_names)
    assert restored.pricing_device == "cpu" and restored.pricing_gpu_backend is None
    assert restored.pricing_gpu_endpoint_bounds and restored.pricing_gpu_endpoint_max_rows == 64


def test_cpu_default_and_explicit_gpu_configuration() -> None:
    """CPU remains the default, while Metal/CUDA selections retain every resource control."""
    default = ColGenParams()
    assert default.pricing_device == "cpu" and default.pricing_gpu_backend is None
    for backend in ("metal", "cuda"):
        selected = ColGenParams(
            pricing_device="gpu",
            pricing_gpu_backend=backend,
            pricing_cpu_recompute=False,
            pricing_gpu_batch_size=3,
            pricing_gpu_max_labels=1000,
            pricing_gpu_chunk_size=128,
            pricing_gpu_endpoint_bounds=False,
            pricing_gpu_endpoint_max_rows=32,
            pricing_gpu_endpoint_cache_bytes=12345,
            pricing_gpu_endpoint_cache_work=45678,
            n_pricing_workers=2,
        )
        assert selected.pricing_gpu_backend == backend
        assert not selected.pricing_cpu_recompute
        assert selected.pricing_gpu_batch_size == 3
        assert selected.pricing_gpu_max_labels == 1000
        assert selected.pricing_gpu_chunk_size == 128
        from freespace_sim.planner.colgen.pricing_pool import _gpu_options
        options = _gpu_options(selected)
        assert not options.endpoint_bounds and options.endpoint_max_rows == 32
        assert options.endpoint_cache_bytes == 12345 and options.endpoint_cache_work == 45678


@pytest.mark.parametrize(
    "overrides",
    [
        {"pricing_device": "invalid"},
        {"pricing_device": "gpu"},
        {"pricing_gpu_backend": "metal"},
        {"pricing_device": "gpu", "pricing_gpu_backend": "invalid"},
        {"pricing_device": "gpu", "pricing_gpu_backend": "metal", "columns_per_flight": 2},
        {"pricing_device": "gpu", "pricing_gpu_backend": "metal", "cheap_pricing": True},
        {"pricing_cpu_recompute": 1},
        {"pricing_gpu_batch_size": 0},
        {"pricing_gpu_batch_size": True},
        {"pricing_gpu_max_labels": 2**27},
        {"pricing_gpu_chunk_size": 2**27},
        {"pricing_gpu_chunk_size": 1.5},
        {"pricing_gpu_endpoint_bounds": 1},
        {"pricing_gpu_endpoint_max_rows": 65},
        {"pricing_gpu_endpoint_max_rows": -1},
        {"pricing_gpu_endpoint_cache_bytes": True},
        {"pricing_gpu_endpoint_cache_work": -1},
    ],
)
def test_invalid_or_unsupported_pricing_selection_fails(overrides: dict) -> None:
    """Unsupported modes and resource limits cannot silently alter requested pricing."""
    with pytest.raises((ValueError, TypeError)):
        ColGenParams(**overrides)


@pytest.mark.parametrize(
    "extra",
    [
        ["--pricing-device", "gpu"],
        ["--pricing-gpu-backend", "metal"],
        ["--pricing-gpu-max-labels", "0"],
        ["--pricing-device", "gpu", "--pricing-gpu-backend", "metal", "--columns-per-flight", "2"],
        [
            "--pricing-device",
            "gpu",
            "--pricing-gpu-backend",
            "metal",
            "--version",
            "iteration_ip_cheap",
        ],
    ],
)
def test_cli_rejects_invalid_pricing_before_running(
    monkeypatch, tmp_path: Path, extra: list[str]
) -> None:
    """Invalid GPU CLI choices fail before any warmup, solver or output mutation."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "child",
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--scenario",
            "colgen_test",
            "--seed",
            "0",
            "--flights",
            "1",
            "--version",
            "iteration_ip_eager",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as failed:
        benchmark.main()
    assert failed.value.code == 2
    assert not (tmp_path / "out").exists()


def test_paired_benchmark_forwards_gpu_controls_and_preserves_kernel_source(
    monkeypatch, tmp_path: Path
) -> None:
    """The child receives GPU options, and benchmark provenance includes the shared shader."""
    source = tmp_path / "source"
    package = source / "freespace_sim"
    package.mkdir(parents=True)
    (package / "pricing_gpu.h").write_text("// shared GPU source\n")
    output = tmp_path / "out"
    seen = []
    real_run = benchmark.subprocess.run

    class CapturedInvocation(BaseException):
        """Stop after observing the intended subprocess without running a simulation."""

    def capture(command, **kwargs):
        if "child" not in command:
            return real_run(command, **kwargs)
        seen.append(command)
        raise CapturedInvocation

    monkeypatch.setattr(benchmark.subprocess, "run", capture)
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "run",
            "--baseline",
            str(source),
            "--fixed",
            str(source),
            "--out",
            str(output),
            "--versions",
            "fixed",
            "--scenarios",
            "colgen_test",
            "--seeds",
            "0",
            "--pricing-device",
            "gpu",
            "--pricing-gpu-backend",
            "metal",
            "--no-pricing-cpu-recompute",
            "--pricing-gpu-batch-size",
            "3",
            "--pricing-gpu-max-labels",
            "1000",
            "--pricing-gpu-chunk-size",
            "128",
        ],
    )
    with pytest.raises(CapturedInvocation):
        benchmark.main()
    parsed = []
    monkeypatch.setattr(benchmark, "child", lambda args: parsed.append(args) or 0)
    monkeypatch.setattr(sys, "argv", seen[0][1:])
    assert benchmark.main() == 0
    actual = benchmark._pricing_options(parsed[0])
    assert actual == dict(
        pricing_device="gpu",
        pricing_gpu_backend="metal",
        pricing_cpu_recompute=False,
        pricing_gpu_batch_size=3,
        pricing_gpu_max_labels=1000,
        pricing_gpu_chunk_size=128,
        pricing_gpu_endpoint_bounds=True,
        pricing_gpu_endpoint_max_rows=64,
        pricing_gpu_endpoint_cache_bytes=1_048_576,
        pricing_gpu_endpoint_cache_work=1_000_000,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(manifest[key] == value for key, value in actual.items())
    assert "freespace_sim/pricing_gpu.h" in manifest["fixed_source_sha256"]


def test_cpu_default_benchmark_remains_compatible_with_archived_parameters(
    monkeypatch, tmp_path: Path
) -> None:
    """A default CPU command adds no new required kwargs to archived parameter classes."""
    parsed = []
    monkeypatch.setattr(benchmark, "child", lambda args: parsed.append(args) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "child",
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--scenario",
            "colgen_test",
            "--seed",
            "0",
            "--flights",
            "1",
            "--version",
            "iteration_ip_eager",
        ],
    )
    assert benchmark.main() == 0
    assert benchmark._pricing_kwargs(parsed[0]) == {}


def test_gpu_benchmark_warmup_and_main_run_use_requested_backend(
    monkeypatch, tmp_path: Path
) -> None:
    """A GPU benchmark warms the selected backend and passes the same selection to production."""
    pytest.importorskip("numba")
    from freespace_sim import sim
    from freespace_sim.planner.colgen import dp_kernel, solver

    calls = []

    class ReachedMainSimulation(BaseException):
        """Stop before running the production workload after observing its options."""

    def observe_run(cfg, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ReachedMainSimulation
        return SimpleNamespace(verified=True, planner_stats={"gpu_dispatches": 1})

    def refuse_cpu_warmup():
        raise AssertionError("GPU benchmarks must not warm the CPU pricing kernel")

    # The child instruments these for an isolated process; restore them in this unit test.
    monkeypatch.setattr(solver, "_initial_feasible_selection", solver._initial_feasible_selection)
    monkeypatch.setattr(solver.ColGenSolver, "solve", solver.ColGenSolver.solve)
    monkeypatch.setattr(sim, "run", observe_run)
    monkeypatch.setattr(dp_kernel, "warm_kernel", refuse_cpu_warmup)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "child",
            "--root",
            str(Path(benchmark.__file__).resolve().parents[2]),
            "--out",
            str(tmp_path / "out"),
            "--scenario",
            "colgen_test",
            "--seed",
            "0",
            "--flights",
            "1",
            "--demand-seconds",
            "4",
            "--version",
            "iteration_ip_eager",
            "--pricing-device",
            "gpu",
            "--pricing-gpu-backend",
            "metal",
            "--no-pricing-cpu-recompute",
            "--pricing-gpu-batch-size",
            "3",
            "--pricing-gpu-max-labels",
            "1000",
            "--pricing-gpu-chunk-size",
            "128",
        ],
    )
    with pytest.raises(ReachedMainSimulation):
        benchmark.main()
    assert len(calls) == 2
    for call in calls:
        params = call["planner_params"]
        assert params.pricing_device == "gpu" and params.pricing_gpu_backend == "metal"
        assert not params.pricing_cpu_recompute
        assert params.pricing_gpu_batch_size == 3
        assert params.pricing_gpu_max_labels == 1000 and params.pricing_gpu_chunk_size == 128
    assert calls[0]["planner_params"].seed_nominal_routes
    assert len(calls[0]["requests"]) == 1
    assert calls[0]["static_terminals"] == []
