"""The full-cohort entry point resolves field overrides before any solve work."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments import colgen_full_cohort as cohort
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.zeta import ZetaOptions


def _resolve(tmp_path, content, params=(), zeta=()):
    """Resolve one options file (raw text or JSON-able object) plus command-line pairs."""
    path = tmp_path / "options.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    args = argparse.Namespace(options_file=path, param=list(params), zeta_option=list(zeta))
    return path, cohort.resolve_overrides(args)


def test_options_file_merges_under_command_line_and_builds_both_dataclasses(tmp_path):
    """File fields reach ColGenParams and ZetaOptions; command-line pairs win; the file is recorded.

    Command-line words are typed by the field: ``false`` is a bool and an int widens to float.
    """
    content = {"params": {"lp_gap": 0.01, "turn_below_90": True},
               "zeta_options": {"completion_bounds": ["arrival"], "traversal": "best_first",
                                "max_labels": 1000}}
    path, (params, zeta, record) = _resolve(tmp_path, content,
                                            params=["lp_gap=0.002", "time_limit_s=1200"],
                                            zeta=["max_labels=5", "forward_cruise=false"])
    assert params == {"lp_gap": 0.002, "turn_below_90": True, "time_limit_s": 1200.0}
    assert isinstance(params["time_limit_s"], float)
    assert zeta == {"completion_bounds": ("arrival",), "traversal": "best_first", "max_labels": 5,
                    "forward_cruise": False}
    assert record == {"path": str(path), "content": content,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert ColGenParams(**params).turn_below_90 is True
    options = ZetaOptions(**{**cohort.ZETA_DEFAULTS, **zeta, "turn_below_90": True})
    assert options.completion_bounds == ("arrival",) and options.max_labels == 5


@pytest.mark.parametrize("content,zeta,message", [
    ({"params": {"lp_gapp": 0.01}}, (), r"\['lp_gapp'\] are not ColGenParams fields"),
    ({"zeta_options": {"price_snapp": True}}, (), r"\['price_snapp'\] are not ZetaOptions fields"),
    ({"zeta": {}}, (), "only 'params' and 'zeta_options'"),
    ({"params": [1]}, (), "must be a JSON object"),
    ({"zeta_options": {"turn_below_90": True}}, (), "params field turn_below_90"),
    ("{not json", (), "invalid JSON"),
    ({"zeta_options": {"forward_cruise": "false"}}, (), "forward_cruise='false': expected bool"),
    ({}, ("max_labels=True",), "max_labels=True: expected int"),
    ({"params": {"lp_gap": None}}, (), "lp_gap: None is not allowed"),
    ({"zeta_options": {"completion_bounds": "arrival"}}, (), "expected tuple"),
], ids=["unknown-param", "unknown-zeta-option", "unknown-section", "non-object", "turn-in-zeta",
        "bad-json", "string-for-bool", "bool-for-int", "none-for-float", "string-for-tuple"])
def test_options_reject_unknown_names_and_wrong_types_before_any_work(tmp_path, content, zeta, message):
    """A typo, misplaced field or wrongly typed value stops the run with a message naming it."""
    with pytest.raises(SystemExit, match=message):
        _resolve(tmp_path, content, zeta=zeta)


def test_reports_are_written_once_and_never_partially(tmp_path):
    """A second writer to the same report fails and leaves the first report and no temp file."""
    out = tmp_path / "run" / "zeta.json"
    cohort.write_new(out, b"first")
    with pytest.raises(SystemExit, match="never overwrite"):
        cohort.write_new(out, b"second")
    assert out.read_bytes() == b"first"
    assert [path.name for path in out.parent.iterdir()] == ["zeta.json"]


@pytest.mark.parametrize("group,limits,env,expected", [
    ("0::/job/step", {"/sys/fs/cgroup/memory.max": "max",
                      "/sys/fs/cgroup/job/memory.max": str(16 * 2**30)}, {}, 16),
    ("0::/job", {"/sys/fs/cgroup/job/memory.max": str(32 * 2**30)},
     {"SLURM_MEM_PER_NODE": "8192"}, 8),
    ("5:cpu,memory:/job", {"/sys/fs/cgroup/memory/job/memory.limit_in_bytes": str(4 * 2**30)},
     {}, 4),
    ("0::/../../job", {"/sys/fs/cgroup/memory.max": str(12 * 2**30)}, {}, 12),
    ("", {}, {"SLURM_MEM_PER_CPU": "2048", "SLURM_CPUS_ON_NODE": "3"}, 6),
    ("0::/job", {"/sys/fs/cgroup/job/memory.max": "max"}, {}, 64),
], ids=["parent-cgroup", "slurm-tighter", "v1", "namespace", "per-cpu", "physical"])
def test_memory_limit_respects_job_and_ancestor_limits(monkeypatch, group, limits, env, expected):
    """A search budget uses the smallest actual allocation, including an enclosing cgroup."""
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cohort.os, "sysconf", lambda key: {
        "SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 64 * 2**30 // 4096,
    }[key])

    def read(path):
        """Expose the simulated process's allocation files, with other files absent."""
        if str(path) == "/proc/self/cgroup":
            return group
        if str(path) in limits:
            return limits[str(path)]
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", read)
    assert cohort.memory_limit()[0] == expected * 2**30


@pytest.mark.parametrize("requested", [None, 4096, 100 * 2**30])
def test_search_sizing_cannot_bypass_allocation_or_tighter_budget(monkeypatch, requested):
    """Explicit settings may tighten the cap; None, huge budgets and cache eviction cannot lift it."""
    monkeypatch.setattr(cohort, "memory_limit", lambda: (64 * 2**30, "job"))
    params = ColGenParams(n_pricing_workers=44)
    baseline, record = cohort.size_search_memory(params, 2, None, None)
    assert record["effective_workers"] == 2
    assert record["search_memory_bytes"] == baseline.zeta_search_memory_bytes
    configured = replace(params, zeta_search_memory_bytes=requested, zeta_worker_cache_bytes=0)
    sized, details = cohort.size_search_memory(configured, 2, 128.0, None)
    assert sized.zeta_search_memory_bytes == min(
        baseline.zeta_search_memory_bytes,
        requested if requested is not None else baseline.zeta_search_memory_bytes,
    )
    assert details["memory_limit_bytes"] == 64 * 2**30
    lower, _ = cohort.size_search_memory(params, 2, 32.0, None)
    assert lower.zeta_search_memory_bytes < baseline.zeta_search_memory_bytes
    larger_parent, _ = cohort.size_search_memory(params, 2, None, 16.0)
    assert larger_parent.zeta_search_memory_bytes < baseline.zeta_search_memory_bytes
    with pytest.raises(ValueError, match="fewer workers"):
        cohort.size_search_memory(params, 2, 4.0, None)


@pytest.mark.parametrize("limit,reserve", [(0.0, None), (float("inf"), None),
                                           (None, -1.0), (None, float("nan"))])
def test_search_sizing_rejects_invalid_memory_settings(limit, reserve):
    """Malformed limits fail before starting workers or allocating search state."""
    with pytest.raises(ValueError, match="finite and positive"):
        cohort.size_search_memory(ColGenParams(), 400, limit, reserve)


def test_sequence_preserves_memory_settings_and_frozen_input(tmp_path, monkeypatch):
    """Both child arms receive the same memory settings and the same frozen input bytes."""
    source = tmp_path / "input.json"
    source.write_bytes(b"{}")
    args = argparse.Namespace(out=tmp_path / "result.json", input=source, workers=44,
                              gurobi_threads=None, options_file=None, memory_limit_gib=32.0,
                              parent_reserve_gib=8.0, count=400, param=[], zeta_option=[],
                              flight_records=False)
    commands = []
    monkeypatch.setattr(cohort.subprocess, "run", lambda command, **kw: commands.append(command))
    cohort.run_sequence(args, ["dp", "zeta"])
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--memory-limit-gib") + 1] == "32.0"
        assert command[command.index("--parent-reserve-gib") + 1] == "8.0"
        assert Path(command[command.index("--input") + 1]).read_bytes() == b"{}"
