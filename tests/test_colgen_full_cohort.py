"""The full-cohort entry point resolves field overrides before any solve work."""
from __future__ import annotations

import argparse
import hashlib
import json

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
