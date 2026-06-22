"""Readouts — standalone consumers of persisted runs (per-run folder / cross-run index).

Each test saves run(s) to a tmp results root, then invokes a readout's ``main`` with patched argv and
checks it emits its artifact. The readouts must never re-simulate — they only read what's on disk.
"""

import sys

from freespace_sim import runs
from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.sim import run

from experiments.readouts import compare, curve, figures, histograms, replay, uss_breakdown


def _two_uss_run(lam=120.0, seed=1):
    cfg = SimConfig(planner="straight", lam_per_hour=lam, horizon_s=600.0, seed=seed,
                    region_size_m=(3000.0, 3000.0))
    return run(cfg, demand=UniformPoissonDemand(uss_ids=("uss_a", "uss_b")))


def _save(tmp_path, *, lam=120.0, seed=1, tag="t"):
    return runs.save_run(_two_uss_run(lam, seed), root=tmp_path, label=tag,
                         scenario="metro_2uss", demand="uniform", write_replay=False)


def _main(monkeypatch, mod, argv):
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()


def test_replay_readout_writes_html(tmp_path, monkeypatch):
    folder = _save(tmp_path)
    assert not (folder / "replay.html").exists()         # execute persisted data only
    _main(monkeypatch, replay, ["replay", str(folder)])
    assert (folder / "replay.html").stat().st_size > 0


def test_figures_readout_writes_pngs(tmp_path, monkeypatch):
    folder = _save(tmp_path)
    _main(monkeypatch, figures, ["figures", str(folder), "--no-3d"])
    assert (folder / "snapshot.png").stat().st_size > 0
    assert (folder / "heatmap.png").stat().st_size > 0


def test_figures_readout_uss_slice(tmp_path, monkeypatch):
    folder = _save(tmp_path)
    _main(monkeypatch, figures, ["figures", str(folder), "--no-3d", "--uss", "uss_a"])
    assert (folder / "snapshot_uss_a.png").stat().st_size > 0


def test_uss_breakdown_readout(tmp_path, monkeypatch, capsys):
    folder = _save(tmp_path)
    _main(monkeypatch, uss_breakdown, ["uss_breakdown", str(folder)])
    assert (folder / "uss_breakdown.png").stat().st_size > 0
    assert "uss_a" in capsys.readouterr().out                # printed the per-USS table


def test_curve_readout_writes_to_sweep_dir(tmp_path, monkeypatch):
    # two λ points under one tag → a curve the readout builds from the index alone (no re-sim),
    # self-located into <root>/sweeps/<tag>/ (not loose in the results root)
    _save(tmp_path, lam=60.0, tag="sw")
    _save(tmp_path, lam=180.0, tag="sw")
    _main(monkeypatch, curve, ["curve", "--tag", "sw", "--root", str(tmp_path)])
    assert (tmp_path / "sweeps" / "sw" / "curve.png").stat().st_size > 0


def test_compare_readout_from_index(tmp_path, monkeypatch, capsys):
    _save(tmp_path, seed=0, tag="cmp")
    _save(tmp_path, seed=1, tag="cmp")
    _main(monkeypatch, compare, ["compare", "--tag", "cmp", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "straight" in out and "n_runs" in out             # grouped table printed
    assert (tmp_path / "sweeps" / "cmp" / "compare.csv").stat().st_size > 0


def test_histograms_readout_per_run_writes_into_folder(tmp_path, monkeypatch):
    # per-run: one run folder in → its delay distributions written next to it
    folder = _save(tmp_path)
    _main(monkeypatch, histograms, ["histograms", str(folder)])
    for f in ("delay_hist.png", "delay_pct_hist.png", "delay_sources.png"):
        assert (folder / f).stat().st_size > 0


def test_histograms_readout_collects_into_out_dir(tmp_path, monkeypatch):
    # when the shell feeds many runs into one folder, files are run-name-prefixed (no clobber)
    folder = _save(tmp_path)
    coll = tmp_path / "sweeps" / "mysweep"
    _main(monkeypatch, histograms, ["histograms", str(folder), "--out-dir", str(coll)])
    assert (coll / f"{folder.name}_delay_hist.png").stat().st_size > 0
