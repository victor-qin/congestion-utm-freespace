"""SuperCloud agent runs without a network: caps, shipped code and the generated job."""

import subprocess
from dataclasses import replace

import pytest

from experiments import supercloud as sc

LICENCE = "/state/partition1/llgrid/pkg/gurobi/gurobi1201/gurobi.lic"
CAPS = sc.Caps(max_cores_per_job=48, max_gpus_per_job=2, max_hours_per_job=48, max_jobs=3,
               max_cores_total=64, max_gpus_total=2)
CFG = sc.SuperCloudConfig(host="supercloud", remote_root="/home/gridsan/u/agent-runs",
                          conda_env="congestion-utm", allow_gurobi=False,
                          gurobi_license_file=LICENCE, caps=CAPS)
RUN_DIR = "/home/gridsan/u/agent-runs/runs/r1"
# Two agent jobs holding 40 cores and 1 GPU, plus the user's own job, which caps must ignore:
# counting it (48 cores, 2 GPUs) would refuse even the default request below.
SQUEUE = "\n".join([
    "101|agent-20261111T000000Z-a|20|gres/gpu:volta:1|RUNNING",
    "102|agent-20261111T000001Z-b|20|N/A|PENDING",
    "103|my|own|job|48|gpu:volta:2|RUNNING",
])


def _req(**changes) -> sc.JobRequest:
    """A request within every cap; tests override one field at a time."""
    base = sc.JobRequest(name="t", module="experiments.run", args=(), cores=8, gpus=0, hours=2.0)
    return replace(base, **changes)


@pytest.mark.parametrize("req, cfg, refusal", [
    (_req(), CFG, None),
    (_req(module="pytest", gpus=1, cores=20), CFG, None),
    (_req(module="os"), CFG, "not allowed"),
    (_req(name="../x"), CFG, "name"),
    (_req(cores=49), CFG, "cores per job"),
    (_req(gpus=3), CFG, "GPUs per job"),
    (_req(hours=48.5), CFG, "hours per job"),
    (_req(gurobi=True), CFG, "Gurobi"),
    (_req(gurobi=True), replace(CFG, allow_gurobi=True), None),
    (_req(cores=25), CFG, "total cores"),
    (_req(gpus=2), CFG, "total GPUs"),
    (_req(), replace(CFG, caps=replace(CAPS, max_jobs=2)), "agent jobs already queued"),
    (_req(env=(("lower", "1"),)), CFG, "UPPER_SNAKE_CASE"),
])
def test_check_request_enforces_caps(req, cfg, refusal):
    live = sc.parse_squeue(SQUEUE)
    assert [(j.cores, j.gpus) for j in live] == [(20, 1), (20, 0)]
    problems = sc.check_request(req, cfg, live)
    if refusal is None:
        assert problems == []
    else:
        assert any(refusal in p for p in problems), problems


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def test_snapshot_ships_working_tree_without_touching_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "tracked.py").write_text("v1\n")
    (repo / "staged.py").write_text("s1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tracked.py").write_text("v2\n")
    (repo / "staged.py").write_text("s2\n")
    _git(repo, "add", "staged.py")
    (repo / "new.py").write_text("n\n")
    (repo / "ignored.txt").write_text("x\n")

    def state():
        return (_git(repo, "rev-parse", "HEAD"), _git(repo, "symbolic-ref", "HEAD"),
                _git(repo, "status", "--porcelain"), _git(repo, "diff", "--cached"))

    before = state()
    sha = sc.snapshot_commit(repo, "r1")
    assert state() == before
    assert _git(repo, "rev-parse", "refs/supercloud/r1").strip() == sha
    assert _git(repo, "rev-parse", f"{sha}^") == before[0]

    stage = tmp_path / "stage"
    sc.stage_snapshot(repo, sha, stage)
    shipped = {p.relative_to(stage).as_posix(): p.read_text()
               for p in stage.rglob("*") if p.is_file()}
    assert shipped == {".gitignore": "ignored.txt\n", "tracked.py": "v2\n",
                       "staged.py": "s2\n", "new.py": "n\n"}

    with pytest.raises(sc.SuperCloudError):
        sc.snapshot_commit(repo, "r1")  # a reused run id must not overwrite the recorded code
    (repo / "big.bin").write_bytes(b"0" * 2048)
    with pytest.raises(sc.SuperCloudError, match="big.bin"):
        sc.snapshot_commit(repo, "r2", max_file_bytes=1024)


@pytest.mark.parametrize("gurobi, gpus", [(False, 0), (True, 1)])
def test_job_runs_the_snapshot(gurobi, gpus):
    req = _req(args=("--scenario", "a b; rm -rf ~"), gurobi=gurobi, gpus=gpus, cores=20,
               env=(("COLGEN_GUROBI_THREADS", "8"),))
    script = sc.render_job_script(CFG, req, "r1", RUN_DIR)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "#SBATCH" not in script  # LLsub drops command-line options when any is present
    assert 'export PYTHONPATH="$PWD"' in script and "not this run's snapshot" in script
    assert (f"export GRB_LICENSE_FILE={LICENCE}" in script) is gurobi
    # Extra variables come before PYTHONPATH is set, so none can redirect the import.
    assert script.index("export COLGEN_GUROBI_THREADS=8") < script.index("export PYTHONPATH")
    assert script.rstrip().endswith(
        "exec \"$PY\" -m experiments.run --scenario 'a b; rm -rf ~'")

    args = sc.sbatch_args(req, "r1", RUN_DIR)
    assert args[args.index("-p") + 1] == ("xeon-g6-volta" if gpus else "xeon-p8")
    assert ("gpu:volta:1" in args) is bool(gpus)
    assert args[args.index("-t") + 1] == "120"
