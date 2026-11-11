"""SuperCloud agent runs without a network: caps, shipped code, the generated job and the lock."""

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

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


@pytest.fixture(autouse=True)
def _private_lock(tmp_path, monkeypatch):
    """Keep tests off the real machine-wide submit lock next to the user's config."""
    monkeypatch.setattr(sc, "LOCK_PATH", tmp_path / "supercloud.lock")


def _req(**changes) -> sc.JobRequest:
    """A request within every cap; tests override one field at a time."""
    base = sc.JobRequest(name="t", module="experiments.run", args=(), cores=8, gpus=0, hours=2.0)
    return replace(base, **changes)


@pytest.mark.parametrize("req, cfg, refusal", [
    (_req(), CFG, None),
    (_req(module="pytest", gpus=1, cores=20), CFG, None),
    (_req(module="os"), CFG, "not allowed"),
    (_req(name="../x"), CFG, "name"),
    (_req(name="a..b"), CFG, "name"),
    (_req(name="x.lock"), CFG, "name"),
    (_req(cores=49), CFG, "cores per job"),
    (_req(gpus=3), CFG, "GPUs per job"),
    (_req(hours=48.5), CFG, "hours per job"),
    (_req(hours=math.nan), CFG, "finite"),
    (_req(gurobi=True), CFG, "Gurobi"),
    (_req(gurobi=True), replace(CFG, allow_gurobi=True), None),
    (_req(env=(("COLGEN_GUROBI_THREADS", "8"),)), CFG, None),
    (_req(env=(("PYTHONHASHSEED", "0"),)), CFG, None),  # reproducibility, not a code redirect
    (_req(env=(("lower", "1"),)), CFG, "UPPER_SNAKE_CASE"),
    (_req(env=(("GRB_LICENSE_FILE", LICENCE),)), CFG, "reserved"),
    (_req(env=(("PYTHONHOME", "/x"),)), CFG, "reserved"),
    (_req(module="experiments.colgen_four_round", args=("--capture", "c.pkl")), CFG, None),
    (_req(module="experiments.colgen_four_round", args=("--source", "/x")), CFG, "--source"),
    (_req(module="experiments.colgen_four_round", args=("--sour=/x",)), CFG, "--source"),
    (_req(cores=25), CFG, "total cores"),
    (_req(gpus=2), CFG, "total GPUs"),
    (_req(), replace(CFG, caps=replace(CAPS, max_jobs=2)), "agent jobs already queued"),
])
def test_check_request_enforces_caps(req, cfg, refusal):
    live = sc.parse_squeue(SQUEUE)
    assert [(j.cores, j.gpus) for j in live] == [(20, 1), (20, 0)]
    problems = sc.check_request(req, cfg, live)
    if refusal is None:
        assert problems == []
    else:
        assert any(refusal in p for p in problems), problems


@pytest.mark.parametrize("key, value", [
    ("allow_gurobi", '"false"'),        # a string would read as truthy and enable Gurobi
    ("max_jobs", "true"),               # bool is an int subclass
    ("remote_root", '"~/agent-runs"'),  # shell quoting would keep the "~" literal
    ("max_jobs", ""),                   # TOML syntax error
])
def test_load_config_rejects_malformed_settings(tmp_path, key, value):
    top = {"host": '"supercloud"', "remote_root": '"/home/gridsan/u/agent-runs"',
           "conda_env": '"congestion-utm"', "allow_gurobi": "false",
           "gurobi_license_file": f'"{LICENCE}"'}
    caps = {"max_cores_per_job": "48", "max_gpus_per_job": "2", "max_hours_per_job": "48",
            "max_jobs": "6", "max_cores_total": "192", "max_gpus_total": "4"}
    path = tmp_path / "supercloud.toml"

    def write(top: dict, caps: dict) -> None:
        path.write_text("".join(f"{k} = {v}\n" for k, v in top.items()) + "[caps]\n"
                        + "".join(f"{k} = {v}\n" for k, v in caps.items()))

    write(top, caps)
    assert sc.load_config(path).caps.max_jobs == 6
    write({**top, key: value} if key in top else top, {**caps, key: value} if key in caps else caps)
    with pytest.raises(sc.SuperCloudError):
        sc.load_config(path)


def test_parse_command_accepts_python_dash_m():
    assert sc._parse_command(["--", "python", "-m", "pytest", "-q"]) == ("pytest", ("-q",))
    assert sc._parse_command(["experiments.run", "--seed", "1"]) == ("experiments.run",
                                                                     ("--seed", "1"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    return path


def test_snapshot_ships_working_tree_without_touching_it(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("ignored.txt\n")
    for name, text in (("tracked.py", "v1\n"), ("staged.py", "s1\n"), ("gone.py", "g\n")):
        (repo / name).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tracked.py").write_text("v2\n")
    (repo / "staged.py").write_text("s2\n")
    _git(repo, "add", "staged.py")
    (repo / "new.py").write_text("n\n")
    (repo / "ignored.txt").write_text("x\n")
    (repo / "gone.py").unlink()

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
    (repo / "big.bin").unlink()
    # A link out of the tree would dangle on the cluster: refused with a message, not a traceback.
    (repo / "link").symlink_to("/etc/hosts")
    with pytest.raises(sc.SuperCloudError, match="link"):
        sc.stage_snapshot(repo, sc.snapshot_commit(repo, "r3"), tmp_path / "stage3")


@pytest.mark.parametrize("gurobi, gpus", [(False, 0), (True, 1)])
def test_job_runs_the_snapshot(tmp_path, gurobi, gpus):
    # The decoy package plays the cluster env's editable install of ~/congestion-utm-freespace,
    # which is on the env python's path behind PYTHONPATH.
    decoy = tmp_path / "decoy"
    (decoy / "freespace_sim").mkdir(parents=True)
    (decoy / "freespace_sim" / "__init__.py").write_text("")
    home = tmp_path / "home"
    env_python = home / ".conda" / "envs" / CFG.conda_env / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.write_text("#!/bin/bash\n"
                          f'export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}{decoy}"\n'
                          f'exec {shlex.quote(sys.executable)} "$@"\n')
    env_python.chmod(0o755)
    run_dir = tmp_path / "run dir"
    (run_dir / "freespace_sim").mkdir(parents=True)
    (run_dir / "freespace_sim" / "__init__.py").write_text("")
    (run_dir / "probe.py").write_text(
        "import json, os, sys, freespace_sim\n"
        "print(json.dumps({'pkg': os.path.realpath(freespace_sim.__file__), 'argv': sys.argv[1:],"
        " 'threads': os.environ.get('COLGEN_GUROBI_THREADS'),"
        " 'licence': os.environ.get('GRB_LICENSE_FILE')}))\n")
    # PY is a legal --env name; it must not be able to replace the job's interpreter.
    req = _req(module="probe", args=("--scenario", "a b; rm -rf ~"), gurobi=gurobi, gpus=gpus,
               cores=20, env=(("COLGEN_GUROBI_THREADS", "8"), ("PY", "/bin/false")))
    script = sc.render_job_script(CFG, req, "r1", str(run_dir))
    assert "#SBATCH" not in script  # LLsub drops command-line options when any is present
    # sbatch copies the login shell's environment into the job; an inherited licence must not
    # reach a job that did not ask for --gurobi.
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "USER": "u",
           "GRB_LICENSE_FILE": "/inherited/gurobi.lic"}

    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out == {"pkg": os.path.realpath(run_dir / "freespace_sim" / "__init__.py"),
                   "argv": ["--scenario", "a b; rm -rf ~"], "threads": "8",
                   "licence": LICENCE if gurobi else None}

    shutil.rmtree(run_dir / "freespace_sim")  # now only the decoy is importable
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env,
                          timeout=60)
    assert done.returncode != 0 and "FATAL" in done.stderr

    args = sc.sbatch_args(req, "r1", RUN_DIR)
    assert args[args.index("-p") + 1] == ("xeon-g6-volta" if gpus else "xeon-p8")
    assert ("gpu:volta:1" in args) is bool(gpus)
    assert args[args.index("-t") + 1] == "120"


def test_parse_sacct_takes_state_from_the_job_and_memory_from_its_steps():
    rows = sc.parse_sacct("\n".join([
        "6219869|COMPLETED|00:01:50|", "6219869.batch|COMPLETED|00:01:50|1.25G",
        "6219869.extern|COMPLETED|00:01:50|0.01G", "7|PENDING|00:00:00|"]))
    assert rows == {"6219869": {"state": "COMPLETED", "elapsed": "00:01:50", "max_rss_gb": 1.25},
                    "7": {"state": "PENDING", "elapsed": "00:00:00", "max_rss_gb": 0.0}}


def test_concurrent_submits_from_two_worktrees_cannot_exceed_caps(tmp_path, monkeypatch):
    main = _init_repo(tmp_path / "main")
    (main / "a.py").write_text("committed\n")
    _git(main, "add", ".")
    _git(main, "commit", "-q", "-m", "base")
    side = tmp_path / "side"
    _git(main, "worktree", "add", "-q", str(side), "-b", "side")
    (side / "a.py").write_text("uncommitted in worktree\n")
    expected = {"one": "committed\n", "two": "uncommitted in worktree\n"}

    queue = ["101|agent-old|8|N/A|RUNNING"]  # one live agent job; max_jobs=2 leaves one slot
    removed = []
    guard = threading.Lock()
    both_uploading = threading.Barrier(2)

    def fake_ssh(cfg, command):
        words = command.split()
        if words[0] == "squeue":
            snapshot = "\n".join(queue)
            time.sleep(0.2)  # a slow queue query: without the lock both submits read one slot
            return snapshot
        with guard:
            if words[:2] == ["rm", "-rf"]:
                removed.append(shlex.split(command)[2])
            if words[0] == "sbatch":
                queue.append(f"{200 + len(queue)}|{words[words.index('-J') + 1]}|8|N/A|PENDING")
                return queue[-1].split("|")[0] + "\n"
        return ""

    def fake_rsync(src, dest):
        both_uploading.wait(timeout=30)  # both have passed the pre-check before either locks

    monkeypatch.setattr(sc, "_ssh", fake_ssh)
    monkeypatch.setattr(sc, "_rsync", fake_rsync)
    cfg = replace(CFG, caps=replace(CAPS, max_jobs=2))
    outcomes = {}

    def run(name, tree):
        try:
            outcomes[name] = sc.submit(cfg, _req(name=name), tree)
        except sc.SuperCloudError as exc:
            outcomes[name] = exc

    threads = [threading.Thread(target=run, args=args) for args in (("one", main), ("two", side))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    accepted = {n: r for n, r in outcomes.items() if isinstance(r, dict)}
    refused = [r for r in outcomes.values() if isinstance(r, sc.SuperCloudError)]
    assert len(accepted) == 1 and len(refused) == 1, outcomes
    assert "agent jobs already queued" in str(refused[0])
    (name, record), = accepted.items()
    # One ledger in the MAIN checkout, whichever worktree won.
    ledger = (main / "results" / "supercloud" / "runs.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in ledger] == [record["run_id"]]
    # The refused snapshot's ref and upload are gone; the accepted one holds that worktree's
    # on-disk files.
    assert _git(main, "for-each-ref", "--format=%(refname)", "refs/supercloud").split() == [
        f"refs/supercloud/{record['run_id']}"]
    assert _git(main, "show", f"{record['sha']}:a.py") == expected[name]
    other = next(n for n in expected if n != name)
    assert len(removed) == 1 and removed[0].startswith(f"{CFG.remote_root}/runs/")
    assert removed[0].endswith(f"-{other}")


def test_gpu_probe_respects_caps_and_is_cancelled_when_interrupted(monkeypatch, capsys):
    commands = []
    queue = "101|agent-a|20|gpu:volta:2|RUNNING"  # every GPU the caps allow is in use

    def fake_ssh(cfg, command):
        commands.append(command)
        if command.startswith("squeue"):
            return queue
        if command.startswith("sbatch"):
            return "555\n"
        if command.startswith("sacct"):
            raise KeyboardInterrupt  # the agent's tool timed out while doctor waited
        return ""

    monkeypatch.setattr(sc, "_ssh", fake_ssh)
    monkeypatch.setattr(sc, "_rsync", lambda src, dest: None)
    assert sc.doctor(CFG, gpu=True) is False
    assert "FAIL  gpu kernel: submit refused" in capsys.readouterr().out
    assert not any(c.startswith("sbatch") for c in commands)

    queue = ""
    with pytest.raises(KeyboardInterrupt):
        sc.doctor(CFG, gpu=True)
    assert any(c.startswith("scancel") and c.endswith(" 555") for c in commands)


def test_sigterm_during_submit_leaves_nothing_half_recorded(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "a.py").write_text("a\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    config = tmp_path / "supercloud.toml"
    config.write_text('host = "supercloud"\nremote_root = "/r"\nconda_env = "e"\n'
                      f'allow_gurobi = false\ngurobi_license_file = "{LICENCE}"\n[caps]\n'
                      "max_cores_per_job = 4\nmax_gpus_per_job = 0\nmax_hours_per_job = 1\n"
                      "max_jobs = 1\nmax_cores_total = 4\nmax_gpus_total = 0\n")
    uploading = tmp_path / "uploading"
    driver = (
        "import sys, time\nfrom pathlib import Path\nfrom experiments import supercloud as sc\n"
        f"sc.CONFIG_PATH = Path({str(config)!r})\n"
        f"sc.LOCK_PATH = Path({str(tmp_path / 'lock')!r})\n"
        "sc._ssh = lambda cfg, command: ''\n"
        f"def slow_rsync(src, dest):\n    Path({str(uploading)!r}).touch(); time.sleep(60)\n"
        "sc._rsync = slow_rsync\n"
        "sys.exit(sc.main(['submit', '--name', 'x', '--cores', '1', '--hours', '1', '--',"
        " 'pytest']))\n")
    root = Path(sc.__file__).resolve().parents[1]
    proc = subprocess.Popen([sys.executable, "-c", driver], cwd=repo, text=True,
                            env={**os.environ, "PYTHONPATH": str(root)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while not uploading.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert uploading.exists(), proc.communicate()
    proc.send_signal(signal.SIGTERM)  # what an agent harness sends when a tool call times out
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130, err
    assert _git(repo, "for-each-ref", "refs/supercloud") == ""
