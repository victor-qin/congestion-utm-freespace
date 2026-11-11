"""Submit, track and fetch agent runs on MIT SuperCloud.

    uv run python -m experiments.supercloud doctor [--init] [--gpu]
    uv run python -m experiments.supercloud submit --name NAME --cores N --hours H [--gpus G]
        [--gurobi] [--env NAME=VALUE ...] -- experiments.run --scenario ... --no-progress
    uv run python -m experiments.supercloud status [RUN_ID ...]
    uv run python -m experiments.supercloud fetch RUN_ID
    uv run python -m experiments.supercloud cancel RUN_ID
    uv run python -m experiments.supercloud list

``submit`` ships the caller's worktree exactly as it is on disk, uncommitted edits included, as
a snapshot commit kept locally under ``refs/supercloud/<run_id>``. Nothing is committed to any
branch. The snapshot is extracted and rsynced to ``<remote_root>/runs/<run_id>``, so the cluster
needs no git. The runs ledger lives in the MAIN checkout's ``results/supercloud/``, which every
worktree shares.

Settings and caps live outside the repo in ``~/.config/congestion-utm/supercloud.toml``, and
the caps belong to the user. Template:

    host = "supercloud"                            # ssh alias in ~/.ssh/config
    remote_root = "/home/gridsan/<user>/agent-runs"
    conda_env = "congestion-utm"                   # job runs ~/.conda/envs/<env>/bin/python
    allow_gurobi = true
    # SuperCloud's token-server licence (full academic); --gurobi jobs export it as GRB_LICENSE_FILE
    gurobi_license_file = "/state/partition1/llgrid/pkg/gurobi/gurobi1201/gurobi.lic"

    [caps]
    max_cores_per_job = 48
    max_gpus_per_job = 2
    max_hours_per_job = 48
    max_jobs = 6
    max_cores_total = 192
    max_gpus_total = 4
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CONFIG_PATH = Path("~/.config/congestion-utm/supercloud.toml").expanduser()
JOB_PREFIX = "agent-"
ALLOWED_MODULES = ("experiments.run", "pytest")
CPU_PARTITION = "xeon-p8"
GPU_PARTITION = "xeon-g6-volta"
MAX_SNAPSHOT_FILE_BYTES = 20 * 1024 * 1024
REQUIRED_IMPORTS = ("numpy", "numba", "fcl", "pulp", "scipy", "pandas", "pyarrow", "trimesh")
OPTIONAL_IMPORTS = ("gurobipy", "cupy")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Matches gpu:volta:2, gres:gpu:volta:2, gres/gpu:volta=2 and gpu:2 across Slurm versions.
_GPU_RE = re.compile(r"gpu(?::[A-Za-z0-9_-]+)?[:=](\d+)")
# Login-shell profiles may print to stdout, so remote output is read after this marker only.
_MARK = "__SUPERCLOUD_OUTPUT__"
_REMOTE_TIMEOUT_S = 600


class SuperCloudError(RuntimeError):
    """A SuperCloud operation was refused or failed; the message says why."""


@dataclass(frozen=True)
class Caps:
    """
    Hard limits on agent jobs, set by the user in the config file.

    Per-job limits bound one submission. The totals bound all agent jobs (running plus pending)
    together, counted from the cluster queue at submit time; the user's own jobs are not counted.
    """

    max_cores_per_job: int
    max_gpus_per_job: int
    max_hours_per_job: float
    max_jobs: int
    max_cores_total: int
    max_gpus_total: int


@dataclass(frozen=True)
class SuperCloudConfig:
    """Connection settings and caps, read from ``CONFIG_PATH``."""

    host: str
    remote_root: str
    conda_env: str
    allow_gurobi: bool
    gurobi_license_file: str
    caps: Caps


@dataclass(frozen=True)
class JobRequest:
    """
    One job an agent asks for: a module from ``ALLOWED_MODULES``, its arguments, its resources,
    and extra environment variables (e.g. ``COLGEN_GUROBI_THREADS``) exported before it starts.
    """

    name: str
    module: str
    args: tuple[str, ...]
    cores: int
    gpus: int
    hours: float
    gurobi: bool = False
    env: tuple[tuple[str, str], ...] = ()

    @property
    def partition(self) -> str:
        """GPU jobs need the V100 nodes; every other job goes to the CPU nodes."""
        return GPU_PARTITION if self.gpus > 0 else CPU_PARTITION


@dataclass(frozen=True)
class LiveJob:
    """An agent job currently in the cluster queue, running or pending."""

    job_id: str
    name: str
    cores: int
    gpus: int
    state: str


def load_config(path: Path = CONFIG_PATH) -> SuperCloudConfig:
    """
    Read the user's SuperCloud settings and caps.

    Parameters
    ------------
    - path (Path): TOML file in the format shown in this module's docstring.

    Return
    --------
    - config (SuperCloudConfig): parsed settings; raises SuperCloudError if missing or malformed.
    """
    if not path.exists():
        raise SuperCloudError(f"no config at {path}; ask the user (template: experiments/supercloud.py)")
    raw = tomllib.loads(path.read_text())
    try:
        return SuperCloudConfig(caps=Caps(**raw.pop("caps")), **raw)
    except (KeyError, TypeError) as exc:
        raise SuperCloudError(f"{path} is malformed: {exc!r}") from exc


def parse_squeue(text: str) -> list[LiveJob]:
    """
    Parse ``squeue -h -o '%i|%j|%C|%b|%T'`` output, keeping only agent jobs.

    Parameters
    ------------
    - text (str): squeue output, one job per line.

    Return
    --------
    - jobs (list[LiveJob]): jobs whose name starts with ``JOB_PREFIX``.
    """
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # Split from both ends so a '|' inside some other job's name cannot shift the fields.
        job_id, rest = line.split("|", 1)
        name, cores, gres, state = rest.rsplit("|", 3)
        if name.startswith(JOB_PREFIX):
            gpus = sum(int(n) for n in _GPU_RE.findall(gres))
            jobs.append(LiveJob(job_id, name, int(cores), gpus, state))
    return jobs


def check_request(req: JobRequest, cfg: SuperCloudConfig, live: list[LiveJob]) -> list[str]:
    """
    List every rule a job request breaks: allowed commands, per-job caps and agent-wide totals.

    Parameters
    ------------
    - req (JobRequest): the job being submitted.
    - cfg (SuperCloudConfig): settings holding the caps and whether Gurobi jobs are allowed.
    - live (list[LiveJob]): agent jobs already running or pending.

    Return
    --------
    - problems (list[str]): one message per broken rule; empty means the job may be submitted.
    """
    caps = cfg.caps
    problems = []
    if not _NAME_RE.match(req.name):
        problems.append(f"name {req.name!r} must be 1-41 chars of letters, digits, '_', '.', '-'")
    if req.module not in ALLOWED_MODULES:
        problems.append(f"module {req.module!r} is not allowed; use one of {ALLOWED_MODULES}")
    if req.cores < 1 or req.gpus < 0 or req.hours <= 0:
        problems.append("cores must be >= 1, gpus >= 0 and hours > 0")
    if req.cores > caps.max_cores_per_job:
        problems.append(f"{req.cores} cores exceeds {caps.max_cores_per_job} cores per job")
    if req.gpus > caps.max_gpus_per_job:
        problems.append(f"{req.gpus} GPUs exceeds {caps.max_gpus_per_job} GPUs per job")
    if req.hours > caps.max_hours_per_job:
        problems.append(f"{req.hours} hours exceeds {caps.max_hours_per_job} hours per job")
    if req.gurobi and not cfg.allow_gurobi:
        problems.append("Gurobi jobs are disabled (allow_gurobi = false in the config)")
    bad_env = [name for name, _ in req.env if not _ENV_NAME_RE.match(name)]
    if bad_env:
        problems.append(f"env names {bad_env} must be UPPER_SNAKE_CASE")
    if len(live) + 1 > caps.max_jobs:
        problems.append(f"{len(live)} agent jobs already queued; cap is {caps.max_jobs} in total")
    used_cores = sum(j.cores for j in live)
    if used_cores + req.cores > caps.max_cores_total:
        problems.append(f"{used_cores} + {req.cores} cores exceeds {caps.max_cores_total} total cores")
    used_gpus = sum(j.gpus for j in live)
    if used_gpus + req.gpus > caps.max_gpus_total:
        problems.append(f"{used_gpus} + {req.gpus} GPUs exceeds {caps.max_gpus_total} total GPUs")
    return problems


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    """Run git in ``repo`` and return its stripped stdout; raise with git's stderr on failure."""
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SuperCloudError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def snapshot_commit(worktree: Path, run_id: str,
                    max_file_bytes: int = MAX_SNAPSHOT_FILE_BYTES) -> str:
    """
    Commit the worktree as it is on disk without touching HEAD, any branch, the index or files.

    Staged, unstaged and untracked files are included, and gitignored files are not. A temporary
    index keeps the real one untouched. The commit's parent is HEAD and it is kept alive by the
    local ref ``refs/supercloud/<run_id>``, so the exact code a job ran stays recoverable.

    Parameters
    ------------
    - worktree (Path): top level of the checkout to snapshot.
    - run_id (str): names the ref; must not already exist.
    - max_file_bytes (int): refuse the snapshot if any file is larger, to keep data out of it.

    Return
    --------
    - sha (str): the snapshot commit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(worktree, "read-tree", "HEAD", env=env)
        _git(worktree, "add", "-A", env=env)
        tree = _git(worktree, "write-tree", env=env)
    too_big = []
    for entry in _git(worktree, "ls-tree", "-r", "-l", "-z", tree).split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        size = meta.split()[3]
        if size != "-" and int(size) > max_file_bytes:
            too_big.append(f"{path} ({int(size) / 1e6:.1f} MB)")
    if too_big:
        raise SuperCloudError(f"files over {max_file_bytes / 1e6:.0f} MB would be shipped; "
                              f"gitignore or move them: {', '.join(too_big)}")
    head = _git(worktree, "rev-parse", "HEAD")
    sha = _git(worktree, "commit-tree", "--no-gpg-sign", tree, "-p", head,
               "-m", f"supercloud snapshot {run_id}")
    # Empty old-value: fail rather than overwrite if this run_id was already used.
    _git(worktree, "update-ref", f"refs/supercloud/{run_id}", sha, "")
    return sha


def stage_snapshot(worktree: Path, sha: str, dest: Path) -> None:
    """
    Extract a snapshot commit's files into ``dest``. Shipping this copy, rather than the live
    worktree, means edits made during the upload cannot leak into the run.

    Parameters
    ------------
    - worktree (Path): checkout whose object store holds ``sha``.
    - sha (str): snapshot commit from ``snapshot_commit``.
    - dest (Path): directory to fill; created if missing.

    Return
    --------
    - None
    """
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["git", "-C", str(worktree), "archive", "--format=tar", sha],
                            stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(dest, filter="data")
    if proc.wait() != 0:
        raise SuperCloudError(f"git archive {sha} failed")


def render_job_script(cfg: SuperCloudConfig, req: JobRequest, run_id: str, run_dir: str) -> str:
    """
    Write the batch script that runs one request from its snapshot directory.

    Every Slurm option goes on the sbatch command line instead, because LLsub ignores
    command-line options when a script carries any directive line. The script does not
    activate conda or load modules: in a batch shell `module` is undefined and activation does
    not reliably fix PATH (see experiments/batch/supercloud_array.sh), so it calls the env's
    python by absolute path. PYTHONPATH outranks the env's editable install of the home
    checkout, and the import check turns a wrong-copy import into a hard failure.

    Parameters
    ------------
    - cfg (SuperCloudConfig): supplies the conda env and the Gurobi licence file.
    - req (JobRequest): module, arguments, extra env vars and whether to point at the licence.
    - run_id (str): printed in the log header.
    - run_dir (str): absolute cluster path holding the snapshot.

    Return
    --------
    - script (str): bash source for the job.
    """
    exports = list(req.env)
    if req.gurobi:
        exports.append(("GRB_LICENSE_FILE", cfg.gurobi_license_file))
    # Exported before PYTHONPATH is set below, so no extra variable can redirect the import.
    extra = "".join(f"export {name}={shlex.quote(value)}\n" for name, value in exports)
    command = shlex.join(["-m", req.module, *req.args])
    return f"""#!/bin/bash
# Generated by experiments/supercloud.py for run {run_id}.
set -euo pipefail
unset PYTHONHOME
PY="$HOME/.conda/envs/{cfg.conda_env}/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Node-local and outside $TMPDIR, which SuperCloud deletes when the job ends.
export NUMBA_CACHE_DIR="/state/partition1/user/${{USER:-$(id -un)}}/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null || unset NUMBA_CACHE_DIR
{extra}cd {shlex.quote(run_dir)}
export PYTHONPATH="$PWD"
"$PY" - "$PWD" <<'PYCHECK'
import os, sys
assert sys.version_info >= (3, 12), sys.version
import freespace_sim
where = os.path.realpath(freespace_sim.__file__)
root = os.path.realpath(sys.argv[1]) + os.sep
if not where.startswith(root):
    sys.exit(f"FATAL: freespace_sim imported from {{where}}, not this run's snapshot {{root}}")
PYCHECK
echo "run={run_id} job=${{SLURM_JOB_ID:-none}} host=$(hostname) start=$(date -u +%FT%TZ)"
exec "$PY" {command}
"""


def sbatch_args(req: JobRequest, run_id: str, run_dir: str) -> list[str]:
    """Slurm options for one request: name, partition, cores, GPUs, time limit and log path."""
    args = ["-J", JOB_PREFIX + run_id, "-p", req.partition, "-c", str(req.cores),
            "-t", str(math.ceil(req.hours * 60)), "-D", run_dir,
            "-o", f"{run_dir}/.supercloud/slurm-%j.log"]
    if req.gpus:
        args += ["--gres", f"gpu:volta:{req.gpus}"]
    return args


def _ssh(cfg: SuperCloudConfig, command: str) -> str:
    """
    Run ``command`` in a login bash on the cluster's login node.

    Parameters
    ------------
    - cfg (SuperCloudConfig): supplies the ssh host alias.
    - command (str): bash source; callers quote any untrusted parts.

    Return
    --------
    - stdout (str): output printed by ``command`` itself; raises SuperCloudError on failure.
    """
    remote = f"echo {_MARK}; {command}"
    try:
        proc = subprocess.run(["ssh", cfg.host, "bash -lc " + shlex.quote(remote)],
                              capture_output=True, text=True, timeout=_REMOTE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SuperCloudError(f"ssh {cfg.host} timed out after {_REMOTE_TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise SuperCloudError(f"ssh {cfg.host} exited {proc.returncode} running {command!r}: "
                              f"{proc.stderr.strip()}")
    return proc.stdout.split(_MARK + "\n", 1)[-1]


def _rsync(src: str, dest: str) -> None:
    """Copy ``src`` to ``dest`` with rsync over ssh; raise with rsync's stderr on failure."""
    try:
        proc = subprocess.run(["rsync", "-a", src, dest], capture_output=True, text=True,
                              timeout=_REMOTE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SuperCloudError(f"rsync {src} -> {dest} timed out") from exc
    if proc.returncode != 0:
        raise SuperCloudError(f"rsync {src} -> {dest} failed: {proc.stderr.strip()}")


def live_agent_jobs(cfg: SuperCloudConfig) -> list[LiveJob]:
    """Agent jobs the cluster currently has running or pending for this user."""
    return parse_squeue(_ssh(cfg, "squeue -h -u \"$USER\" -o '%i|%j|%C|%b|%T'"))


def toplevel(path: Path) -> Path:
    """Top level of the checkout containing ``path``."""
    return Path(_git(path, "rev-parse", "--show-toplevel"))


def ledger_dir(worktree: Path) -> Path:
    """
    The main checkout's ``results/supercloud``. It is resolved through git's common dir so every
    worktree shares one ledger and one submit lock.
    """
    common = Path(_git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    path = common.parent / "results" / "supercloud"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` so concurrent submits on this machine serialize."""
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def read_ledger(worktree: Path) -> list[dict]:
    """Every run recorded in the shared ledger, oldest first."""
    path = ledger_dir(worktree) / "runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_run(worktree: Path, run_id: str) -> dict:
    """The ledger record for ``run_id``; raises SuperCloudError if there is none."""
    for record in read_ledger(worktree):
        if record["run_id"] == run_id:
            return record
    raise SuperCloudError(f"no run {run_id!r} in {ledger_dir(worktree) / 'runs.jsonl'}")


def submit(cfg: SuperCloudConfig, req: JobRequest, worktree: Path) -> dict:
    """
    Check the caps, ship a snapshot of ``worktree`` to the cluster and submit it to Slurm.

    The lock spans the cap check through the ledger write, so two agents on this machine cannot
    both pass the check and then exceed the caps together.

    Parameters
    ------------
    - cfg (SuperCloudConfig): connection settings and caps.
    - req (JobRequest): what to run and with which resources.
    - worktree (Path): top level of the checkout whose files the job should run.

    Return
    --------
    - record (dict): the ledger entry, including ``run_id``, ``job_id`` and ``sha``.
    """
    ledger = ledger_dir(worktree)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{req.name}"
    run_dir = f"{cfg.remote_root}/runs/{run_id}"
    with _locked(ledger / ".submit.lock"):
        problems = check_request(req, cfg, live_agent_jobs(cfg))
        if problems:
            raise SuperCloudError("submit refused:\n  " + "\n  ".join(problems))
        sha = snapshot_commit(worktree, run_id)
        record = {
            "run_id": run_id, "sha": sha, "base_sha": _git(worktree, "rev-parse", "HEAD"),
            "branch": _git(worktree, "rev-parse", "--abbrev-ref", "HEAD"),
            "worktree": str(worktree), "module": req.module, "args": list(req.args),
            "cores": req.cores, "gpus": req.gpus, "hours": req.hours, "gurobi": req.gurobi,
            "env": dict(req.env),
            "partition": req.partition, "run_dir": run_dir,
            "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / run_id
            stage_snapshot(worktree, sha, stage)
            meta = stage / ".supercloud"
            meta.mkdir()
            (meta / "job.sh").write_text(render_job_script(cfg, req, run_id, run_dir))
            (meta / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
            _ssh(cfg, f"mkdir -p {shlex.quote(cfg.remote_root + '/runs')}")
            _rsync(f"{stage}/", f"{cfg.host}:{run_dir}/")
        out = _ssh(cfg, shlex.join(["sbatch", "--parsable", *sbatch_args(req, run_id, run_dir),
                                    f"{run_dir}/.supercloud/job.sh"]))
        record["job_id"] = out.strip().splitlines()[-1].split(";")[0]
        with open(ledger / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def status(cfg: SuperCloudConfig, records: list[dict]) -> list[dict]:
    """
    Slurm state, elapsed time and peak memory for recorded runs.

    Parameters
    ------------
    - cfg (SuperCloudConfig): connection settings.
    - records (list[dict]): ledger entries to look up.

    Return
    --------
    - rows (list[dict]): ``run_id``, ``job_id``, ``state``, ``elapsed`` and ``max_rss_gb`` per run.
    """
    if not records:
        return []
    ids = ",".join(r["job_id"] for r in records)
    out = _ssh(cfg, f"sacct -n -P --units=G -j {shlex.quote(ids)} -o JobID,State,Elapsed,MaxRSS")
    found: dict[str, dict] = {}
    for line in out.splitlines():
        if line.count("|") != 3:
            continue
        job, state, elapsed, max_rss = line.split("|")
        row = found.setdefault(job.split(".")[0], {"state": "", "elapsed": "", "max_rss_gb": 0.0})
        if "." not in job:
            row["state"], row["elapsed"] = state, elapsed
        # Peak memory is reported on the job's steps (e.g. 123.batch), not the allocation line.
        try:
            row["max_rss_gb"] = max(row["max_rss_gb"], float(max_rss.rstrip("G")))
        except ValueError:
            pass
    missing = {"state": "UNKNOWN", "elapsed": "", "max_rss_gb": 0.0}
    return [{"run_id": r["run_id"], "job_id": r["job_id"], **found.get(r["job_id"], missing)}
            for r in records]


def fetch(cfg: SuperCloudConfig, record: dict, dest_root: Path) -> Path:
    """
    Copy a run's ``results/`` and its job files and Slurm log back to this machine.

    Parameters
    ------------
    - cfg (SuperCloudConfig): connection settings.
    - record (dict): ledger entry of the run.
    - dest_root (Path): local parent directory; the run lands in ``dest_root/<run_id>``.

    Return
    --------
    - dest (Path): local directory holding the copied files.
    """
    run_dir = record["run_dir"]
    present = _ssh(cfg, f"cd {shlex.quote(run_dir)} && for d in .supercloud results; "
                        "do if [ -d \"$d\" ]; then echo \"$d\"; fi; done").split()
    dest = dest_root / record["run_id"]
    for sub in present:
        (dest / sub).mkdir(parents=True, exist_ok=True)
        _rsync(f"{cfg.host}:{run_dir}/{sub}/", f"{dest / sub}/")
    return dest


def cancel(cfg: SuperCloudConfig, record: dict) -> None:
    """Cancel a recorded run; ``--name`` makes Slurm refuse any job that is not this agent job."""
    _ssh(cfg, shlex.join(["scancel", f"--name={JOB_PREFIX}{record['run_id']}", record["job_id"]]))


def doctor(cfg: SuperCloudConfig, init: bool, gpu: bool = False) -> bool:
    """
    Check every piece a submit depends on and print one line per check.

    Parameters
    ------------
    - cfg (SuperCloudConfig): connection settings.
    - init (bool): also create the remote runs directory.
    - gpu (bool): also run a short job on a V100 node (waits for the GPU queue).

    Return
    --------
    - ok (bool): True when every check passed; the optional imports only inform.
    """
    py = f'"$HOME/.conda/envs/{cfg.conda_env}/bin/python"'
    probe = ("import importlib, json, sys\n"
             "out = {'python': sys.version.split()[0]}\n"
             f"for m in {REQUIRED_IMPORTS + OPTIONAL_IMPORTS!r}:\n"
             "    try:\n"
             "        out[m] = getattr(importlib.import_module(m), '__version__', 'ok')\n"
             "    except Exception as exc:\n"
             "        out[m] = 'MISSING: ' + type(exc).__name__ + ': ' + str(exc)[:80]\n"
             "print(json.dumps(out))\n")

    def env_check() -> str:
        found = json.loads(_ssh(cfg, f"{py} -c {shlex.quote(probe)}").strip().splitlines()[-1])
        broken = [m for m in REQUIRED_IMPORTS if str(found[m]).startswith("MISSING")]
        if tuple(int(x) for x in found["python"].split(".")[:2]) < (3, 12) or broken:
            raise SuperCloudError(json.dumps(found))
        return json.dumps(found)

    def gurobi_check() -> str:
        # Above the pip wheel's 2000-variable trial limit, which fails only at optimize().
        solve = ("import gurobipy as gp\n"
                 "m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1\n"
                 "x = m.addVars(3000, ub=1.0)\n"
                 "m.addConstrs(x[i] + x[i + 1] <= 1 for i in range(2999))\n"
                 "m.setObjective(x.sum(), gp.GRB.MAXIMIZE); m.optimize()\n"
                 "print('solved 3000 vars, status', m.Status, 'gurobipy', gp.gurobi.version())\n")
        lic = shlex.quote(cfg.gurobi_license_file)
        return _ssh(cfg, f"GRB_LICENSE_FILE={lic} {py} -c {shlex.quote(solve)}").strip().splitlines()[-1]

    def gpu_check() -> str:
        # Same NVRTC option as the pricing kernels (gpu_runtime.py), compiled for the node's GPU.
        kernel = ("import cupy as cp\n"
                  "k = cp.RawKernel(r'extern \"C\" __global__ void inc(float* x) "
                  "{ x[threadIdx.x] += 1.0f; }', 'inc', options=('--fmad=false',))\n"
                  "x = cp.zeros(32, dtype=cp.float32); k((1,), (32,), (x,))\n"
                  "props = cp.cuda.runtime.getDeviceProperties(0)\n"
                  "print(props['name'].decode(), 'sm', cp.cuda.Device().compute_capability,\n"
                  "      'cuda', cp.cuda.runtime.runtimeGetVersion(),\n"
                  "      'driver', cp.cuda.runtime.driverGetVersion(), 'sum', float(x.sum()))\n")
        out = _ssh(cfg, f"srun -p {GPU_PARTITION} --gres=gpu:volta:1 -c 20 -t 10 "
                        f"-J {JOB_PREFIX}doctor-gpu {py} -c {shlex.quote(kernel)}")
        if "sum 32.0" not in out:
            raise SuperCloudError(out.strip())
        return out.strip().splitlines()[-1]

    def partitions_check() -> str:
        names = {p.rstrip("*") for p in _ssh(cfg, "sinfo -h -o %P").split()}
        missing = {CPU_PARTITION, GPU_PARTITION} - names
        if missing:
            raise SuperCloudError(f"missing {sorted(missing)}; cluster has {sorted(names)}")
        return f"{CPU_PARTITION}, {GPU_PARTITION} present"

    def rsync_check() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            probe_dir = Path(tmp) / "out"
            probe_dir.mkdir()
            (probe_dir / "probe.txt").write_text("supercloud doctor\n")
            remote = f"{cfg.host}:{cfg.remote_root}/.doctor/"
            _rsync(f"{probe_dir}/", remote)
            _rsync(remote, f"{Path(tmp) / 'back'}/")
            if (Path(tmp) / "back" / "probe.txt").read_text() != "supercloud doctor\n":
                raise SuperCloudError("round trip changed the file")
        return "round trip ok"

    checks = [
        ("ssh login", lambda: _ssh(cfg, "hostname").strip()),
        ("runs directory", lambda: _ssh(
            cfg, (f"mkdir -p {shlex.quote(cfg.remote_root)}/runs && " if init else "")
            + f"test -w {shlex.quote(cfg.remote_root)}/runs && echo writable")),
        ("partitions", partitions_check),
        ("python env", env_check),
        ("rsync", rsync_check),
    ]
    if cfg.allow_gurobi:
        checks.append(("gurobi licence", gurobi_check))
    if gpu:
        checks.append(("gpu kernel", gpu_check))
    ok = True
    for label, check in checks:
        try:
            print(f"OK    {label}: {str(check()).strip()}")
        except (SuperCloudError, OSError, ValueError) as exc:
            ok = False
            print(f"FAIL  {label}: {exc}")
    return ok


def _parse_command(command: list[str]) -> tuple[str, tuple[str, ...]]:
    """Split ``-- [python -m] MODULE ARGS...`` into the module and its arguments."""
    if command[:1] == ["--"]:
        command = command[1:]
    if command[:2] == ["python", "-m"]:
        command = command[2:]
    if not command:
        raise SuperCloudError("nothing to run: end with `-- experiments.run ...` or `-- pytest ...`")
    return command[0], tuple(command[1:])


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m experiments.supercloud",
                                     description="Submit, track and fetch agent runs on SuperCloud.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_doc = sub.add_parser("doctor", help="check ssh, cluster paths, python env and rsync")
    p_doc.add_argument("--init", action="store_true", help="create the remote runs directory")
    p_doc.add_argument("--gpu", action="store_true",
                       help="also compile and run a CUDA kernel on a V100 node (queues a job)")
    p_sub = sub.add_parser("submit", help="snapshot this worktree and submit one job")
    p_sub.add_argument("--name", required=True, help="short label, part of the run id")
    p_sub.add_argument("--cores", type=int, required=True,
                       help="CPU cores; also buys memory (xeon-p8: 4 GB per core)")
    p_sub.add_argument("--gpus", type=int, default=0, help="V100 GPUs (about 20 cores each)")
    p_sub.add_argument("--hours", type=float, required=True, help="wall-time limit")
    p_sub.add_argument("--gurobi", action="store_true",
                       help="use SuperCloud's full Gurobi licence (sets GRB_LICENSE_FILE)")
    p_sub.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                       help="export a variable for the job, e.g. COLGEN_GUROBI_THREADS=8")
    p_sub.add_argument("command", nargs=argparse.REMAINDER,
                       help="-- experiments.run ARGS... | -- pytest ARGS...")
    p_st = sub.add_parser("status", help="Slurm state of recorded runs (default: last 10)")
    p_st.add_argument("run_ids", nargs="*")
    for name, text in (("fetch", "copy results and logs back"), ("cancel", "cancel a run")):
        sub.add_parser(name, help=text).add_argument("run_id")
    sub.add_parser("list", help="every run in the ledger")
    args = parser.parse_args(argv)

    try:
        worktree = toplevel(Path.cwd())
        if args.cmd == "list":
            for r in read_ledger(worktree):
                print(f"{r['run_id']}  job {r['job_id']}  {r['branch']}  "
                      f"{r['module']} {shlex.join(r['args'])}")
            return 0
        cfg = load_config()
        if args.cmd == "doctor":
            return 0 if doctor(cfg, args.init, args.gpu) else 1
        if args.cmd == "submit":
            module, mod_args = _parse_command(args.command)
            env = tuple(tuple(item.split("=", 1)) for item in args.env)
            if any(len(pair) != 2 for pair in env):
                raise SuperCloudError(f"--env takes NAME=VALUE, got {args.env}")
            req = JobRequest(args.name, module, mod_args, args.cores, args.gpus, args.hours,
                             args.gurobi, env)
            record = submit(cfg, req, worktree)
            print(f"submitted run {record['run_id']} as job {record['job_id']} "
                  f"(snapshot {record['sha'][:12]}, {record['partition']})")
            return 0
        if args.cmd == "status":
            records = ([find_run(worktree, rid) for rid in args.run_ids] if args.run_ids
                       else read_ledger(worktree)[-10:])
            for row in status(cfg, records):
                print(f"{row['run_id']}  job {row['job_id']}  {row['state'] or 'UNKNOWN'}  "
                      f"elapsed {row['elapsed'] or '-'}  peak {row['max_rss_gb']:.1f} GB")
            return 0
        record = find_run(worktree, args.run_id)
        if args.cmd == "fetch":
            print(fetch(cfg, record, ledger_dir(worktree)))
        else:
            cancel(cfg, record)
            print(f"cancelled {record['run_id']} (job {record['job_id']})")
        return 0
    except SuperCloudError as exc:
        print(f"supercloud: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
