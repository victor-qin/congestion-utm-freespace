"""Sync run folders to/from the cloud — push a saved run up, or pull run(s) back down.

Storage is a Hugging Face Hub **dataset repo**: free, private/public toggle per repo, durable (HF
hosts the git-LFS/S3), and its Dataset Viewer renders the parquet artifacts (flights / per_uss /
index) in-browser. Large files route through HF's LFS automatically. This is deliberately *separate*
from ``experiments.run`` (which only runs + persists a scenario) — sync is an orthogonal operation on
the ``results/`` folders that run produces.

Requires the optional ``cloud`` extra and a one-time login::

    uv sync --extra cloud
    uv run huggingface-cli login            # or set $HF_TOKEN

Repo id comes from ``--remote`` or ``$FREESPACE_HF_REPO``. Usage::

    # push one saved run (private by default; --no-private makes the repo public):
    uv run python -m experiments.push_pull_results push results/<run-folder> --remote you/freespace-runs

    # pull run(s) back into ./results/ — a run-folder name, 'all', or just 'index':
    uv run python -m experiments.push_pull_results pull all   --remote you/freespace-runs
    uv run python -m experiments.push_pull_results pull index --remote you/freespace-runs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# huggingface_hub is imported lazily so a plain checkout (without the 'cloud' extra) still imports
# this module — the friendly install hint fires only when you actually try to sync.
def _require_hf():
    try:
        import huggingface_hub as hf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "cloud sync needs huggingface_hub: `uv sync --extra cloud`, "
            "then authenticate once with `uv run huggingface-cli login` (or set $HF_TOKEN)."
        ) from exc
    return hf


def _resolve_remote(remote: str | None) -> str:
    repo = remote or os.environ.get("FREESPACE_HF_REPO")
    if not repo:
        raise SystemExit("no cloud repo: pass --remote <user/dataset> or set $FREESPACE_HF_REPO")
    return repo


def push_run(folder: str, remote: str | None, *, private: bool) -> str:
    """Upload one run folder to ``results/<name>/`` in the dataset repo, creating the repo at the
    requested visibility if absent. Returns the browsable URL."""
    hf, repo, f = _require_hf(), _resolve_remote(remote), Path(folder)
    if not f.is_dir():
        raise SystemExit(f"not a run folder: {folder}")
    api = hf.HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(f),
                      path_in_repo=f"results/{f.name}", commit_message=f"push run {f.name}")
    return f"https://huggingface.co/datasets/{repo}/tree/main/results/{f.name}"


def pull_runs(target: str, remote: str | None, *, root: str = ".") -> str:
    """Download run(s) from the dataset repo into ``./results/``. ``target`` is a run-folder name, or
    ``'all'`` for every run, or ``'index'`` for just ``results/index.parquet``. Returns the local
    results path."""
    hf, repo = _require_hf(), _resolve_remote(remote)
    patterns = {"all": ["results/**"], "index": ["results/index.parquet"]}.get(
        target, [f"results/{target}/**"])
    hf.snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=root)
    return str(Path(root) / "results")


def _add_remote(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--remote", default=None, help="HF Hub dataset repo id (else $FREESPACE_HF_REPO)")


def main() -> None:
    p = argparse.ArgumentParser(description="Push/pull run folders to a Hugging Face Hub dataset repo.")
    sub = p.add_subparsers(dest="action", required=True)

    pp = sub.add_parser("push", help="upload a saved run folder to the cloud repo")
    pp.add_argument("folder", help="path to a results/<run> folder")
    _add_remote(pp)
    pp.add_argument("--private", action=argparse.BooleanOptionalAction, default=True,
                    help="repo visibility: --private (default) or --no-private for public")

    pl = sub.add_parser("pull", help="download run(s) from the cloud repo into ./results/")
    pl.add_argument("target", help="a run-folder name, 'all', or 'index'")
    _add_remote(pl)

    args = p.parse_args()
    if args.action == "push":
        url = push_run(args.folder, args.remote, private=args.private)
        print(f"pushed → {url}", file=sys.stderr)
        print(url)   # stdout: the browsable URL
    else:
        path = pull_runs(args.target, args.remote)
        print(f"pulled {args.target!r} → {path}", file=sys.stderr)
        print(path)  # stdout: the local results path


if __name__ == "__main__":
    main()
