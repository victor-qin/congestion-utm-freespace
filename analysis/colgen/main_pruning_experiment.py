"""Build pinned source exports and compare PR146 with and without PR144 on Metal.

This creates no Git checkouts and changes no branches or PRs. Each measurement
uses a separate process, its own source directory and the shared benchmark lock.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MAIN = "0cbd284a4d46b42416b5ab1ef09ebf7f903ac13c"
FEATURE = "78f1d4d2504bb8b67a535865049012bd16583918"
PACKAGE = Path("freespace_sim/planner/colgen")
SHARED = Path("/Users/victor/Documents/Research/congestion-utm-freespace")
LOCK = SHARED / ".agent/gpu-followups-20261106/metal-benchmark.lock"
CAPTURE = SHARED / ".agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl"
LATE = SHARED / ".agent/runs/colgen-capture-20261105-211249/pricing_captures/round_0011.pkl"
FEATURE_FILES = (
    "gpu_integer.py", "gpu_integer_bounds.py", "gpu_integer_search.py",
    "gpu_integer_source.py", "gpu_approach.py", "pricing_integer.h",
    "pricing_integer_bounds.h", "pricing_integer_search.h",
)
MODIFIED_MAIN = {
    str(PACKAGE / name) for name in (
        "gpu_runtime.py", "gpu_pricing.py", "pricing_gpu.h", "params.py",
        "pricing_platform.py", "pricing_pool.py",
    )
}
HARNESSES = ("common.py", "platform_replay.py", "gpu_validation.py", "execution_replay.py")


@dataclass(frozen=True)
class Variant:
    """One source and numerical configuration.

    Attributes
    ------------
    - source (str): isolated source export name.
    - flags (tuple[str, ...]): explicitly enabled pricing improvements.
    """

    source: str
    flags: tuple[str, ...]


VARIANTS = {
    "main": Variant("main", ()),
    "main-integer": Variant("port", ("--integer-bounds",)),
    "main-pr146": Variant("port", ("--integer-bounds", "--approach-bounds")),
    "stacked-reference": Variant("stacked", ()),
    "stacked-integer": Variant("stacked", ("--integer-bounds",)),
    "stacked-pr146": Variant("stacked", ("--integer-bounds", "--approach-bounds")),
}


def git_bytes(*args: str) -> bytes:
    """Read pinned Git objects without changing the repository or worktree."""
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(data: bytes) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    """Write a complete JSON generation atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def export(commit: str, target: Path) -> None:
    """Extract a pinned commit into a fresh ordinary source directory.

    Parameters
    ------------
    - commit (str): immutable source object; never a moving branch name.
    - target (Path): nonexistent destination inside this bound worktree.

    Return
    --------
    - output (None): writes tracked files, without a .git directory or checkout.
    """
    target.mkdir()
    with tarfile.open(fileobj=io.BytesIO(git_bytes("archive", commit))) as archive:
        archive.extractall(target, filter="data")


def install_harness(target: Path) -> None:
    """Adapt analysis arguments to each source's actual public options.

    Parameters
    ------------
    - target (Path): source export receiving identical measurement helpers.

    Return
    --------
    - output (None): installs analysis-only adapters; production code is untouched.
    """
    helper = '''
def compatible_options(cls, **kwargs):
    """Reject unsupported enabled features; omit absent disabled options."""
    names = {field.name for field in fields(cls)}
    for name, value in kwargs.items():
        if name not in names and value not in (None, False):
            raise ValueError(f"Unsupported enabled option: {name}")
    return cls(**{key: value for key, value in kwargs.items() if key in names})


def compatible_replace(value, **kwargs):
    """Preserve every original field while validating requested overrides."""
    current = {field.name: getattr(value, field.name) for field in fields(value)}
    return compatible_options(type(value), **(current | kwargs))

'''
    for name in HARNESSES:
        path = Path("analysis/colgen") / name
        text = git_bytes("show", f"{FEATURE}:{path}").decode()
        if name in ("gpu_validation.py", "execution_replay.py"):
            text = text.replace("PricingOptions(", "compatible_options(PricingOptions, ")
            text = text.replace("pricing_platform.compatible_options(PricingOptions,",
                                "compatible_options(pricing_platform.PricingOptions,")
            text = text.replace("replace(params,", "compatible_replace(params,")
            text = text.replace(
                'ComputeRuntime("metal", integer_bounds=args.integer_bounds)',
                '(ComputeRuntime("metal", integer_bounds=True) if args.integer_bounds '
                'else ComputeRuntime("metal"))',
            )
            text = text.replace('if __name__ == "__main__":', helper + '\nif __name__ == "__main__":')
        if name == "execution_replay.py":
            # The port binds integer arrays inside run; include those copies in
            # the main runtime estimate. Stacked runtime supplies actual counters.
            old = '        yield measured\n'
            new = '''        if getattr(runtime, "integer_bounds", False) and not hasattr(runtime, "counters"):
            from freespace_sim.planner.colgen.gpu_integer_search import IntegerData
            original_arguments = IntegerData.arguments

            def numeric_arguments(owner, step):
                arrays = original_arguments(owner, step)
                for index, array in enumerate(arrays):
                    cached = runtime._static.get(f"integer_{index}")
                    if cached is None or cached[0] is not array:
                        measured["run_upload_bytes"] += array.nbytes
                return arrays

            stack.enter_context(patch.object(IntegerData, "arguments", numeric_arguments))
        yield measured
'''
            assert text.count(old) == 1
            text = text.replace(old, new)
        (target / path).write_text(text)


def build(destination: Path) -> dict:
    """Construct main, main plus the numerical port, and the original stacked PR.

    Parameters
    ------------
    - destination (Path): fresh experiment directory inside this worktree.

    Return
    --------
    - output (dict): exact source hashes, provenance and absence checks.
    """
    destination.mkdir(parents=True, exist_ok=False)
    sources = destination / "sources"
    sources.mkdir()
    for name, commit in (("main", MAIN), ("port", MAIN), ("stacked", FEATURE)):
        export(commit, sources / name)
    port = sources / "port"
    for relative in [*(PACKAGE / name for name in FEATURE_FILES), Path("tests/test_colgen_gpu_integer.py")]:
        (port / relative).write_bytes(git_bytes("show", f"{FEATURE}:{relative}"))
    patch = Path(__file__).with_name("main_pruning_port.patch")
    subprocess.run(["patch", "--batch", "-p1", "-i", str(patch)], cwd=port, check=True)
    for name in ("gpu_buffers.py", "gpu_resident.py", "gpu_prepare.py", "pricing_resident.h"):
        assert not (port / PACKAGE / name).exists(), f"PR144-only module present: {name}"
    changed = []
    diff = []
    main = sources / "main"
    for path in sorted((port / "freespace_sim").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(port)
        original = main / relative
        if original.exists() and original.read_bytes() != path.read_bytes():
            assert str(relative) in MODIFIED_MAIN, f"Unexpected main modification: {relative}"
            changed.append(str(relative))
            diff.extend(difflib.unified_diff(original.read_text().splitlines(True),
                        path.read_text().splitlines(True), fromfile="main/" + str(relative),
                        tofile="port/" + str(relative)))
        elif not original.exists():
            assert relative.parent == PACKAGE and relative.name in FEATURE_FILES
    (destination / "production-port.diff").write_text("".join(diff))
    hashes = {}
    for name in ("main", "port", "stacked"):
        source = sources / name
        install_harness(source)
        hashes[name] = {str(p.relative_to(source)): digest(p.read_bytes())
                        for folder in ("freespace_sim", "analysis/colgen", "tests")
                        for p in sorted((source / folder).rglob("*"))
                        if p.is_file() and p.suffix in (".py", ".h")}
    manifest = dict(main_commit=MAIN, feature_commit=FEATURE,
                    patch_sha256=digest(patch.read_bytes()), sources=hashes,
                    main_production_files_modified_in_port=changed,
                    pr144_production_modules_absent=True,
                    description="Pinned main plus adapted PR146; not a retargeted Git branch")
    save(destination / "build.json", manifest)
    return manifest


def run_locked(source: Path, module: str, arguments: list[str], log: Path) -> int:
    """Run a guarded child from its own source while holding the shared benchmark lock.

    Parameters
    ------------
    - source (Path): exact module import root and working directory.
    - module (str): analysis or pytest entry point.
    - arguments (list[str]): explicit measurement or functional-test arguments.
    - log (Path): combined child output; echoed after completion.

    Return
    --------
    - output (int): child exit code; incomplete solver results remain explicit artifacts.
    """
    bootstrap = (
        "import pathlib,runpy,sys; import freespace_sim; "
        "assert pathlib.Path(freespace_sim.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
        f"sys.argv=[{module!r},*sys.argv[1:]]; runpy.run_module({module!r},run_name='__main__')"
    )
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE="1", COLGEN_GUROBI_THREADS="1")
    with LOCK.open("a") as lock:
        print(f"Waiting for CPU/Metal lock: {log.name}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"Running {log.name}; output: {log}", flush=True)
        with log.open("w") as output:
            process = subprocess.run([sys.executable, "-c", bootstrap, *arguments], cwd=source,
                                     env=env, stdout=output, stderr=subprocess.STDOUT)
    print(log.read_text(), flush=True)
    return process.returncode


def replay_summary(paths: dict[str, list[Path]]) -> dict:
    """Report warm timing only alongside matched certified objectives.

    Parameters
    ------------
    - paths (dict[str, list[Path]]): complete raw reports for each experiment arm.

    Return
    --------
    - output (dict): phase details, warm medians and conditional relative timings.
    """
    summary = {}
    baseline = None
    capture_hashes = None
    for name, files in paths.items():
        phases, warm = [], []
        matched = bool(files)
        for path in files:
            report = json.loads(path.read_text())
            matched &= bool(report["runs"])
            if capture_hashes is None:
                capture_hashes = report["capture_sha256"]
            if report["capture_sha256"] != capture_hashes:
                raise ValueError("Cannot compare different captured price snapshots")
            for run in report["runs"]:
                rows = run["results"]
                rc = [(r["flight_id"], r["reduced_cost"]) for r in rows]
                gaps = [r["upper_bound"] - r["reduced_cost"]
                        for r in rows if isinstance(r["upper_bound"], (int, float))
                        and isinstance(r["reduced_cost"], (int, float))]
                complete = (bool(rows) and all(r["certified"] and r["search_complete"] for r in rows)
                            and len(gaps) == len(rows)
                            and all(math.isfinite(gap) and 0 <= gap <= 1.001e-6 for gap in gaps))
                # Compare corresponding snapshots, not early versus changed prices.
                key = "changed" if run["label"] in ("round11_changed", "round11_warm") else "early"
                if baseline is None:
                    baseline = {}
                reference = baseline.setdefault(key, rc)
                same = len(reference) == len(rc) and all(
                    a[0] == b[0] and isinstance(a[1], (int, float)) and isinstance(b[1], (int, float))
                    and abs(a[1] - b[1]) <= 1.001e-6 for a, b in zip(reference, rc)
                )
                matched &= complete and same
                phases.append(dict(raw=str(path), phase=run["label"], wall_s=run["elapsed_s"],
                                   certified=sum(r["certified"] for r in rows), flights=len(rows),
                                   matches_first_objectives=same, complete=complete,
                                   max_certificate_gap=max(gaps, default=None),
                                   process_peak_rss_bytes=run.get("process_peak_rss_bytes"),
                                   labels=sum(r["stats"].get("labels", 0) for r in rows),
                                   cpu_numeric_checks=sum(r["stats"].get("cpu_recomputations", 0)
                                                          for r in rows)))
                if run["label"].startswith("round1_warm_"):
                    warm.append(run["elapsed_s"])
        summary[name] = dict(phases=phases, warm_median_s=statistics.median(warm) if warm else None,
                             all_complete_matching_objectives=matched)
    if summary:
        first = next(iter(summary.values()))
        for value in summary.values():
            comparable = first["all_complete_matching_objectives"] and value["all_complete_matching_objectives"]
            value["matched_warm_reduction_percent_vs_first"] = (
                100 * (1 - value["warm_median_s"] / first["warm_median_s"])
                if comparable and value["warm_median_s"] and first["warm_median_s"] else None
            )
    return summary


def main() -> int:
    """Build and optionally run the selected pinned comparison.

    Parameters
    ------------
    - none: options are parsed from the command line.

    Return
    --------
    - output (int): nonzero for failed children; raw incomplete outcomes are preserved.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh directory inside this worktree")
    parser.add_argument("--mode", choices=("build", "test", "replay", "solve"), default="replay")
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+",
                        default=["main", "main-pr146", "stacked-pr146"])
    parser.add_argument("--cohort", choices=("hard88", "late18"), default="hard88")
    parser.add_argument("--smoke", action="store_true", help="Only flights 940 and 2224; not hard88 evidence")
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--budget", type=float, default=900.)
    parser.add_argument("--audit-integer", action="store_true", help="Development shadow audit; affects timings")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".agent"):
        parser.error("Output must be a fresh directory under this worktree's .agent")
    if args.repeats < 1 or args.warm_repeats < 1 or args.budget <= 0:
        parser.error("Repeat counts and budget must be positive")
    if len(args.variants) != len(set(args.variants)):
        parser.error("Select each experiment variant at most once")
    if args.smoke and args.cohort != "hard88":
        parser.error("Smoke uses the original early capture only")
    build(out)
    if args.mode == "build":
        print(f"Built and verified sources: {out / 'build.json'}")
        return 0
    if args.mode == "test":
        return run_locked(out / "sources/port", "pytest", [
            "tests/test_colgen_gpu_integer.py", "tests/test_colgen_gpu_endpoint.py",
            "tests/test_colgen_pricing_platform.py", "-q", "--disable-warnings",
        ], out / "tests.log")
    cohort = json.loads(git_bytes("show", f"{FEATURE}:analysis/colgen/arrival_union_results.json"))
    ids = [940, 2224] if args.smoke else [r["flight_id"] for r in cohort[args.cohort]["after"]["rows"]]
    assert len(ids) == len(set(ids))
    captures = [CAPTURE] + ([LATE] if args.cohort == "late18" and args.mode == "replay" else [])
    save(out / "run.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
         mode=args.mode, variants=args.variants, flight_ids=ids, smoke=args.smoke,
         capture_sha256={str(p): digest(p.read_bytes()) for p in captures},
         max_labels=250_000, chunk_size=2048, batch_size=1, repeats=args.repeats,
         warm_repeats=args.warm_repeats, budget=args.budget, audit=args.audit_integer))
    reports: dict[str, list[Path]] = {name: [] for name in args.variants}
    failures = []
    for repeat in range(args.repeats):
        # Reverse the arm order on alternate process repeats to reduce ordering bias.
        order = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
        for name in order:
            variant = VARIANTS[name]
            result = out / f"{name}-{repeat}"
            common = ["--capture", str(CAPTURE), "--flight-ids", *map(str, ids),
                      "--budget", str(args.budget), "--batch-size", "1",
                      "--max-labels", "250000", "--chunk-size", "2048", *variant.flags]
            if args.audit_integer and "--integer-bounds" in variant.flags:
                common += ["--audit-integer"]
            if args.mode == "replay":
                result = result.with_suffix(".json")
                command = [*common, "--out", str(result), "--warm-repeats", str(args.warm_repeats)]
                if len(captures) == 2:
                    command += ["--capture", str(LATE)]
                module = "analysis.colgen.execution_replay"
            else:
                command = ["solve", *common, "--out", str(result)]
                module = "analysis.colgen.gpu_validation"
            code = run_locked(out / "sources" / variant.source, module, command, out / f"{name}-{repeat}.log")
            if code:
                failures.append(dict(variant=name, repeat=repeat, exit_code=code))
            if args.mode == "replay" and result.exists():
                reports[name].append(result)
    if args.mode == "replay":
        summary = replay_summary(reports)
        if failures:
            for value in summary.values():
                value["matched_warm_reduction_percent_vs_first"] = None
            for failure in failures:
                summary[failure["variant"]]["all_complete_matching_objectives"] = False
        save(out / "comparison.json", summary)
    save(out / "completion.json", dict(failed_children=failures))
    print(f"Results: {out}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
