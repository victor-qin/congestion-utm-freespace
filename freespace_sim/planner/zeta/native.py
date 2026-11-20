"""Opt-in Cython build of the shared compact search, with a process-safe binary cache."""

from __future__ import annotations

from collections.abc import Callable
import fcntl
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

_LOAD_LOCK = threading.Lock()
_LOADED: tuple[Callable, tuple[str, str]] | None = None


def _remaining(deadline: float | None) -> float:
    """Bound each build operation by the shared deadline and a two-minute maximum."""
    remaining = 120.0 if deadline is None else min(120.0, deadline - time.perf_counter())
    if remaining <= 0.0:
        raise TimeoutError("Zeta compiled-loop preparation deadline expired")
    return remaining


def _run(command: list[str], deadline: float | None) -> str:
    """Run a bounded compiler command and reap its process group on interruption.

    Parameters
    ------------
    - command (list[str]): compiler invocation passed directly without a shell.
    - deadline (float | None): absolute pricing cutoff, also limiting command startup.

    Return
    --------
    - output (str): captured standard output; failures raise with compiler diagnostics.
    """
    try:
        budget = _remaining(deadline)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=budget)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"Zeta compiled-loop build failed: {stderr}")
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Zeta compiled-loop compiler deadline expired") from error
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"Zeta compiled-loop build failed: {detail}") from error
    return stdout


# Compiled module only: rebinds the loop's price floor (imported from price_floor.py) to the
# libm twin declared in compact.pxd, so each call avoids four interpreted math.nextafter calls.
_COMPILED_PRICE_FLOOR = b"""


def price_floor(cost: float, positive: float, negative: float, error: float,
                tolerance: float) -> float:
    '''Return the C floor with the reference operation order and rounding.'''
    return _zeta_price_floor(cost, positive, negative, error, tolerance)
"""


def _load(deadline: float | None) -> tuple[Callable, tuple[str, str]]:
    """Build or load this source's extension without changing the interpreted control.

    Parameters
    ------------
    - deadline (float | None): absolute cutoff for compilation, loading and cache-lock waits.

    Return
    --------
    - output (tuple): pricing entry point and its module-name/binary-path worker artifact.
      Missing build tools and compiler failures raise explicitly; there is no silent fallback.
    """
    try:
        import Cython
    except ImportError as error:
        raise RuntimeError("Zeta compiled_loop requires the 'zeta-native' extra (Cython)") from error
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("Zeta compiled_loop currently supports macOS and Linux")
    source = Path(__file__).with_name("compact.py").read_bytes()
    declarations = Path(__file__).with_name("compact.pxd").read_bytes()
    history_source = Path(__file__).with_name("history.py").read_bytes()
    coverage_source = Path(__file__).with_name("route_cache.py").read_bytes()
    state_source = Path(__file__).with_name("state_index.py").read_bytes()
    # One implementation serves interpreted and compiled comparisons. The first module's
    # future import applies to the appended helpers too; it cannot appear a second time.
    combined_source = source + b"\n" + b"\n".join(
        helper.replace(b"from __future__ import annotations\n", b"", 1)
        for helper in (history_source, coverage_source, state_source)
    ) + _COMPILED_PRICE_FLOOR
    compiler = shlex.split(os.environ.get("CC", sysconfig.get_config_var("CC") or "cc"))
    flags = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing"]
    directives = "annotation_typing=True,infer_types=False"
    metadata = {
        "source_sha256": hashlib.sha256(source).hexdigest(), "cython": Cython.__version__,
        "declarations_sha256": hashlib.sha256(declarations).hexdigest(),
        "history_source_sha256": hashlib.sha256(history_source).hexdigest(),
        "coverage_source_sha256": hashlib.sha256(coverage_source).hexdigest(),
        "state_source_sha256": hashlib.sha256(state_source).hexdigest(),
        "floor_source_sha256": hashlib.sha256(_COMPILED_PRICE_FLOOR).hexdigest(),
        "python": sys.version, "abi": sysconfig.get_config_var("SOABI"),
        "platform": sys.platform, "machine": platform.machine(),
        "compiler": compiler, "compiler_version": _run(compiler + ["--version"], deadline),
        "flags": flags, "directives": directives,
        "package_layout": 6,
    }
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    short_name = "_zeta_loop_" + fingerprint
    name = __package__ + "." + short_name
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = Path(os.environ.get("ZETA_LOOP_CACHE_DIR", str(base / "freespace-sim" / "zeta-loop")))
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / (short_name + sysconfig.get_config_var("EXT_SUFFIX"))
    with (cache / (short_name + ".lock")).open("a") as lock:
        lock_deadline = time.perf_counter() + _remaining(deadline)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.01, _remaining(lock_deadline)))
        _remaining(deadline)
        if not binary.exists():
            started = time.perf_counter()
            include = Path(sys.base_prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            configured = Path(sysconfig.get_path("include"))
            if (configured / "Python.h").is_file():
                include = configured
            if not (include / "Python.h").is_file():
                raise RuntimeError("Zeta compiled_loop requires the running Python's development headers")
            with tempfile.TemporaryDirectory(prefix=short_name + "-", dir=cache) as temporary:
                directory = Path(temporary)
                package = directory.joinpath(*__package__.split("."))
                package.mkdir(parents=True)
                pyfile = package / (short_name + ".py")
                pyfile.write_bytes(combined_source)
                pyfile.with_suffix(".pxd").write_bytes(declarations)
                cfile = pyfile.with_suffix(".c")
                _run([sys.executable, "-m", "cython", "-3", "-I", str(directory), "-X", directives,
                      "--module-name", name, "-o", str(cfile), str(pyfile)], deadline)
                linkage = (["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin"
                           else ["-shared", "-fPIC"])
                built = directory / binary.name
                _run(compiler + linkage + flags + ["-I" + str(include), str(cfile), "-o", str(built)],
                     deadline)
                metadata["build_wall_s"] = time.perf_counter() - started
                (cache / (short_name + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
                built.replace(binary)
    _remaining(deadline)
    artifact = name, str(binary)
    return _import_artifact(artifact), artifact


def _import_artifact(artifact: tuple[str, str]) -> Callable:
    """Import a parent-prepared binary without spawning a compiler in a pricing worker.

    Parameters
    ------------
    - artifact (tuple[str, str]): fully qualified module name and completed binary path.

    Return
    --------
    - output (Callable): native pricing entry point with direct label writes enabled.
    """
    name, binary = artifact
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load compiled Zeta loop at {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if module._Labels.__dictoffset__ != 0:
            raise RuntimeError("Cython did not apply the native label-storage declarations")
        if module._StateIndex.__dictoffset__ != 0:
            raise RuntimeError("Cython did not apply the native state-index declarations")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return partial(module.search_compact, _native_loop=True,
                   _compiled_history_type=module.PriceHistory,
                   _compiled_coverage_type=module.RouteCoverageCache,
                   _compiled_state_type=module._StateIndex)


def prepare_compiled_loop(deadline: float | None = None) -> tuple[str, str]:
    """Prepare the binary in the parent before workers can be terminated by a sweep timeout.

    Parameters
    ------------
    - deadline (float | None): absolute monotonic cutoff shared with pricing and worker startup.

    Return
    --------
    - output (tuple[str, str]): importable module name and completed binary path for workers.
    """
    global _LOADED
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    if not _LOAD_LOCK.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("Zeta compiled-loop preparation lock deadline expired")
    try:
        if _LOADED is None:
            _LOADED = _load(deadline)
        return _LOADED[1]
    finally:
        _LOAD_LOCK.release()


def install_compiled_loop(artifact: tuple[str, str]) -> None:
    """Install the parent's prepared artifact at worker startup, without a build subprocess."""
    global _LOADED
    with _LOAD_LOCK:
        _LOADED = _import_artifact(artifact), artifact


def load_compiled_loop(deadline: float | None = None) -> Callable:
    """Reuse the installed loop or prepare it within the current pricing deadline."""
    prepare_compiled_loop(deadline)
    return _LOADED[0]
