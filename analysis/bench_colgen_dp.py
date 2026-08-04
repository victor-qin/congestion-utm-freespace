"""A/B the compiled colgen pricing DP against the pure-Python reference.

Methodology follows ``analysis/ab_column_clear.py``: each mode runs in a fresh child
interpreter with ``PYTHONHASHSEED=0``, the JIT is warmed before any timer starts, and
the reference is measured *second* so any residual OS-cache advantage goes to the
baseline rather than to the change being advocated for.

Two deliberate departures from that script:

* The two modes are the same tree with ``pricing._dp_kernel`` toggled, not two
  checkouts.  A script file's ``sys.path[0]`` is its own directory, so a two-checkout
  A/B run as ``cd <tree> && python /tmp/bench.py`` silently imports whichever copy is
  installed in the venv -- and measures it against itself.  The resolved module path
  is printed on every line so a wrong-tree run is visible in the results.
* Parity is asserted on the priced columns, not on a whole-run digest, because
  ``colgen_test`` is time-limited: it always spends its budget, so wall-clock for a
  full solve measures the budget, not the work.  This benchmark therefore prices a
  fixed set of subproblems instead of running a solve.

Usage:
    uv run python analysis/bench_colgen_dp.py [--flights N] [--require-speedup X]
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (max_ground_delay_s, detour_slack_hops, destination cell) -- three problem scales.
CASES = (
    (3600.0, 12, (6, 0)),
    (200.0, 6, (5, 0)),
    (48.0, 4, (4, -1)),
)


def _build(case, flights):
    import numpy as np

    from freespace_sim.config import SimConfig
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.pricing import DualView
    from freespace_sim.types import FlightRequest, vec

    ground_delay_s, slack, dest = case
    cfg = SimConfig(
        planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0), terminal_airspace_always_active=True,
        max_ground_delay_s=ground_delay_s, max_detour_factor=10.0,
    )
    radius = hg.circumradius(cfg)

    def point(cell):
        x, y = hg.hex_center(*cell, radius)
        return vec(x, y, cfg.ground_level_m)

    params = ColGenParams(solver="highs", detour_slack_hops=slack)
    graphs = [
        build_flight_graph(
            FlightRequest(i, point((0, 0)), point(dest), 0.0, 0.0), cfg, (), params
        )
        for i in range(flights)
    ]
    rng = np.random.default_rng(7)
    duals: dict = {}
    base = graphs[0]
    for q in range(-4, 11):
        for r in range(-6, 7):
            if rng.random() < 0.30:
                for step in range(base.base_step, min(base.max_step, base.base_step + 60), 3):
                    if rng.random() < 0.25:
                        duals[RowKey.cell((q, r), 0, step)] = float(rng.gamma(2.0, 3.0))
    return cfg, params, graphs, DualView(duals, cfg)


def _child(mode: str, flights: int) -> dict:
    import freespace_sim
    from freespace_sim.planner.colgen import pricing

    module_path = os.path.realpath(freespace_sim.__file__)
    if mode == "reference":
        pricing._dp_kernel = None
    elif pricing._dp_kernel is not None:
        # Compile every signature the timed region will use, before any timer.
        pricing._dp_kernel.warm_kernel()

    rows = []
    for case in CASES:
        cfg, params, graphs, view = _build(case, flights)
        # Warm: pay seed construction and (for the kernel) topology packing, exactly
        # as column generation's first iteration does, outside the measured region.
        pricing.price_flight(graphs[0], view, 0.0, cfg, params, require_improving=False)
        started = time.perf_counter()
        answers = []
        for graph in graphs:
            reduced_cost, column = pricing.price_flight(
                graph, view, 0.0, cfg, params, require_improving=False
            )
            answers.append(
                (
                    round(float(reduced_cost), 8),
                    None if column is None else column.cell_path,
                    None if column is None else column.departure_step,
                    None if column is None else column.origin_lane_idx,
                    None if column is None else column.dest_lane_idx,
                )
            )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "case": f"gd={case[0]:g} slack={case[1]}",
                "elapsed": elapsed,
                "per_call_ms": elapsed / len(graphs) * 1000.0,
                "answers": answers,
            }
        )
    return {"mode": mode, "module_path": module_path, "rows": rows}


def _run_child(mode: str, flights: int, result_path: Path) -> dict:
    # Pickle is safe here and is the established pattern in analysis/ab_column_clear.py:
    # the only writer is the child this function just spawned, the path is inside the
    # repo's gitignored .context/, and nothing external ever supplies this file.
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()),
         "--_mode", mode, "--_result", str(result_path), "--flights", str(flights)],
        check=True, env=env, cwd=str(REPO_ROOT),
    )
    return pickle.loads(result_path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=int, default=12,
                        help="priced subproblems per case (default 12)")
    parser.add_argument("--require-speedup", type=float, default=None,
                        help="exit non-zero unless every case reaches this speedup")
    parser.add_argument("--_mode", help=argparse.SUPPRESS)
    parser.add_argument("--_result", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._mode:
        Path(args._result).write_bytes(pickle.dumps(_child(args._mode, args.flights)))
        return

    tmp = REPO_ROOT / ".context"
    tmp.mkdir(exist_ok=True)
    result_path = tmp / "bench_colgen_dp.pickle"
    print("Running kernel first, then reference, in fresh interpreters "
          "(any residual OS-cache advantage goes to the reference).", flush=True)
    kernel = _run_child("kernel", args.flights, result_path)
    reference = _run_child("reference", args.flights, result_path)

    if kernel["module_path"] != reference["module_path"]:
        raise SystemExit(
            f"ABORT: modes loaded different trees\n  {kernel['module_path']}\n  "
            f"{reference['module_path']}"
        )
    print(f"module: {kernel['module_path']}")

    # Parity first, speed second: a fast wrong answer is not a result.
    mismatches = []
    for left, right in zip(kernel["rows"], reference["rows"]):
        for index, (a, b) in enumerate(zip(left["answers"], right["answers"])):
            if abs(a[0] - b[0]) > 1e-8 or a[1:] != b[1:]:
                mismatches.append((left["case"], index, a, b))
    if mismatches:
        case, index, a, b = mismatches[0]
        print(f"PARITY: FAILED ({len(mismatches)} differing columns)")
        print(f"  first: {case} subproblem {index}\n    kernel={a}\n    reference={b}")
        raise SystemExit(1)
    n = sum(len(row["answers"]) for row in kernel["rows"])
    print(f"PARITY: EXACT ✓  ({n} priced columns; reduced cost within 1e-8 AND identical "
          f"cell_path / departure_step / lanes)")

    print("\ntimings exclude JIT warm-up, graph construction, and seed construction; "
          "they include dual packing, search, and canonical certification")
    worst = None
    for left, right in zip(kernel["rows"], reference["rows"]):
        speedup = right["elapsed"] / left["elapsed"]
        worst = speedup if worst is None else min(worst, speedup)
        print(f"  {left['case']:<22} reference {right['per_call_ms']:8.2f} ms/call   "
              f"kernel {left['per_call_ms']:8.2f} ms/call   speedup x{speedup:.2f}")
    print(f"\nSPEED: worst case x{worst:.2f}")
    if args.require_speedup is not None and worst < args.require_speedup:
        raise SystemExit(f"FAILED --require-speedup {args.require_speedup}: got x{worst:.2f}")


if __name__ == "__main__":
    main()
