"""EXECUTE — run ONE scenario and persist it. The single "experiment box".

This is the entire execute stage: pick a named scenario (optionally override fields), run the
simulation, and write a complete, reloadable run folder + one index row. It produces **no plots and
no replay** — those are readout scripts (``experiments/readouts/``). It runs **one** scenario; sweeps
and planner comparisons are pure-shell loops over this script (see ``experiments/batch/``), joined by
a shared ``--tag``.

    # one run, captured to results/<stamp>_<tag>_<hash>/ ; the folder path is the last stdout line
    uv run python -m experiments.run --scenario dallas_hub_2uss --planner astar_shortcut

    # override any field; capture the folder in a shell variable for a readout
    FOLDER=$(uv run python -m experiments.run --scenario metro_2uss --lam 240 --tag demo | tail -1)

Scenario identity (``--scenario`` / ``--tag`` / demand pattern) is written into the index so cross-run
readouts can filter to exactly the runs a batch produced.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from freespace_sim import metrics, runs
from freespace_sim.scenarios import SCENARIOS, get_scenario, with_overrides
from freespace_sim.sim import RETURN_ANCHORS, run

log = logging.getLogger("experiments.run")


def _scaled_lam_per_uss(spec, total_lam: float) -> dict[str, float]:
    """Scale an explicit per-USS rate map while preserving its operator proportions."""
    rates = spec.demand.lam_per_uss
    if rates is None:
        raise ValueError("scenario has no explicit lam_per_uss rates to scale")
    current_total = sum(float(rate) for rate in rates.values())
    if current_total <= 0.0:
        raise ValueError("cannot scale lam_per_uss rates whose total is not positive")
    scale = float(total_lam) / current_total
    return {uss_id: float(rate) * scale for uss_id, rate in rates.items()}


def _kernel_status(planner_name: str) -> str:
    """One-line compiled-kernel status for the startup INFO block. The module lands in
    ``sys.modules`` so the sim's later import is free. Only the astar family has a kernel —
    anything else reports n/a rather than paying the numba import.

    Load the ``astar`` package outside the fallback guard: its ``__init__`` pulls the whole family
    (planner, both occupancy modules, and through them volumes/cost/ledger), so any break in that
    graph must propagate as itself. Probe numba separately because its compatibility checks raise
    plain ``ImportError`` (``name=None``), while a missing transitive dependency names e.g.
    ``llvmlite``; neither can be classified reliably from ``ImportError.name``. Once numba imports,
    load the kernel outside the guard too, so a broken kernel import is never mislabeled as an
    optional-dependency fallback.
    """
    if "astar" not in planner_name:
        return "n/a (planner has no compiled kernel)"
    if planner_name == "astar_ref":
        return "pure-Python reference (explicitly requested via astar_ref)"

    import freespace_sim.planner.astar  # noqa: F401  # validate the non-numba import graph first
    try:
        from numba import njit as _njit  # noqa: F401  # exact dependency kernel.py imports
    except ImportError:
        return ("REFERENCE FALLBACK — numba unavailable, ~5-7x slower search. "
                "Run via plain `uv run` (numba is in tool.uv default-groups) or `uv sync`.")

    from freespace_sim.planner.astar import kernel  # noqa: F401
    return "compiled (numba kernel active)"


class _StderrTee:
    """Copy everything written to fd 2 into a file, without taking it off the terminal.

    Exists because the compiled pricing path's diagnosis lives only on stderr. A run can
    report ``kernel_fell_back: 37`` and not say whether the answer is to install numba,
    raise ``MAX_LABEL_CAPACITY``, or widen the time limit — and those call for opposite
    responses, which is the whole reason ``pricing._warn_budget_growth`` distinguishes
    them. Cluster runs are the case with no terminal to read, and ``[[run-archive-workflow]]``
    syncs the run FOLDER, so a file inside it is archived while a ``slurm-*.out`` in the
    submit directory is not.

    **fd-level, and not a ``sys.stderr`` swap.** The pricing sweep runs in *spawned*
    workers, which inherit file descriptors but not Python objects — and a worker never
    runs ``basicConfig``, which is exactly why those warnings are ``print(file=sys.stderr)``
    rather than ``logging`` in the first place. Replacing ``sys.stderr`` would capture the
    parent and silently miss every worker, i.e. the half that matters.

    **The pump stops on a flag, never on EOF.** ``multiprocessing`` launches its resource
    tracker with ``sys.stderr.fileno()`` in ``fds_to_pass`` and that child outlives the
    parent (``resource_tracker._launch``), so from the first pool onward a process we do
    not control holds a duplicate of this pipe's write end for the rest of the run.
    Draining until ``read()`` returns ``b''`` would block forever *after the solve
    finished*, on precisely the parallel runs this exists to instrument. Restoring fd 2
    first is necessary and nowhere near sufficient.
    """

    __slots__ = ("path", "_read_fd", "_saved_fd", "_stop", "_thread")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._read_fd, write_fd = os.pipe()
        self._saved_fd = os.dup(2)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._stop = threading.Event()
        # Daemon on purpose: the stop flag is what ends this thread, so if it ever wedges
        # the failure should be a lost tail rather than an interpreter that will not exit.
        self._thread = threading.Thread(target=self._pump, name="stderr-tee", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        with open(self.path, "wb") as sink:
            while True:
                # Polled rather than blocking, because this thread must never be the reason
                # a writer stalls: a full 64 KB pipe blocks everyone writing to fd 2,
                # workers included, and a worker that blocks in a warning hangs the sweep.
                ready, _, _ = select.select([self._read_fd], [], [], 0.2)
                if ready:
                    chunk = os.read(self._read_fd, 65536)
                    if chunk:
                        os.write(self._saved_fd, chunk)
                        sink.write(chunk)
                        sink.flush()
                        continue
                # Only once the pipe has nothing left AND shutdown was asked for, so the
                # last warnings before teardown are not dropped.
                if self._stop.is_set():
                    return

    def close(self) -> None:
        """Restore fd 2, let the pump drain, and release both descriptors.

        The order is the whole of it. Restoring *before* signalling means anything written
        during teardown reaches the real stderr instead of a pipe nobody is reading; and
        the descriptors are released only once the pump has actually stopped, because
        closing one it is still selecting on would hand its number to the next ``open()``.
        """

        os.dup2(self._saved_fd, 2)
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            sys.stderr.write(f"stderr tee did not stop; {self.path} may be truncated\n")
            return
        os.close(self._read_fd)
        os.close(self._saved_fd)


def spec_from_args(args):
    """Layer CLI overrides on top of the chosen registry scenario (frozen → copies, never mutates)."""
    spec = get_scenario(args.scenario)
    top: dict = {}
    if args.region is not None:
        top["region_m"] = (float(args.region[0]), float(args.region[1]))
    if args.lam is not None:
        top["lam_per_hour"] = args.lam
    if args.horizon is not None:
        top["horizon_s"] = args.horizon
    if args.demand_duration is not None:
        top["demand_duration_s"] = args.demand_duration
    if args.seed is not None:
        top["seed"] = args.seed
    if args.planner is not None:
        top["planner"] = args.planner
    if args.terminal_airspace_always_active is not None:
        top["terminal_airspace_always_active"] = args.terminal_airspace_always_active

    demand: dict = {}
    if args.lam is not None and spec.demand.lam_per_uss is not None:
        demand["lam_per_uss"] = _scaled_lam_per_uss(spec, args.lam)
    if args.demand is not None:
        demand["pattern"] = args.demand
    if args.uss is not None:
        demand["uss"] = tuple(args.uss)
    if args.hubs is not None:
        demand["hubs"] = tuple(int(h) for h in args.hubs)
    if args.direction is not None:
        demand["direction"] = args.direction
    if args.radius is not None:
        demand["radius_m"] = args.radius
    if args.pads_per_hub is not None:
        demand["pads_per_hub"] = args.pads_per_hub
    if args.terminal_radius is not None:
        demand["terminal_radius_m"] = args.terminal_radius
    if args.corridor_overlap is not None:
        demand["corridor_overlap_m"] = args.corridor_overlap
    if args.return_flights is not None:
        demand["return_flights"] = args.return_flights
    if args.turnaround is not None:
        demand["turnaround_s"] = args.turnaround

    return with_overrides(spec, demand_overrides=demand or None, **top)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from ``main`` so tests can construct a real args namespace.

    Hand-mirroring this flag set in a test fixture silently rots the moment a flag is added —
    ``spec_from_args`` reads attributes that the fixture doesn't have. Parse from here instead.
    """
    p = argparse.ArgumentParser(description="Run one scenario and persist it (the execute box).")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="metro_uniform",
                   help="named world from the registry (override individual fields with the flags below)")
    p.add_argument("--region", type=float, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--lam", type=float, default=None, help="arrival rate (req/h)")
    p.add_argument("--horizon", type=float, default=None, help="sim horizon (s)")
    p.add_argument("--demand-duration", type=float, default=None,
                   help="offered-load window (s); demand is generated over this, the run continues to "
                        "--horizon so the return tail is never clipped. Must be <= --horizon, so shrinking "
                        "a density_* scenario for a smoke test needs BOTH (e.g. --horizon 900 "
                        "--demand-duration 120). Unset keeps the scenario's own value.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--planner", default=None, help="override planner")
    p.add_argument("--terminal-airspace-always-active", action=argparse.BooleanOptionalAction,
                   default=None, dest="terminal_airspace_always_active",
                   help="permanently wall each hub's column+lanes off from foreign traffic (foreign "
                        "transit → air detour instead of ground-block); needs a wall-aware planner "
                        "(A* family, terminal-aware MILP, or colgen)")
    p.add_argument("--demand", choices=("uniform", "hub", "hub_radius"), default=None,
                   help="demand pattern")
    p.add_argument("--uss", nargs="+", default=None, help="USS labels (multi-operator demand)")
    p.add_argument("--hubs", nargs="+", type=int, default=None, help="per-USS hub counts (hub patterns)")
    p.add_argument("--direction", choices=("delivery", "pickup"), default=None)
    # hub_radius knobs
    p.add_argument("--radius", type=float, default=None, help="customer demand radius around a hub (m)")
    p.add_argument("--pads-per-hub", type=int, default=None, help="terminal capacity N per hub")
    p.add_argument("--terminal-radius", type=float, default=None,
                   help="shared terminal column radius (m); default = hover footprint")
    p.add_argument("--corridor-overlap", type=float, default=None,
                   help="how far the first corridor box penetrates the terminal (m); default corridor_width/2")
    p.add_argument("--return-flights", action=argparse.BooleanOptionalAction, default=None,
                   help="emit a return flight to the origin pad for each delivery")
    p.add_argument("--turnaround", type=float, default=None, help="return-flight turnaround (s)")
    p.add_argument("--return-anchor", choices=RETURN_ANCHORS, default="nominal",
                   dest="return_anchor",
                   help="what a round-trip return's desired departure waits on. nominal (default): a "
                        "straight-line undelayed estimate of the outbound's arrival, fixed when demand "
                        "is generated — under congestion the return is scheduled before its aircraft "
                        "lands. realized: plan the outbound, then anchor its return to the arrival that "
                        "actually happened (+ turnaround). Same cost; sequential mode only. NOT a "
                        "column in results/index.parquet (only experiment.json records it), so give "
                        "the two anchors distinct --tag values or a cross-run readout cannot tell "
                        "them apart")
    p.add_argument("--tag", default=None, help="run-folder label + index join key (default: scenario name)")
    p.add_argument("--window-frac", type=float, default=0.9,
                   help="steady-state plateau threshold: measure where airborne density ≥ frac×peak "
                        "(issue #25). summary.json always reports both whole-run and this windowed twin")
    p.add_argument("--telemetry", action="store_true",
                   help="capture observer-only congestion telemetry (filed-but-rejected corridors, "
                        "conflict_filed culprits, per-hub metadata, end-of-run walls) into extra parquets")
    p.add_argument("--no-progress", action="store_true", help="silence the live progress line")
    p.add_argument("--no-run-log", action="store_true",
                   help="skip capturing stderr into the run folder as run.log. The capture is an "
                        "fd-level tee so that SPAWNED pricing workers are covered too, which is where "
                        "the compiled-path fallback warnings come from; turn it off for a debugger or "
                        "anything else that dislikes having fd 2 redirected under it")
    p.add_argument("--mode", choices=("sequential", "exact", "relaxed"), default="sequential",
                   help="execution strategy for the whole simulation (issue #8 Track A). "
                        "sequential (default): the classic serial FCFS loop. exact: speculative "
                        "worker-pool planning, byte-identical to the serial run and faster. relaxed: "
                        "keep any still-feasible speculation — a valid FCFS-class allocation, faster "
                        "and scalable to more workers, at a small delay cost (deterministic via pinned "
                        "prefixes; result depends on workers+window). Parallel modes currently support "
                        "only astar, astar_ref, and astar_shortcut; every other planner runs sequential.")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="worker processes for explicit exact/relaxed mode only (default min(8, cores-2); the "
                        "benchmark sweet spot is ~4 workers for exact, ~8 for relaxed)")
    p.add_argument("--parallel-window", type=int, default=None,
                   help="speculation window for explicit exact/relaxed mode only (default 4×workers); "
                        "result-affecting in relaxed mode")
    # --- colgen (whole-schedule column generation; --planner colgen) ---
    # Exposed because colgen's answer depends on its budget in a way per-flight planners' does not:
    # it reports the best schedule found before the budget ran out, so a run that quietly stopped
    # at iteration 1 still looks like a completed solve. The termination reason is logged for that
    # reason, and these flags are what let a run be given enough budget to actually converge.
    p.add_argument("--colgen-time-limit", type=float, default=None, metavar="S",
                   help="colgen: whole-solve wall budget in seconds (default 1200). Pricing "
                        "dominates the cost and one sweep at 100 flights is already 147 s, so the "
                        "default buys roughly three iterations there — a scenario-scale solve needs "
                        "more, and a solve that stops early says so in its termination reason")
    p.add_argument("--colgen-ip-time-limit", type=float, default=None, metavar="S",
                   help="colgen: wall budget for the FINAL restricted-master IP alone (default "
                        "120). Separate from --colgen-time-limit because the IP otherwise "
                        "inherits everything the CG loop did not spend — a solve that converges "
                        "early hands the MILP hours. Exceeding it is not a failure: the run "
                        "falls back to the validated rounding incumbent and reports a "
                        "non-optimal ip_status, so the schedule is still claim-feasible, just "
                        "uncertified")
    p.add_argument("--colgen-max-eager-rows", type=int, default=None, metavar="N",
                   help="colgen: ceiling on the rows the final IP may pre-materialize "
                        "(default none). The pre-pass is all-or-nothing — over the ceiling it "
                        "materializes NOTHING and the lazy separation loop runs exactly as it "
                        "did before, because a half-materialized set looks eager in the stats "
                        "and still has to search. Worth setting on a pool denser than the ones "
                        "measured here: at 1,500 flights the pre-pass is 495,574 rows and 272 s "
                        "and it scales WITH the pool. 0 pins the old lazy loop for an A/B")
    p.add_argument("--colgen-max-iterations", type=int, default=None, metavar="N",
                   help="colgen: cap on column-generation iterations (default 30)")
    p.add_argument("--colgen-objective", choices=("total_delay", "total_cost"), default=None,
                   help="colgen: what to minimise — total_cost (default) weights ground and "
                        "excess-air seconds by the config's per-second dials (1:3), matching the "
                        "A* cost model; total_delay sums them unweighted, which makes a "
                        "ground-for-air swap exactly free and leaves the pricing search with "
                        "large families of tied columns it cannot order")
    p.add_argument("--colgen-solver", choices=("auto", "gurobi", "highs"), default=None,
                   help="colgen: LP/IP backend for the restricted master (default gurobi — needs "
                        "`uv sync --extra gurobi` and a licence; 'auto' falls back to HiGHS instead "
                        "of failing, 'highs' pins it). Result-affecting on a degenerate master — the "
                        "two backends return different optimal dual vertices, which changes both the "
                        "pricing subproblems and how tight the reported LP bound is. HiGHS also runs "
                        "the final IP cold (scipy.optimize.milp takes no incumbent), which dominates "
                        "the solve at scale")
    p.add_argument("--colgen-gap-metric", choices=("revenue", "cost"), default=None,
                   help="colgen: which scale the lp_gap/ip_gap thresholds are measured on. cost "
                        "(default) normalises by total cost: far stricter, and its termination gate "
                        "additionally requires that no new columns arrived. revenue is the paper's "
                        "eq. (10)/(11), normalised by an objective whose scale includes n*M — n "
                        "cancels, so the gate reduces to 'stop unless the average flight can still "
                        "save tau*M seconds' and moves with M, which is why it is no longer the "
                        "default")
    # The knob that sizes the pricing search itself. The budget flags above decide how long a
    # solve may run; this decides how much work each iteration IS, and pricing is ~98% of the
    # cost — so without it a run cannot be tuned, only given more time.
    p.add_argument("--colgen-max-air-overrun", type=int, default=None, metavar="HOPS",
                   help="colgen: hop budget over the lattice geodesic for a priced route "
                        "(default 3). Also the half-width of the O-D ellipse the flight is "
                        "priced over, because the budget implies it — a route within the budget "
                        "cannot reach a cell outside that ellipse. The dominant term in how much "
                        "search a sweep does; suboptimal by construction, since a route needing "
                        "more hops becomes unreachable — widen it first if a congested scenario "
                        "denies flights that ought to be placeable")
    # The three knobs `ColGenParams` grew for the compiled/parallel pricing path. Without
    # them the params object is reachable only from Python: `n_pricing_workers` defaults to
    # 0, so no invocation of this CLI could ever run the pool, and the ladder defaults ON and
    # could not be turned down. "Off by default" and "unreachable" look identical from the
    # params object and are not the same thing.
    #
    # `--colgen-greedy-budget-rate` is now the mirror image -- its stage defaults OFF, so its
    # flag turns something ON. Worth stating because the flip means an unset flag no longer
    # implies "the documented behaviour happens", and the help strings below are the only
    # place a caller sees which way each one points.
    p.add_argument("--colgen-warm-start", choices=("astar",), default=None,
                   help="colgen: run this planner first and hand colgen its schedule, as pool "
                        "columns AND as the starting incumbent (default off). Result-affecting "
                        "and REPORTED IN STATS, because it changes what the run is measuring: "
                        "on density_faa_wing_zipline x1500, unaided colgen is 235,388 (+11.3%% "
                        "vs A*) while A*-seeded colgen is 196,398 (-7.1%%), so leaving this on "
                        "silently would turn 'colgen beats A*' into 'refining A* beats A*'. "
                        "Costs one extra planner pass (64 s against a 3,750 s solve at x1500); "
                        "flights whose route the column model cannot express — an air hold, or "
                        "a route outside the O-D ellipse — are dropped and logged, not forced.")
    p.add_argument("--colgen-workers", type=int, default=None, metavar="N",
                   help="colgen: fan each pricing sweep across N worker processes (default 0, "
                        "in-process). Note this is NOT --workers, which sizes the simulation's "
                        "speculative pool. Answer-identical to sequential ONLY ON A SWEEP THAT "
                        "FINISHES: the accepted prefix and the reduced-cost order both reproduce "
                        "the sequential loop, but the pricing deadline is a wall clock, so a pool "
                        "gets further through the flights before the same instant and keeps a "
                        "LONGER prefix -- more pricing inside the budget, and a different column "
                        "set. Fast (3.5x at 4 workers on density x50) but MEMORY is the binding "
                        "constraint and it is linear: 3.9 GB in-process, 12.5 GB at 4 workers, "
                        "22.7 GB at 8, sampled across the process tree at only 50 flights. Size "
                        "this against the RAM you have, not the cores; an OOM-killed worker hangs "
                        "the sweep rather than failing it")
    p.add_argument("--colgen-seed-ladder", type=int, default=None, metavar="STEPS",
                   help="colgen: seed each flight's column with STEPS retimed copies of itself "
                        "before the first LP (default 20; 0 disables). Pure clock translation, so "
                        "a rung adds ground delay and nothing else, and LP duality then PROVES "
                        "those retimes non-improving rather than making pricing rediscover them "
                        "one iteration at a time. Buys objective and costs pricing time: measured "
                        # `%%` because argparse %-formats help strings -- a bare `%` here is a
                        # ValueError at parser construction, i.e. every invocation.
                        "-6%% to -19%% objective for 1.6-13.7x the pricing. At 100 flights the "
                        "ladder is 2000 of the final 2186 columns")
    p.add_argument("--colgen-greedy-budget-rate", type=float, default=None, metavar="S",
                   help="colgen: seconds PER FLIGHT for the initial greedy feasible-selection "
                        "stage. DEFAULT 0, WHICH DISABLES THE STAGE — pass a rate to enable it "
                        "(0.7 was the old default, 350 s at 500 flights). Off because it is a "
                        "head start column generation closes on its own: run to convergence at "
                        "500 flights it bought 0.129 percent of objective for 57 percent more "
                        "wall, and pricing was 16 percent SLOWER with it on. Beware measuring it "
                        "at a truncated iteration count, where it reads 2 percent better — that "
                        "compares convergence rate, not solution quality. A rate rather than a "
                        "total because the stage splits its budget across the flights still to "
                        "try, so a flat budget starves large batches. Worth enabling for a solve "
                        "whose --colgen-time-limit genuinely binds, where a better heuristic is "
                        "the answer rather than a starting point")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the execute CLI's cross-argument execution-mode contract."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "sequential" and (
        args.workers is not None or args.parallel_window is not None
    ):
        parser.error("--workers and --parallel-window require --mode exact or --mode relaxed")
    # `_colgen_overrides` first: it is a namespace read, while `_effective_planner` resolves
    # the registry and builds a `SimConfig`.  Ordering it this way keeps that work off every
    # invocation that has nothing to do with colgen, and keeps any error it could raise out
    # of runs that never mentioned the planner.
    if _colgen_overrides(args) and _effective_planner(args) != "colgen":
        # Silently ignoring them is the bad outcome: a sweep that meant to raise the solver
        # budget would report a converged-looking run at the default one.
        parser.error(
            "--colgen-* flags need a colgen run: pass --planner colgen, or use a scenario "
            "whose spec selects it"
        )
    return args


def _effective_planner(args) -> str:
    """The planner this invocation will actually run.

    NOT ``args.planner``, which is an OVERRIDE defaulting to ``None``: a ``ScenarioSpec``
    carries its own planner and falls back to ``SimConfig``'s. Keying the colgen flags on
    the override alone means a scenario that selects colgen itself cannot be given a solver
    budget -- ``--colgen-time-limit`` hard-errors asking for a flag the scenario already
    implies -- and, worse, that run silently uses ``ColGenParams()`` defaults, which is a
    smoke-test budget rather than a converging one.

    Reads the REGISTRY spec, not ``spec_from_args``: the CLI's other overrides can make a
    spec invalid (``--horizon`` below its demand window raises from ``SimConfig``), and that
    belongs in ``main`` where the error is about the run, not here where it would surface as
    a traceback out of argument parsing.
    """

    if args.planner is not None:
        return args.planner
    return get_scenario(args.scenario).config().planner


def _colgen_overrides(args) -> dict:
    """The colgen knobs the CLI actually set, as ``ColGenParams`` keyword arguments."""
    return {
        name: value
        for name, value in (
            ("time_limit_s", args.colgen_time_limit),
            ("ip_time_limit_s", args.colgen_ip_time_limit),
            ("max_eager_ip_rows", args.colgen_max_eager_rows),
            ("max_iterations", args.colgen_max_iterations),
            ("objective", args.colgen_objective),
            ("solver", args.colgen_solver),
            ("gap_metric", args.colgen_gap_metric),
            ("max_air_overrun_hops", args.colgen_max_air_overrun),
            ("n_pricing_workers", args.colgen_workers),
            ("seed_ladder_steps", args.colgen_seed_ladder),
            ("greedy_budget_s_per_flight", args.colgen_greedy_budget_rate),
            ("warm_start_planner", args.colgen_warm_start),
        )
        if value is not None
    }


def colgen_params_from_args(args, planner: str):
    """Build the planner's params object, or ``None`` when this run is not a colgen run.

    ``planner`` is passed in rather than re-derived so the decision is made from the config
    that will actually run -- ``parse_args``'s own check is an early UX guard, and the two
    must not be able to drift.

    ``None`` rather than a default-constructed ``ColGenParams`` so every other planner keeps
    taking the no-params path through :func:`~freespace_sim.planner.get_planner`.
    """
    if planner != "colgen":
        return None
    from freespace_sim.planner.colgen import ColGenParams

    return ColGenParams(**_colgen_overrides(args))


def main() -> None:
    args = parse_args()
    # everything human-facing goes to stderr; stdout is reserved for the folder path (shell capture)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    # Installed before any work, so the invocation banner is inside the capture, and torn down
    # in a `finally` so a run that DIES still leaves its log somewhere findable -- which is the
    # case with the most to say.
    tee = None if args.no_run_log else _StderrTee(Path(tempfile.mkdtemp()) / "run.log")
    # A LIST rather than `_execute`'s return value, because `_execute` does real work AFTER
    # `save_run` returns: it reads `summary.json` back off disk and logs the steady-state
    # twin.  An exception anywhere in that tail leaves the return value unassigned even
    # though the run folder exists and is complete, and the log would then be stranded in a
    # temp directory under a message claiming the run "failed before save_run" -- the
    # opposite of what happened, and pointing at the wrong thing to debug.  `_execute`
    # appends the moment the folder exists.
    saved: list[Path] = []
    try:
        _execute(args, saved)
    finally:
        if tee is not None:
            _archive_log(tee, saved[0] if saved else None)


def _archive_log(tee: _StderrTee, folder: Path | None) -> None:
    """Move the captured stderr into the run folder, or say where it was left."""

    tee.close()
    if folder is None:
        # No folder to put it in: the run never reached `save_run`.  Keep the file and name
        # it, rather than deleting the only record of why.
        sys.stderr.write(f"run failed before save_run; stderr log kept at {tee.path}\n")
        return
    shutil.move(tee.path, folder / "run.log")
    os.rmdir(tee.path.parent)


def _execute(args, saved: list[Path] | None = None) -> Path:
    spec = spec_from_args(args)
    # to_json_dict, not asdict: the latter loses every tuple to a JSON list and leaves `demand` a
    # plain dict, so the archived recipe could not be rebuilt. See ScenarioSpec.from_json_dict.
    scenario_payload = spec.to_json_dict()
    cfg = spec.config()
    demand = spec.demand_model()
    tag = args.tag or spec.name
    log.info("invocation: python -m experiments.run %s", " ".join(sys.argv[1:]) or "(no arguments)")
    log.info("scenario=%s tag=%s planner=%s demand=%s region=%s λ=%s/h planner-envelope=%ss seed=%s",
             spec.name, tag, cfg.planner, spec.demand.pattern, cfg.region_size_m,
             cfg.lam_per_hour, cfg.horizon_s, cfg.seed)
    if spec.description:
        log.info("description: %s", spec.description)
    log.info("active demand duration=%ss", cfg.effective_demand_duration_s)
    log.info("A* kernel: %s", _kernel_status(cfg.planner))

    pcfg = None
    if args.mode != "sequential":
        from freespace_sim.parallel import PARALLEL_PLANNERS, ParallelConfig
        if cfg.planner in PARALLEL_PLANNERS:
            kw = {"mode": args.mode, "window": args.parallel_window}
            if args.workers is not None:                 # else ParallelConfig's own capped default
                kw["n_workers"] = args.workers
            pcfg = ParallelConfig(**kw)
            log.info("mode=%s: %d workers, window=%d", pcfg.mode, pcfg.n_workers, pcfg.resolved_window)
        else:                                            # MILP/straight/etc. have no envelope-recording
            log.info("planner %r has no parallel kernel — running sequential (--mode %s ignored)",
                     cfg.planner, args.mode)
    else:
        # `--mode` names the FCFS COMMIT loop and nothing else.  Spelled out for planners
        # that could never use it, because the bare line reads as a claim about the whole
        # run: a colgen solve fanning its pricing sweep across eight processes still prints
        # `mode=sequential`, correctly -- the simulation loop IS serial -- and every reader
        # so far has taken that to mean the run is single-process.
        from freespace_sim.parallel import PARALLEL_PLANNERS
        log.info(
            "mode=sequential: serial FCFS planning%s",
            "" if cfg.planner in PARALLEL_PLANNERS else
            f" — {cfg.planner!r} has no speculative parallel kernel, so --mode/--workers "
            "never apply to it; a planner with its own internal pool reports that separately"
        )

    if args.return_anchor == "realized":
        from freespace_sim.planner import WHOLE_SCHEDULE_PLANNERS

        # The coupling keys off FlightRequest.paired_outbound_id, which ONLY hub_radius sets. Testing
        # `return_flights` alone is not enough: it defaults True on DemandSpec but the uniform and hub
        # patterns ignore it entirely, so metro_2uss/dallas_hub_2uss would sail past the guard and the
        # flag would be a silent no-op — the failure it exists to prevent.
        if demand is None or spec.demand.pattern != "hub_radius" or not spec.demand.return_flights:
            raise SystemExit(
                "--return-anchor realized needs round-trip returns, which only the hub_radius demand "
                f"pattern emits (scenario {spec.name!r} is pattern={spec.demand.pattern!r}, "
                f"return_flights={spec.demand.return_flights}); drop the flag or pick a hub_radius "
                "scenario such as a density_* world")
        if pcfg is not None:
            # run() raises on this too; catching it here keeps the CLI failure a one-line message
            # rather than a traceback, and does it before the world is built.
            raise SystemExit(
                f"--return-anchor realized is sequential-only (got --mode {args.mode}): a speculative "
                "worker may plan a return before its outbound has committed, and the exact-mode "
                "envelope check cannot detect that. Use --mode sequential or --return-anchor nominal.")
        if cfg.planner in WHOLE_SCHEDULE_PLANNERS:
            # Also raised by run(); same reason as above for catching it here.
            raise SystemExit(
                f"--return-anchor realized is not implemented for --planner {cfg.planner}: a "
                "whole-schedule solver plans every flight at once, so no outbound commits before its "
                "return and the coupling loop never runs — the flag would silently leave every return "
                "on the nominal anchor. Use a per-flight planner, or --return-anchor nominal.")
        log.info("return anchor: realized — each return departs when its outbound's landing column "
                 "actually cleared (touchdown + pad dwell) + %.0fs turnaround, not a nominal estimate",
                 spec.demand.turnaround_s)

    t0 = time.time()
    res = run(cfg, demand=demand, progress=not args.no_progress, telemetry=args.telemetry,
              parallel=pcfg, planner_params=colgen_params_from_args(args, cfg.planner),
              return_anchor=args.return_anchor)
    wall = time.time() - t0
    sim_lo, sim_hi = metrics.simulation_window(res)
    log.info(
        "realized simulation: first activity %.1fs, final landing %.1fs, duration %.1fs",
        sim_lo,
        sim_hi,
        sim_hi - sim_lo,
    )
    # SimConfig only validates demand_duration_s <= horizon_s, but what actually has to fit under the
    # horizon is preroll + demand window + lead spread + trip. The preroll is a max-order statistic over
    # the departure-lead draws (~2300 s for amazon_uss's N(1800, 300)), so a shrunken --horizon can pass
    # validation and still push most departures past it — where the compiled A* box guard silently
    # dispatches to the ~5-7x slower pure-Python reference. On a cluster that is the difference between
    # a 6-hour job and a 30-hour one, so say it out loud rather than let the allocation absorb it.
    late = sum(1 for i in res.intents if i.request.t_departure > cfg.horizon_s)
    if late and "astar" in cfg.planner:
        log.warning("%d/%d departures (%.0f%%) are past horizon_s=%.0fs — those flights fall back to "
                    "the slow reference A* (box guard). Raise --horizon or lower --demand-duration.",
                    late, len(res.intents), 100.0 * late / len(res.intents), cfg.horizon_s)
    if pcfg is not None and pcfg.stats:
        log.info("parallel stats: %s", pcfg.stats)

    folder = runs.save_run(
        res, label=tag, experiment="run", scenario=spec.name, demand=spec.demand.pattern,
        experiment_args={"scenario": spec.name, "tag": tag, "overrides": vars(args)},
        scenario_spec=scenario_payload,
        wall_seconds=wall, write_replay=False,   # execute persists data only; replay is a readout
        window_frac=args.window_frac,
    )
    # Published to the caller HERE, not via the return value: everything below this line can
    # raise, and the folder is already complete and on disk.  See `main`.
    if saved is not None:
        saved.append(folder)
    s = res.summary()
    log.info("n=%s acc=%s den=%s verified=%s (%.1fs) → %s",
             s["n_requests"], s["n_accepted"], s["n_denied"], res.verified, wall, folder)
    # the steady-state twin vs the whole-run number (read back from the summary save_run just wrote,
    # so it's the exact persisted value — no recompute). window == full horizon when no plateau exists.
    import json
    summ = json.loads((folder / "summary.json").read_text())
    st = summ.get("steady_state", {})
    log.info("steady window [%.0f,%.0f]s · mean delay %.1fs (whole-run %.1fs)",
             st.get("window_lo", 0), st.get("window_hi", 0),
             st.get("mean_total_delay_s", 0), summ.get("mean_total_delay_s", 0))
    print(folder)   # LAST stdout line: the run folder, for `FOLDER=$(... | tail -1)`
    return folder


if __name__ == "__main__":
    main()
