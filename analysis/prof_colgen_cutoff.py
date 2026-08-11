"""Per-flight cutoff quality: what the pricing search enters with, and what it costs.

``prof_colgen_stages.py`` says where a solve's time goes by stage.  This says where it
goes by *flight*, and pairs each flight's cost with the one quantity issue #90 argues is
the cause: the **gap between the incumbent the search starts from and the optimum it
ends at**.

The discriminator this exists to run.  A search that declines its label budget has two
candidate explanations that call for opposite fixes:

* **Weak cutoff.**  ``completion_can_compete`` returns ``True`` unconditionally when
  ``incumbent is None`` (pricing.py:1489) and the completion envelope's *length* is frozen
  against the incumbent at first use, so a low entry incumbent keeps orders of magnitude
  more labels alive.  The fix is a better incumbent -- a bootstrap round.
* **Wide dominance state.**  ``state_history_depth = max(2, revisit_depth)``
  (pricing.py:1291) sets how many trailing cells the dominance key carries; a wider key
  merges fewer labels.  The fix is a narrower key, which is a different (and
  answer-affecting) change.

They are told apart by ``gap`` below.  If the entry incumbent is already at or near the
final reduced cost, the cutoff was fine and the labels went somewhere else.

Every number here is per ``(sweep, flight)``:

``entry_rc``    the incumbent ``price_flight`` hands the search: ``seed_column``, then
                ``_shifted_seed_incumbent``, then the master's ``known_column``.
                ``-inf`` means no incumbent at all, the worst case for pruning.
``final_rc``    what ``price_flight`` returned.
``gap``         ``final_rc - entry_rc``.  How much of the answer the cutoff did not know.
``roots``       root variants surviving ``prepare_variants``' two gates.  A cutoff prunes
                here first and most visibly.
``labels``      ``DagResult.n_labels`` -- the pool the compiled search actually filled.
``att``         ``DagResult.attempts``; >1 means a budget filled and the search restarted
                from its first layer, discarding every certification it had paid for.
``status``      the kernel's verdict.  ``LABEL_LIMIT`` is the decline this issue is about.
``boot_s``      time in the bootstrap (``--bootstrap-departures``), kept apart from the
                search it informs; ``labels``/``roots``/``status`` describe the REAL search
                only, since a shared row would let the second call overwrite the first.
``fallback_s``  time in ``_best_column``, the pure-Python reference, after a decline.

Sequential only, for the same reason ``prof_colgen_stages`` is: none of this crosses a
process boundary.

    uv run python analysis/prof_colgen_cutoff.py --scenario density_faa_wing_zipline \
        --flights 12 --iterations 2
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import dp_prepare as dp_prepare_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing_pool as pricing_pool_mod  # noqa: E402
from freespace_sim.planner.colgen import solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

try:
    from freespace_sim.planner.colgen import dp_kernel as dp_kernel_mod  # noqa: E402
except ImportError:  # pragma: no cover - depends on the install
    dp_kernel_mod = None

ROWS: list[dict] = []
SWEEP = [1]
# `--carry-forward`: the column pricing returned for this flight LAST sweep, kept so the
# next one can start from it.  The solver drops it today -- `known_column` is the greedy
# heuristic's selection (`solver.py:1048`), not pricing's own previous answer.
CARRIED: dict[int, object] = {}
# One row under construction, keyed by nothing: the three wrapped functions are called in a
# fixed order within one `price_flight` and the sweep is sequential, so a single slot is
# enough and a dict keyed by flight id would only hide an ordering surprise.
CURRENT: dict = {}


def install_probes(carry_forward: bool = False) -> None:
    """Wrap the functions that see a flight's cutoff, its search, and its fallback.

    Every module object holding a direct-name binding is patched, for the reason
    ``prof_colgen_stages`` documents: ``pricing_pool`` imports ``price_flight`` for the
    sweep and ``solver`` imports it again for the greedy's repair path, so patching only
    ``pricing`` would instrument neither.

    ``carry_forward`` prototypes the cheapest candidate cutoff in the issue-#90 design
    space: hand each flight the column ITS OWN pricing returned last sweep, when that
    scores better under the current duals than the heuristic column the solver passes.
    Deliberately expressed through the existing ``known_column`` seam rather than a new
    one, so it lands on both searches at once -- ``price_flight`` folds it into
    ``incumbent`` (pricing.py:2847) before the fork, and hands that same ``incumbent`` to
    ``_best_column_compiled`` and to the ``_best_column`` fallback.  Optimality-safe for
    the same reason ``known_column`` is: the score is achievable, so pruning against it
    cannot discard anything strictly better.
    """

    price_flight = pricing_mod.price_flight

    def _carried_known(fg, duals, pi_f, cfg, params, kwargs):
        """Pick the better of the solver's heuristic column and last sweep's priced one."""

        carried = CARRIED.get(fg.request.flight_id)
        if carried is None:
            return
        view = duals if isinstance(duals, pricing_mod.DualView) else pricing_mod.DualView(
            duals, cfg
        )
        model = pricing_mod.cost_model(cfg, params)
        benefit = pricing_mod._benefit(params)

        def score(column):
            return model.reduced_cost(
                benefit=benefit,
                cost=column.delay_s,
                dual_cost=view.claim_cost(column.claims),
                pi_f=float(pi_f),
            )

        known = kwargs.get("known_column")
        if known is None or score(carried) > score(known):
            kwargs["known_column"] = carried

    def timed_price_flight(fg, *args, **kwargs):
        if carry_forward and not kwargs.get("forbidden_rows") and len(args) >= 4:
            _carried_known(fg, args[0], args[1], args[2], args[3], kwargs)
        row = dict(
            sweep=SWEEP[0],
            flight=fg.request.flight_id,
            # Repair prices under an exclusion set and zero duals inside the greedy, not in
            # the sweep.  Same function, different question; kept apart rather than summed.
            repair=bool(kwargs.get("forbidden_rows")),
            entry_rc=None,
            mid_rc=None,
            final_rc=None,
            roots=None,
            bootstrap_s=0.0,
            bootstrap_labels=0,
            labels=None,
            attempts=None,
            status=None,
            budget=None,
            search_s=0.0,
            fallback_s=0.0,
            total_s=0.0,
        )
        outer = dict(CURRENT)
        CURRENT.clear()
        CURRENT.update(row)
        started = time.perf_counter()
        try:
            reduced_cost, column = price_flight(fg, *args, **kwargs)
            CURRENT["final_rc"] = float(reduced_cost)
            if carry_forward and column is not None and not kwargs.get("forbidden_rows"):
                CARRIED[fg.request.flight_id] = column
            return reduced_cost, column
        finally:
            CURRENT["total_s"] = time.perf_counter() - started
            ROWS.append(dict(CURRENT))
            # Restore, so a nested call (repair reaches `price_flight` again) cannot leave
            # the enclosing row half-written.
            CURRENT.clear()
            CURRENT.update(outer)

    best_compiled = pricing_mod._best_column_compiled

    def timed_best_compiled(*args, incumbent=None, **kwargs):
        # Two compiled searches per flight once the bootstrap is on, and they must not share
        # a row: these were plain assignments, so the main search's numbers silently
        # overwrote the bootstrap's and the overhead being tuned for was invisible.
        # `max_departure_step` is what tells them apart -- only the bootstrap restricts.
        bootstrap = kwargs.get("max_departure_step") is not None
        if not bootstrap:
            CURRENT["entry_rc"] = -math.inf if incumbent is None else float(incumbent[0])
        CURRENT["in_bootstrap"] = bootstrap
        started = time.perf_counter()
        try:
            return best_compiled(*args, incumbent=incumbent, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            if bootstrap:
                CURRENT["bootstrap_s"] += elapsed
            else:
                CURRENT["search_s"] = elapsed
            CURRENT["in_bootstrap"] = False

    best_reference = pricing_mod._best_column

    def timed_best_reference(*args, **kwargs):
        # `seed=True` is `seed_column`'s rare last-resort geodesic and `max_departure_step`
        # is the bootstrap's own restricted reference; neither is the decline fallback, and
        # charging either would report a fallback that never happened.
        restricted = kwargs.get("max_departure_step") is not None
        fallback = not kwargs.get("seed", False) and not restricted
        started = time.perf_counter()
        try:
            return best_reference(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            if CURRENT:
                if fallback:
                    CURRENT["fallback_s"] += elapsed
                elif restricted:
                    CURRENT["bootstrap_s"] += elapsed

    prepare_variants = dp_prepare_mod.prepare_variants

    def counted_prepare_variants(*args, **kwargs):
        variants = prepare_variants(*args, **kwargs)
        if CURRENT and kwargs.get("max_departure_step") is None:
            CURRENT["roots"] = int(variants.departure_step.size)
        return variants

    pricing_mod.price_flight = timed_price_flight
    pricing_pool_mod.price_flight = timed_price_flight
    solver_mod.price_flight = timed_price_flight
    pricing_mod._best_column_compiled = timed_best_compiled
    pricing_mod._best_column = timed_best_reference
    dp_prepare_mod.prepare_variants = counted_prepare_variants

    if dp_kernel_mod is not None:
        price_dag = dp_kernel_mod.price_dag

        def probed_price_dag(*args, **kwargs):
            result = price_dag(*args, **kwargs)
            if CURRENT and CURRENT.get("in_bootstrap"):
                CURRENT["bootstrap_labels"] += int(result.n_labels)
            elif CURRENT:
                CURRENT["labels"] = int(result.n_labels)
                CURRENT["attempts"] = int(result.attempts)
                CURRENT["status"] = dp_kernel_mod.STATUS_NAMES.get(
                    result.status, str(result.status)
                )
                CURRENT["budget"] = result.budget
                # What the LAST attempt's mid-sweep certifications reached before it ran
                # out.  `price_dag` resets this to the entry incumbent on every restart
                # (dp_kernel.py:2078-2084) and `_best_column_compiled` never reads it on a
                # decline, so on a declining flight this number is computed, paid for, and
                # thrown away -- twice over.  Against `entry_rc` it says how much cutoff
                # the ladder is discarding.
                CURRENT["mid_rc"] = (
                    None if result.incumbent is None else float(result.incumbent[0])
                )
            return result

        dp_kernel_mod.price_dag = probed_price_dag
        # `_best_column_compiled` resolved `kernel.price_dag` through the module object, so
        # the rebind above is enough -- but `pricing` also imports the module lazily inside
        # `_dp_kernel()`, which returns that same object.  Nothing else holds a direct name.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--solver", default="highs", choices=("highs", "gurobi"))
    parser.add_argument("--gap-metric", default="cost", choices=("cost", "revenue"))
    parser.add_argument(
        "--max-label-log2", type=int, default=None,
        help="override dp_kernel.MAX_LABEL_CAPACITY with 2**N, to find where a declining "
             "flight's label demand actually tops out. Answer-neutral by construction "
             "(dp_kernel.py:120) -- a budget bounds work, never the search -- so the "
             "objective must not move across a sweep of this, and if it does, something "
             "else is wrong. SEQUENTIAL ONLY: this rebinds a module global in THIS "
             "process, and a spawned pricing worker imports dp_kernel fresh and gets the "
             "shipped value, so a parallel arm would silently measure the default while "
             "reporting the override.",
    )
    parser.add_argument(
        "--bootstrap-departures", type=int, default=0, metavar="N",
        help="run the pricing bootstrap over the first N departure steps before the real "
             "search (0 = off). Answer-affecting but optimality-safe, so the OBJECTIVE must "
             "not move across a sweep of this -- if it does, the bootstrap is not producing "
             "certified incumbents and the premise is wrong.",
    )
    parser.add_argument(
        "--carry-forward", action="store_true",
        help="prototype: reuse each flight's own previous priced column as its incumbent "
             "when it beats the solver's heuristic column under the current duals. "
             "Optimality-safe, and it reaches BOTH searches because it goes through "
             "`known_column`, which price_flight folds in before the compiled/reference "
             "fork. It can still return a DIFFERENT equally-optimal column, so read the "
             "objective, not just the wall clock.",
    )
    args = parser.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    if len(cfg.flight_levels_m) != 1:
        raise SystemExit(f"{args.scenario} has {len(cfg.flight_levels_m)} flight levels")
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        solver=args.solver,
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        bootstrap_departures=args.bootstrap_departures,
    )

    if args.max_label_log2 is not None:
        if dp_kernel_mod is None:
            raise SystemExit("--max-label-log2 needs the compiled kernel (numba missing)")
        dp_kernel_mod.MAX_LABEL_CAPACITY = 1 << args.max_label_log2

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"{args.solver} gap={args.gap_metric} sequential (fixed work, no clock)"
          f" bootstrap={args.bootstrap_departures}"
          f"{' carry-forward' if args.carry_forward else ''}")
    if dp_kernel_mod is not None:
        print(f"ceilings  MAX_LABEL_CAPACITY={dp_kernel_mod.MAX_LABEL_CAPACITY:,} "
              f"MAX_LOG2CAP={dp_kernel_mod.MAX_LOG2CAP}")

    install_probes(carry_forward=args.carry_forward)

    def _record(state: dict) -> None:
        SWEEP[0] = int(state.get("iteration") or SWEEP[0]) + 1

    started = time.perf_counter()
    result = ColGenSolver().solve(
        requests, cfg, static_terms, params, on_iteration=_record
    )
    wall = time.perf_counter() - started

    sweep_rows = [row for row in ROWS if not row["repair"]]
    repair_rows = [row for row in ROWS if row["repair"]]

    print("\n--- PER FLIGHT (pricing sweep, slowest first) ---")
    header = (f"{'sw':>3} {'flight':>7} {'entry_rc':>13} {'mid_rc':>13} {'final_rc':>13} "
              f"{'gap':>11} {'roots':>7} {'labels':>12} {'att':>4} {'status':>12} "
              f"{'boot_s':>8} {'search_s':>9} {'fallb_s':>9} {'total_s':>9}")
    print(header)
    for row in sorted(sweep_rows, key=lambda r: (r["sweep"], -r["total_s"])):
        entry = row["entry_rc"]
        final = row["final_rc"]
        mid = row["mid_rc"]
        gap = (
            None
            if final is None or entry is None or not math.isfinite(entry)
            else final - entry
        )
        print(
            f"{row['sweep']:>3} {row['flight']:>7} "
            f"{'-inf' if entry is None or not math.isfinite(entry) else format(entry, '13.4f')} "
            f"{'n/a' if mid is None else format(mid, '13.4f')} "
            f"{'n/a' if final is None else format(final, '13.4f')} "
            f"{'n/a' if gap is None else format(gap, '11.4f')} "
            f"{'n/a' if row['roots'] is None else row['roots']:>7} "
            f"{'n/a' if row['labels'] is None else format(row['labels'], ',d'):>12} "
            f"{'n/a' if row['attempts'] is None else row['attempts']:>4} "
            f"{str(row['status']):>12} "
            f"{row['bootstrap_s']:>8.2f} "
            f"{row['search_s']:>9.2f} {row['fallback_s']:>9.2f} {row['total_s']:>9.2f}"
        )

    if repair_rows:
        print(f"\n({len(repair_rows)} repair calls inside the greedy, "
              f"{sum(row['total_s'] for row in repair_rows):.1f}s total, excluded above)")

    print("\n--- STRAGGLER SHARE (per sweep) ---")
    print(f"{'sw':>3} {'flights':>8} {'sum_s':>10} {'max_s':>10} {'max_flight':>11} "
          f"{'share':>7} {'amdahl_cap':>11}")
    for sweep in sorted({row["sweep"] for row in sweep_rows}):
        rows = [row for row in sweep_rows if row["sweep"] == sweep]
        total = sum(row["total_s"] for row in rows)
        worst = max(rows, key=lambda r: r["total_s"])
        share = worst["total_s"] / total if total else 0.0
        print(f"{sweep:>3} {len(rows):>8} {total:>10.1f} {worst['total_s']:>10.1f} "
              f"{worst['flight']:>11} {100 * share:>6.1f}% {1 / share if share else 0:>11.2f}x")
    print("`amdahl_cap` is the speedup ANY worker pool can reach with that flight serial: "
          "1/share, unreachable in practice because the rest must also fit.")

    print(f"\nWALL {wall:.2f}s   iters={result.stats['iterations']} "
          f"cols={result.stats['n_columns']} obj={result.stats.get('objective')!r}")
    print(f"kernel {pricing_mod.kernel_stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
