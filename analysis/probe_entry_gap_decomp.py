"""Where the entry-cutoff gap actually lives, term by term.

``prof_colgen_cutoff`` reports ``gap = final_rc - entry_rc`` as one number.  This splits it.
Reduced cost is (``objective.py:60-65``, with ``objective.py:41-58`` expanding ``cost``):

    rc = benefit - cost - dual_cost - pi_f
    cost      = w_ground*ground_delay_s + w_air*air_hold_s + w_air*air_detour_s
    dual_cost = sum over the column's claimed rows of that row's dual

``benefit`` and ``pi_f`` are per-flight constants shared by every column of that flight, so
they CANCEL in a difference between two columns of the same flight:

    rc_final - rc_entry = (cost_entry - cost_final) + (dual_entry - dual_final)

That identity is the whole point.  It says the gap is either delay the entry column wastes
or duals the entry column pays -- and those call for completely different fixes.  A delay
gap means the seed departed or routed badly and a better seed search closes it.  A dual gap
means the optimum is dodging priced rows that no geodesic dodges, and no amount of seed
search over geodesics will ever find it.

Reported per (sweep, flight) for the entry incumbent and the final priced column:
``dep`` departure step, ``hops`` arc count, ``ground``/``hold``/``detour`` the three cost
terms in seconds, ``cost`` their weighted sum, ``dual`` the claim sum, ``rc`` the total.
Then the four deltas, which must satisfy ``d_rc == d_cost + d_dual`` to printing precision.

    uv run python analysis/probe_entry_gap_decomp.py --scenario density_faa_wing_zipline \
        --flights 12 --iterations 2 --bootstrap-roots 4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen.translate import column_to_intent  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

ROWS: list[dict] = []
SWEEP = [1]
CURRENT: dict = {}


def decompose(column, fg, cfg, view, benefit: float, pi_f: float, model) -> dict:
    """One column's reduced cost, split into the terms the formula is built from."""

    if column is None:
        return {}
    intent = column_to_intent(column, fg.request, cfg)
    detour_s = intent.air_detour_m / cfg.nominal_speed_mps
    dual = view.claim_cost(column.claims) if column.claims else 0.0
    cost = model.evaluate(
        ground_s=intent.ground_delay_s, air_hold_s=intent.air_hold_s, air_detour_s=detour_s
    )
    return dict(
        dep=int(column.departure_step),
        hops=len(column.cell_path) - 1,
        ground=float(intent.ground_delay_s),
        hold=float(intent.air_hold_s),
        detour=float(detour_s),
        cost=float(cost),
        stored_cost=float(column.delay_s),
        dual=float(dual),
        rc=float(model.reduced_cost(benefit=benefit, cost=cost, dual_cost=dual, pi_f=pi_f)),
    )


def install_probes() -> None:
    price_flight = pricing_mod.price_flight
    best_compiled = pricing_mod._best_column_compiled
    best_reference = pricing_mod._best_column

    def probed_price_flight(fg, duals, pi_f, cfg, params, **kwargs):
        view = (
            duals
            if isinstance(duals, pricing_mod.DualView)
            else pricing_mod.DualView(duals, cfg)
        )
        model = pricing_mod.cost_model(cfg, params)
        benefit = pricing_mod._benefit(params)
        outer = dict(CURRENT)
        CURRENT.clear()
        CURRENT.update(
            sweep=SWEEP[0],
            flight=fg.request.flight_id,
            repair=bool(kwargs.get("forbidden_rows")),
            ctx=(fg, cfg, view, benefit, float(pi_f), model),
            entry=None,
            final=None,
        )
        try:
            rc, column = price_flight(fg, duals, pi_f, cfg, params, **kwargs)
            CURRENT["final"] = decompose(column, fg, cfg, view, benefit, float(pi_f), model)
            CURRENT["final_rc"] = float(rc)
            return rc, column
        finally:
            row = dict(CURRENT)
            row.pop("ctx", None)
            ROWS.append(row)
            CURRENT.clear()
            CURRENT.update(outer)

    def _capture_entry(incumbent, bootstrap: bool) -> None:
        # Only the REAL search's entry incumbent; the bootstrap restricts roots and its own
        # entry is the pre-bootstrap one, which would overwrite the number being measured.
        if bootstrap or CURRENT.get("entry") is not None or not CURRENT.get("ctx"):
            return
        if incumbent is None:
            CURRENT["entry"] = {}
            return
        fg, cfg, view, benefit, pi_f, model = CURRENT["ctx"]
        CURRENT["entry"] = decompose(incumbent[1], fg, cfg, view, benefit, pi_f, model)
        CURRENT["entry_rc"] = float(incumbent[0])

    def probed_best_compiled(*args, incumbent=None, **kwargs):
        _capture_entry(incumbent, kwargs.get("keep_roots") is not None)
        return best_compiled(*args, incumbent=incumbent, **kwargs)

    def probed_best_reference(*args, incumbent=None, **kwargs):
        _capture_entry(incumbent, kwargs.get("keep_roots") is not None)
        return best_reference(*args, incumbent=incumbent, **kwargs)

    pricing_mod.price_flight = probed_price_flight
    pricing_mod._best_column_compiled = probed_best_compiled
    pricing_mod._best_column = probed_best_reference
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod

    pool_mod.price_flight = probed_price_flight
    solver_mod.price_flight = probed_price_flight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--bootstrap-roots", type=int, default=4)
    parser.add_argument(
        "--objective", default="total_cost", choices=("total_delay", "total_cost"),
        help="total_cost (the shipped default) is w_ground=1, w_air=3 (config.py:64-66), "
             "so one ground step costs dt=4 and one air hop costs 3*dt=12. total_delay "
             "weights them equally, which makes every ground-for-air swap EXACTLY tied.",
    )
    args = parser.parse_args()

    install_probes()
    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        bootstrap_roots=args.bootstrap_roots,
        objective=args.objective,
        # SEQUENTIAL, and not optional here.  This probe monkeypatches module-level
        # functions in THIS process, and `n_pricing_workers` now DEFAULTS TO 4.  The pool
        # uses the `spawn` context, so a worker re-imports the module and binds the REAL
        # function: the patch would reach only the parent, which prices nothing, and the
        # report would come back empty with no error saying why.
        n_pricing_workers=0,
    )

    def _record(state: dict) -> None:
        SWEEP[0] = int(state.get("iteration") or SWEEP[0]) + 1

    ColGenSolver().solve(requests, cfg, static_terms, params, on_iteration=_record)

    model = pricing_mod.cost_model(cfg, params)
    print(f"\nobjective={args.objective}  w_ground={model.ground_weight}  "
          f"w_air={model.air_weight}  dt_s={cfg.dt_s}  speed={cfg.nominal_speed_mps}")
    print(f"quanta: one ground step = {model.ground_weight * cfg.dt_s:g}, "
          f"one air hop = {model.air_weight * cfg.dt_s:g}  "
          f"(cell pitch = speed*dt = {cfg.nominal_speed_mps * cfg.dt_s:g} m)")
    header = (
        f"{'sw':>3} {'fl':>4} {'which':>6} {'dep':>6} {'hops':>5} {'ground':>9} {'hold':>8} "
        f"{'detour':>10} {'cost':>10} {'dual':>9} {'rc':>11}"
    )
    print(header)
    for row in sorted(ROWS, key=lambda r: (r["sweep"], r["flight"])):
        if row.get("repair"):
            continue
        for which in ("entry", "final"):
            d = row.get(which) or {}
            if not d:
                print(f"{row['sweep']:>3} {row['flight']:>4} {which:>6}   (none)")
                continue
            print(
                f"{row['sweep']:>3} {row['flight']:>4} {which:>6} {d['dep']:>6} "
                f"{d['hops']:>5} {d['ground']:>9.3f} {d['hold']:>8.3f} {d['detour']:>10.4f} "
                f"{d['cost']:>10.4f} {d['dual']:>9.4f} {d['rc']:>11.4f}"
            )
        e, f = row.get("entry") or {}, row.get("final") or {}
        if e and f:
            d_cost = e["cost"] - f["cost"]
            d_dual = e["dual"] - f["dual"]
            print(
                f"{'':>3} {'':>4} {'DELTA':>6} {f['dep'] - e['dep']:>6} "
                f"{f['hops'] - e['hops']:>5} {f['ground'] - e['ground']:>9.3f} "
                f"{f['hold'] - e['hold']:>8.3f} {f['detour'] - e['detour']:>10.4f} "
                f"{-d_cost:>10.4f} {-d_dual:>9.4f} {f['rc'] - e['rc']:>11.4f}"
                f"   | d_rc={f['rc'] - e['rc']:.4f} = d_cost {d_cost:+.4f}"
                f" + d_dual {d_dual:+.4f}"
                f"  resid={abs((f['rc'] - e['rc']) - (d_cost + d_dual)):.2e}"
            )
        print()

    pairs = [r for r in ROWS if not r.get("repair") and r.get("entry") and r.get("final")]
    if pairs:
        print("--- WHERE THE GAP LIVES ---")
        print(f"{'sw':>3} {'fl':>4} {'gap':>11} {'from delay':>12} {'from duals':>12} "
              f"{'delay share':>12} {'gap/dt':>8}")
        for r in sorted(pairs, key=lambda r: (r["sweep"], r["flight"])):
            e, f = r["entry"], r["final"]
            gap = f["rc"] - e["rc"]
            d_cost, d_dual = e["cost"] - f["cost"], e["dual"] - f["dual"]
            share = d_cost / gap if abs(gap) > 1e-12 else math.nan
            print(
                f"{r['sweep']:>3} {r['flight']:>4} {gap:>11.4f} {d_cost:>12.4f} "
                f"{d_dual:>12.4f} {share:>11.1%} {gap / cfg.dt_s:>8.3f}"
            )
        tot_gap = sum(f["rc"] - e["rc"] for e, f in ((r["entry"], r["final"]) for r in pairs))
        tot_cost = sum(e["cost"] - f["cost"] for e, f in ((r["entry"], r["final"]) for r in pairs))
        tot_dual = sum(e["dual"] - f["dual"] for e, f in ((r["entry"], r["final"]) for r in pairs))
        print(f"\ntotal gap {tot_gap:.4f} = delay {tot_cost:+.4f} + duals {tot_dual:+.4f}"
              f"   (delay is {tot_cost / tot_gap:.1%})" if abs(tot_gap) > 1e-12 else "")


if __name__ == "__main__":
    main()
