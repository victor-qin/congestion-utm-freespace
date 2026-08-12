"""Size the proposed backward completion DP before writing it.

Two numbers decide whether a dual-aware completion bound is worth building, and both are
per flight rather than per solve:

``cells x steps``  the state count of a backward DP over ``(cell, absolute step)``.  This is
                   the honest cost: the DP must be time-indexed because ``visit_cost`` is a
                   function of ``(cell, step)``, not of the cell alone.
``neg_credit``     ``DualView.max_negative_credit``.  The existing gate adds this to every
                   bound (``dp_prepare.py:1233``), so it is the slack a tighter completion
                   term would have to overcome.  If it is ~0 the DP replaces a pure loss;
                   if it is large the DP's win is capped by it.

    uv run python analysis/probe_backward_dp_size.py --scenario density_faa_wing_zipline \
        --flights 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import dp_prepare as dp_prepare_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

ROWS: list[dict] = []
SWEEP = [0]


def install_probe() -> None:
    price_flight = pricing_mod.price_flight

    def probed(fg, duals, pi_f, cfg, params, **kwargs):
        topology, rows = dp_prepare_mod.prepared_for(fg, cfg)
        row = {"sweep": SWEEP[0], "flight": fg.request.flight_id, "ok": topology.ok}
        # `price_flight` accepts either shape and normalizes internally; do the same here
        # rather than assume, so the probe cannot silently read a mapping as a view.
        view = duals if isinstance(duals, pricing_mod.DualView) else pricing_mod.DualView(
            duals, cfg
        )
        if topology.ok:
            n_cells = topology.n_cells
            n_steps = topology.max_step - topology.min_step + 1
            reachable = int(np.count_nonzero(topology.rev_remaining < dp_prepare_mod.UNREACHABLE))
            # Cells carrying at least one dual anywhere on their clock.
            prepared_duals = dp_prepare_mod.prepare_duals(view, fg, topology, rows)
            priced_cells = int(np.count_nonzero(prepared_duals.cell_series >= 0))
            row.update(
                cells=n_cells,
                reachable=reachable,
                steps=n_steps,
                arcs=int(topology.arc_target.shape[0]),
                air_hops=topology.air_hop_limit,
                shortest=topology.shortest_hops,
                priced_rows=int(prepared_duals.row_id.shape[0]),
                priced_cells=priced_cells,
                neg_credit=view.max_negative_credit,
                states=n_cells * n_steps,
                reach_states=reachable * n_steps,
            )
        ROWS.append(row)
        return price_flight(fg, duals, pi_f, cfg, params, **kwargs)

    pricing_mod.price_flight = probed
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod

    pool_mod.price_flight = probed
    solver_mod.price_flight = probed

    solve_sweep = solver_mod.ColGenSolver._price_sweep if hasattr(
        solver_mod.ColGenSolver, "_price_sweep"
    ) else None
    return solve_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument(
        "--objective", default="total_cost", choices=("total_delay", "total_cost"),
        help="total_cost is the shipped default (w_ground=1, w_air=3). The DUALS depend on "
             "it, so this probe's answer does too: the master prices a different problem "
             "under each, and dual sparsity is what decides whether the backward DP has "
             "anything to say.",
    )
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    install_probe()
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
        objective=args.objective,
    )
    ColGenSolver().solve(requests, cfg, static_terms, params)

    print(
        f"{'sw':>2} {'fl':>3} {'cells':>6} {'reach':>6} {'steps':>6} {'arcs':>7} "
        f"{'hops':>5} {'short':>5} {'p_rows':>7} {'p_cells':>7} "
        f"{'states':>10} {'reach_st':>10} {'neg_credit':>12}"
    )
    for row in ROWS:
        if not row.get("ok"):
            print(f"{row['sweep']:>2} {row['flight']:>3}  (unsupported)")
            continue
        print(
            f"{row['sweep']:>2} {row['flight']:>3} {row['cells']:>6} {row['reachable']:>6} "
            f"{row['steps']:>6} {row['arcs']:>7} {row['air_hops']:>5} {row['shortest']:>5} "
            f"{row['priced_rows']:>7} {row['priced_cells']:>7} "
            f"{row['states']:>10,} {row['reach_states']:>10,} {row['neg_credit']:>12.6g}"
        )
    ok = [r for r in ROWS if r.get("ok")]
    if ok:
        print()
        print(f"flights           {len(ok)}")
        print(f"max states        {max(r['states'] for r in ok):,}")
        print(f"max reach states  {max(r['reach_states'] for r in ok):,}")
        print(f"mean states       {sum(r['states'] for r in ok) / len(ok):,.0f}")
        print(f"max neg_credit    {max(r['neg_credit'] for r in ok):.6g}")
        print(f"max priced rows   {max(r['priced_rows'] for r in ok):,}")
        print(f"arc-steps (max)   {max(r['arcs'] * r['steps'] for r in ok):,}")


if __name__ == "__main__":
    main()
