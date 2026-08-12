"""What rank does the OPTIMUM's root actually have, under the bootstrap's ranking?

``bootstrap_roots=K`` searches the K best roots by ``PreparedVariants.score`` and hands
whatever it finds to the real search as a cutoff.  Whether that is worth doing is a
measured question with two very different answers:

* if the optimum's root sits at rank 1-2, the ranking WORKS and K is simply how much of a
  good list to take;
* if it sits at rank 40 of 64, the ranking is close to noise, the bootstrap's gain comes
  from stumbling on any merely-improving column, and the lever to pull is a better
  ranking rather than a larger K.

The x50 cost-currency logs bound this without settling it: on ``density_faa`` flight 14,
K=2 lifts ``entry_rc`` to 2.8485 against a true optimum of 25.5275, so the top two roots
demonstrably do NOT contain the optimum's root -- but they do not say where it is.

This measures it directly: for every root that survives ``prepare_variants``, run a search
restricted to that ONE root and record the reduced cost it reaches.  That is the root's
true value.  Then compare against the ``score`` ordering the bootstrap would have used.

``score`` is ``-w_ground*ground_delay - w_air*origin_leg - start_dual_cost`` -- cost so far,
with no lookahead at all.  Measured previously in delay currency, ``start_dual_cost`` was
CONSTANT across roots and ``origin_leg`` correlated ~0 with quality, leaving score carrying
effectively one bit: depart earlier.  This says whether that one bit is enough.

Every restricted search runs with ``record_budget=False``, so the probe cannot corrupt the
graph's ladder memo and change the very search it is measuring.  ``envelopes`` is passed to
``prepare_variants`` because ranking over the UN-gated root set is a silent no-op -- the
gate can reject the winner, ``prepare_variants`` returns empty, and the search reports
``(-inf, None)`` successfully.

    uv run python analysis/probe_root_truth.py --scenario density_faa_wing_zipline \
        --flights 50 --flight 14
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import dp_prepare  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

ROWS: list[dict] = []
SWEEP = [1]


def capture(fg, view, pi_f, cfg, params, benefit, model, forbidden, incumbent, out_path):
    """Price every surviving root on its own, and rank them two ways."""

    topology, rows = dp_prepare.prepared_for(fg, cfg)
    if not (topology.ok and rows.ok):
        return None
    envelopes = dp_prepare.CompletionEnvelopes(
        fg, cfg, view, benefit=benefit, pi_f=pi_f, model=model,
        forbidden_rows=forbidden, incumbent=incumbent,
    )
    variants = dp_prepare.prepare_variants(
        fg, cfg, view, topology, rows, benefit=benefit, pi_f=pi_f,
        cost_cutoff=None if incumbent is None else incumbent[0],
        model=model, forbidden_rows=forbidden, envelopes=envelopes,
    )
    n = int(variants.departure_step.size)
    if n == 0:
        return None

    # The bootstrap's own ordering, reproduced exactly: descending score, STABLE.
    order = np.argsort(-variants.score, kind="stable")

    pairs = []
    for rank, i in enumerate(order.tolist()):
        root = (int(variants.departure_step[i]), int(variants.lane_idx[i]))
        started = time.perf_counter()
        outcome = pricing_mod._best_column_compiled(
            fg, view, pi_f, cfg, benefit, forbidden,
            incumbent=None,            # no cutoff: the root's UNAIDED value
            model=model,
            keep_roots=frozenset({root}),
            record_budget=False,       # must not touch the ladder memo
        )
        if isinstance(outcome, pricing_mod.Declined):
            rc, column, why = -math.inf, None, outcome.value
        else:
            rc, column = outcome
            why = None
        pairs.append(dict(
            rank_by_score=rank,
            departure_step=root[0],
            lane_idx=root[1],
            score=float(variants.score[i]),
            ground_w=float(variants.ground_w[i]) if hasattr(variants, "ground_w") else None,
            rc=float(rc) if math.isfinite(rc) else None,
            found=column is not None,
            declined=why,
            secs=time.perf_counter() - started,
        ))
    return dict(
        n_roots=n,
        # Geometry, so the label count can be checked against `roots x cells x hops` --
        # the size of the space-time DAG the search enumerates once per root, since the
        # dominance key carries `paid_class` and `first_hop` and so never merges labels
        # that came from different roots.
        cells=int(topology.n_cells),
        arcs=int(topology.arc_target.shape[0]),
        shortest_hops=int(topology.shortest_hops),
        air_hop_limit=int(topology.air_hop_limit),
        steps=int(topology.max_step - topology.min_step + 1),
        revisit_depth=int(topology.revisit_depth),
        state_history_depth=int(topology.state_history_depth),
        pairs=pairs,
    )


def install(target_flight: int | None, out_path: Path) -> None:
    price_flight = pricing_mod.price_flight
    done = {"hit": False}

    def probed(fg, duals, pi_f, cfg, params, **kwargs):
        flight_id = fg.request.flight_id
        want = target_flight is None or flight_id == target_flight
        if not want or done["hit"] or kwargs.get("forbidden_rows"):
            return price_flight(fg, duals, pi_f, cfg, params, **kwargs)
        done["hit"] = True
        view = (
            duals if isinstance(duals, pricing_mod.DualView)
            else pricing_mod.DualView(duals, cfg)
        )
        model = pricing_mod.cost_model(cfg, params)
        benefit = pricing_mod._benefit(params)
        forbidden = kwargs.get("forbidden_rows", frozenset())
        started = time.perf_counter()
        captured = capture(
            fg, view, float(pi_f), cfg, params, benefit, model, forbidden, None, out_path
        )
        capture_s = time.perf_counter() - started

        rc, column = price_flight(fg, duals, pi_f, cfg, params, **kwargs)
        if captured is not None:
            captured.update(
                sweep=SWEEP[0], flight=flight_id, optimum_rc=float(rc),
                capture_s=capture_s,
            )
            ROWS.append(captured)
            out_path.write_text(json.dumps(ROWS, indent=1))
        return rc, column

    pricing_mod.price_flight = probed
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod

    pool_mod.price_flight = probed
    solver_mod.price_flight = probed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--flight", type=int, default=14,
        help="flight id to capture (the x50 cost-currency straggler is 14 on density_faa, "
             "48 on density_future)",
    )
    parser.add_argument("--objective", default="total_cost",
                        choices=("total_delay", "total_cost"))
    parser.add_argument("--out", default=".context/issue90/root_truth_cost.json")
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    install(args.flight, out_path)

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        max_iterations=args.iterations, time_limit_s=86400.0, objective=args.objective,
    )
    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} flight={args.flight} "
          f"objective={args.objective}")

    def _record(state: dict) -> None:
        SWEEP[0] = int(state.get("iteration") or SWEEP[0]) + 1

    ColGenSolver().solve(requests, cfg, static_terms, params, on_iteration=_record)

    for row in ROWS:
        pairs = row["pairs"]
        opt = row["optimum_rc"]
        scored = [p for p in pairs if p["rc"] is not None]
        best = max((p["rc"] for p in scored), default=None)
        print(f"\n=== sweep {row['sweep']} flight {row['flight']}: {row['n_roots']} roots, "
              f"capture {row['capture_s']:.1f}s ===")
        print(f"geometry  cells={row['cells']:,} arcs={row['arcs']:,} "
              f"shortest={row['shortest_hops']} air_hop_limit={row['air_hop_limit']} "
              f"(overrun {row['air_hop_limit'] - row['shortest_hops']}) "
              f"steps={row['steps']} revisit_depth={row['revisit_depth']} "
              f"state_depth={row['state_history_depth']}")
        print(f"DAG size  roots x cells x hops = {row['n_roots']:,} x {row['cells']:,} x "
              f"{row['air_hop_limit']} = "
              f"{row['n_roots'] * row['cells'] * row['air_hop_limit']:,}")
        print(f"full-search optimum rc   {opt:.4f}")
        print(f"best single-root rc      {best if best is None else round(best, 4)}")
        if best is None:
            continue
        # Rank (by score) of the roots that actually reach the best value.
        winners = [p["rank_by_score"] for p in scored if p["rc"] >= best - 1e-9]
        print(f"roots reaching that      {len(winners)}  at score-ranks {winners[:10]}")
        print(f"BEST ROOT'S SCORE-RANK   {min(winners)} of {row['n_roots']}"
              f"   -> smallest K that finds it: {min(winners) + 1}")
        reach_opt = [p["rank_by_score"] for p in scored if p["rc"] >= opt - 1e-9]
        print(f"roots reaching the OPTIMUM {len(reach_opt)}"
              + (f" at ranks {reach_opt[:10]}" if reach_opt else "  (none -- the optimum "
                 "needs a cutoff the single-root searches never get)"))
        rcs = [p["rc"] for p in scored]
        ranks = [p["rank_by_score"] for p in scored]
        if len(rcs) > 2:
            corr = float(np.corrcoef(np.array(ranks, float), np.array(rcs, float))[0, 1])
            print(f"corr(score-rank, rc)     {corr:+.3f}   "
                  f"(negative = the ranking is informative)")
        top = sorted(scored, key=lambda p: -p["rc"])[:8]
        print(f"\n{'score_rank':>10} {'dep':>6} {'lane':>5} {'score':>12} {'rc':>10} {'s':>7}")
        for p in top:
            print(f"{p['rank_by_score']:>10} {p['departure_step']:>6} {p['lane_idx']:>5} "
                  f"{p['score']:>12.4f} {p['rc']:>10.4f} {p['secs']:>7.2f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
