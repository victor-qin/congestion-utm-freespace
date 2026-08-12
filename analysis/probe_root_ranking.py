"""Would ranking roots by the COMPLETION BOUND beat ranking them by ``variants.score``?

The bootstrap searches the K best roots by ``PreparedVariants.score`` and hands what it
finds to the real search as a cutoff.  Written out against the quantity it is trying to
approximate::

    variants.score = -w_ground*ground_delay - w_air*origin_leg - start_dual_cost

    real rc        =  benefit - pi_f - [w_ground*ground_delay + w_air*air_hold
                                        + w_air*air_detour]
                              - SUM over rows in claims(whole column) of pi_row

    candidate      =  benefit - pi_f - delay_lbs[h0]
                              - max(start_dual_cost, dest_positive[h0])
                              + max_negative_credit,   h0 = hex_remaining[root cell]

``score`` is pure ``g(n)``: cost already incurred at the root, with the whole-route air term
truncated to the origin fold leg and the whole-column dual sum truncated to the origin
claims.  Measured previously, its two lane-varying terms are inert -- ``start_dual_cost``
constant across roots, ``origin_leg`` correlating ~0 with quality -- leaving one live bit,
"depart earlier", which is why ``bootstrap_roots=1`` provably fails (``entry_rc`` stays at
exactly 0.0000) while 2 works.

The candidate is ``completion_can_compete``'s own ``hop_rc_bound`` (dp_prepare.py:1228) at
the root's minimum feasible hop count -- an UPPER bound on what the root can achieve, i.e.
``g + h`` rather than ``g``.  It is already computed for every surviving root, because
``prepare_variants`` calls ``envelopes.can_compete`` on each one (dp_prepare.py:1486); this
probe only reads out the number instead of the verdict.

TEST 1 of the plan: do the two orderings even DIFFER?  If ``dest_positive[h0]`` is shared
across roots with the same corridor start and ``max_negative_credit`` is 0 (both measured
likely), the candidate may collapse to ``delay_lbs[h0]`` alone -- still a genuinely new bit
(route length, not just departure time), but worth confirming before touching the search.
A null here kills the idea for the cost of one probe.

    uv run python analysis/probe_root_ranking.py --scenario density_faa_wing_zipline \
        --flights 50 --flights-probed 8
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

from freespace_sim.planner.colgen import dp_prepare  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

ROWS: list[dict] = []
EVAL_ROWS: list[dict] = []
EVALUATE = [False, set()]


def candidate_bounds(fg, cfg, view, benefit, pi_f, model, forbidden, topology, variants,
                     envelopes) -> np.ndarray:
    """``hop_rc_bound`` per root, at that root's own minimum feasible hop count."""

    n = int(variants.departure_step.size)
    out = np.full(n, -math.inf, dtype=np.float64)
    for i in range(n):
        dep = int(variants.departure_step[i])
        lane_raw = int(variants.lane_idx[i])
        lane = None if lane_raw < 0 else lane_raw
        h0 = int(topology.hex_remaining[int(variants.cell[i])])
        if h0 < 1:
            h0 = 1
        delay_lbs, corridor_start = envelopes._delay_envelope(dep, lane)
        if h0 >= len(delay_lbs):
            continue  # the envelope's LENGTH already rejects this root
        dest_positive = envelopes._destination_cost(corridor_start + h0, h0)
        if not math.isfinite(dest_positive):
            continue
        paid_positive = max(0.0, float(variants.start_dual_cost[i]))
        out[i] = (
            benefit - pi_f - delay_lbs[h0]
            - max(paid_positive, dest_positive)
            + view.max_negative_credit
        )
    return out


def install(limit: int) -> None:
    price_flight = pricing_mod.price_flight
    seen = [0]

    def probed(fg, duals, pi_f, cfg, params, **kwargs):
        if seen[0] >= limit or kwargs.get("forbidden_rows"):
            return price_flight(fg, duals, pi_f, cfg, params, **kwargs)
        view = (duals if isinstance(duals, pricing_mod.DualView)
                else pricing_mod.DualView(duals, cfg))
        model = pricing_mod.cost_model(cfg, params)
        benefit = pricing_mod._benefit(params)
        forbidden = kwargs.get("forbidden_rows", frozenset())
        topology, rows = dp_prepare.prepared_for(fg, cfg)
        if topology.ok and rows.ok:
            envelopes = dp_prepare.CompletionEnvelopes(
                fg, cfg, view, benefit=benefit, pi_f=float(pi_f), model=model,
                forbidden_rows=forbidden, incumbent=None,
            )
            variants = dp_prepare.prepare_variants(
                fg, cfg, view, topology, rows, benefit=benefit, pi_f=float(pi_f),
                cost_cutoff=None, model=model, forbidden_rows=forbidden,
                envelopes=envelopes,
            )
            n = int(variants.departure_step.size)
            if n > 1:
                seen[0] += 1
                cand = candidate_bounds(fg, cfg, view, benefit, float(pi_f), model,
                                        forbidden, topology, variants, envelopes)
                score = np.asarray(variants.score, dtype=np.float64)
                # The bootstrap's own ordering: descending, STABLE.
                order_s = np.argsort(-score, kind="stable")
                order_c = np.argsort(-cand, kind="stable")
                finite = np.isfinite(cand)
                ROWS.append(dict(
                    flight=fg.request.flight_id, n_roots=n,
                    score_spread=float(score.max() - score.min()),
                    cand_spread=(float(cand[finite].max() - cand[finite].min())
                                 if finite.any() else 0.0),
                    dual_spread=float(variants.start_dual_cost.max()
                                      - variants.start_dual_cost.min()),
                    leg_spread=float(variants.origin_leg_w_s.max()
                                     - variants.origin_leg_w_s.min()),
                    neg_credit=float(view.max_negative_credit),
                    top1_same=bool(order_s[0] == order_c[0]),
                    top2_same=set(order_s[:2].tolist()) == set(order_c[:2].tolist()),
                    top4_overlap=len(set(order_s[:4].tolist())
                                     & set(order_c[:4].tolist())),
                    corr=(float(np.corrcoef(score[finite], cand[finite])[0, 1])
                          if finite.sum() > 2 else float("nan")),
                    n_finite=int(finite.sum()),
                ))
                if EVALUATE[0] and fg.request.flight_id in EVALUATE[1]:
                    # WHICH ranking is better, not merely whether they differ: run the
                    # bootstrap's own restricted search over each ordering's top K and
                    # compare the cutoff it comes back with.  Higher rc = better cutoff.
                    # `incumbent=None` so the number is the roots' UNAIDED value, and a
                    # deadline because an unaided search over a fat root has no cutoff to
                    # stop it.
                    import time as _t
                    for name, order in (("score", order_s), ("cand", order_c)):
                        for K in (1, 2, 4):
                            keep = frozenset(
                                (int(variants.departure_step[i]), int(variants.lane_idx[i]))
                                for i in order[:K]
                            )
                            t0 = _t.perf_counter()
                            try:
                                out = pricing_mod._best_column_compiled(
                                    fg, view, float(pi_f), cfg, benefit, forbidden,
                                    incumbent=None, model=model, keep_roots=keep,
                                    record_budget=False,
                                    deadline=_t.monotonic() + 120.0,
                                )
                            except Exception as exc:  # deadline raises
                                out = f"({type(exc).__name__})"
                            if isinstance(out, tuple):
                                rc = out[0]
                                got = "None" if out[1] is None else f"{rc:.4f}"
                            else:
                                got = str(out)
                            EVAL_ROWS.append(dict(
                                flight=fg.request.flight_id, ranking=name, K=K,
                                result=got, secs=_t.perf_counter() - t0,
                            ))
        return price_flight(fg, duals, pi_f, cfg, params, **kwargs)

    pricing_mod.price_flight = probed
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod
    pool_mod.price_flight = probed
    solver_mod.price_flight = probed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=50)
    parser.add_argument("--flights-probed", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--objective", default="total_cost")
    parser.add_argument(
        "--evaluate", type=int, nargs="*", default=None, metavar="FLIGHT",
        help="for these flight ids, actually RUN the bootstrap over each ranking's top K "
             "and report the cutoff each finds -- the only way to say which ordering is "
             "better rather than merely different.",
    )
    args = parser.parse_args()

    if args.evaluate is not None:
        EVALUATE[0] = True
        EVALUATE[1] = set(args.evaluate)
    install(args.flights_probed)
    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(max_iterations=args.iterations, time_limit_s=86400.0,
                          gap_metric="cost", objective=args.objective, bootstrap_roots=0)
    ColGenSolver().solve(requests, cfg, static_terms, params)

    print(f"\n{'fl':>4} {'roots':>6} {'fin':>5} {'corr':>7} {'top1=':>6} {'top2=':>6} "
          f"{'top4∩':>6} {'score_sp':>10} {'cand_sp':>10} {'dual_sp':>9} {'leg_sp':>9} "
          f"{'negcr':>7}")
    for r in ROWS:
        print(f"{r['flight']:>4} {r['n_roots']:>6} {r['n_finite']:>5} {r['corr']:>7.3f} "
              f"{str(r['top1_same']):>6} {str(r['top2_same']):>6} {r['top4_overlap']:>6} "
              f"{r['score_spread']:>10.3f} {r['cand_spread']:>10.3f} "
              f"{r['dual_spread']:>9.3f} {r['leg_spread']:>9.3f} {r['neg_credit']:>7.3g}")
    if ROWS:
        n = len(ROWS)
        print()
        print(f"flights probed          {n}")
        print(f"top-1 root SAME         {sum(r['top1_same'] for r in ROWS)}/{n}")
        print(f"top-2 set SAME          {sum(r['top2_same'] for r in ROWS)}/{n}")
        print(f"mean top-4 overlap      {sum(r['top4_overlap'] for r in ROWS)/n:.2f}/4")
        print(f"mean corr(score, cand)  {np.nanmean([r['corr'] for r in ROWS]):+.3f}")
        print(f"start_dual_cost spread  max {max(r['dual_spread'] for r in ROWS):.4f}"
              f"   (0 => inert across roots, as previously measured)")
        print(f"origin_leg spread       max {max(r['leg_spread'] for r in ROWS):.4f}")
        print()
        print("READ: if top-2 is SAME everywhere the candidate cannot change what the")
        print("bootstrap searches at K=2, and the idea is dead for this instance family.")
    if EVAL_ROWS:
        print(f"\n--- WHICH RANKING FINDS THE BETTER CUTOFF (higher rc wins) ---")
        print(f"{'fl':>4} {'ranking':>8} {'K':>2} {'cutoff rc':>14} {'secs':>8}")
        for r in EVAL_ROWS:
            print(f"{r['flight']:>4} {r['ranking']:>8} {r['K']:>2} {r['result']:>14} "
                  f"{r['secs']:>8.2f}")


if __name__ == "__main__":
    main()
