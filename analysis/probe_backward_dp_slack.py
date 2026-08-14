"""How much slack is left in the completion bound's DUAL term, before building the DP.

``probe_backward_dp_size`` sizes the state space.  This answers the prior question: is
there anything to win?  The existing gate (``dp_prepare.py:1228``) bounds a label's
achievable reduced cost by

    benefit - pi_f - delay_lb[total_hops] - max(paid_positive, dest_positive)
                                          + max_negative_credit

and charges **nothing** for the duals the label must still pay between where it stands and
a destination.  A backward DP over ``(cell, step)`` would supply exactly that term.  Its
value is capped by how large that term can be, so measure the term itself:

``D[c][s]``  the minimum total ``visit_cost`` payable from cell ``c`` at step ``s`` to any
             destination, over the any-role arc superset with no revisit ban.  A relaxation
             of the real completion set, so ``D`` lower-bounds the true remaining cost --
             which is what a pruning bound needs.

Reported per flight: the share of live ``(cell, step)`` states where ``D > 0`` at all, and
the distribution of ``D`` where it is positive.  If ``D`` is zero almost everywhere the DP
cannot tighten anything and the idea is dead before a line of kernel goes in.

    uv run python analysis/probe_backward_dp_slack.py --scenario colgen_test --flights 12
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
SWEEP = [1]


class _Enough(Exception):
    """Stop the solve once the probe has its sample.

    Dual DENSITY is what this probe measures, and it is a property of the instance
    size, not of how long the solve runs -- so a 100-flight first sweep answers the
    question at 1/50th the cost of letting that solve finish.
    """


def dense_visit_cost(
    duals: dp_prepare_mod.PreparedDuals, n_cells: int, step0: int, n_steps: int
) -> np.ndarray:
    """``visit_cost(cell, step)`` for every ``(cell, step)``, as one dense array.

    Built from the same prefix series the search reads, one cell at a time, so a cell with
    no dual anywhere costs one ``continue`` rather than a row of arithmetic.
    """

    out = np.zeros((n_cells, n_steps), dtype=np.float64)
    steps = np.arange(step0, step0 + n_steps)
    lo = steps + duals.offsets_lo
    hi = steps + duals.offsets_hi + 1
    for cell in range(n_cells):
        series = int(duals.cell_series[cell])
        if series < 0:
            continue  # no dual anywhere on this cell's clock
        lo_index = int(duals.series_start[series])
        hi_index = int(duals.series_start[series + 1])
        length = hi_index - lo_index
        if length <= 1:
            continue
        first = int(duals.series_first[series])
        stop = first + length - 1
        prefix = duals.series_prefix[lo_index:hi_index]
        lo_c = np.clip(lo, first, stop)
        hi_c = np.clip(hi, first, stop)
        # `range_sum`'s own arithmetic, one row at a time instead of one cell-step at a
        # time.  Same two floats subtracted, so the values are identical.
        out[cell] = np.where(hi_c > lo_c, prefix[hi_c - first] - prefix[lo_c - first], 0.0)
    return out


def backward_dp(
    topology: dp_prepare_mod.PreparedTopology, vcost: np.ndarray, step0: int, n_steps: int
) -> np.ndarray:
    """``D[cell, step]`` — cheapest remaining dual cost to any destination.

    Backwards over absolute step, because ``visit_cost`` is a function of ``(cell, step)``
    and hops advance in lockstep with steps (every arc is exactly one step; there is no
    wait arc).  Within a layer every cell is independent, so the whole layer is one
    ``np.minimum.reduceat`` over the CSR — which is also the shape a kernel would take.
    """

    n_cells = topology.n_cells
    arc_start = topology.arc_start.astype(np.int64)
    arc_target = topology.arc_target.astype(np.int64)
    dest = topology.dest_mask != 0

    inf = np.inf
    D = np.full((n_cells, n_steps), inf, dtype=np.float64)
    D[dest, n_steps - 1] = 0.0
    # Cells with no outgoing arc would make `reduceat` read past the end; mask them out.
    has_arc = np.diff(arc_start) > 0
    heads = arc_start[:-1][has_arc]

    for i in range(n_steps - 2, -1, -1):
        nxt = D[:, i + 1]
        # Cost of stepping into `arc_target[a]` at step i+1 and completing from there.
        edge = vcost[arc_target, i + 1] + nxt[arc_target]
        layer = np.full(n_cells, inf, dtype=np.float64)
        if heads.size:
            layer[has_arc] = np.minimum.reduceat(edge, heads)
        # Arriving here is always an option for a destination cell: the visit that entered
        # it was charged on the arc in, exactly as `label_score` charges it.
        layer[dest] = np.minimum(layer[dest], 0.0)
        D[:, i] = layer
    return D


def deadline_dp(
    topology: dp_prepare_mod.PreparedTopology,
    vcost: np.ndarray,
    deadline_i: int,
    budget: int,
) -> np.ndarray:
    """``B[cell, j]`` — cheapest remaining dual cost, arriving no later than a deadline.

    This is the bound that can actually be positive, and :func:`backward_dp` is the one
    that cannot.  A label may only take ``air_hop_limit - hops`` more hops, and
    ``air_hop_limit`` is the geodesic plus ``max_air_overrun_hops`` (3 by default) -- so the
    freedom to detour around a priced cell is three hops, not unlimited.  Dropping that
    constraint is what makes the unbounded DP answer zero everywhere: with all duals
    non-negative and a wide corridor, an unbudgeted walker always finds a free way round.

    Indexed by ``j``, the offset back from the deadline, so column ``j`` holds states at
    absolute step ``deadline - j`` with exactly ``j`` hops of budget left.  A variant's
    deadline is ``corridor_start + air_hop_limit``, a constant per root, which is what
    makes a per-deadline 2D sweep equivalent to a 3D ``(cell, step, budget)`` table.
    """

    n_cells = topology.n_cells
    arc_start = topology.arc_start.astype(np.int64)
    arc_target = topology.arc_target.astype(np.int64)
    dest = topology.dest_mask != 0
    has_arc = np.diff(arc_start) > 0
    heads = arc_start[:-1][has_arc]

    B = np.full((n_cells, budget + 1), np.inf, dtype=np.float64)
    B[dest, 0] = 0.0  # at the deadline itself, only a destination is a valid state
    for j in range(1, budget + 1):
        step = deadline_i - j
        if step < 0:
            break
        nxt = B[:, j - 1]
        edge = vcost[arc_target, step + 1] + nxt[arc_target]
        layer = np.full(n_cells, np.inf, dtype=np.float64)
        if heads.size:
            layer[has_arc] = np.minimum.reduceat(edge, heads)
        layer[dest] = np.minimum(layer[dest], 0.0)  # arriving early is always allowed
        B[:, j] = layer
    return B


def install_probe(limit: int, deadline_samples: int) -> None:
    price_flight = pricing_mod.price_flight
    seen = [0]

    def probed(fg, duals, pi_f, cfg, params, **kwargs):
        if seen[0] < limit:
            seen[0] += 1
            topology, rows = dp_prepare_mod.prepared_for(fg, cfg)
            if topology.ok:
                view = (
                    duals
                    if isinstance(duals, pricing_mod.DualView)
                    else pricing_mod.DualView(duals, cfg)
                )
                prepared = dp_prepare_mod.prepare_duals(view, fg, topology, rows)
                step0 = topology.min_step
                n_steps = topology.max_step - step0 + 1
                vcost = dense_visit_cost(prepared, topology.n_cells, step0, n_steps)
                D = backward_dp(topology, vcost, step0, n_steps)
                live = np.isfinite(D)
                positive = live & (D > 0.0)
                vals = D[positive]

                # The budgeted twin, over a spread of deadlines.  Every root's deadline is
                # `corridor_start + air_hop_limit`, and corridor starts run the length of
                # the departure window, so an even sample over that window samples the
                # deadlines the real search actually uses.
                budget = int(topology.air_hop_limit)
                b_live = b_pos = 0
                b_max = 0.0
                b_vals: list[float] = []
                if deadline_samples > 0 and budget > 0:
                    first = budget + 1
                    last = n_steps - 1
                    if last > first:
                        for d in np.linspace(first, last, deadline_samples).astype(int):
                            B = deadline_dp(topology, vcost, int(d), budget)
                            fin = np.isfinite(B)
                            pos_b = fin & (B > 0.0)
                            b_live += int(fin.sum())
                            b_pos += int(pos_b.sum())
                            if pos_b.any():
                                b_max = max(b_max, float(B[pos_b].max()))
                                b_vals.append(float(B[pos_b].mean()))
                ROWS.append(
                    dict(
                        sweep=SWEEP[0],
                        flight=fg.request.flight_id,
                        cells=topology.n_cells,
                        steps=n_steps,
                        budget=budget,
                        live=int(live.sum()),
                        pos=int(positive.sum()),
                        share=float(positive.sum()) / max(1, int(live.sum())),
                        vmax=float(vals.max()) if vals.size else 0.0,
                        vmean=float(vals.mean()) if vals.size else 0.0,
                        b_live=b_live,
                        b_pos=b_pos,
                        b_share=b_pos / max(1, b_live),
                        b_max=b_max,
                        b_mean=(sum(b_vals) / len(b_vals)) if b_vals else 0.0,
                        vcost_max=float(vcost.max()),
                        vcost_sum=float(vcost.sum()),
                        neg=view.max_negative_credit,
                    )
                )
                if seen[0] >= limit:
                    raise _Enough
        return price_flight(fg, duals, pi_f, cfg, params, **kwargs)

    pricing_mod.price_flight = probed
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod

    pool_mod.price_flight = probed
    solver_mod.price_flight = probed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="colgen_test")
    parser.add_argument(
        "--objective", default="total_cost", choices=("total_delay", "total_cost"),
        help="total_cost is the shipped default (w_ground=1, w_air=3). The DUALS depend on "
             "it, so this probe's answer does too: the master prices a different problem "
             "under each, and dual sparsity is what decides whether the backward DP has "
             "anything to say.",
    )
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--deadline-samples", type=int, default=48, metavar="N",
        help="how many arrival deadlines to sample for the BUDGETED bound (0 = skip). "
             "Every root's deadline is corridor_start + air_hop_limit, and corridor starts "
             "span the departure window, so an even sample over that window is a sample "
             "over the deadlines the search really uses.",
    )
    parser.add_argument(
        "--bootstrap-roots", type=int, default=0, metavar="K",
        help="run the pricing bootstrap, purely to make a density solve tractable. The "
             "DP measured here is a function of the duals and the graph, and the bootstrap "
             "is objective-neutral, so it changes how long this takes and not what it says.",
    )
    parser.add_argument(
        "--probe-flights", type=int, default=24,
        help="how many price_flight calls to instrument; the DP is O(cells*steps) in "
             "numpy and would otherwise dominate a long run",
    )
    args = parser.parse_args()

    install_probe(args.probe_flights, args.deadline_samples)
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
        bootstrap_roots=args.bootstrap_roots,
        # SEQUENTIAL, and not optional here.  This probe monkeypatches module-level
        # functions in THIS process, and `n_pricing_workers` now DEFAULTS TO 4.  The pool
        # uses the `spawn` context, so a worker re-imports the module and binds the REAL
        # function: the patch would reach only the parent, which prices nothing, and the
        # report would come back empty with no error saying why.
        n_pricing_workers=0,
    )
    try:
        ColGenSolver().solve(requests, cfg, static_terms, params)
    except _Enough:
        print(f"(stopped after {len(ROWS)} probed flights)")

    print(
        f"\n{'fl':>3} {'cells':>6} {'steps':>6} {'bud':>4} {'live':>11} {'D>0':>8} "
        f"{'share':>7} | {'b_live':>10} {'B>0':>10} {'b_share':>8} {'B_max':>9} "
        f"{'B_mean':>9} | {'vc_max':>8} {'vc_sum':>9}"
    )
    for row in ROWS:
        print(
            f"{row['flight']:>3} {row['cells']:>6} {row['steps']:>6} {row['budget']:>4} "
            f"{row['live']:>11,} {row['pos']:>8,} {row['share']:>7.2%} | "
            f"{row['b_live']:>10,} {row['b_pos']:>10,} {row['b_share']:>8.3%} "
            f"{row['b_max']:>9.4f} {row['b_mean']:>9.4f} | "
            f"{row['vcost_max']:>8.3f} {row['vcost_sum']:>9.2f}"
        )
    if ROWS:
        live = sum(r["live"] for r in ROWS)
        pos = sum(r["pos"] for r in ROWS)
        b_live = sum(r["b_live"] for r in ROWS)
        b_pos = sum(r["b_pos"] for r in ROWS)
        print()
        print(f"flights probed          {len(ROWS)}")
        print(f"UNBUDGETED live states  {live:,}")
        print(f"  with D>0              {pos:,}  ({pos / max(1, live):.3%})")
        print(f"  max D                 {max(r['vmax'] for r in ROWS):.6f}")
        print(f"BUDGETED live states    {b_live:,}")
        print(f"  with B>0              {b_pos:,}  ({b_pos / max(1, b_live):.3%})")
        print(f"  max B                 {max(r['b_max'] for r in ROWS):.6f}")
        print(f"max single visit        {max(r['vcost_max'] for r in ROWS):.6f}")


if __name__ == "__main__":
    main()
