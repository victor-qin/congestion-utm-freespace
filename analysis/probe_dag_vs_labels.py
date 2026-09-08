"""Measure issue #94's claim: labels/memory today vs a geometry-pinned DAG sweep.

Per flight, per sweep, this records BOTH sides of the comparison:

* **today** -- ``n_labels`` (main + bootstrap searches) from ``last_search_record()``,
  the final label budget and dominance-table rung, and the derived bytes
  (40 B/label arena + 32 B/slot table, the constants ``dp_kernel`` documents).
* **issue #94's side-graph DAG** -- the union space-time ellipse ``slots`` =
  sum over cells of the (departure fan + hop slack) step window, computed from the
  packed topology exactly as the search sees it (forward BFS over the CSR arcs +
  ``rev_remaining``); side-nodes = 6 x slots, side-arcs = 3 x side-nodes, and the
  dense-sweep bytes (1 B parent per node + two live 6-wide value layers).

Leading-only ("asym") filing -- every corridor transit volume live ``[t0, t1 +
time_buffer_s]`` -- is what production now ships, so ``--filing asym`` is UNPATCHED and is
the default: ``derive_cell_window`` MEASURES a 3-period footprint ``(-1, 1)``,
``revisit_depth`` is 2, and the state key carries one fewer ``recent[]`` entry than it did
before, which was refinement 2 of #94.

``--filing legacy-sym`` restores the historical ``[t0 - buf, t1 + buf]`` pads for the A/B.
Pairwise transit separation is the SUM of the two facing pads, so that arm doubles the
enforced gap (4 s -> 8 s at defaults) and solves a TIGHTER capacity model; the objective
may move. The acceptance test (``windows._cross_check_conflicts``) runs unmodified on both
arms and will raise if either filing under-covers a ledger template conflict.

Pricing-side only: the commit path (``translate._retime_lattice_reservation``) also stamps
sub-box windows, but it cannot feed back into pricing, which is all this probe measures.
See ``probe_ledger_e2e`` for the arm that patches both halves.

Sequential only, same reason as ``probe_backward_dp_size``: the probe monkeypatches
module globals in THIS process and a spawned worker would silently bind the real ones.

    uv run python analysis/probe_dag_vs_labels.py --flights 12 --iterations 3 \
        --max-label-log2 27
    uv run python analysis/probe_dag_vs_labels.py --flights 12 --iterations 3 \
        --max-label-log2 27 --filing legacy-sym
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim import volumes as volumes_mod  # noqa: E402
from freespace_sim.planner.colgen import dp_prepare as dp_prepare_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen import windows as windows_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

try:
    from freespace_sim.planner.colgen import dp_kernel as dp_kernel_mod  # noqa: E402
except ImportError:  # pragma: no cover - depends on the install
    dp_kernel_mod = None

LABEL_BYTES = 40   # dp_kernel: one float64 + eight int32 across the nine arrays
SLOT_BYTES = 32    # dp_kernel: four tables + two layer buffers per dominance slot
SIDES = 6          # hex entry sides; #94's side-graph multiplies (cell, step) by this
OUT_DEGREE = 3     # side-graph out-degree: straight + the two 60-degree turns

ROWS: list[dict] = []
SWEEP = [0]


def install_legacy_sym_filing() -> None:
    """Restore the pre-2026-08-14 symmetric time padding on corridor transit volumes.

    Total pad (and therefore the enforced pairwise gap between any two transits, which
    depends only on the SUM of their facing pads) grows from 1*buf back to 2*buf; the
    trailing edge leaves the discrete clock, which is what re-widens the measured cell
    window from 3 periods to 4.
    """

    real = volumes_mod.corridor_segment_volume

    def sym_corridor_segment_volume(p0, t0, p1, t1, cfg, *, terminal_id=None):
        vol = real(p0, t0, p1, t1, cfg, terminal_id=terminal_id)
        return replace(vol, t_start=vol.t_start - cfg.time_buffer_s)

    # Every module that bound the builder by name and can run during a solve.  The commit
    # path (`translate._retime_lattice_reservation`) stamps sub-box windows too, and this
    # arm leaves it shipped-asymmetric -- it cannot feed back into pricing, which is all
    # this probe measures. `probe_ledger_e2e` is the arm that patches both halves.
    from freespace_sim.planner import terminal_capacity as terminal_capacity_mod
    from freespace_sim.planner.colgen import network as network_mod

    for mod in (volumes_mod, windows_mod, network_mod, terminal_capacity_mod):
        mod.corridor_segment_volume = sym_corridor_segment_volume
    windows_mod.derive_cell_window.cache_clear()
    windows_mod.validate_edge_locality.cache_clear()


def _forward_hops(topology) -> np.ndarray:
    """Hops from the nearest origin cell, over the packed CSR arcs (any role)."""

    unreachable = dp_prepare_mod.UNREACHABLE
    fwd = np.full(topology.n_cells, unreachable, dtype=np.int64)
    queue: deque[int] = deque()
    for cell in np.unique(topology.origin_cell):
        cell = int(cell)
        if fwd[cell] != 0:
            fwd[cell] = 0
            queue.append(cell)
    arc_start, arc_target = topology.arc_start, topology.arc_target
    while queue:
        source = queue.popleft()
        next_hops = fwd[source] + 1
        for a in range(int(arc_start[source]), int(arc_start[source + 1])):
            target = int(arc_target[a])
            if next_hops < fwd[target]:
                fwd[target] = next_hops
                queue.append(target)
    return fwd


def _sizing(topology) -> dict:
    """Issue #94 arithmetic for one flight, from the same packed arrays the DP uses."""

    unreachable = dp_prepare_mod.UNREACHABLE
    fwd = _forward_hops(topology)
    rev = topology.rev_remaining.astype(np.int64)
    n_dep = int(topology.latest_departure_step) - int(topology.base_step) + 1
    mask = (fwd < unreachable) & (rev < unreachable)
    slack = topology.air_hop_limit - fwd[mask] - rev[mask]
    width = np.clip(n_dep + slack, 0, None)
    slots = int(width.sum())
    side_nodes = SIDES * slots
    n_cells = int(topology.n_cells)
    return {
        "cells": n_cells,
        "reach_cells": int(mask.sum()),
        "arcs": int(topology.arc_target.shape[0]),
        "n_dep": n_dep,
        "air_hops": int(topology.air_hop_limit),
        "shortest": int(topology.shortest_hops),
        "revisit_depth": int(topology.revisit_depth),
        "state_depth": int(topology.state_history_depth),
        "slots": slots,
        "side_nodes": side_nodes,
        "side_arcs": OUT_DEGREE * side_nodes,
        # 1 B parent per node for path reconstruction + two live dense value layers.
        "dag_bytes": side_nodes * 1 + 2 * SIDES * n_cells * 8,
    }


def install_probe() -> None:
    price_flight = pricing_mod.price_flight

    def probed(fg, duals, pi_f, cfg, params, **kwargs):
        topology, _rows = dp_prepare_mod.prepared_for(fg, cfg)
        row = {"sweep": SWEEP[0], "flight": int(fg.request.flight_id), "ok": topology.ok}
        if topology.ok:
            row.update(_sizing(topology))
        pricing_mod.clear_search_record()
        started = time.perf_counter()
        try:
            return price_flight(fg, duals, pi_f, cfg, params, **kwargs)
        finally:
            row["wall_s"] = time.perf_counter() - started
            record = pricing_mod.last_search_record()
            row["n_labels"] = int(record.get("n_labels", 0))
            row["bootstrap_labels"] = int(record.get("bootstrap_labels", 0))
            row["label_budget"] = int(record.get("label_budget", 0))
            row["status"] = record.get("status")
            row["declined"] = bool(record.get("declined", False))
            row["declined_reason"] = record.get("declined_reason")
            row["fallback_s"] = float(record.get("fallback_s", 0.0))
            budget = getattr(getattr(fg, "_search_cache", None), "dag_budget", None)
            row["log2cap"] = int(budget[1]) if budget is not None else None
            ROWS.append(row)

    pricing_mod.price_flight = probed
    from freespace_sim.planner.colgen import pricing_pool as pool_mod
    from freespace_sim.planner.colgen import solver as solver_mod

    pool_mod.price_flight = probed
    solver_mod.price_flight = probed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--ladder", type=int, default=None,
                        help="override seed_ladder_steps (default: shipped 20)")
    parser.add_argument("--bootstrap-roots", type=int, default=None,
                        help="override bootstrap_roots (default: shipped 1). 0 reproduces "
                             "the pre-#93 regime where #90's straggler exhausts the pool")
    parser.add_argument("--max-label-log2", type=int, default=None,
                        help="raise dp_kernel.MAX_LABEL_CAPACITY so stragglers COMPLETE "
                             "and report their true demand instead of declining")
    parser.add_argument("--filing", default="asym", choices=("asym", "legacy-sym"))
    parser.add_argument(
        "--gap-metric", default="cost", choices=("cost", "revenue"),
        help="`cost` by default, NOT the shipped `revenue`: the revenue gate is diluted "
             "by n*M and reads `stop` at iteration 1 under the ladder (#84 known limit), "
             "which would leave nothing but seeding duals to measure.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.filing == "legacy-sym":
        install_legacy_sym_filing()
    if args.max_label_log2 is not None:
        if dp_kernel_mod is None:
            raise SystemExit("--max-label-log2 requires the compiled kernel")
        dp_kernel_mod.MAX_LABEL_CAPACITY = 1 << args.max_label_log2

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    offsets = windows_mod.derive_cell_window(cfg)
    print(f"tree      {_loaded.parent.parent}")
    print(f"filing    {args.filing}  cell window {offsets} "
          f"({offsets[1] - offsets[0] + 1} periods, revisit_depth={offsets[1] - offsets[0]})")
    if dp_kernel_mod is not None:
        print(f"ceilings  MAX_LABEL_CAPACITY={dp_kernel_mod.MAX_LABEL_CAPACITY:,} "
              f"MAX_LOG2CAP={dp_kernel_mod.MAX_LOG2CAP}")

    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    overrides = {}
    if args.ladder is not None:
        overrides["seed_ladder_steps"] = args.ladder
    if args.bootstrap_roots is not None:
        overrides["bootstrap_roots"] = args.bootstrap_roots
    params = ColGenParams(
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        # Sequential, mandatory: the probe patches module globals in this process.
        n_pricing_workers=0,
        **overrides,
    )
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"ladder={params.seed_ladder_steps} K={params.bootstrap_roots}/"
          f"{params.bootstrap_ranking} obj={params.objective}")

    install_probe()

    def _record(state: dict) -> None:
        SWEEP[0] = state.get("iteration", SWEEP[0])

    started = time.perf_counter()
    result = ColGenSolver().solve(requests, cfg, static_terms, params, on_iteration=_record)
    wall = time.perf_counter() - started
    print(f"\nWALL {wall:.1f}s  iters={result.stats['iterations']} "
          f"cols={result.stats['n_columns']} objective={result.stats.get('objective')!r} "
          f"lp={result.stats.get('lp_objective')!r}")
    print(f"kernel_priced={result.stats.get('kernel_priced')} "
          f"kernel_fell_back={result.stats.get('kernel_fell_back')}")

    ok = [r for r in ROWS if r.get("ok")]
    print(
        f"\n{'sw':>2} {'fl':>3} {'cells':>6} {'ndep':>5} {'slots':>10} {'side_nodes':>11} "
        f"{'labels':>12} {'boot_lab':>10} {'lab/slot':>9} {'lab/node':>9} "
        f"{'MB_now':>8} {'MB_dag':>7} {'cap':>4} {'status':>12} {'wall_s':>8}"
    )
    for r in ok:
        labels = r["n_labels"] + r["bootstrap_labels"]
        table_bytes = SLOT_BYTES * (1 << r["log2cap"]) if r.get("log2cap") else 0
        now_bytes = LABEL_BYTES * max(r["n_labels"], r["bootstrap_labels"]) + table_bytes
        r["now_bytes"] = now_bytes
        r["labels_total"] = labels
        print(
            f"{r['sweep']:>2} {r['flight']:>3} {r['cells']:>6} {r['n_dep']:>5} "
            f"{r['slots']:>10,} {r['side_nodes']:>11,} {labels:>12,} "
            f"{r['bootstrap_labels']:>10,} "
            f"{labels / r['slots'] if r['slots'] else float('nan'):>9.2f} "
            f"{labels / r['side_nodes'] if r['side_nodes'] else float('nan'):>9.2f} "
            f"{now_bytes / 1e6:>8.1f} {r['dag_bytes'] / 1e6:>7.1f} "
            f"{r['log2cap'] if r['log2cap'] is not None else '-':>4} "
            f"{(r['status'] or '-')[:12]:>12} {r['wall_s']:>8.2f}"
        )
    if ok:
        worst = max(ok, key=lambda r: r["labels_total"])
        print(f"\ncalls             {len(ok)} (flights x sweeps)")
        print(f"sum labels        {sum(r['labels_total'] for r in ok):,}")
        print(f"sum slots         {sum(r['slots'] for r in ok):,}")
        print(f"peak labels/call  {worst['labels_total']:,} (flight {worst['flight']}, "
              f"sweep {worst['sweep']})")
        print(f"peak now MB       {max(r['now_bytes'] for r in ok) / 1e6:,.1f}")
        print(f"peak dag MB       {max(r['dag_bytes'] for r in ok) / 1e6:,.1f}")
        print(f"sum wall_s        {sum(r['wall_s'] for r in ok):,.1f}")
        print(f"declines          {sum(1 for r in ok if r['declined'])}")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps({"filing": args.filing, "rows": ROWS}, indent=1))
        print(f"rows -> {args.json_out}")


if __name__ == "__main__":
    main()
