"""Does collapsing the DP's history depth change the priced column?

``analysis/probe_window_state_width.py`` measures what the W=4 visit window costs
the label pool.  This asks the other half: run the *whole* compiled gate --
kernel proposal plus exact Python certification -- with the history depth
overridden, and compare the certified reduced cost and column against the
unmodified run.

Why the answer could legitimately be "unchanged": ``_canonical_candidate``
recertifies every proposal from the real de-duplicated claim set, so the relaxed
search can only change *which* columns get proposed, never how one is priced.
And a path that revisits a cell within W steps is under-valued by the additive
DP score, so relaxing the ban searches a superset of the W-separated universe
with a bound that stays exact on that universe.

Why it could still change: a self-overlapping column that certifies better than
the W-separated optimum would now be reachable, and it is outside the universe
the committed tests declare.  That is the case this probe is looking for.

Usage:
    uv run python analysis/probe_window_answer_parity.py /tmp/fail_3176.pkl
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import time
from pathlib import Path

import freespace_sim
from freespace_sim.planner.colgen import dp_prepare, pricing


def _load(path: Path):
    # Locally-produced capture of one in-process pricing call, written by this
    # repo's own straggler tooling; never read from an untrusted source.
    with path.open("rb") as handle:
        return pickle.load(handle)


def _run(case, revisit, depth, monkey):
    """Price the captured flight with the history depths overridden."""

    fg, view, cfg, pi_f = case["graph"], case["duals"], case["cfg"], case["pi_f"]
    benefit = case["params"].M

    real_prepare = dp_prepare.prepare_topology

    def patched(graph, config):
        topology = real_prepare(graph, config)
        if revisit is None:
            return topology
        return dataclasses.replace(
            topology, revisit_depth=revisit, state_history_depth=depth
        )

    monkey(patched)
    # The topology is cached per graph, so clear it between arms or the first
    # arm's depths silently survive into every later one.
    fg._search_cache.topology = None

    seed = pricing.seed_column(fg, cfg)
    seed_rc = benefit - seed.delay_s - view.claim_cost(seed.claims) - pi_f
    incumbent = pricing._shifted_seed_incumbent(
        seed, fg, view, pi_f, cfg, benefit, frozenset(), (seed_rc, seed)
    )

    start = time.perf_counter()
    best, proved = pricing._best_column_compiled(
        fg, view, pi_f, cfg, benefit, frozenset(),
        seed=False, incumbent=incumbent, deadline=None,
    )
    wall = time.perf_counter() - start
    stats = dict(pricing._LAST_KERNEL_STATS)
    return best, proved, wall, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="?", default=Path("/tmp/fail_3176.pkl"))
    args = parser.parse_args()

    print(f"freespace_sim: {freespace_sim.__file__}")
    case = _load(args.case)

    original = dp_prepare.prepare_topology

    def monkey(fn):
        # ``_topology_for`` does ``from . import dp_prepare`` inside the function and
        # resolves the attribute at call time, so patching the module is enough.
        dp_prepare.prepare_topology = fn

    arms = [
        ("W=4 baseline     ", None, None),
        ("W=2 / no ban, d=2", 0, 2),
        ("no ban, d=1      ", 0, 1),
    ]
    results = []
    try:
        for name, revisit, depth in arms:
            best, proved, wall, stats = _run(case, revisit, depth, monkey)
            rc = None if best is None else best[0]
            path = None if best is None else tuple(map(tuple, best[1].cell_path))
            results.append((name, rc, path, best))
            print(
                f"{name}  rc={rc!r}  proved={proved}  wall={wall:.2f}s  "
                f"labels={stats.get('labels'):,}  hops={None if path is None else len(path)}  "
                f"status={stats.get('status')}"
            )
    finally:
        dp_prepare.prepare_topology = original

    print()
    base_name, base_rc, base_path, base_best = results[0]
    for name, rc, path, best in results[1:]:
        same_rc = base_rc is not None and rc is not None and abs(rc - base_rc) <= 1e-8
        verdict = "SAME COLUMN" if path == base_path else (
            "same rc, different column" if same_rc else "DIFFERENT"
        )
        delta = "n/a" if (rc is None or base_rc is None) else f"{rc - base_rc:+.10g}"
        print(f"{name} vs baseline: {verdict}   d(rc)={delta}")
        if best is not None and base_best is not None:
            print(
                f"    departure_step {base_best[1].departure_step} -> {best[1].departure_step}"
                f"   delay_s {base_best[1].delay_s:.6f} -> {best[1].delay_s:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
