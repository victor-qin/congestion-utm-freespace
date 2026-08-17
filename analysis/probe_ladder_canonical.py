"""Does `_canonical_column` ever change a departure-ladder column? (issue #92, item 4)

The ladder builds each rung by pure clock translation -- `_shift_column` adds `delta` to
every claim's `step` and nothing else -- and then hands the result to `_canonical_column`,
which re-derives the claim set from geometry and keeps its own answer if the two disagree.
If the lattice is time-homogeneous the two agree always, the re-derivation is pure
verification, and skipping it on the ladder path is answer-neutral.

That "if" is the whole question. Issue #92 lists the skip as its only item that is *not*
answer-neutral by construction, so this probe is the precondition, not the follow-up: it
counts ladder rungs where the re-derivation actually moved the claims.

A DISAGREEMENT here is the useful outcome. It means the shift is not a symmetry of the
claim rule -- most likely at a horizon or wall boundary where a later departure is graded
differently -- and it names the flight and step to look at.

Reports the other `_canonical_column` call sites separately: they price real routes rather
than translations, so a disagreement there is expected and says nothing about the ladder.

    uv run python analysis/probe_ladder_canonical.py --scenario colgen_test --flights 12
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

COUNTS: collections.Counter = collections.Counter()
DISAGREEMENTS: list[tuple[int, int, int, int]] = []

# Set while the stack is inside `_add_departure_ladder`, which is the only call site whose
# input is a pure translation.
_IN_LADDER = False


def install() -> None:
    canonical = solver_mod._canonical_column
    ladder = solver_mod._add_departure_ladder

    def probed_canonical(column, graph, cfg):
        result = canonical(column, graph, cfg)
        site = "ladder" if _IN_LADDER else "other"
        changed = result.claims != column.claims
        COUNTS[f"{site}.calls"] += 1
        if changed:
            COUNTS[f"{site}.changed"] += 1
            if _IN_LADDER:
                DISAGREEMENTS.append((
                    column.flight_id,
                    column.departure_step,
                    len(column.claims),
                    len(result.claims),
                ))
        return result

    def probed_ladder(*args, **kwargs):
        global _IN_LADDER
        previous, _IN_LADDER = _IN_LADDER, True
        try:
            return ladder(*args, **kwargs)
        finally:
            _IN_LADDER = previous

    solver_mod._canonical_column = probed_canonical
    solver_mod._add_departure_ladder = probed_ladder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="colgen_test")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--ladder-steps", type=int, default=None, metavar="K")
    args = parser.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    if len(cfg.flight_levels_m) != 1:
        raise SystemExit(f"{args.scenario} has {len(cfg.flight_levels_m)} flight levels")
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]

    overrides = {"max_iterations": args.iterations, "time_limit_s": 1e9}
    if args.ladder_steps is not None:
        overrides["seed_ladder_steps"] = args.ladder_steps
    params = ColGenParams(**overrides)

    install()
    # As `prof_colgen_stages.py` / `ab_colgen_parity.py` do. Without the scenario's permanent
    # terminal volumes the ladder is built over a different route set, so a clean run here
    # would attest to an invariant production never exercises.
    static_terms = list(demand.terminals(cfg))
    ColGenSolver().solve(requests, cfg, static_terms, params)

    ladder_calls = COUNTS["ladder.calls"]
    ladder_changed = COUNTS["ladder.changed"]
    print(f"\n--- {args.scenario} x{args.flights}, {args.iterations} iterations ---")
    print(f"ladder  calls {ladder_calls:6d}   claims CHANGED {ladder_changed:6d}")
    print(f"other   calls {COUNTS['other.calls']:6d}   claims changed {COUNTS['other.changed']:6d}"
          "   (expected nonzero: real routes, not translations)")
    if ladder_changed:
        print("\nLADDER DISAGREEMENTS (flight, departure_step, n_claims_in, n_claims_out):")
        for row in DISAGREEMENTS[:20]:
            print(f"  {row}")
        print("\nVERDICT: the shift is NOT a symmetry of the claim rule here. Issue #92's"
              " item 4 (skip the ladder re-derivation) is NOT answer-neutral as written.")
        return 1
    print("\nVERDICT: no ladder rung changed. Consistent with the invariant on this"
          " scenario -- necessary, not sufficient; widen the suite before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
