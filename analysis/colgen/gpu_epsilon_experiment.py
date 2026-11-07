"""Diagnostic ablations of GPU pruning and completed-route validation.

No mode in this script disables the solver's final independent feasibility
checks. Skipping completion validation still constructs and scores proposed
routes on the CPU; it does not implement GPU geometry or remove CPU preparation.
Prepared initial incumbents retain the normal canonical seed checks.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np

from analysis.colgen import execution_replay, gpu_validation
from freespace_sim.planner.colgen import pricing, pricing_pool
from freespace_sim.planner.colgen.gpu_pricing import _Batch
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing_platform import PricingOptions, PricingPlatform
from freespace_sim.planner.colgen.translate import Column


def unchecked_complete(
    batch: _Batch, record: np.ndarray, path: tuple[tuple[int, int], ...],
) -> None:
    """Accept a preliminary route without its canonical acceptance/claim checks.

    Parameters
    ------------
    - batch (_Batch): explicitly uncertified epsilon search.
    - record (np.ndarray): GPU-generated completion state.
    - path (tuple[tuple[int, int], ...]): reconstructed discrete predecessor path.

    Return
    --------
    - output (None): updates an unchecked incumbent; the caller must audit afterward.
    """
    if batch.options.certify_search:
        raise ValueError("Unchecked completions must never enter a certified search")
    root = int(record[1])
    flight, variant = batch.root_flights[root], batch.root_variants[root]
    problem, variants = batch.problems[flight], batch.variants[flight]
    lane = int(variants.lane_idx[variant])
    label = pricing._Label(0.0, int(variants.departure_step[variant]),
                           None if lane < 0 else lane, path, frozenset())
    model, view = batch.models[flight], batch.views[flight]
    for destination in batch.destinations[flight].get(path[-1], ()):
        batch.check()
        claims = pricing._path_claims(problem.graph, problem.cfg, label, destination)
        cost = pricing._path_delay_s(problem.graph, problem.cfg, label, model)
        rc = model.reduced_cost(benefit=batch.benefits[flight], cost=cost,
                               dual_cost=view.claim_cost(claims), pi_f=problem.pi_f)
        batch.stats["unchecked_completions"] = batch.stats.get("unchecked_completions", 0) + 1
        if rc > batch.best[flight][0]:
            column = Column(flight_id=problem.graph.request.flight_id,
                            departure_step=label.departure_step, level=0,
                            origin_lane_idx=label.origin_lane_idx, dest_lane_idx=destination,
                            cell_path=path, delay_s=cost, claims=claims)
            batch.best[flight] = rc, column


@contextmanager
def experiment(
    epsilon: float, skip_completion_validation: bool, verify_border: bool = False,
) -> Iterator[None]:
    """Apply explicit diagnostic settings without changing production solver options.

    Parameters
    ------------
    - epsilon (float): experimental absolute pruning margin.
    - skip_completion_validation (bool): also skip canonical completion checks.
    - verify_border (bool): retain original CPU checks for near-cutoff proposals.

    Return
    --------
    - output (Iterator[None]): scoped ablation; original full CPU pricing stays forbidden.
    """
    initialize = PricingPlatform.__init__
    solver_options = pricing_pool._gpu_options

    def selected(options: PricingOptions) -> PricingOptions:
        """Keep solver factory/reuse validation and direct replay on the identical mode."""
        return replace(options, gpu_prune_epsilon=epsilon, gpu_prune_audit=False,
                       gpu_prune_verify_border=verify_border)

    def selected_solver_options(params: ColGenParams) -> PricingOptions:
        """Preserve strict solver option matching with the diagnostic settings included."""
        return selected(solver_options(params))

    def initialize_experiment(self: PricingPlatform, options: PricingOptions = PricingOptions()) -> None:
        """Keep the same search, with explicit uncertified epsilon pruning."""
        if options.device == "gpu":
            options = selected(options)
        initialize(self, options)

    with ExitStack() as stack:
        stack.enter_context(patch.object(PricingPlatform, "__init__", initialize_experiment))
        stack.enter_context(patch.object(pricing_pool, "_gpu_options", selected_solver_options))
        if skip_completion_validation:
            stack.enter_context(patch.object(_Batch, "complete", unchecked_complete))
        yield


def main(argv: list[str] | None = None) -> int:
    """Run an existing checked harness with a clearly identified experimental ablation.

    Parameters
    ------------
    - argv (list[str] | None): experiment options, followed by the selected harness options.

    Return
    --------
    - output (int): original harness result; uncertified solver runs fail its certificate check.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "solve"))
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--skip-completion-validation", action="store_true")
    parser.add_argument("--verify-border", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    PricingOptions(device="gpu", gpu_backend="metal", gpu_prune_epsilon=args.epsilon)
    with experiment(args.epsilon, args.skip_completion_validation, args.verify_border):
        if args.mode == "replay":
            extra = ["--verify-border"] if args.verify_border else []
            status = execution_replay.main([*remaining, "--epsilon", str(args.epsilon), *extra])
        else:
            status = gpu_validation.main(["solve", *remaining])
    # Preserve the validation harness's failures instead of treating an
    # uncertified final LP gap as a successful certificate.
    if "--out" in remaining:
        output = Path(remaining[remaining.index("--out") + 1])
        if args.mode == "replay":
            report = json.loads(output.read_text())
            report["skip_completion_validation"] = args.skip_completion_validation
            output.write_text(json.dumps(report, indent=2))
        else:
            (output / "experiment.txt").write_text(
                f"epsilon={args.epsilon}\n"
                f"verify_border={args.verify_border}\n"
                f"skip_completion_validation={args.skip_completion_validation}\n"
                "Initial seed checks and final solver feasibility checks retained.\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
