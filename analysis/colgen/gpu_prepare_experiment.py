"""Disable only PR144's fixed packing/start/membership reuse for a checked comparison."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.colgen import gpu_validation
from freespace_sim.config import SimConfig
from freespace_sim.planner.colgen import gpu_pricing
from freespace_sim.planner.colgen.dp_prepare import PreparedTopology, PricingPreparation
from freespace_sim.planner.colgen.network import FlightGraph


def uncached_packing(
    graph: FlightGraph, cfg: SimConfig, topology: PreparedTopology,
) -> SimpleNamespace:
    """Rebuild the local GPU layout without retaining PR144's fixed graph packing.

    Parameters
    ------------
    - graph (FlightGraph): current flight, whose existing CPU caches remain enabled.
    - cfg (SimConfig): current configuration.
    - topology (PreparedTopology): existing CPU topology shared by both comparison modes.

    Return
    --------
    - output (SimpleNamespace): fresh adjacency and physical cell order for this batch.
    """
    packed = np.full((topology.n_cells, 14), -1, np.int32)
    packed[:, 6:12] = 0
    packed[:, 12] = np.minimum(topology.rev_remaining, topology.air_hop_limit + 1)
    packed[:, 13] = topology.dest_mask
    for cell in range(topology.n_cells):
        start, stop = map(int, topology.arc_start[cell:cell + 2])
        packed[cell, :stop - start] = topology.arc_target[start:stop]
        packed[cell, 6:6 + stop - start] = topology.arc_roles[start:stop]
    return SimpleNamespace(graph=packed, cells=tuple(zip(map(int, topology.cell_q),
                                                        map(int, topology.cell_r))))


@contextmanager
def without_preparation_cache() -> Iterator[None]:
    """Use original call-local preparation, preserving snapshot prices and all checks.

    Parameters
    ------------
    - None.

    Return
    --------
    - output (Iterator[None]): temporary analysis-only ablation, restored on exceptions too.
    """
    with patch.object(gpu_pricing, "GPUPreparation", PricingPreparation), patch.object(
        gpu_pricing, "packing_for", uncached_packing,
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """Run the unchanged validation harness with only fixed preparation reuse disabled.

    Parameters
    ------------
    - argv (list[str] | None): gpu_validation arguments, including solve/prepare/replay mode.

    Return
    --------
    - output (int): unchanged strict validation status, including incomplete-search failures.
    """
    with without_preparation_cache():
        return gpu_validation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
