"""Shared colgen test helpers.

What lives here is here rather than copied into each colgen test module because it encodes
a correctness invariant that must not drift between copies.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.network import FlightGraph, _graph_max_step


@pytest.fixture
def check_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-check every canonicalization shortcut against the untouched reference path.

    Pricing reuses pre-check work when it certifies a route; with this on, each such answer is
    recomputed the slow way and must match bit for bit (``pricing._CHECK_CANONICAL``).  The
    environment variable reaches pool workers, which read it at import.  Opt in per module with
    ``pytestmark = pytest.mark.usefixtures("check_canonical")`` after importing this fixture.
    """
    monkeypatch.setattr(pricing, "_CHECK_CANONICAL", True)
    monkeypatch.setenv("COLGEN_CHECK_CANONICAL", "1")


def with_air_hops(fg: FlightGraph, max_air_hops: int) -> FlightGraph:
    """Return the same corridor under a different hop budget.

    ``ColGenParams`` deliberately exposes one knob: ``max_air_overrun_hops`` sizes the O-D
    ellipse AND caps route length, because the budget implies the ellipse.  A handful of
    tests need those two apart -- to lift the ceiling off a fixed corridor so some other
    contract is what decides the answer, or to hold a corridor wider than the budget so a
    containment claim is about the SEARCH rather than about the graph.  This is how they say
    so: at the ``FlightGraph`` level, where the two are separate fields, rather than through
    a params pairing that no longer exists.

    ``max_step`` MUST be recomputed alongside.  It is denominated in ``max_air_hops`` (see
    :func:`_graph_max_step`), so lifting the ceiling without it leaves the clock short of the
    new budget and the horizon, not the ceiling, binds at the latest departures -- the exact
    departure-dependent cap the ceiling exists to remove, reintroduced inside a test that
    would then quietly measure the wrong thing.

    ``dataclasses.replace`` shares ``corridor_cells`` and ``forbidden_hops`` by reference, so
    the corridor is provably identical between two graphs built this way, and re-creates the
    ``init=False`` search cache cold.
    """

    if max_air_hops < fg.shortest_hops:
        # Below the geodesic nothing can complete, `price_flight` returns None, and a caller
        # asserting on the column gets a confusing failure several frames away from the typo.
        raise ValueError(
            f"max_air_hops={max_air_hops} is below the graph's shortest path "
            f"({fg.shortest_hops}); no route could complete"
        )
    return replace(
        fg,
        max_air_hops=max_air_hops,
        max_step=_graph_max_step(
            fg.latest_departure_step,
            fg.takeoff_steps,
            fg.origin_lanes,
            max_air_hops,
        ),
    )
