"""Shared colgen test helpers.

Only one thing lives here, and it is here rather than copied into each colgen test module
because it encodes a correctness invariant that must not drift between copies.
"""

from __future__ import annotations

from dataclasses import replace

from freespace_sim.planner.colgen.network import FlightGraph, _graph_max_step


def with_air_hops(fg: FlightGraph, max_air_hops: int) -> FlightGraph:
    """Return the same corridor under a different hop budget.

    ``ColGenParams`` deliberately exposes one knob: ``max_air_overrun_hops`` sizes the O-D
    ellipse AND caps route length, because the budget implies the ellipse.  A handful of
    tests need those two apart -- to lift the ceiling off a fixed corridor so some other
    contract is what decides the answer, or to hold a corridor wider than the budget so a
    containment claim is about the SEARCH rather than about the graph.  This is how they say
    so: at the ``FlightGraph`` level, where the two are separate fields, rather than through
    a params pairing that no longer exists (issue #78).

    ``max_step`` MUST be recomputed alongside.  It is denominated in ``max_air_hops`` (see
    :func:`_graph_max_step`), so lifting the ceiling without it leaves the clock short of the
    new budget and the horizon, not the ceiling, binds at the latest departures -- the exact
    departure-dependent cap ``9816f61`` was written to remove, reintroduced inside a test that
    would then quietly measure the wrong thing.

    ``dataclasses.replace`` shares ``corridor_cells`` and ``forbidden_hops`` by reference, so
    the corridor is provably identical between two graphs built this way, and re-creates the
    ``init=False`` search cache cold.
    """

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
