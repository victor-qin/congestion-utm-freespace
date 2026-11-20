"""Exact strictly-below-90-degree cruise turn rule shared by the DP and Zeta searches.

Cruise waypoints are pointy-top axial hex centres (``hexgrid.hex_center``), so the
Euclidean dot product of two waypoint offsets is ``pitch**2 / 2`` times the integer
returned by :func:`axial_dot`. The rule therefore needs no trigonometry or tolerance:
exactly perpendicular links have dot zero and are rejected, reversals are negative and
straight continuations are positive. Only consecutive horizontal cruise links are
checked; the first cruise link, vertical climb/descent and endpoint lanes or stubs are not.
"""

from __future__ import annotations

from collections.abc import Sequence

Cell = tuple[int, int]


def axial_dot(first: Cell, second: Cell) -> int:
    """Return twice the Euclidean dot product of two axial offsets, in pitch-squared units.

    Parameters
    ------------
    - first (Cell): incoming waypoint offset ``(dq, dr)``.
    - second (Cell): outgoing waypoint offset ``(dq, dr)``.

    Return
    --------
    - output (int): ``2*dq1*dq2 + dq1*dr2 + dr1*dq2 + 2*dr1*dr2``; its sign is the sign
      of the physical dot product for ``hexgrid.hex_center`` geometry.
    """
    return (2 * first[0] * second[0] + first[0] * second[1]
            + first[1] * second[0] + 2 * first[1] * second[1])


def turn_allowed(incoming: Cell, outgoing: Cell) -> bool:
    """Accept a continuation only when its deflection is strictly below 90 degrees."""
    return axial_dot(incoming, outgoing) > 0


def turn_allowed_at(previous: Cell, current: Cell, following: Cell) -> bool:
    """Apply :func:`turn_allowed` to the two links through one interior waypoint."""
    return turn_allowed((current[0] - previous[0], current[1] - previous[1]),
                        (following[0] - current[0], following[1] - current[1]))


def path_turns_valid(path: Sequence[Cell]) -> bool:
    """Check every interior cruise waypoint of an ordered waypoint path.

    Parameters
    ------------
    - path (Sequence[Cell]): cruise waypoints from the first to the last hex centre.

    Return
    --------
    - output (bool): whether every consecutive link pair has a positive dot product.
      Paths with fewer than three waypoints have no interior turn.
    """
    return all(turn_allowed_at(previous, current, following)
               for previous, current, following in zip(path, path[1:], path[2:]))
