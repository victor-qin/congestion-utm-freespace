"""The colgen objective, in one place.

Every reduced cost, every admissible bound, and every master coefficient in column
generation is built from the same two quantities: seconds spent waiting on the ground and
seconds spent in the air beyond the reference trajectory.  Before this module each of
those combinations was written out longhand at nineteen sites across six files, all of
them implicitly weighting the two terms equally.  Changing the objective therefore meant
finding all nineteen, and missing one in the compiled kernel would have been silent --
``label_score`` *is* the objective as far as dominance is concerned, so a stale weight
there discards the true optimum while still reporting ``proved=True``.

:class:`CostModel` is the single source of truth.  Python call sites call
:meth:`CostModel.evaluate`; the compiled kernel cannot (it is ``@njit``), so it is instead
fed *pre-weighted* inputs -- see :func:`scaled_dt_s` and ``dp_prepare.prepare_variants``.
That is the important structural point: the kernel does not hold a second copy of the
objective, it holds objective-agnostic arithmetic over scaled units.  There is
consequently nothing to keep in sync, and ``test_colgen_objective.py`` pins that claim by
asserting the kernel's own accumulated score equals :meth:`evaluate` on the column it
returns.

Term order matters.  ``evaluate`` sums ground, then hold, then detour, left to right,
because that is the association the pre-existing expressions used.  At unit weights the
multiplications are exact and the sum is bit-identical to what the code did before, which
is what makes the (1.0, 1.0) refactor verifiable rather than merely plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CostModel", "DELAY_MODEL", "cost_model", "scaled_dt_s"]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Weights turning (ground seconds, air seconds) into the quantity colgen minimises.

    ``ground_weight`` prices time held on the pad; ``air_weight`` prices time flown beyond
    the reference trajectory, and equally the loiter that ``air_hold_s`` reports.
    """

    ground_weight: float = 1.0
    air_weight: float = 1.0

    def evaluate(
        self,
        *,
        ground_s: float,
        air_hold_s: float = 0.0,
        air_detour_s: float = 0.0,
    ) -> float:
        """The objective value of one trajectory, in the master's currency.

        Summed in the order (ground, hold, detour) so that at unit weights the result is
        bit-identical to the longhand expressions this replaced.
        """

        return (
            self.ground_weight * ground_s
            + self.air_weight * air_hold_s
            + self.air_weight * air_detour_s
        )

    def reduced_cost(
        self, *, benefit: float, cost: float, dual_cost: float, pi_f: float
    ) -> float:
        """Pricing's objective: what one column is worth against the current duals."""

        return benefit - cost - dual_cost - pi_f

    @property
    def separable(self) -> bool:
        """Whether the compiled kernel can be driven by pre-weighted inputs.

        True exactly when the objective is linear and splits into one ground scalar and
        one air scalar, which is what lets ``_search_dag`` charge ``air_weight * dt_s``
        per arc and stay correct without knowing the objective exists.  A future
        non-linear objective -- a penalty above a delay threshold, a per-flight priority
        multiplier, anything quadratic -- must return False here so
        :func:`dp_kernel.search_dag` refuses it loudly instead of pruning against a bound
        that no longer matches what it is optimising.
        """

        return True


#: The historical objective: one second of ground delay costs the same as one second of
#: excess flight.  Note this is *not* the cost model the A* planner uses (config's
#: 1:3:3:4 per-second weights); colgen has always summed unweighted seconds, and keeping
#: that as the default is what makes this refactor a no-op until someone opts in.
DELAY_MODEL = CostModel(ground_weight=1.0, air_weight=1.0)


def cost_model(cfg: Any, params: Any = None) -> CostModel:
    """Resolve the objective for one solve.

    ``params.objective`` selects it; anything other than ``"total_cost"`` keeps the
    unweighted delay objective, so existing callers and every stored scenario behave
    exactly as before.
    """

    objective = getattr(params, "objective", "total_delay")
    if objective != "total_cost":
        return DELAY_MODEL

    ground = float(cfg.cost_ground_delay_per_s)
    lateral = float(cfg.cost_air_lateral_per_s)
    hold = float(cfg.cost_air_hold_per_s)
    # One air scalar has to cover both, because the kernel charges a single weight per
    # step.  Config ships them equal (3.0 each); if that ever diverges the kernel would
    # silently price loiter as cruise, so refuse rather than guess.
    if hold != lateral:
        raise ValueError(
            "objective='total_cost' needs cost_air_hold_per_s == cost_air_lateral_per_s "
            f"(got {hold} and {lateral}); the compiled DP charges one air weight per step"
        )
    if not (ground > 0.0 and lateral > 0.0):
        raise ValueError("cost weights must be positive")
    return CostModel(ground_weight=ground, air_weight=lateral)


def scaled_dt_s(model: CostModel, dt_s: float) -> float:
    """The per-hop air charge the compiled kernel should accumulate.

    ``_search_dag`` adds ``dt_s`` per arc and ``_delay_lower_bound`` multiplies hop counts
    by it.  Handing it ``air_weight * dt_s`` -- together with the fold, leg and reference
    times that ``prepare_variants`` pre-weights the same way -- makes that untouched
    arithmetic compute ``ground_weight * ground + air_weight * air`` exactly.
    """

    return model.air_weight * dt_s
