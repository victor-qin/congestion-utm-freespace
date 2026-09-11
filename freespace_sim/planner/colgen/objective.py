"""The colgen objective, in one place.

Every reduced cost, every admissible bound, and every master coefficient in column
generation is built from the same two quantities: seconds spent waiting on the ground and
seconds spent in the air beyond the reference trajectory (the en-route detour ruler; see
context/figures/enroute_rulers.png).  Because a pricing bound IS the objective as far as
dominance is concerned, a weight that goes stale at even one call site fails silently: it
discards the true optimum while the search still reports it proved optimality.

:class:`CostModel` is therefore the single source of truth, and every site calls
:meth:`evaluate` rather than restating the sum.  Any future accelerated pricing path
inherits the same rule -- it may hold objective-agnostic arithmetic over pre-weighted
inputs, but never a second copy of the weights.

Term order is a byte-exact contract, not a preference.  ``evaluate`` sums ground, then
hold, then detour, left to right, matching the association the longhand expressions it
replaced used; at unit weights the multiplications are exact and the sum is bit-identical
to the prior code, which is what makes the (1.0, 1.0) default verifiable rather than merely
plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CostModel", "DELAY_MODEL", "cost_model"]


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

    def intent_cost(self, intent: Any, cfg: Any) -> float:
        """Price translated trajectory metrics in this solve's objective units."""

        return self.evaluate(
            ground_s=intent.ground_delay_s,
            air_hold_s=intent.air_hold_s,
            air_detour_s=intent.air_detour_m / cfg.nominal_speed_mps,
        )

    def reduced_cost(
        self, *, benefit: float, cost: float, dual_cost: float, pi_f: float
    ) -> float:
        """Pricing's objective: what one column is worth against the current duals."""

        return benefit - cost - dual_cost - pi_f


#: The equal-weight objective: one second of ground delay costs the same as one second of
#: excess flight.  This is NOT the cost model the A* planner uses (config's 1:3:3:4
#: per-second weights), nor is it the colgen default -- ``ColGenParams.objective`` ships
#: ``"total_cost"``.  It stays the module default so every function taking
#: ``model: CostModel = DELAY_MODEL`` keeps its historical behaviour when called without
#: one, which is what the kernel-parity tests rely on.
#:
#: Reach for it deliberately, not by omission.  Equal weights make ``ground + flown``
#: invariant under a ground-for-air swap, so large sets of columns tie EXACTLY -- see
#: ``ColGenParams.objective`` for what that costs the pricing search.
DELAY_MODEL = CostModel(ground_weight=1.0, air_weight=1.0)


def cost_model(cfg: Any, params: Any = None) -> CostModel:
    """Resolve the objective for one solve.

    ``params.objective`` selects it; anything other than ``"total_cost"`` keeps the
    unweighted delay objective, so existing callers and every stored scenario behave
    exactly as before.

    Parameters
    ------------
    - cfg (Any): the sim config; read for ``cost_ground_delay_per_s`` /
      ``cost_air_lateral_per_s`` / ``cost_air_hold_per_s`` when ``"total_cost"`` is selected.
    - params (Any): the colgen params; ``params.objective`` picks the model, defaulting to the
      unweighted delay objective when ``params`` is ``None`` or the attribute is absent.

    Return
    --------
    - output (CostModel): :data:`DELAY_MODEL` for the delay objective, else a model carrying
      the config's ground/air weights.  Raises ``ValueError`` if the config's air-hold and
      lateral weights differ, or if either resolved weight is non-positive.
    """

    objective = getattr(params, "objective", "total_delay")
    if objective != "total_cost":
        return DELAY_MODEL

    ground = float(cfg.cost_ground_delay_per_s)
    lateral = float(cfg.cost_air_lateral_per_s)
    hold = float(cfg.cost_air_hold_per_s)
    # One air scalar has to cover both, because the model carries a single air weight
    # and pricing charges it per step of flight.  Config ships them equal (3.0 each); if
    # that ever diverges, pricing would silently value loiter as cruise, so refuse rather
    # than guess -- splitting them means splitting the weight, not reinterpreting one.
    if hold != lateral:
        raise ValueError(
            "objective='total_cost' needs cost_air_hold_per_s == cost_air_lateral_per_s "
            f"(got {hold} and {lateral}); the cost model carries one air weight per step"
        )
    if not (ground > 0.0 and lateral > 0.0):
        raise ValueError("cost weights must be positive")
    return CostModel(ground_weight=ground, air_weight=lateral)

