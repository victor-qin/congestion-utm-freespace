"""CPU-only flight preparation and explicit data transfer to the single GPU owner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
import time

import numpy as np

from . import pricing
from .dp_prepare import PreparedTopology, PreparedVariants, prepare_variants, prepared_for
from .gpu_bounds import CompletionBoundData, CompletionBounds, completion_upper_bound
from .gpu_endpoint import EndpointBounds, _Membership
from .gpu_prepare import GPUPreparation, packing_for
from .gpu_prices import summary_for
from .objective import cost_model
from .pricing_platform import PricingOptions, PricingProblem
from .translate import Column

BOUND_CLOSURE_TOL = 1e-6


class PreparationStopped(Exception):
    """Preparation stopped without resolving the flight's search domain."""


def check_deadline(deadline: float | None) -> None:
    """Reject work beyond the shared monotonic deadline before publishing it."""
    if deadline is not None and time.monotonic() >= deadline:
        raise PreparationStopped("deadline")


@dataclass(slots=True)
class PreparedFlight:
    """CPU preparation bound to exactly one invocation's original problem and prices.

    Attributes
    ------------
    - topology, graph (PreparedTopology | None, np.ndarray | None): fixed complete domain.
    - variants, bounds, endpoint: current roots and original FP64 verification data.
    - best (tuple): canonically validated incumbent and reduced cost.
    - status (str): incomplete preparation never authorizes search certification.
    - stats (dict): preparation work and elapsed time, excluding transport.
    """

    topology: PreparedTopology | None = None
    graph: np.ndarray | None = None
    variants: PreparedVariants | None = None
    bounds: CompletionBounds | None = None
    endpoint: EndpointBounds | None = None
    best: tuple[float, Column | None] = (-math.inf, None)
    negative: float = 0.0
    status: str = "complete"
    stats: dict[str, float | int] = field(default_factory=dict)
    problem: PricingProblem | None = None
    options: PricingOptions | None = None

    def check(self, problem: PricingProblem, view: pricing.DualView, options: PricingOptions) -> None:
        """Reject reuse under a different graph, snapshot, objective, restriction or preparation mode.

        Parameters
        ------------
        - problem (PricingProblem): current GPU invocation inputs.
        - view (pricing.DualView): current normalized immutable snapshot.
        - options (PricingOptions): endpoint/domain preparation controls.

        Return
        --------
        - output (None): raises ValueError rather than consuming stale weighted data.
        """
        original = self.problem
        if (original is None or original.graph is not problem.graph or original.cfg != problem.cfg
                or original.params != problem.params or original.duals is not view
                or original.pi_f != problem.pi_f or original.known_column is not problem.known_column
                or original.forbidden_rows != problem.forbidden_rows
                or original.require_improving != problem.require_improving or self.options != options):
            raise ValueError("Prepared flight does not match the current problem/snapshot/options")


def prepare_flight(
    p: PricingProblem, view: pricing.DualView, options: PricingOptions, deadline: float | None,
) -> PreparedFlight:
    """Prepare the unchanged legal domain without creating or calling a GPU runtime.

    Parameters
    ------------
    - p (PricingProblem): immutable graph, objective, current restrictions and assignment dual.
    - view (pricing.DualView): original immutable prices for this invocation.
    - options (PricingOptions): existing endpoint preparation budgets.
    - deadline (float | None): covers seed validation and all CPU preparation.

    Return
    --------
    - output (PreparedFlight): complete roots/bounds or an explicit incomplete status.
    """
    result = PreparedFlight(problem=replace(p, duals=view), options=options)
    started, cpu_started = time.monotonic(), time.process_time()
    model, benefit = cost_model(p.cfg, p.params), pricing._benefit(p.params)
    screened = 0

    def check() -> None:
        """Check this invocation's deadline throughout lazy preparation."""
        check_deadline(deadline)

    def keep_departure(departure: int) -> bool:
        """Retain every departure not closed by the original FP64 ground-only bound."""
        nonlocal screened
        check()
        ground = max(0.0, math.nextafter((departure - p.graph.base_step) * p.cfg.dt_s, -math.inf))
        cost = max(0.0, math.nextafter(ground * model.ground_weight, -math.inf))
        upper = completion_upper_bound(benefit, p.pi_f, cost, (), result.negative)
        keep = not math.isfinite(upper) or upper > result.best[0]
        screened += int(not keep)
        return keep

    try:
        check()
        if p.cfg != p.graph._cfg or not math.isfinite(p.pi_f):
            raise ValueError("Pricing needs the graph's configuration and a finite flight dual")
        if not all(math.isfinite(w) and w >= 0 for w in (model.ground_weight, model.air_weight)):
            raise ValueError("Certified bounds require finite nonnegative objective weights")
        result.negative = summary_for(view, check).negative
        topology, rows = prepared_for(p.graph, p.cfg)
        if not topology.ok or not rows.ok:
            raise PreparationStopped("unsupported")
        remaining_hops = np.minimum(topology.rev_remaining, topology.air_hop_limit + 1)
        remaining = int(remaining_hops.max(initial=0))
        if (min(topology.min_step, topology.base_step) + min(0, view.offsets[0]) < -(2**31)
                or max(topology.max_step, topology.air_hop_limit) + remaining
                + max(1, view.offsets[1]) >= 2**31):
            raise ValueError("GPU time/hop arithmetic exceeds int32 range")
        check()
        try:
            seed = pricing._shortest_seed(p.graph, p.cfg, deadline=deadline, model=model)
        except ValueError:
            seed = None
        for column in (seed, p.known_column):
            if column is None:
                continue
            column = pricing.certify_column(column, p.graph, p.cfg, model)
            if column.claims.isdisjoint(p.forbidden_rows):
                rc = model.reduced_cost(benefit=benefit, cost=column.delay_s,
                                        dual_cost=view.claim_cost(column.claims), pi_f=p.pi_f)
                if rc > result.best[0]:
                    result.best = rc, column
        check()
        preparation = GPUPreparation(
            p.graph, p.cfg, view, benefit=benefit, pi_f=p.pi_f, model=model,
            forbidden_rows=p.forbidden_rows, retain_root_claims=True,
        )
        packing = packing_for(p.graph, p.cfg, topology)
        hits, builds = packing.claim_hits, packing.claim_builds
        variants = prepare_variants(
            p.graph, p.cfg, view, topology, rows, benefit=benefit, pi_f=p.pi_f,
            model=model, forbidden_rows=p.forbidden_rows, preparation=preparation,
            screen_departure=keep_departure,
        )
        if not variants.ok:
            raise PreparationStopped("unsupported")
        bounds = CompletionBounds(p, topology, variants, view, preparation=preparation)
        if (options.endpoint_bounds and options.endpoint_max_rows and any(
            completion_upper_bound(
                benefit, p.pi_f,
                bounds.cost_lower(root, int(remaining_hops[variants.cell[root]])),
                (bounds.root_positive(root),), result.negative,
            ) > result.best[0] + BOUND_CLOSURE_TOL for root in range(variants.n_variants)
        )):
            result.endpoint = EndpointBounds(bounds, options, check)
        result.topology, result.graph = topology, packing.graph
        result.variants, result.bounds = variants, bounds
        result.stats.update(claim_hits=packing.claim_hits - hits,
                            claim_builds=packing.claim_builds - builds)
        check()
    except PreparationStopped as stop:
        result.status = str(stop)
    except pricing.PricingTimeout:
        result.status = "deadline"
    result.stats.update(screened_departures=screened, cpu_prepare_s=time.monotonic() - started,
                        worker_cpu_s=time.process_time() - cpu_started)
    return result


@dataclass(slots=True)
class PreparedFlightData:
    """Transfer only arrays, exact memberships and a candidate; never graph or global prices.

    Attributes
    ------------
    - fixed (tuple | None): topology/local adjacency, omitted after owner acknowledgement.
    - variants, bounds: original root arrays and FP64 verification payload.
    - endpoint (dict | None): selected exact membership and snapshot-local mask/tree data.
    - column (Column | None): incumbent revalidated by the receiving owner.
    - negative, status, stats: original credit, completion and preparation measurements.
    """

    fixed: tuple[PreparedTopology, np.ndarray] | None
    variants: PreparedVariants | None
    bounds: CompletionBoundData | None
    endpoint: dict | None
    column: Column | None
    negative: float
    status: str
    stats: dict[str, float | int]


def export_flight(prepared: PreparedFlight, *, include_topology: bool) -> PreparedFlightData:
    """Detach original FP64 data from worker-owned graphs, prices and deadline callbacks.

    Parameters
    ------------
    - prepared (PreparedFlight): CPU-only preparation for the matching task.
    - include_topology (bool): include fixed arrays when absent from the owner cache.

    Return
    --------
    - output (PreparedFlightData): compact pickle-safe payload with no GPU ownership.
    """
    endpoint = None
    if prepared.endpoint is not None:
        e = prepared.endpoint
        endpoint = {name: getattr(e, name) for name in (
            "first", "last", "size", "leaves", "entries", "stats", "rows", "bits",
            "weights", "root_masks", "visit_masks",
        )}
        # Later mask discovery only uses selected rows. Unpriced memberships stay
        # in the worker's graph cache and are reconsidered on every new snapshot.
        endpoint["membership"] = _Membership(
            e.membership.cfg, e.first, e.last, e.membership.destinations,
            e.membership.visit_offsets, {row: e.membership.arrivals[row] for row in e.rows},
        )
    return PreparedFlightData(
        (prepared.topology, prepared.graph) if include_topology and prepared.topology else None,
        prepared.variants, prepared.bounds.export() if prepared.bounds is not None else None,
        endpoint, prepared.best[1], prepared.negative, prepared.status, prepared.stats,
    )


def restore_flight(
    data: PreparedFlightData, p: PricingProblem, view: pricing.DualView,
    fixed: tuple[PreparedTopology, np.ndarray] | None, options: PricingOptions,
    check: Callable[[], None],
) -> PreparedFlight:
    """Reattach a version-matched payload and revalidate its incumbent canonically.

    Parameters
    ------------
    - data (PreparedFlightData): worker result whose complete task tag has been checked.
    - p (PricingProblem): owner graph, current objective and restrictions.
    - view (pricing.DualView): owner snapshot with unchanged FP64 prices.
    - fixed (tuple | None): cached topology/adjacency for exactly this graph version.
    - options (PricingOptions): this call's endpoint budgets.
    - check (Callable): owner deadline callback used by subsequent lazy mask discovery.

    Return
    --------
    - output (PreparedFlight): owner verification objects, without repeating preparation.
    """
    check()
    result = PreparedFlight(variants=data.variants, negative=data.negative,
                            status=data.status, stats=data.stats,
                            problem=replace(p, duals=view), options=options)
    if data.column is not None:
        model = cost_model(p.cfg, p.params)
        column = pricing.certify_column(data.column, p.graph, p.cfg, model)
        if not column.claims.isdisjoint(p.forbidden_rows):
            raise ValueError("GPU preparation returned a forbidden incumbent")
        result.best = (model.reduced_cost(
            benefit=pricing._benefit(p.params), cost=column.delay_s,
            dual_cost=view.claim_cost(column.claims), pi_f=p.pi_f,
        ), column)
    if data.status == "complete":
        if fixed is None or data.variants is None or data.bounds is None:
            raise ValueError("Incomplete GPU preparation payload")
        result.topology, result.graph = fixed
        result.bounds = CompletionBounds.from_data(data.bounds, p, result.topology,
                                                   data.variants, view)
        if data.endpoint is not None:
            e = EndpointBounds.__new__(EndpointBounds)
            e.__dict__.update(data.endpoint)
            e.bounds, e.options, e.check = result.bounds, options, check
            result.endpoint = e
    check()
    return result
