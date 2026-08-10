"""Fan one iteration's pricing sweep across worker processes.

The sweep's subproblems are independent given the iteration's duals -- each one searches a
different flight's DAG and touches nothing shared -- so the only thing parallelism can
change is *when* results arrive. This module exists to make sure that is all it changes.

**Determinism, two rules, both load-bearing.**

1. *Longest completed prefix.* The sequential sweep ``break``s at the first
   :class:`PricingTimeout`, so flights after it are never priced. A pool has no such
   ordering: a worker can finish flight 40 while flight 5 times out. Results are collected
   by ``pricing_order`` index and everything at or past the first timeout is DISCARDED, so
   the accepted set is exactly the prefix the sequential loop would have produced.
2. *Index order, not completion order.* ``master.upper_bound`` sums the reduced costs with
   plain ``sum`` (master.py), not ``math.fsum``, and float addition is not associative --
   so appending them as workers finish perturbs the global bound at ulp level and the
   solve can terminate an iteration earlier or later. They are appended by index.

``n_workers=0`` runs the sequential loop in-process and is byte-identical to no pool at
all; it is the default and the parity baseline.

**Processes, not threads.** Pricing is ~90% Python outside the numba kernel, so threads
would contend on the GIL for the part that is not compiled. The costs processes bring are
measured rather than assumed: a ``FlightGraph`` pickles to 38.6 KB but REBUILDS in 0.11 ms
against 0.40 ms to pickle, so workers are handed the flight *requests* and build their own
graphs. The caches (`_search_cache`) do not survive pickling either way, so nothing is lost
by rebuilding that would have been kept by shipping.

**``mp.Pool``, never ``ProcessPoolExecutor``.** The latter deadlocks on CPython 3.14.2 when
``max_tasks_per_child`` fires -- see ``[[colgen-parallel-pricing-pool]]``. For the same
reason no worker-recycling budget is set here.

**macOS spawn hazard, deliberately accepted.** Under the ``spawn`` start method a caller
script that does work at module level rather than under ``if __name__ == "__main__":`` dies
in ``_check_not_importing_main``. Every shipped entry point already guards its main.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

from ...config import SimConfig
from ...types import FlightRequest
from .network import StaticTerminalCatalog, build_flight_graph
from .params import ColGenParams
from .pricing import DualView, PricingTimeout, price_flight
from .translate import Column

__all__ = [
    "ParallelPricingConfig",
    "SweepResult",
    "price_sweep",
]


@dataclass(frozen=True, slots=True)
class ParallelPricingConfig:
    """How wide to fan the sweep. ``n_workers=0`` is the sequential loop."""

    n_workers: int = 0
    chunksize: int = 1

    def __post_init__(self) -> None:
        for name in ("n_workers", "chunksize"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.n_workers < 0:
            raise ValueError("n_workers must be non-negative")
        if self.chunksize < 1:
            raise ValueError("chunksize must be positive")


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One sweep's accepted prefix, in ``pricing_order`` index order.

    ``reduced_costs`` and ``columns`` are the same length and cover only the accepted
    prefix; ``timeout_flight_id`` names the flight that ended it, or is ``None`` when the
    sweep completed.
    """

    flight_ids: tuple[int, ...]
    reduced_costs: tuple[float, ...]
    columns: tuple[Column | None, ...]
    timeout_flight_id: int | None

    @property
    def complete(self) -> bool:
        return self.timeout_flight_id is None


# --------------------------------------------------------------------------- worker side

# Populated once per worker per sweep by `_init_worker`. A plain module global because that
# is the only state a spawned worker can carry between tasks without re-pickling it.
_WORKER: dict[str, Any] = {}


def _init_worker(
    requests: list[FlightRequest],
    cfg: SimConfig,
    params: ColGenParams,
    catalog: StaticTerminalCatalog,
    duals: dict,
    flight_duals: dict[int, float],
    known_columns: dict[int, Column],
    deadline: float | None,
) -> None:
    """Rebuild this worker's view of the batch. Runs once per worker per sweep.

    The catalog is shipped rather than rebuilt from raw terminals: every graph in a solve
    must see the identical wall catalogue, and re-deriving it here would be a second source
    of truth for something the parent already snapshotted.  It pickles to 12.2 KB.
    """

    _WORKER.clear()
    _WORKER["graphs"] = {
        request.flight_id: build_flight_graph(request, cfg, catalog, params)
        for request in requests
    }
    _WORKER["cfg"] = cfg
    _WORKER["params"] = params
    # `DualView` is rebuilt here rather than pickled so the worker owns its own caches.
    _WORKER["dual_view"] = DualView(duals, cfg)
    _WORKER["flight_duals"] = flight_duals
    _WORKER["known_columns"] = known_columns
    # `time.monotonic` is a system-wide clock on both Linux and macOS, so a deadline taken
    # in the parent is directly comparable here. It would NOT be across hosts.
    _WORKER["deadline"] = deadline


def _price_one(flight_id: int):
    """Price one flight in a worker, returning ``(flight_id, priced, rc, column)``.

    ``priced`` is an explicit BOOLEAN and not an identity sentinel, which is a correctness
    requirement rather than a style choice: a module-level ``object()`` pickles happily and
    arrives in the parent as a DIFFERENT instance, so ``result is _SENTINEL`` is always
    False across a process boundary.  The sequential path would never have caught it --
    nothing is pickled there -- so the bug would surface only on the first production
    timeout under a pool.

    A timeout is reported rather than raised for a related reason: propagating it would
    make `mp.Pool` pickle and re-raise the exception in the parent, and a timeout here is
    an ordinary outcome, not a failure.
    """

    try:
        reduced_cost, column = price_flight(
            _WORKER["graphs"][flight_id],
            _WORKER["dual_view"],
            _WORKER["flight_duals"][flight_id],
            _WORKER["cfg"],
            _WORKER["params"],
            known_column=_WORKER["known_columns"].get(flight_id),
            deadline=_WORKER["deadline"],
        )
    except PricingTimeout:
        return flight_id, False, 0.0, None
    return flight_id, True, float(reduced_cost), column


# --------------------------------------------------------------------------- parent side


def price_sweep(
    pricing_order: list[int],
    requests: list[FlightRequest],
    graphs: dict,
    cfg: SimConfig,
    params: ColGenParams,
    catalog: StaticTerminalCatalog,
    duals: dict,
    dual_view: DualView,
    flight_duals: dict[int, float],
    known_columns: dict[int, Column],
    *,
    deadline: float | None = None,
    config: ParallelPricingConfig | None = None,
) -> SweepResult:
    """Price every flight in ``pricing_order``, sequentially or across processes.

    ``graphs`` is used only by the sequential path; workers build their own from
    ``requests`` (cheaper than pickling, see the module docstring).
    """

    config = config or ParallelPricingConfig()
    if config.n_workers == 0:
        return _sweep_sequential(
            pricing_order, graphs, cfg, params, dual_view, flight_duals,
            known_columns, deadline,
        )
    return _sweep_parallel(
        pricing_order, requests, cfg, params, catalog, duals, flight_duals,
        known_columns, deadline, config,
    )


def _sweep_sequential(
    pricing_order, graphs, cfg, params, dual_view, flight_duals, known_columns, deadline
) -> SweepResult:
    """The original in-process loop, kept verbatim as the parity baseline."""

    flight_ids: list[int] = []
    reduced_costs: list[float] = []
    columns: list[Column | None] = []
    for flight_id in pricing_order:
        try:
            reduced_cost, column = price_flight(
                graphs[flight_id],
                dual_view,
                flight_duals[flight_id],
                cfg,
                params,
                known_column=known_columns.get(flight_id),
                deadline=deadline,
            )
        except PricingTimeout:
            return SweepResult(
                tuple(flight_ids), tuple(reduced_costs), tuple(columns), flight_id
            )
        flight_ids.append(flight_id)
        reduced_costs.append(float(reduced_cost))
        columns.append(column)
    return SweepResult(tuple(flight_ids), tuple(reduced_costs), tuple(columns), None)


def _sweep_parallel(
    pricing_order, requests, cfg, params, catalog, duals, flight_duals,
    known_columns, deadline, config
) -> SweepResult:
    """Fan the sweep across a pool built and torn down for this sweep alone.

    The pool is per-sweep because the duals change every iteration and they reach the
    workers through the initializer -- once per worker rather than once per task, which is
    the difference between pickling them ~n_workers times and ~n_flights times.
    """

    # `spawn` explicitly rather than by platform default: `fork` would inherit the parent's
    # numba runtime and thread state, which is exactly the combination that is unsafe.
    ctx = mp.get_context("spawn")
    init_args = (
        list(requests), cfg, params, catalog, duals, flight_duals,
        dict(known_columns), deadline,
    )
    flight_ids: list[int] = []
    reduced_costs: list[float] = []
    columns: list[Column | None] = []
    timeout_flight_id: int | None = None
    with ctx.Pool(
        processes=config.n_workers, initializer=_init_worker, initargs=init_args
    ) as pool:
        # `imap` and not `imap_unordered`: results must arrive in `pricing_order` index
        # order so the accepted prefix is the one the sequential loop would have produced,
        # and so the reduced costs reach `master.upper_bound`'s non-associative `sum` in a
        # fixed order. See the module docstring.
        for flight_id, priced, reduced_cost, column in pool.imap(
            _price_one, pricing_order, config.chunksize
        ):
            if not priced:
                # Past the first gap nothing is accepted, so breaking here abandons the
                # outstanding tasks; leaving the `with` block terminates the pool.
                timeout_flight_id = flight_id
                break
            flight_ids.append(flight_id)
            reduced_costs.append(float(reduced_cost))
            columns.append(column)
    return SweepResult(
        tuple(flight_ids), tuple(reduced_costs), tuple(columns), timeout_flight_id
    )
