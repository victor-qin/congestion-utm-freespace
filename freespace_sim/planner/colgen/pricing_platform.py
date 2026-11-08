"""CPU/Metal/CUDA pricing interface for fixed-price batches and solver GPU sweeps.

The full solver preserves its CPU worker pool and reuses this platform for GPU
sweeps and repair. Certificates cover the finite graph, hop and departure domain.
Disabling CPU recomputation always disables the GPU optimality certificate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ...config import SimConfig
from . import pricing
from .network import FlightGraph, RowKey
from .translate import Column


@dataclass(frozen=True, slots=True)
class PricingOptions:
    """Select a device and bounded experimental GPU search resources."""

    device: str = "cpu"
    gpu_backend: str | None = None
    cpu_recompute: bool = True
    max_labels: int = 250_000
    chunk_size: int = 2048
    # Experimental absolute RC allowance, not a proved floating-point error bound.
    gpu_prune_epsilon: float | None = None
    gpu_prune_audit: bool = False
    gpu_prune_verify_border: bool = False
    endpoint_bounds: bool = True
    endpoint_max_rows: int = 64
    endpoint_cache_bytes: int = 1_048_576
    endpoint_cache_work: int = 1_000_000
    gpu_integer_bounds: bool = False
    gpu_integer_audit: bool = False
    gpu_approach_bounds: bool = False
    prepare_workers: int = 0
    prepare_batch_size: int = 8
    prepare_max_bytes: int = 128 * 1024 * 1024
    prepare_cache_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject ambiguous device selection and invalid allocation limits."""
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if (self.device == "cpu" and self.gpu_backend is not None) or (
            self.device == "gpu" and self.gpu_backend not in ("metal", "cuda")
        ):
            raise ValueError("Select gpu_backend='metal' or 'cuda' exactly when device='gpu'")
        if not isinstance(self.cpu_recompute, bool):
            raise TypeError("cpu_recompute must be boolean")
        if not isinstance(self.gpu_integer_bounds, bool) or not isinstance(self.gpu_integer_audit, bool):
            raise TypeError("integer bound/audit options must be boolean")
        if self.gpu_integer_bounds:
            if self.device != "gpu" or not self.cpu_recompute or self.gpu_prune_epsilon is not None:
                raise ValueError("integer bounds require GPU canonical checks and no epsilon mode")
        elif self.gpu_integer_audit:
            raise ValueError("integer audit requires integer bounds")
        if not isinstance(self.gpu_approach_bounds, bool):
            raise TypeError("gpu_approach_bounds must be boolean")
        if self.gpu_approach_bounds and (self.device != "gpu" or not self.cpu_recompute
                                         or self.gpu_prune_epsilon is not None):
            raise ValueError("Approach bounds require GPU canonical checks and no epsilon mode")
        if not isinstance(self.gpu_prune_audit, bool):
            raise TypeError("gpu_prune_audit must be boolean")
        if not isinstance(self.gpu_prune_verify_border, bool):
            raise TypeError("gpu_prune_verify_border must be boolean")
        if self.gpu_prune_epsilon is not None:
            if (self.device != "gpu" or not self.cpu_recompute or
                isinstance(self.gpu_prune_epsilon, bool) or
                not math.isfinite(self.gpu_prune_epsilon) or self.gpu_prune_epsilon < 0):
                raise ValueError("GPU epsilon needs a finite nonnegative margin and cpu_recompute=True")
        elif self.gpu_prune_audit or self.gpu_prune_verify_border:
            raise ValueError("GPU pruning audit/border verification requires gpu_prune_epsilon")
        if type(self.prepare_workers) is not int or not 0 <= self.prepare_workers <= 64:
            raise ValueError("prepare_workers must be an integer between 0 and 64")
        if self.prepare_workers and self.device != "gpu":
            raise ValueError("CPU preparation workers require GPU pricing")
        for name, lower, upper in (("prepare_batch_size", 1, 1024),
                                   ("prepare_max_bytes", 4096, 2**31 - 1),
                                   ("prepare_cache_bytes", 0, 2**31 - 1)):
            if type(getattr(self, name)) is not int or not lower <= getattr(self, name) <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if not isinstance(self.endpoint_bounds, bool):
            raise TypeError("endpoint_bounds must be boolean")
        for name in ("endpoint_max_rows", "endpoint_cache_bytes", "endpoint_cache_work"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**31:
                raise ValueError(f"{name} must be a nonnegative integer below 2**31")
        if self.endpoint_max_rows > 64:
            raise ValueError("endpoint_max_rows must be at most 64")
        for name in ("max_labels", "chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**27:
                raise ValueError(f"{name} must be a positive integer below 2**27")

    @property
    def verify_prunes(self) -> bool:
        """Prepare original-price prefix sums when any proposal can require CPU checking."""
        if self.gpu_integer_bounds:
            return self.gpu_integer_audit
        return self.cpu_recompute and (self.gpu_prune_epsilon is None or self.gpu_prune_audit
                                       or self.gpu_prune_verify_border)

    @property
    def certify_search(self) -> bool:
        """Only original FP64 checks, including epsilon audits, authorize a certificate."""
        return self.cpu_recompute and (self.gpu_prune_epsilon is None or self.gpu_prune_audit)


@dataclass(frozen=True, slots=True)
class PricingProblem:
    """One flight with a fixed graph, objective and snapshot of master prices."""

    graph: FlightGraph
    cfg: SimConfig
    params: Any
    duals: Mapping[RowKey | tuple, float] | pricing.DualView
    pi_f: float
    known_column: Column | None = None
    forbidden_rows: frozenset[RowKey] = frozenset()
    require_improving: bool = True

    def __post_init__(self) -> None:
        """Keep route filtering identical across CPU and GPU repair calls."""
        if not isinstance(self.require_improving, bool):
            raise TypeError("require_improving must be a boolean")


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Validated incumbent, search status and an optional certified upper bound.

    A certified upper bound can exceed the incumbent's reduced cost. It covers
    all omitted alternatives, including near-tie branches closed by the GPU
    search; only the upper bound can prove absence of an improving route.
    """

    flight_id: int
    reduced_cost: float
    column: Column | None
    upper_bound: float | None
    search_complete: bool
    certified: bool
    backend: str
    cpu_recompute: bool
    status: str
    stats: dict[str, Any] = field(default_factory=dict)


class PricingPlatform:
    """Reuse one selected backend across fixed-price batches without CPU fallback."""

    def __init__(self, options: PricingOptions = PricingOptions()) -> None:
        """Store backend options; GPU initialization is deferred until the first batch."""
        self.options = options
        self._runtime = None
        self._preparation_pool = None

    def close_preparation(self) -> None:
        """Release CPU workers and caches; a subsequent pricing call may start fresh workers."""
        if self._preparation_pool is not None:
            self._preparation_pool.close()
            self._preparation_pool = None

    def close(self) -> None:
        """Release preparation workers and all owner-held device caches."""
        self.close_preparation()
        if self._runtime is not None:
            self._runtime.clear_cache()
            self._runtime = None

    def price_batches(
        self, problems: Sequence[PricingProblem], batch_size: int,
        deadline: float | None = None,
    ) -> Iterator[tuple[PricingResult, ...]]:
        """Prepare bounded synchronous groups, then submit the original GPU batches in order.

        Parameters
        ------------
        - problems (Sequence[PricingProblem]): full ordered sweep; raw prices snapshot when
          iteration begins and remain fixed across all its groups.
        - batch_size (int): search batch size, independent of CPU preparation group size.
        - deadline (float | None): shared deadline, including worker startup and transport.

        Return
        --------
        - output (Iterator[tuple[PricingResult, ...]]): unchanged search batches; no worker
          is preparing while the owner searches. Close the iterator when stopping early.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if not self.options.prepare_workers:
            for start in range(0, len(problems), batch_size):
                yield self.price_batch(problems[start:start + batch_size], deadline)
            return
        from .gpu_prepare_pool import (
            GPUPreparationLimit, GPUPreparationPool, GPUPreparationWorkerLost,
        )
        from .gpu_prepared import PreparedFlight, PreparationStopped
        from .gpu_pricing import price_gpu_batch
        from .gpu_runtime import ComputeRuntime
        from .windows import derive_cell_window

        if batch_size > self.options.prepare_batch_size:
            raise GPUPreparationLimit("GPU batch exceeds preparation group size")
        normalized, current = {}, []
        for p in problems:
            view = p.duals
            if not isinstance(view, pricing.DualView):
                key = id(view), derive_cell_window(p.cfg)
                if key not in normalized:
                    normalized[key] = pricing.DualView(view, p.cfg)
                view = normalized[key]
            current.append(replace(p, duals=view))
        width = (self.options.prepare_batch_size // batch_size) * batch_size
        finished = False
        try:
            for start in range(0, len(current), width):
                group = current[start:start + width]
                if (self._preparation_pool is not None
                        and self._preparation_pool.options != self.options):
                    self.close_preparation()
                if self._preparation_pool is None or self._preparation_pool.closed:
                    self._preparation_pool = GPUPreparationPool(self.options)
                pool = self._preparation_pool
                before = dict(pool.stats)
                started = time.monotonic()
                try:
                    prepared = pool.prepare(group, deadline)
                except PreparationStopped:
                    prepared = tuple(PreparedFlight(status="deadline") for _ in group)
                except GPUPreparationWorkerLost:
                    prepared = tuple(PreparedFlight(status="preparation_worker_lost") for _ in group)
                except GPUPreparationLimit:
                    prepared = tuple(PreparedFlight(status="preparation_limit") for _ in group)
                preparation_s = time.monotonic() - started
                measured = {key: value - before[key] for key, value in pool.stats.items()
                            if not key.startswith("peak_") and key != "owner_cache_bytes"}
                measured.update({key: value for key, value in pool.stats.items()
                                 if key.startswith("peak_") or key == "owner_cache_bytes"})
                measured.update(wall_s=preparation_s, flights=len(group), overlap=False)
                # This point is the stage barrier: every job/reply in this group has
                # completed, or its workers were reaped on timeout. No refill occurs.
                if self._runtime is None:
                    self._runtime = (ComputeRuntime(self.options.gpu_backend, integer_bounds=True)
                                     if self.options.gpu_integer_bounds else
                                     ComputeRuntime(self.options.gpu_backend))
                for offset in range(0, len(group), batch_size):
                    batch = group[offset:offset + batch_size]
                    values = price_gpu_batch(batch, self.options, self._runtime, deadline,
                                             prepared[offset:offset + batch_size])
                    yield tuple(replace(value, stats=dict(value.stats,
                        preparation_group=measured if offset == 0 else {},
                        preparation_flight=prepared[offset + i].stats,
                    )) for i, value in enumerate(values))
                del prepared
            finished = True
        finally:
            if not finished:
                self.close_preparation()

    def price_batch(
        self, problems: Sequence[PricingProblem], deadline: float | None = None
    ) -> tuple[PricingResult, ...]:
        """Price each flight against fixed prices and return explicit certificate status.

        Parameters
        ------------
        - problems (Sequence[PricingProblem]): independent flights; order is preserved.
          Share one DualView per LP snapshot to reuse sparse prices across calls. Raw
          mappings are snapshotted anew on each call and may be changed between calls.
        - deadline (float | None): shared monotonic deadline including GPU initialization.

        Return
        --------
        - output (tuple[PricingResult, ...]): incomplete searches never assert optimality.
        """
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if not problems:
            return ()
        if self.options.prepare_workers:
            batches = self.price_batches(problems, len(problems), deadline)
            try:
                result, = tuple(batches)
                return result
            finally:
                batches.close()
        if self.options.device == "gpu":
            from .gpu_pricing import price_gpu_batch
            from .gpu_runtime import ComputeRuntime

            if self._runtime is None:
                self._runtime = (ComputeRuntime(self.options.gpu_backend, integer_bounds=True)
                                 if self.options.gpu_integer_bounds else
                                 ComputeRuntime(self.options.gpu_backend))
            return price_gpu_batch(problems, self.options, self._runtime, deadline)
        results = []
        for problem in problems:
            start = time.monotonic()
            pricing.clear_search_record()
            try:
                rc, column = pricing.price_flight(
                    problem.graph,
                    problem.duals,
                    problem.pi_f,
                    problem.cfg,
                    problem.params,
                    known_column=problem.known_column,
                    forbidden_rows=problem.forbidden_rows,
                    require_improving=problem.require_improving,
                    deadline=deadline,
                )
                complete = True
            except pricing.PricingTimeout:
                rc, column, complete = -math.inf, None, False
            results.append(
                PricingResult(
                    flight_id=problem.graph.request.flight_id,
                    reduced_cost=rc,
                    column=column,
                    upper_bound=rc if complete else None,
                    search_complete=complete,
                    certified=complete,
                    backend="cpu",
                    cpu_recompute=True,
                    status="complete" if complete else "deadline",
                    stats={**pricing.last_search_record(), "elapsed_s": time.monotonic() - start},
                )
            )
        return tuple(results)
