"""Restricted master problem and LP/MIP backend adapters.

The mathematical model is expressed in maximize sense throughout this module:
each trajectory has value ``M - delay_s`` and every flight/capacity constraint is
an upper-bound row.  SciPy/HiGHS minimizes, so only that adapter negates the
objective and returned inequality marginals.
"""

from __future__ import annotations

import math
import operator
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

from .network import RowIndex, RowKey
from .params import ColGenParams
from .translate import Column


class BackendTimeout(TimeoutError):
    """Raised when an LP backend reaches the caller-provided solve budget."""


@dataclass(frozen=True, slots=True)
class BackendLpResult:
    """One backend-independent LP result, in maximize sense."""

    objective: float
    flight_duals: dict[int, float]
    row_duals: dict[RowKey, float]
    x: np.ndarray


@dataclass(frozen=True, slots=True)
class BackendIpResult:
    """One backend-independent integer result, in maximize sense.

    ``upper_bound`` is the backend's bound on the current restricted master;
    positive infinity means that a limited solve supplied no usable bound.
    """

    objective: float
    x: np.ndarray
    upper_bound: float = math.inf
    status: str = "unknown"
    optimal: bool = False

    @property
    def bound(self) -> float:
        """Compatibility alias for the explicitly oriented upper bound."""

        return self.upper_bound


@runtime_checkable
class LpBackend(Protocol):
    """Small incremental seam used by :class:`RestrictedMaster`.

    Backends own flight rows from construction time.  Capacity rows are added
    lazily, and ``column_rows`` contains only the already-materialized rows that
    a newly added column claims.
    """

    name: str
    flight_ids: tuple[int, ...]
    time_limit_s: float

    def add_column(
        self,
        objective: float,
        flight_id: int,
        column_rows: Sequence[RowKey],
    ) -> None: ...

    def add_row(
        self,
        row: RowKey,
        rhs: float,
        column_indices: Sequence[int],
    ) -> None: ...

    def solve_lp(self) -> BackendLpResult: ...

    def solve_ip(self, warm_start: np.ndarray | None = None) -> BackendIpResult: ...


def _flight_tuple(flight_ids: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for flight_id in flight_ids:
        try:
            normalized.append(operator.index(flight_id))
        except TypeError as exc:
            raise TypeError("flight ids must be integers") from exc
    if len(normalized) != len(set(normalized)):
        raise ValueError("flight ids must be unique")
    return tuple(normalized)


def _validate_column_indices(indices: Sequence[int], n_columns: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in indices:
        try:
            value = operator.index(index)
        except TypeError as exc:
            raise TypeError("column indices must be integers") from exc
        if not 0 <= value < n_columns:
            raise IndexError(f"column index {value} is outside [0, {n_columns})")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise AssertionError("capacity-row coefficients must be in {0, 1}")
    return tuple(normalized)


class HighsBackend:
    """Cold-rebuild SciPy/HiGHS LP and MIP backend.

    SciPy exposes HiGHS in minimization sense.  We pass ``-rho`` and negate
    ``ineqlin.marginals`` exactly once here, so callers see maximize objectives
    and non-negative duals for upper-bound rows.
    """

    name = "highs"

    def __init__(
        self,
        flight_ids: Iterable[int],
        *,
        ip_gap: float = 1e-3,
        time_limit_s: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.flight_ids = _flight_tuple(flight_ids)
        self._flight_pos = {flight_id: i for i, flight_id in enumerate(self.flight_ids)}
        self._objectives: list[float] = []
        self._column_flights: list[int] = []
        self._rows: list[RowKey] = []
        self._row_rhs: dict[RowKey, float] = {}
        self._row_columns: dict[RowKey, set[int]] = {}
        self.ip_gap = float(ip_gap)
        self.time_limit_s = float(time_limit_s)
        # Retained as part of the common deterministic-backend contract.  The
        # scipy.optimize wrappers currently expose no HiGHS random-seed option.
        self.seed = operator.index(seed)

    def add_column(
        self,
        objective: float,
        flight_id: int,
        column_rows: Sequence[RowKey],
    ) -> None:
        if flight_id not in self._flight_pos:
            raise KeyError(f"unknown flight id {flight_id}")
        value = float(objective)
        if not math.isfinite(value):
            raise ValueError("column objective must be finite")
        if len(column_rows) != len(set(column_rows)):
            raise AssertionError("capacity-row coefficients must be in {0, 1}")
        unknown = [row for row in column_rows if row not in self._row_rhs]
        if unknown:
            raise KeyError(f"column references unmaterialized rows: {unknown[:3]!r}")
        index = len(self._objectives)
        self._objectives.append(value)
        self._column_flights.append(flight_id)
        for row in column_rows:
            self._row_columns[row].add(index)

    def add_row(
        self,
        row: RowKey,
        rhs: float,
        column_indices: Sequence[int],
    ) -> None:
        if row in self._row_rhs:
            raise ValueError(f"capacity row {row!r} is already materialized")
        bound = float(rhs)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("capacity-row rhs must be finite and non-negative")
        indices = _validate_column_indices(column_indices, len(self._objectives))
        self._rows.append(row)
        self._row_rhs[row] = bound
        self._row_columns[row] = set(indices)

    def _matrix(self) -> tuple[csc_matrix, np.ndarray]:
        n_flights = len(self.flight_ids)
        n_columns = len(self._objectives)
        row_indices: list[int] = []
        column_indices: list[int] = []

        for column, flight_id in enumerate(self._column_flights):
            row_indices.append(self._flight_pos[flight_id])
            column_indices.append(column)
        for offset, row in enumerate(self._rows, start=n_flights):
            for column in sorted(self._row_columns[row]):
                row_indices.append(offset)
                column_indices.append(column)

        data = np.ones(len(row_indices), dtype=float)
        matrix = csc_matrix(
            (data, (row_indices, column_indices)),
            shape=(n_flights + len(self._rows), n_columns),
        )
        rhs = np.array(
            [1.0] * n_flights + [self._row_rhs[row] for row in self._rows],
            dtype=float,
        )
        return matrix, rhs

    def solve_lp(self) -> BackendLpResult:
        if not self._objectives:
            return BackendLpResult(
                0.0,
                {flight_id: 0.0 for flight_id in self.flight_ids},
                {row: 0.0 for row in self._rows},
                np.empty(0, dtype=float),
            )
        matrix, rhs = self._matrix()
        result = linprog(
            c=-np.asarray(self._objectives, dtype=float),
            A_ub=matrix,
            b_ub=rhs,
            # The per-flight row already implies x <= 1.  Repeating that as a
            # variable bound lets its marginal absorb the flight-row dual and
            # corrupts pricing's pi_f.
            bounds=(0.0, None),
            method="highs",
            options={"presolve": True, "time_limit": self.time_limit_s},
        )
        if result.status == 1:
            raise BackendTimeout(f"HiGHS LP reached its limit: {result.message}")
        if not result.success or result.x is None or result.fun is None:
            raise RuntimeError(f"HiGHS LP failed (status {result.status}): {result.message}")

        # Marginals are d(min objective)/d(rhs), hence non-positive for the
        # min/<= formulation.  Negation restores the reference max-sense dual.
        maximize_duals = -np.asarray(result.ineqlin.marginals, dtype=float)
        n_flights = len(self.flight_ids)
        flight_duals = {
            flight_id: float(maximize_duals[index])
            for index, flight_id in enumerate(self.flight_ids)
        }
        row_duals = {
            row: float(maximize_duals[n_flights + index]) for index, row in enumerate(self._rows)
        }
        x = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        return BackendLpResult(float(-result.fun), flight_duals, row_duals, x)

    def solve_ip(self, warm_start: np.ndarray | None = None) -> BackendIpResult:
        # scipy.optimize.milp deliberately has no incumbent parameter.  Validate
        # shape for a common backend contract, then run cold.
        if warm_start is not None and np.asarray(warm_start).shape != (len(self._objectives),):
            raise ValueError("MIP warm start has the wrong number of columns")
        if not self._objectives:
            return BackendIpResult(0.0, np.empty(0, dtype=float), 0.0, "optimal", True)

        matrix, rhs = self._matrix()
        constraint = LinearConstraint(matrix, -np.inf, rhs)
        result = milp(
            c=-np.asarray(self._objectives, dtype=float),
            integrality=np.ones(len(self._objectives), dtype=np.uint8),
            bounds=Bounds(0.0, 1.0),
            constraints=constraint,
            options={
                "disp": False,
                "presolve": True,
                "mip_rel_gap": self.ip_gap,
                "time_limit": self.time_limit_s,
            },
        )
        if result.x is None and result.status == 1:
            # The binary RMP always has the all-zero solution.  HiGHS may hit
            # an extremely tight time cap before publishing even that trivial
            # incumbent; return it explicitly so RestrictedMaster can retain
            # its independently validated rounding incumbent.
            return BackendIpResult(
                0.0,
                np.zeros(len(self._objectives), dtype=float),
                math.inf,
                "time_limit_no_incumbent",
                False,
            )
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"HiGHS MIP failed (status {result.status}): {result.message}")
        x = np.rint(np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0))
        objective = float(np.dot(np.asarray(self._objectives, dtype=float), x))
        optimal = bool(result.success and result.status == 0)
        raw_bound = getattr(result, "mip_dual_bound", None)
        bound = -float(raw_bound) if raw_bound is not None else math.inf
        if not math.isfinite(bound):
            bound = objective if optimal else math.inf
        bound = max(objective, bound)
        status = "optimal" if optimal else f"status_{result.status}"
        return BackendIpResult(objective, x, bound, status, optimal)


class _GurobiUnavailable(RuntimeError):
    """Raised only for import, environment, or license failures."""


class GurobiBackend:
    """Persistent native-maximize Gurobi backend with a real MIP start."""

    name = "gurobi"

    def __init__(
        self,
        flight_ids: Iterable[int],
        *,
        ip_gap: float = 1e-3,
        time_limit_s: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.flight_ids = _flight_tuple(flight_ids)
        self.ip_gap = float(ip_gap)
        self.time_limit_s = float(time_limit_s)
        self.seed = operator.index(seed)
        try:
            import gurobipy as gp
        except ImportError as exc:  # pragma: no cover - depends on local optional package
            raise _GurobiUnavailable("gurobipy is not installed") from exc

        self._gp = gp
        try:
            model = gp.Model("colgen-rmp")
        except gp.GurobiError as exc:  # pragma: no cover - depends on local license
            raise _GurobiUnavailable(f"Gurobi could not start: {exc}") from exc
        self._model = model
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = self.seed
        model.ModelSense = gp.GRB.MAXIMIZE
        self._flight_rows = {
            flight_id: model.addConstr(gp.LinExpr() <= 1.0, name=f"flight[{flight_id}]")
            for flight_id in self.flight_ids
        }
        self._capacity_rows: dict[RowKey, object] = {}
        self._variables: list[object] = []
        self._objectives: list[float] = []
        model.update()

    def add_column(
        self,
        objective: float,
        flight_id: int,
        column_rows: Sequence[RowKey],
    ) -> None:
        if flight_id not in self._flight_rows:
            raise KeyError(f"unknown flight id {flight_id}")
        value = float(objective)
        if not math.isfinite(value):
            raise ValueError("column objective must be finite")
        if len(column_rows) != len(set(column_rows)):
            raise AssertionError("capacity-row coefficients must be in {0, 1}")
        try:
            constraints = [self._flight_rows[flight_id]] + [
                self._capacity_rows[row] for row in column_rows
            ]
        except KeyError as exc:
            raise KeyError(f"column references unmaterialized row {exc.args[0]!r}") from exc
        coefficients = [1.0] * len(constraints)
        gp_column = self._gp.Column(coefficients, constraints)
        variable = self._model.addVar(
            lb=0.0,
            ub=self._gp.GRB.INFINITY,
            obj=value,
            vtype=self._gp.GRB.CONTINUOUS,
            column=gp_column,
            name=f"route[{len(self._variables)}]",
        )
        self._variables.append(variable)
        self._objectives.append(value)

    def add_row(
        self,
        row: RowKey,
        rhs: float,
        column_indices: Sequence[int],
    ) -> None:
        if row in self._capacity_rows:
            raise ValueError(f"capacity row {row!r} is already materialized")
        bound = float(rhs)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("capacity-row rhs must be finite and non-negative")
        indices = _validate_column_indices(column_indices, len(self._variables))
        self._model.update()
        expression = self._gp.quicksum(self._variables[index] for index in indices)
        constraint = self._model.addConstr(
            expression <= bound,
            name=f"capacity[{len(self._capacity_rows)}]",
        )
        self._capacity_rows[row] = constraint
        self._model.update()

    def solve_lp(self) -> BackendLpResult:
        for variable in self._variables:
            variable.VType = self._gp.GRB.CONTINUOUS
            variable.UB = self._gp.GRB.INFINITY
        self._model.Params.TimeLimit = self.time_limit_s
        self._model.update()
        self._model.optimize()
        if self._model.Status == self._gp.GRB.TIME_LIMIT:
            raise BackendTimeout("Gurobi LP reached its time limit")
        if self._model.Status != self._gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi LP failed with status {self._model.Status}")
        flight_duals = {
            flight_id: float(constraint.Pi) for flight_id, constraint in self._flight_rows.items()
        }
        row_duals = {row: float(constraint.Pi) for row, constraint in self._capacity_rows.items()}
        x = np.asarray([variable.X for variable in self._variables], dtype=float)
        return BackendLpResult(float(self._model.ObjVal), flight_duals, row_duals, x)

    def solve_ip(self, warm_start: np.ndarray | None = None) -> BackendIpResult:
        if warm_start is not None:
            warm_start = np.asarray(warm_start, dtype=float)
            if warm_start.shape != (len(self._variables),):
                raise ValueError("MIP warm start has the wrong number of columns")
        for index, variable in enumerate(self._variables):
            variable.VType = self._gp.GRB.BINARY
            variable.UB = 1.0
            variable.Start = (
                self._gp.GRB.UNDEFINED if warm_start is None else float(warm_start[index])
            )
        self._model.Params.MIPGap = self.ip_gap
        self._model.Params.TimeLimit = self.time_limit_s
        self._model.update()
        self._model.optimize()
        if self._model.SolCount < 1 and self._model.Status == self._gp.GRB.TIME_LIMIT:
            # As with HiGHS, a sub-millisecond limit can expire before Gurobi
            # processes the supplied MIP start.  RestrictedMaster owns the
            # validated heuristic and will prefer it to this zero solution.
            return BackendIpResult(
                0.0,
                np.zeros(len(self._variables), dtype=float),
                math.inf,
                "time_limit_no_incumbent",
                False,
            )
        if self._model.SolCount < 1:
            raise RuntimeError(f"Gurobi MIP found no incumbent (status {self._model.Status})")
        x = np.rint(
            np.clip(np.asarray([variable.X for variable in self._variables], dtype=float), 0.0, 1.0)
        )
        objective = float(np.dot(np.asarray(self._objectives, dtype=float), x))
        bound = float(self._model.ObjBound)
        if not math.isfinite(bound) or abs(bound) >= 0.5 * float(self._gp.GRB.INFINITY):
            bound = math.inf
        optimal = self._model.Status == self._gp.GRB.OPTIMAL
        status = "optimal" if optimal else f"status_{self._model.Status}"
        return BackendIpResult(objective, x, max(objective, bound), status, optimal)


def create_backend(
    flight_ids: Iterable[int],
    params: ColGenParams,
    *,
    seed: int = 0,
) -> LpBackend:
    """Select the requested backend, falling back only for ``solver='auto'``."""

    # Snapshot once: an auto attempt that consumes a generator must not leave
    # the HiGHS fallback with an empty flight catalogue.
    normalized_flight_ids = _flight_tuple(flight_ids)
    # Native solvers measure their relative gap on master revenue, whose
    # scale is dominated by N*M.  Convert the user-facing delay/cancellation
    # tolerance to a conservative absolute-revenue tolerance before handing
    # it to that relative-gap API.  Since an optimal RMP revenue lies in
    # [0, N*M], this makes a native stop imply a transformed-cost gap no
    # larger than ``params.ip_gap`` even when transformed cost is near zero.
    revenue_scale = max(1.0, len(normalized_flight_ids) * params.M)
    kwargs = {
        "ip_gap": params.ip_gap / revenue_scale,
        "time_limit_s": params.time_limit_s,
        "seed": seed,
    }
    if params.solver == "highs":
        return HighsBackend(normalized_flight_ids, **kwargs)
    try:
        return GurobiBackend(normalized_flight_ids, **kwargs)
    except _GurobiUnavailable as exc:
        if params.solver == "gurobi":
            raise RuntimeError(
                f"ColGenParams(solver='gurobi') requested Gurobi, but it is unavailable: {exc}"
            ) from exc
        return HighsBackend(normalized_flight_ids, **kwargs)


def _row_sort_key(row: RowKey) -> tuple[str, ...]:
    return tuple(repr(part) for part in row)


def _column_sort_key(column: Column) -> tuple[object, ...]:
    return (
        column.flight_id,
        column.delay_s,
        column.departure_step,
        column.level,
        -1 if column.origin_lane_idx is None else column.origin_lane_idx,
        -1 if column.dest_lane_idx is None else column.dest_lane_idx,
        column.cell_path,
    )


class RestrictedMaster:
    """Restricted trajectory master with lazily materialized capacity rows."""

    def __init__(
        self,
        flight_ids: Iterable[int],
        row_index: RowIndex,
        params: ColGenParams,
        *,
        seed: int = 0,
        fixed_loads: Mapping[RowKey | tuple[object, ...], int] | None = None,
        backend: LpBackend | None = None,
    ) -> None:
        self.flight_ids = _flight_tuple(flight_ids)
        self.row_index = row_index
        self.params = params
        self.seed = operator.index(seed)
        self.fixed_loads: dict[RowKey, int] = {}
        for raw_row, raw_load in (fixed_loads or {}).items():
            row = raw_row if isinstance(raw_row, RowKey) else RowKey(raw_row)
            try:
                load = operator.index(raw_load)
            except TypeError as exc:
                raise TypeError("fixed row loads must be integers") from exc
            if load < 0:
                raise ValueError("fixed row loads must be non-negative")
            cap = row_index.cap(row)
            if load > cap:
                raise ValueError(f"fixed load {load} exceeds capacity {cap} for row {row!r}")
            if load:
                self.fixed_loads[row] = load

        self._backend: LpBackend = backend or create_backend(
            self.flight_ids,
            params,
            seed=self.seed,
        )
        if tuple(self._backend.flight_ids) != self.flight_ids:
            raise ValueError("backend flight ids do not match the restricted master")
        self._columns: list[Column] = []
        self._column_indices: dict[Column, int] = {}
        self._objectives: list[float] = []
        self._materialized: dict[RowKey, float] = {}
        self.last_flight_duals: dict[int, float] = {flight_id: 0.0 for flight_id in self.flight_ids}
        self.last_row_duals: dict[RowKey, float] = {}
        self.last_lp_objective = 0.0
        self.last_lp_x = np.empty(0, dtype=float)
        self.last_ip_objective: float | None = None
        self.last_ip_bound: float | None = None
        self.last_ip_status: str | None = None
        self.last_ip_optimal: bool | None = None
        self._warm_start: np.ndarray | None = None
        self._heuristic_selection: dict[int, Column] = {}

    @property
    def backend(self) -> LpBackend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def columns(self) -> tuple[Column, ...]:
        return tuple(self._columns)

    @property
    def materialized_rows(self) -> frozenset[RowKey]:
        return frozenset(self._materialized)

    @property
    def heuristic_selection(self) -> dict[int, Column]:
        return dict(self._heuristic_selection)

    @property
    def flight_duals(self) -> dict[int, float]:
        """Per-flight duals from the most recent LP, in maximize sense."""

        return dict(self.last_flight_duals)

    def add_column(self, column: Column) -> int:
        """Add a trajectory variable, returning its stable dense column index."""

        if column.flight_id not in self.flight_ids:
            raise KeyError(f"column belongs to unknown flight {column.flight_id}")
        normalized_claims = frozenset(
            row if isinstance(row, RowKey) else RowKey(row) for row in column.claims
        )
        # The COO/CSC builders sum duplicate entries.  Canonical set ownership
        # and this assertion prevent an accidental coefficient two.
        coefficients = Counter(normalized_claims)
        if any(value not in {0, 1} for value in coefficients.values()):
            raise AssertionError("column coefficients must be in {0, 1}")
        for row in normalized_claims:
            self.row_index.cap(row)  # validates terminal metadata before any solve
        try:
            delay_s = float(column.delay_s)
        except (TypeError, ValueError) as exc:
            raise TypeError("column delay must be a real number") from exc
        if not math.isfinite(delay_s) or delay_s < 0.0:
            raise ValueError("column delay must be finite and non-negative")
        column = replace(column, delay_s=delay_s, claims=normalized_claims)
        existing = self._column_indices.get(column)
        if existing is not None:
            return existing
        objective = float(self.params.M - delay_s)
        if not math.isfinite(objective):
            raise ValueError("column objective must be finite")
        materialized_claims = tuple(
            sorted(
                (row for row in normalized_claims if row in self._materialized), key=_row_sort_key
            )
        )
        index = len(self._columns)
        self._backend.add_column(objective, column.flight_id, materialized_claims)
        self._columns.append(column)
        self._column_indices[column] = index
        self._objectives.append(objective)
        # Existing warm starts stay meaningful when pricing appends a column.
        if self._warm_start is not None:
            self._warm_start = np.pad(self._warm_start, (0, 1))
        return index

    def materialize_rows(self, rows: Iterable[RowKey | tuple[object, ...]]) -> int:
        """Materialize previously implicit rows in deterministic key order."""

        normalized: set[RowKey] = set()
        for raw_row in rows:
            row = raw_row if isinstance(raw_row, RowKey) else RowKey(raw_row)
            if row not in self._materialized:
                normalized.add(row)
        added = 0
        for row in sorted(normalized, key=_row_sort_key):
            cap = self.row_index.cap(row)
            rhs = float(cap - self.fixed_loads.get(row, 0))
            if rhs < 0.0:
                raise ValueError(f"fixed load exceeds capacity for row {row!r}")
            indices = [index for index, column in enumerate(self._columns) if row in column.claims]
            self._backend.add_row(row, rhs, indices)
            self.row_index.intern(row)
            self._materialized[row] = rhs
            added += 1
        return added

    def solve_lp(self) -> tuple[float, dict[RowKey, float], np.ndarray]:
        """Solve the current LP and return maximize objective, row duals, and values."""

        result = self._backend.solve_lp()
        if result.x.shape != (len(self._columns),):
            raise RuntimeError("backend returned an LP vector with the wrong shape")
        self.last_flight_duals = dict(result.flight_duals)
        self.last_row_duals = dict(result.row_duals)
        self.last_lp_objective = float(result.objective)
        self.last_lp_x = np.asarray(result.x, dtype=float)
        return self.last_lp_objective, dict(self.last_row_duals), self.last_lp_x.copy()

    def fractional_loads(self, x: Sequence[float]) -> dict[RowKey, float]:
        """Return fixed plus fractional loads over every claim in the support."""

        values = np.asarray(x, dtype=float)
        if values.shape != (len(self._columns),):
            raise ValueError("x has the wrong number of columns")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must contain finite values")
        loads: Counter[RowKey] = Counter(
            {row: float(load) for row, load in self.fixed_loads.items()}
        )
        for value, column in zip(values, self._columns):
            if value <= 0.0:
                continue
            for row in column.claims:
                loads[row] += float(value)
        return dict(loads)

    def add_violated_rows(self, x: Sequence[float], tol: float = 1e-7) -> int:
        """Materialize implicit capacity rows violated by one LP/IP solution."""

        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError("tol must be finite and non-negative")
        loads = self.fractional_loads(x)
        violated = [
            row
            for row, load in loads.items()
            if load > self.row_index.cap(row) + tol and row not in self._materialized
        ]
        return self.materialize_rows(violated)

    @staticmethod
    def upper_bound(objective: float, best_reduced_costs: Iterable[float]) -> float:
        """Paper bound in the module's normative maximize sense."""

        return float(objective) + sum(max(0.0, float(value)) for value in best_reduced_costs)

    def objective_of(self, selection: Mapping[int, Column] | Iterable[Column]) -> float:
        columns = selection.values() if isinstance(selection, Mapping) else selection
        return float(sum(self.params.M - column.delay_s for column in columns))

    def claim_loads(
        self,
        selection: Mapping[int, Column] | Iterable[Column],
    ) -> dict[RowKey, int]:
        """Return fixed plus integral loads, including unmaterialized rows."""

        columns = selection.values() if isinstance(selection, Mapping) else selection
        loads: Counter[RowKey] = Counter(self.fixed_loads)
        seen_flights: set[int] = set()
        for column in columns:
            if column.flight_id in seen_flights:
                raise ValueError(f"selection contains flight {column.flight_id} more than once")
            seen_flights.add(column.flight_id)
            for row in column.claims:
                loads[row] += 1
        return dict(loads)

    def violated_claim_rows(
        self,
        selection: Mapping[int, Column] | Iterable[Column],
    ) -> frozenset[RowKey]:
        loads = self.claim_loads(selection)
        return frozenset(row for row, load in loads.items() if load > self.row_index.cap(row))

    def is_claim_feasible(
        self,
        selection: Mapping[int, Column] | Iterable[Column],
    ) -> bool:
        try:
            return not self.violated_claim_rows(selection)
        except (KeyError, ValueError):
            return False

    def _can_add(self, column: Column, loads: Mapping[RowKey, int]) -> bool:
        return all(loads.get(row, 0) + 1 <= self.row_index.cap(row) for row in column.claims)

    def round_heuristic(
        self,
        x: Sequence[float],
        rng: np.random.Generator,
        n_tries: int | None = None,
    ) -> dict[int, Column]:
        """Randomized rounding followed by a claim-aware greedy fill."""

        values = np.asarray(x, dtype=float)
        if values.shape != (len(self._columns),):
            raise ValueError("x has the wrong number of columns")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must contain finite values")
        tries = self.params.n_heuristic_tries if n_tries is None else operator.index(n_tries)
        if tries < 1:
            raise ValueError("n_tries must be positive")
        epsilon = self.params.epsilon
        probabilities = np.clip(values, 0.0, 1.0)
        probabilities[probabilities <= epsilon] = 0.0
        probabilities[probabilities >= 1.0 - epsilon] = 1.0
        greedy_order = sorted(
            range(len(self._columns)),
            key=lambda index: (-self._objectives[index], _column_sort_key(self._columns[index])),
        )

        best: dict[int, Column] = {}
        best_objective = -math.inf
        best_signature: tuple[tuple[object, ...], ...] | None = None
        for _ in range(tries):
            selected: dict[int, Column] = {}
            loads: Counter[RowKey] = Counter(self.fixed_loads)

            forced = sorted(
                (index for index, value in enumerate(probabilities) if value >= 1.0),
                key=lambda index: (-values[index], _column_sort_key(self._columns[index])),
            )
            random_order = rng.permutation(len(self._columns)).tolist()
            for index in forced + random_order:
                column = self._columns[index]
                if column.flight_id in selected or probabilities[index] <= 0.0:
                    continue
                if index not in forced and rng.random() >= probabilities[index]:
                    continue
                if not self._can_add(column, loads):
                    continue
                selected[column.flight_id] = column
                loads.update(column.claims)

            for index in greedy_order:
                column = self._columns[index]
                if self._objectives[index] <= 0.0 or column.flight_id in selected:
                    continue
                if self._can_add(column, loads):
                    selected[column.flight_id] = column
                    loads.update(column.claims)

            objective = self.objective_of(selected)
            signature = tuple(
                _column_sort_key(selected[flight_id])
                for flight_id in self.flight_ids
                if flight_id in selected
            )
            if objective > best_objective + 1e-9 or (
                abs(objective - best_objective) <= 1e-9
                and (best_signature is None or signature < best_signature)
            ):
                best = selected
                best_objective = objective
                best_signature = signature
        return {flight_id: best[flight_id] for flight_id in self.flight_ids if flight_id in best}

    def set_heuristic(self, selection: Mapping[int, Column]) -> None:
        """Record a globally claim-feasible incumbent and backend warm-start vector."""

        canonical: dict[int, Column] = {}
        warm = np.zeros(len(self._columns), dtype=float)
        for raw_flight_id, column in selection.items():
            flight_id = operator.index(raw_flight_id)
            if flight_id != column.flight_id:
                raise ValueError("heuristic mapping key does not match column flight_id")
            try:
                index = self._column_indices[column]
            except KeyError as exc:
                raise ValueError("heuristic contains a column outside this master") from exc
            if flight_id in canonical:
                raise ValueError(f"heuristic contains flight {flight_id} more than once")
            canonical[flight_id] = self._columns[index]
            warm[index] = 1.0
        if not self.is_claim_feasible(canonical):
            raise ValueError("heuristic selection violates a capacity claim")
        self._heuristic_selection = {
            flight_id: canonical[flight_id]
            for flight_id in self.flight_ids
            if flight_id in canonical
        }
        self._warm_start = warm

    def _selection_from_x(self, x: Sequence[float]) -> dict[int, Column]:
        values = np.asarray(x, dtype=float)
        if values.shape != (len(self._columns),):
            raise RuntimeError("backend returned an IP vector with the wrong shape")
        selection: dict[int, Column] = {}
        for index in sorted(
            (i for i, value in enumerate(values) if value > 0.5),
            key=lambda i: (-values[i], _column_sort_key(self._columns[i])),
        ):
            column = self._columns[index]
            if column.flight_id in selection:
                raise RuntimeError("backend IP solution selects multiple columns for one flight")
            selection[column.flight_id] = column
        return {
            flight_id: selection[flight_id]
            for flight_id in self.flight_ids
            if flight_id in selection
        }

    def solve_ip(
        self,
        heuristic: Mapping[int, Column] | None = None,
        *,
        deadline: float | None = None,
    ) -> dict[int, Column]:
        """Solve the current binary RMP, separating claim rows until it is clean."""

        if heuristic is not None:
            self.set_heuristic(heuristic)
        if deadline is not None:
            deadline = float(deadline)
            if not math.isfinite(deadline):
                raise ValueError("IP deadline must be finite")
        self.last_ip_objective = None
        self.last_ip_bound = None
        self.last_ip_status = None
        self.last_ip_optimal = None
        original_time_limit_s = self._backend.time_limit_s
        try:
            while True:
                if deadline is not None:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0.0:
                        selection = dict(self._heuristic_selection)
                        self.last_ip_objective = self.objective_of(selection)
                        self.last_ip_bound = math.inf
                        self.last_ip_status = "time_limit_separation"
                        self.last_ip_optimal = False
                        return selection
                    self._backend.time_limit_s = max(
                        1e-6,
                        min(original_time_limit_s, remaining_s),
                    )
                result = self._backend.solve_ip(self._warm_start)
                selection = self._selection_from_x(result.x)
                violated = self.violated_claim_rows(selection)
                if violated:
                    added = self.materialize_rows(violated)
                    if not added:
                        raise RuntimeError(
                            "backend returned an IP solution violating an existing row"
                        )
                    continue

                # HiGHS has no MIP start and may stop on the time cap with a weaker
                # incumbent.  Preserve the already feasible rounding incumbent.
                if self._heuristic_selection and (
                    self.objective_of(self._heuristic_selection)
                    > self.objective_of(selection) + 1e-9
                ):
                    selection = dict(self._heuristic_selection)
                self.last_ip_objective = self.objective_of(selection)
                self.last_ip_bound = max(self.last_ip_objective, float(result.upper_bound))
                self.last_ip_status = result.status
                self.last_ip_optimal = result.optimal
                return selection
        finally:
            self._backend.time_limit_s = original_time_limit_s


__all__ = [
    "BackendIpResult",
    "BackendLpResult",
    "BackendTimeout",
    "GurobiBackend",
    "HighsBackend",
    "LpBackend",
    "RestrictedMaster",
    "create_backend",
]
