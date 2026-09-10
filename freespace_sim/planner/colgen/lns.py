"""Bounded joint repair of an incumbent's LP-supported neighborhood."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


def repair_neighborhood(master, values, incumbent, loads, destroy, deadline):
    """Fix other flights and select exactly one pooled column per released flight.

    Keep incumbent columns even when their LP weight is zero. All other options
    must have positive LP support and fit the frozen reservations. Capacity rows
    with identical candidate incidence are redundant; retain only the tightest.
    The final selection is independently checked against all original claims.
    """
    info = {"flights": 0, "columns": 0, "rows": 0, "improved": False, "n_swapped": 0,
            "status": "no_alternatives"}
    ranked = sorted(incumbent, key=lambda fid: (
        values[master._column_indices[incumbent[fid]]], fid,
    ))[:destroy]
    info["flights"] = len(ranked)
    released_loads = Counter(row for fid in ranked for row in incumbent[fid].claims)
    residual = {}

    def capacity(row):
        if row not in residual:
            residual[row] = master.row_index.cap(row) - loads.get(row, 0) + released_loads[row]
        return residual[row]

    columns = []
    by_flight = defaultdict(list)
    by_row = defaultdict(list)
    for fid in ranked:
        if time.monotonic() >= deadline:
            info["status"] = "construction_deadline"
            return incumbent, info
        for index in master._columns_by_flight[fid]:
            column = master._columns[index]
            if column != incumbent[fid] and values[index] <= master.params.epsilon:
                continue
            if any(capacity(row) < 1 for row in column.claims):
                continue
            local = len(columns)
            columns.append(column)
            by_flight[fid].append(local)
            for row in column.claims:
                by_row[row].append(local)
    info["columns"] = len(columns)
    if len(columns) == len(ranked):
        return incumbent, info
    # A row with claims from only one flight is already enforced by its equality.
    patterns = {}
    for row, indices in by_row.items():
        cap = capacity(row)
        if len({columns[i].flight_id for i in indices}) <= cap:
            continue
        pattern = tuple(indices)
        patterns[pattern] = min(cap, patterns.get(pattern, math.inf))
    info["rows"] = len(patterns)
    if time.monotonic() >= deadline:
        info["status"] = "construction_deadline"
        return incumbent, info
    costs = np.array([c.delay_s for c in columns])
    warm = np.array([incumbent[c.flight_id] == c for c in columns], dtype=float)
    if master.backend.name == "gurobi":
        import gurobipy as gp

        with gp.Model("colgen-lns-repair") as model:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.Seed = master.seed
            model.Params.MIPGap = 0.0
            variables = [model.addVar(vtype=gp.GRB.BINARY, obj=float(cost)) for cost in costs]
            for indices in by_flight.values():
                model.addConstr(gp.LinExpr([1.0] * len(indices),
                                          [variables[i] for i in indices]) == 1)
            for indices, cap in patterns.items():
                model.addConstr(gp.LinExpr([1.0] * len(indices),
                                          [variables[i] for i in indices]) <= cap)
            for variable, value in zip(variables, warm):
                variable.Start = float(value)
            if time.monotonic() >= deadline:
                info["status"] = "construction_deadline"
                return incumbent, info
            model.Params.TimeLimit = max(1e-6, deadline - time.monotonic())
            model.optimize()
            info["status"] = str(model.Status)
            x = np.array([v.X for v in variables]) if model.SolCount else None
    else:
        rows, cols, upper, lower = [], [], [], []
        for indices in by_flight.values():
            rows.extend([len(upper)] * len(indices))
            cols.extend(indices)
            upper.append(1.0)
            lower.append(1.0)
        for indices, cap in patterns.items():
            rows.extend([len(upper)] * len(indices))
            cols.extend(indices)
            upper.append(float(cap))
            lower.append(-math.inf)
        matrix = csc_matrix((np.ones(len(rows)), (rows, cols)),
                            shape=(len(upper), len(columns)))
        if time.monotonic() >= deadline:
            info["status"] = "construction_deadline"
            return incumbent, info
        result = milp(costs, integrality=np.ones(len(columns)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": max(1e-6, deadline - time.monotonic()),
                               "mip_rel_gap": 0.0})
        info["status"] = str(result.status)
        x = result.x
    if x is None or not np.all(np.isfinite(x)):
        return incumbent, info
    candidate = dict(incumbent)
    for fid, indices in by_flight.items():
        selected = [i for i in indices if x[i] > 0.5]
        if len(selected) != 1:
            return incumbent, info
        candidate[fid] = columns[selected[0]]
    if (master.objective_of(candidate) > master.objective_of(incumbent) + 1e-9
            and master.is_claim_feasible(candidate)):
        info["improved"] = True
        info["n_swapped"] = sum(candidate[fid] != incumbent[fid] for fid in ranked)
        return candidate, info
    return incumbent, info
