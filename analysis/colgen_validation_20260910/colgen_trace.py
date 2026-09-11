"""Compact, benchmark-only provenance and diversity tracing for actual CG solves.

Store each route path once and refer to columns by the master's stable integer
indices. Never serialize capacity claims or the live LP model. The observer does
not add columns or change selections, duals, or solver budgets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import time


def write(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def identity(column):
    return (column.flight_id, column.level, column.origin_lane_idx,
            column.dest_lane_idx, column.cell_path)


class ColumnTrace:
    def __init__(self, directory):
        self.out = Path(directory)
        if (self.out / "columns.jsonl").exists():
            raise FileExistsError("Use a fresh output directory for a column trace")
        self.master = None
        self.columns, self.paths = [], []
        self.path_ids, self.geometry_ids = {}, {}
        self.families_by_flight = defaultdict(set)
        self.phase = "initialization"
        self.sweep = 0
        self.rc = {}
        self.lns_calls, self.ip_calls, self.iterations = [], [], []
        self.initial_selection, self.final_selection = {}, {}
        self.initial_available_s = None
        self.rounding_calls = 0
        self.started = time.perf_counter()
        self.logged_columns = self.logged_paths = 0
        self.solutions = set()

    def selection(self, master, selected):
        return {int(fid): master._column_indices[column] for fid, column in selected.items()}

    @staticmethod
    def signature(selected):
        return hashlib.sha256(json.dumps(sorted(selected.items())).encode()).hexdigest()

    def install(self):
        import freespace_sim.planner.colgen.solver as solver
        from freespace_sim.planner.colgen.master import RestrictedMaster

        original_add = RestrictedMaster.add_column
        original_lns = RestrictedMaster.lns_heuristic
        original_round = RestrictedMaster.round_heuristic
        original_ip = RestrictedMaster.solve_ip
        original_set = RestrictedMaster.set_heuristic
        original_sweep = solver.price_sweep

        def add(master, column):
            index = original_add(master, column)
            if self.master is None:
                self.master = master
            assert self.master is master, "Trace expects one measured batch"
            if index < len(self.columns):
                return index
            assert index == len(self.columns)
            key = identity(column)
            geometry = (*key[:4], tuple(cell for cell, _ in groupby(column.cell_path)))
            new_geometry = geometry not in self.geometry_ids
            geometry_id = self.geometry_ids.setdefault(geometry, len(self.geometry_ids))
            self.families_by_flight[column.flight_id].add(geometry_id)
            if key not in self.path_ids:
                path_id = len(self.paths)
                self.path_ids[key] = path_id
                self.paths.append(dict(id=path_id, flight_id=column.flight_id,
                                       level=column.level, origin_lane_idx=column.origin_lane_idx,
                                       dest_lane_idx=column.dest_lane_idx, cells=column.cell_path,
                                       geometry_id=geometry_id))
            self.columns.append(dict(
                index=index, flight_id=column.flight_id, departure_step=column.departure_step,
                path_id=self.path_ids[key], geometry_id=geometry_id, cost=column.delay_s,
                phase=self.phase, sweep=self.sweep, new_geometry=new_geometry,
                reduced_cost=self.rc.get((column.departure_step, key)),
            ))
            return index

        def set_heuristic(master, selected):
            result = original_set(master, selected)
            ids = self.selection(master, selected)
            if self.initial_available_s is None:
                self.initial_selection = ids
                self.initial_available_s = time.perf_counter() - self.started
            self.final_selection = ids
            return result

        def sweep(*args, **kwargs):
            self.sweep += 1
            self.phase = "pricing"
            self.rc = {}
            result = original_sweep(*args, **kwargs)
            self.rc = {(column.departure_step, identity(column)): float(rc)
                       for column, rc in zip(result.columns, result.reduced_costs)
                       if column is not None}
            return result

        def lns(master, x, rng, incumbent, *args, **kwargs):
            before = self.selection(master, incumbent)
            n_columns = len(master._columns)
            started = time.perf_counter()
            result = original_lns(master, x, rng, incumbent, *args, **kwargs)
            elapsed = time.perf_counter() - started
            assert len(master._columns) == n_columns, "LNS unexpectedly generated columns"
            after = self.selection(master, result)
            changed = [dict(flight_id=fid, before=before[fid], after=after[fid])
                       for fid in after if before[fid] != after[fid]]
            signature = self.signature(after)
            self.solutions.add(signature)
            info = dict(call=len(self.lns_calls) + 1, elapsed_s=elapsed,
                        simulation_elapsed_s=time.perf_counter() - self.started,
                        before_cost=sum(self.columns[i]["cost"] for i in before.values()),
                        after_cost=sum(self.columns[i]["cost"] for i in after.values()),
                        covered=len(after), changed=changed, signature=signature,
                        lp_support_columns=sum(float(value) > master.params.epsilon for value in x),
                        geometry_changes=sum(self.columns[c["before"]]["geometry_id"] !=
                                             self.columns[c["after"]]["geometry_id"] for c in changed),
                        stats=dict(master.last_round_stats))
            self.lns_calls.append(info)
            self.flush()
            print(f"LNS {info['call']} cost={info['after_cost']:.3f} "
                  f"changed={len(changed)} geometry_changes={info['geometry_changes']} "
                  f"unique_trials={info['stats']['n_unique_trial_solutions']} "
                  f"wall={elapsed:.3f}s", flush=True)
            return result

        def rounding(master, *args, **kwargs):
            self.rounding_calls += 1
            return original_round(master, *args, **kwargs)

        def ip(master, *args, **kwargs):
            phase = "iteration" if "eager" in kwargs else "final"
            label = f"ITERATION {self.sweep}" if phase == "iteration" else "FINAL"
            before = self.selection(master, master._heuristic_selection)
            info = dict(phase=phase, sweep=self.sweep,
                        start_elapsed_s=time.perf_counter() - self.started,
                        before_selection=before, before_cost=sum(self.columns[i]["cost"]
                                                               for i in before.values()),
                        native_calls=[], requested_budget_s=kwargs.get("budget_s"))
            original_native = master.backend.solve_ip

            def native(warm_start=None):
                call = dict(time_limit_s=master.backend.time_limit_s)
                started = time.perf_counter()
                result = original_native(warm_start)
                call.update(wall_s=time.perf_counter() - started, status=result.status,
                            objective=result.objective, upper_bound=result.upper_bound)
                if master.backend.name == "gurobi":
                    model = master.backend._model
                    call.update(gurobi_runtime_s=float(model.Runtime), nodes=float(model.NodeCount),
                                solution_count=int(model.SolCount))
                info["native_calls"].append(call)
                return result

            master.backend.solve_ip = native
            if master.backend.name == "gurobi":
                log_name = (f"iteration_ip_{self.sweep:03d}_gurobi.log"
                            if phase == "iteration" else "ip_gurobi.log")
                master.backend._model.Params.LogFile = str(self.out / log_name)
                master.backend._model.Params.LogToConsole = 0
                master.backend._model.Params.OutputFlag = 1
            write(self.out / "ip_started.json", info)
            print(f"{label} IP starting: search cap={kwargs.get('budget_s')}s "
                  f"incumbent_route_cost={info['before_cost']:.3f} "
                  f"covered={len(before)}/{len(master.flight_ids)}", flush=True)
            started = time.perf_counter()
            try:
                result = original_ip(master, *args, **kwargs)
            finally:
                master.backend.solve_ip = original_native
            after = self.selection(master, result)
            if phase == "final":
                self.final_selection = after
            info.update(wall_s=time.perf_counter() - started, setup_s=master.last_ip_setup_s,
                        after_selection=after,
                        after_cost=sum(self.columns[i]["cost"] for i in after.values()),
                        status=master.last_ip_status, optimal=master.last_ip_optimal,
                        incumbent_trajectory=master.last_ip_trajectory)
            # Route cost alone understates the objective when a timed IP leaves flights out.
            info.update(before_covered=len(before), after_covered=len(after),
                        before_penalized_cost=info['before_cost'] + master.params.M * (len(master.flight_ids) - len(before)),
                        after_penalized_cost=info['after_cost'] + master.params.M * (len(master.flight_ids) - len(after)))
            self.ip_calls.append(info)
            self.flush()
            print(f"{label} IP ended: {info['status']} penalized_cost={info['after_penalized_cost']:.3f} "
                  f"route_cost={info['after_cost']:.3f} covered={len(after)}/{len(master.flight_ids)} "
                  f"wall={info['wall_s']:.3f}s setup={info['setup_s']:.3f}s", flush=True)
            return result

        RestrictedMaster.add_column = add
        RestrictedMaster.set_heuristic = set_heuristic
        RestrictedMaster.lns_heuristic = lns
        RestrictedMaster.round_heuristic = rounding
        RestrictedMaster.solve_ip = ip
        solver.price_sweep = sweep

    def iteration(self, payload):
        self.flush()
        priced = [c for c in self.columns if c["phase"] == "pricing" and c["sweep"] == self.sweep]
        info = dict(iteration=payload["iteration"], elapsed_s=payload["elapsed_s"],
                    pool_columns=len(self.columns), pool_paths=len(self.paths),
                    pool_spatial_routes=len(self.geometry_ids), priced_columns=len(priced),
                    new_spatial_routes=sum(c["new_geometry"] for c in priced),
                    timing_variants=sum(not c["new_geometry"] for c in priced),
                    flights_with_multiple_spatial_routes=sum(len(ids) > 1
                                                           for ids in self.families_by_flight.values()),
                    lns_published_solutions=len(self.solutions))
        self.iterations.append(info)
        write(self.out / "column_diversity.json", self.iterations)
        print("DIVERSITY " + json.dumps(info), flush=True)

    def flush(self):
        for name, values, start in (("columns.jsonl", self.columns, self.logged_columns),
                                    ("paths.jsonl", self.paths, self.logged_paths)):
            with (self.out / name).open("a") as handle:
                for value in values[start:]:
                    handle.write(json.dumps(value) + "\n")
        self.logged_columns = len(self.columns)
        self.logged_paths = len(self.paths)
        write(self.out / "lns_calls.json", self.lns_calls)
        write(self.out / "ip_calls.json", self.ip_calls)

    def finish(self):
        self.flush()
        if self.master is not None and not any(c["phase"] == "final" for c in self.ip_calls):
            self.final_selection = self.selection(self.master, self.master._heuristic_selection)
        write(self.out / "selections.json", dict(initial=self.initial_selection,
                                                initial_available_s=self.initial_available_s,
                                                final=self.final_selection))
        write(self.out / "trace_summary.json", dict(
            columns=len(self.columns), spatial_routes=len(self.geometry_ids),
            timed_paths=len(self.paths), rounding_calls=self.rounding_calls,
            lns_calls=len(self.lns_calls), unique_published_lns_solutions=len(self.solutions),
            final_ip_calls=sum(call["phase"] == "final" for call in self.ip_calls),
            iteration_ip_calls=sum(call["phase"] == "iteration" for call in self.ip_calls),
            column_origins=dict(Counter(c["phase"] for c in self.columns)),
            heuristic_wall_s=sum(call["elapsed_s"] for call in self.lns_calls),
        ))
