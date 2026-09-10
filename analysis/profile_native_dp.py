"""Opt-in native DP counters; production dispatchers and their signatures stay untouched.

The instrumented twin is generated from the loaded kernel, so the same tool can profile a
frozen checkout via PYTHONPATH. Counters measure work, not elapsed time by category. Only
whole native calls are timed. Compilation and instrumented timings are reported separately;
neither is a production runtime estimate. No global patch survives the context manager.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import time

import numpy as np


class NativeProfile:
    """Generate an analysis-only Numba twin with explicit mutable counter arguments."""

    def __init__(self, kernel):
        self.kernel = kernel
        source = Path(kernel.__file__).read_text()
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                         and d.func.id == 'njit' for d in n.decorator_list)}
        # Only copy the reachable native call graph; unrelated compiled searches remain out.
        reachable = set()
        def collect(name):
            if name in reachable:
                return
            reachable.add(name)
            for node in ast.walk(funcs[name]):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                    collect(node.func.id)
        collect('_price_dag')
        names = ['calls' + name for name in sorted(reachable)] + [
            'hash_probes', 'hash_matches', 'hash_empty', 'dominance_updates',
            'dominance_rejects', 'path_nodes_materialized', 'path_elements_compared',
            'labels_created', 'root_labels_created']
        self.names = names
        self.counts = np.zeros(len(names), np.int64)
        indices = {name: i for i, name in enumerate(names)}
        def bump(name):
            return ast.parse(f'_profile_counts[{indices[name]}] += 1').body[0]

        class Instrument(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id in reachable:
                    node.args.append(ast.Name(id='_profile_counts', ctx=ast.Load()))
                return node

            def visit_For(self, node):
                self.generic_visit(node)
                if self.function == '_state_find' and isinstance(node.target, ast.Name) and node.target.id == '_probe':
                    node.body.insert(0, bump('hash_probes'))
                if self.function == '_path_cmp':
                    node.body.insert(0, bump('path_elements_compared'))
                return node

            def visit_While(self, node):
                self.generic_visit(node)
                if self.function == '_fill_path':
                    node.body.insert(0, bump('path_nodes_materialized'))
                return node

            def visit_AugAssign(self, node):
                self.generic_visit(node)
                if isinstance(node.target, ast.Name) and node.target.id == 'n_labels':
                    return [node, bump('root_labels_created' if self.function == '_seed_layer' else 'labels_created')]
                return node

            def visit_Return(self, node):
                self.generic_visit(node)
                if self.function == '_state_find' and isinstance(node.value, ast.Tuple):
                    flag = node.value.elts[1]
                    if isinstance(flag, ast.Constant) and flag.value is True:
                        return [bump('hash_matches'), node]
                    if isinstance(node.value.elts[0], ast.Name):
                        return [bump('hash_empty'), node]
                if self.function == '_prefer':
                    # Evaluate comparisons once, preserving epsilon/tie behavior exactly.
                    assignment = ast.Assign(targets=[ast.Name(id='_profile_preferred', ctx=ast.Store())], value=node.value)
                    branch = ast.If(test=ast.Name(id='_profile_preferred', ctx=ast.Load()),
                                    body=[bump('dominance_updates')], orelse=[bump('dominance_rejects')])
                    return [assignment, branch, ast.Return(value=ast.Name(id='_profile_preferred', ctx=ast.Load()))]
                return node

        definitions = []
        transformer = Instrument()
        for name, fn in funcs.items():
            if name not in reachable:
                continue
            transformer.function = name
            fn = transformer.visit(fn)
            fn.args.args.append(ast.arg(arg='_profile_counts'))
            fn.body.insert(1, bump('calls' + name))
            for decorator in fn.decorator_list:
                for keyword in decorator.keywords:
                    if keyword.arg == 'cache':
                        keyword.value = ast.Constant(False)
            definitions.append(fn)
        generated = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        namespace = vars(kernel).copy()
        exec(compile(generated, '<native-dp-profile>', 'exec'), namespace)
        self.functions = {name: namespace[name] for name in reachable}
        self.parameters = list(inspect.signature(kernel._price_dag.py_func).parameters)
        self.native_s = 0.0
        self.compile_s = 0.0
        self.max_labels = 0
        self.label_capacity_bytes = 0
        self.state_capacity_bytes = 0

    def run(self, *args):
        """Run the twin, counting all retries/resumes and tracking peak arena sizes."""
        dispatcher = self.functions['_price_dag']
        if not dispatcher.signatures:
            from numba import typeof
            started = time.perf_counter()
            dispatcher.compile(tuple(typeof(a) for a in (*args, self.counts)))
            self.compile_s += time.perf_counter() - started
        started = time.perf_counter()
        status = dispatcher(*args, self.counts)
        self.native_s += time.perf_counter() - started
        arguments = dict(zip(self.parameters, args, strict=True))
        self.max_labels = max(self.max_labels, int(arguments['out_counts'][0]))
        self.label_capacity_bytes = max(self.label_capacity_bytes, sum(
            value.nbytes for name, value in arguments.items() if name.startswith('label_')))
        self.state_capacity_bytes = max(self.state_capacity_bytes, sum(
            value.nbytes for name, value in arguments.items()
            if name.startswith('tbl_') or name in ('layer_items', 'layer_buffer')))
        return status

    @contextmanager
    def installed(self):
        """Temporarily instrument this process; use only in sequential analysis runs."""
        original = self.kernel._price_dag
        self.kernel._price_dag = self.run
        try:
            yield self
        finally:
            self.kernel._price_dag = original

    def report(self):
        """Separate work counters, memory estimates, compilation, and native timing."""
        return dict(source=str(self.kernel.__file__), source_sha256=self.source_sha256,
                    counters=dict(zip(self.names, map(int, self.counts), strict=True)),
                    instrumented_native_s=self.native_s, compilation_s=self.compile_s,
                    peak_labels=self.max_labels, label_used_bytes=self.max_labels * 40,
                    label_capacity_bytes=self.label_capacity_bytes,
                    state_capacity_bytes=self.state_capacity_bytes,
                    note='Counters include retries. Memory is array capacity/touched label estimate, not RSS. Category counts are not category timings.')


def main():
    """Replay trusted local first-sweep captures sequentially against loaded source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--flights', type=int, nargs='+', default=[934, 2692, 1520, 1600])
    args = parser.parse_args()
    from freespace_sim.planner.colgen import dp_kernel, pricing
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    with args.capture.open('rb') as handle:
        captured = pickle.load(handle)  # Trusted local capture supplied explicitly by operator.
    cfg, params = captured['cfg'], captured['params']
    catalog = StaticTerminalCatalog(captured['static_terminals'], cfg)
    dp_kernel.warm_kernel()
    profile = NativeProfile(dp_kernel)
    results = []
    for flight in args.flights:
        request = next(r for r in captured['requests'] if r.flight_id == flight)
        graph = build_flight_graph(request, cfg, catalog, params)
        view = pricing.DualView(captured['duals'], cfg)
        def run():
            return pricing.price_flight(graph, view, captured['flight_duals'][flight], cfg,
                                        params, known_column=captured['known_columns'].get(flight))
        started = time.perf_counter()
        reference = run()
        baseline_s = time.perf_counter() - started
        profile.counts.fill(0)
        profile.native_s = profile.compile_s = 0.0
        profile.max_labels = profile.label_capacity_bytes = profile.state_capacity_bytes = 0
        with profile.installed():
            observed = run()
        assert observed == reference, f'Instrumented parity failed: {flight}'
        native_s = 0.0
        original = dp_kernel._price_dag
        def timed_native(*native_args):
            nonlocal native_s
            started = time.perf_counter()
            try:
                return original(*native_args)
            finally:
                native_s += time.perf_counter() - started
        dp_kernel._price_dag = timed_native
        started = time.perf_counter()
        try:
            warm_reference = run()
        finally:
            dp_kernel._price_dag = original
        warm_s = time.perf_counter() - started
        assert warm_reference == reference, f'Warm baseline parity failed: {flight}'
        record = dict(flight_id=flight, cold_baseline_wall_s=baseline_s,
                      warm_baseline_wall_s=warm_s, warm_baseline_native_s=native_s,
                      parity=True, **profile.report())
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
