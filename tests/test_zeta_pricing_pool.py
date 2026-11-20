"""Zeta's persistent pricing workers change work, never answers."""
from __future__ import annotations

import os
import signal
import time
from dataclasses import fields, replace

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen.network import RowIndex, RowKey, StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.translate import Column
from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions, price_zeta
from freespace_sim.types import FlightRequest, vec


def _cfg(**overrides) -> SimConfig:
    values = {
        "planner": "colgen",
        "flight_levels_m": (100.0,),
        "airspace_ceiling_m": 125.0,
        "region_size_m": (20_000.0, 20_000.0),
        "terminal_airspace_always_active": True,
        "max_ground_delay_s": 48.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _request(flight_id: int, origin, destination, cfg: SimConfig) -> FlightRequest:
    def point(cell):
        x, y = hg.hex_center(*cell, hg.circumradius(cfg))
        return vec(x, y, cfg.ground_level_m)

    return FlightRequest(flight_id, point(origin), point(destination), 0.0, 0.0)


def _params(**overrides) -> ColGenParams:
    values = {
        "solver": "highs",
        "iteration_ip_time_limit_s": 0.0,
        "lp_gap": 1e-4,
        "max_air_overrun_hops": 0,
        "max_iterations": 3,
        "time_limit_s": 120.0,
        "n_heuristic_tries": 16,
        # PINNED to the sequential loop, deliberately not the shipped default of 4.  These
        # are unit tests on five-flight problems: a pool would spawn four processes that
        # each re-import numba and rebuild every graph, which costs seconds per test and
        # buys nothing.  The tests that mean to exercise the pool ask for it by name.  The
        # SHIPPED default is pinned separately, in `test_experiment_run.py`.
        "n_pricing_workers": 0,
    }
    values.update(overrides)
    return ColGenParams(**values)


def test_zeta_native_preparation_deadline_releases_waiters_and_compilers(monkeypatch):
    """Native preparation must bound lock waits and reap its compiler on timeout."""
    import subprocess
    import sys
    from freespace_sim.planner.zeta import native as zeta_native

    started = time.perf_counter()
    zeta_native._LOAD_LOCK.acquire()
    try:
        with pytest.raises(TimeoutError):
            zeta_native.prepare_compiled_loop(started + 0.03)
    finally:
        zeta_native._LOAD_LOCK.release()
    assert time.perf_counter() - started < 2.0

    children = []
    popen = subprocess.Popen

    def record_child(*args, **kwargs):
        """Retain the process handle so the timeout test can verify child cleanup."""
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record_child)
    with pytest.raises(TimeoutError):
        zeta_native._run([sys.executable, "-c", "import time; time.sleep(30)"],
                         time.perf_counter() + 0.05)
    assert children and all(child.poll() is not None for child in children)


def _zeta_pool_case(
    *, capped: bool = False, early_return: bool = False, compiled_loop: bool = False,
    completion_bounds: tuple[str, ...] = (), shared_cache: bool = False,
    strong_history: bool = False, compiled_history: bool = False,
    history_group_skips: bool = False, route_coverage_cache: bool = False,
    compiled_route_coverage: bool = False, merge_shorter: bool = False,
) -> tuple[dict[int, ZetaGraph], ColGenParams]:
    """Build a finite any-angle domain small enough for repeated exact pool comparisons.

    Parameters
    ------------
    - capped (bool): stop each search at one label to exercise incomplete certificates.
    - early_return (bool): stop on a useful column or a proof that none can improve the master.
    - compiled_loop (bool): run the same search through its compiled orchestration.
    - completion_bounds (tuple[str, ...]): optional certified completion relaxations.
    - shared_cache (bool): share price-independent physical legs and exact timing windows.
    - strong_history (bool): compare differing length sequences with a certified price margin.
    - compiled_history (bool): use the compiled exact comparison helper inside the native loop.
    - history_group_skips (bool): skip broad groups that cannot satisfy strict dominance.
    - route_coverage_cache (bool): reuse complete route coverage with fresh prices.
    - compiled_route_coverage (bool): compile the persistent coverage index methods.
    - merge_shorter (bool): align unequal-length arrivals for guarded comparisons.

    Return
    --------
    - output (tuple): two independent flight graphs and their shared pricing parameters.
    """
    cfg = _cfg(max_ground_delay_s=8.0, max_detour_factor=2.0)
    params = _params(n_pricing_workers=2, max_air_overrun_hops=1)
    catalog = StaticTerminalCatalog((), cfg)
    row_index = RowIndex()
    options = ZetaOptions(backend="compiled", lazy_successors=True,
                          max_labels=1 if capped else 250_000, early_return=early_return,
                          compiled_loop=compiled_loop, completion_bounds=completion_bounds,
                          leg_cache=shared_cache, persistent_timing_cache=shared_cache,
                          history_dominance=strong_history, expire_history=strong_history,
                          strong_history_dominance=strong_history, compiled_history=compiled_history,
                          history_group_skips=history_group_skips,
                          route_coverage_cache=route_coverage_cache,
                          compiled_route_coverage=compiled_route_coverage,
                          merge_shorter_histories=merge_shorter,
                          arrival_snap_s=8.0 if merge_shorter else 0.0,
                          speed_fraction=0.75 if merge_shorter else 0.05)
    cases = [
        (_request(9, (0, 0), (2, 0), cfg), {(0, 0), (1, 0), (1, -1), (2, 0)}),
        (_request(3, (0, 0), (0, 2), cfg), {(0, 0), (0, 1), (-1, 1), (0, 2)}),
    ]
    return {
        request.flight_id: ZetaGraph(
            replace(build_flight_graph(request, cfg, catalog, params),
                    corridor_cells=frozenset(cells), max_air_hops=3),
            cfg, row_index, options,
        )
        for request, cells in cases
    }, params


@pytest.mark.parametrize('capped,early_return,compiled_loop,shared_cache,strong_history,compiled_history,history_group_skips,route_coverage_cache,compiled_route_coverage', [
    (False, False, False, False, False, False, False, False, False),
    (True, False, False, False, False, False, False, False, False),
    (False, True, False, False, False, False, False, False, False),
    (True, True, False, False, False, False, False, False, False),
    (False, True, True, False, False, False, False, False, False),
    (True, False, True, False, False, False, False, False, False),
    (False, False, True, True, False, False, False, False, False),
    (False, True, False, True, False, False, False, False, False),
    (True, False, True, True, False, False, False, False, False),
    (False, False, True, False, True, False, False, False, False),
    (False, True, False, False, True, False, False, False, False),
    (True, False, True, False, True, False, False, False, False),
    (False, False, True, False, True, True, False, False, False),
    (False, True, True, False, True, True, False, False, False),
    (True, False, True, False, True, True, False, False, False),
    (False, False, True, False, True, True, True, False, False),
    (False, True, False, False, True, False, True, False, False),
    (True, False, True, False, True, True, True, False, False),
    (False, False, True, False, True, True, True, True, False),
    (False, True, False, False, False, False, False, True, False),
    (True, False, True, False, False, False, False, True, False),
    (False, False, True, False, True, True, True, True, True),
    (False, True, True, False, False, False, False, True, True),
    (True, False, True, False, False, False, False, True, True),
    (False, True, True, True, True, True, True, True, True),
    (True, True, True, True, True, True, True, True, True),
    (False, False, True, True, "merge", True, True, True, True),
    (True, True, True, True, "merge", True, True, True, True),
    (False, True, False, True, "merge", False, True, True, False),
    (False, False, True, True, "lattice", True, True, True, True),
    (True, True, True, True, "lattice", True, True, True, True),
    (False, False, True, True, "shorter", True, True, True, True),
    (False, False, True, True, "forward_dp", True, True, True, True),
    (True, True, True, True, "forward_dp", True, True, True, True),
    (True, True, False, True, "shorter", False, True, True, False),
    (False, False, True, True, "shorter-indexed", True, True, True, True),
    (False, False, True, True, "lattice-indexed", True, True, True, True),
    (False, False, True, True, "forward_dp-indexed", True, True, True, True),
    (True, True, True, True, "forward_dp-indexed", True, True, True, True),
    (False, False, True, True, "merge-snap", True, True, True, True),
    (False, True, False, True, "merge-snap", False, True, True, False),
    (True, True, True, True, "forward_dp-snap", True, True, True, True),
], ids=['exact-best-column', 'work-cap-best-column', 'exact-early-column', 'work-cap-early-column', 'compiled-loop-early-column', 'compiled-loop-work-cap', 'shared-native-exact', 'shared-compact-early', 'shared-native-cap', 'strong-native-exact', 'strong-compact-early', 'strong-native-cap', 'compiled-history-exact', 'compiled-history-early', 'compiled-history-cap', 'skip-groups-native-exact', 'skip-groups-compact-early', 'skip-groups-native-cap', 'route-coverage-native-exact', 'route-coverage-compact-early', 'route-coverage-native-cap', 'compiled-coverage-exact', 'compiled-coverage-early', 'compiled-coverage-cap', 'all-native-early', 'all-native-cap', 'merged-native-exact', 'merged-native-cap', 'merged-compact-early', 'lattice-native-exact', 'lattice-native-cap', 'shorter-native-exact', 'forward-dp-exact', 'forward-dp-cap', 'shorter-compact-cap', 'indexed-full-exact', 'indexed-radius-exact', 'indexed-forward-dp-exact', 'indexed-forward-dp-cap', 'snap-merged-native-exact', 'snap-merged-compact-early', 'snap-forward-dp-cap'])
@pytest.mark.parametrize("completion_bounds", [(), ("arrival", "resource", "selected", "crossing")])
def test_zeta_persistent_workers_match_serial_with_changed_prices_and_claims(
    capped, early_return, compiled_loop, shared_cache, strong_history, compiled_history,
    history_group_skips, route_coverage_cache, compiled_route_coverage, completion_bounds,
):
    """Parallel seeds and later prices retain serial columns while reusing the same workers."""
    pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    from freespace_sim.planner.colgen.zeta_pool import ZetaPricingPool, build_seed_columns

    price_snap = isinstance(strong_history, str) and strong_history.endswith("-snap")
    if price_snap:
        strong_history = strong_history.removesuffix("-snap")
    native_state_index = isinstance(strong_history, str) and strong_history.endswith("-indexed")
    if native_state_index:
        strong_history = strong_history.removesuffix("-indexed")
    forward_dp = strong_history == "forward_dp"
    merge_equivalent = strong_history == "merge"
    merge_shorter = strong_history == "shorter"
    lattice = strong_history == "lattice"
    strong_history = bool(strong_history)
    graphs, params = _zeta_pool_case(capped=capped, early_return=early_return,
                                     compiled_loop=compiled_loop, completion_bounds=completion_bounds,
                                     shared_cache=shared_cache, strong_history=strong_history,
                                     compiled_history=compiled_history,
                                     history_group_skips=history_group_skips,
                                     route_coverage_cache=route_coverage_cache,
                                     compiled_route_coverage=compiled_route_coverage,
                                     merge_shorter=merge_shorter)
    if forward_dp:
        graphs = {fid: type(graph)(graph.fg, graph.cfg, graph.row_index,
                                  replace(graph.options, traversal="forward_dp", forward_cruise=True,
                                          connection_mode="hex12", lattice_distance_bound=True,
                                          merge_equivalent_histories=True, merge_shorter_histories=True))
                  for fid, graph in graphs.items()}
    if merge_equivalent:
        for graph in graphs.values():
            graph.options = replace(graph.options, merge_equivalent_histories=True)
    if lattice:
        graphs = {fid: type(graph)(graph.fg, graph.cfg, graph.row_index,
                                  replace(graph.options, lattice_distance_bound=True,
                                          connection_radius_m=2.0*graph.pitch))
                  for fid, graph in graphs.items()}
    if native_state_index:
        for graph in graphs.values():
            graph.options = replace(graph.options, native_state_index=True)
    if price_snap:
        for graph in graphs.values():
            graph.options = replace(graph.options, price_snap=True)
    serial_graphs = {
        fid: type(graph)(graph.fg, graph.cfg, graph.row_index,
                         replace(graph.options, native_state_index=False))
        for fid, graph in graphs.items()
    } if native_state_index else graphs
    nominal = {fid: graph.seed_columns(params) for fid, graph in serial_graphs.items()}
    supplied = {
        fid: (replace(columns[0], departure_step=1, delay_s=-100.0, claims=frozenset()),) * 2
        for fid, columns in nominal.items()
    }
    expected_seeds = tuple(
        (fid, build_seed_columns(serial_graphs[fid], params, supplied[fid])) for fid in sorted(graphs)
    )
    known = {fid: result.columns for fid, result in expected_seeds}
    rows = sorted({row for columns in known.values() for column in columns for row in column.claims})
    prices = {row: (-0.25, 0.5, 3.0)[index % 3] for index, row in enumerate(rows)}
    if price_snap:
        # LP-like noise on every fourth row; workers must withhold exactly what serial does.
        prices = {row: (2e-9 if index % 4 == 0 else value)
                  for index, (row, value) in enumerate(prices.items())}
    block = max(known[3][0].claims)
    flights = {3: 7.25, 9: 19.5}
    if early_return:
        known = {fid: () for fid in graphs}
    sweeps = []
    returned_early = False
    proved_no_improvement = False
    inputs = [
        (prices, frozenset(), flights),
        ({row: -2.0 * value for row, value in prices.items()}, frozenset({block}), flights),
        (prices, frozenset(), {fid: params.M + 1_000_000.0 for fid in graphs}),
    ]
    shorter_prices = {RowKey.cell(10_000, 10_000, 0, 0): -100.0}
    if merge_shorter and not capped:
        inputs.append((shorter_prices, frozenset(), flights))
    with ZetaPricingPool(graphs, params) as pool:
        seeded = pool.run_seed_sweep(supplied)
        assert seeded.complete and seeded.failure is None
        assert seeded.results == expected_seeds
        assert all(result.complete and result.provided_primary in result.columns
                   for _fid, result in seeded.results)
        seed_identity = {(r["flight_id"], r["worker"], r["pid"])
                         for r in seeded.flight_records}
        blocked_seeds = pool.run_seed_sweep(supplied, frozenset({block}))
        assert blocked_seeds.complete and blocked_seeds.failure is None
        assert blocked_seeds.results == tuple(
            (fid, build_seed_columns(serial_graphs[fid], params, supplied[fid], frozenset({block})))
            for fid in sorted(graphs)
        )
        assert all(column.claims.isdisjoint({block}) for _fid, result in blocked_seeds.results
                   for column in result.columns)
        for duals, forbidden, flight_duals in inputs:
            expected = tuple(
                (fid, price_zeta(serial_graphs[fid], duals, flight_duals[fid], params,
                                 known_columns=known[fid], forbidden_rows=forbidden))
                for fid in sorted(graphs)
            )
            sweep = pool.run_sweep(duals, flight_duals, known, forbidden)
            sweeps.append(sweep)
            assert sweep.complete and sweep.failure is None
            assert tuple(fid for fid, _result in sweep.results) == tuple(sorted(graphs))
            counters = {field.name: 0 for field in fields(expected[0][1])
                        if field.name in {"wall_s", "bound_build_s",
                                          "topology_setup_s", "incumbent_setup_s"}
                        or field.name.startswith(("shared_leg_cache_", "shared_footprint_cache_",
                                                  "persistent_timing_cache_"))}
            assert tuple((fid, replace(result, **counters)) for fid, result in sweep.results) == tuple(
                (fid, replace(result, **counters)) for fid, result in expected
            )
            if capped or not early_return or flight_duals is not flights:
                assert all(result.complete is not capped
                           or ((merge_equivalent or merge_shorter) and result.reason == "merge_uncertainty"
                               and result.merge_error_bound > 0.0 and not result.complete)
                           for _fid, result in sweep.results)
            if price_snap and duals is not shorter_prices:
                assert all(result.snapped_rows > 0 for _fid, result in sweep.results)
            if duals is shorter_prices and not completion_bounds:
                assert any(result.shorter_histories_merged > 0 for _fid, result in sweep.results)
            for fid, result in sweep.results:
                if result.reason == "improving_column":
                    returned_early = True
                    assert early_return and not result.complete
                    assert result.column not in known[fid]
                    assert result.reduced_cost > params.epsilon
                if result.reason == "no_improvement":
                    proved_no_improvement = True
                    assert early_return and result.complete
                    assert result.reduced_cost <= result.upper_bound <= params.epsilon
                if result.column is not None:
                    assert result.column.claims.isdisjoint(forbidden)
                    known[fid] = (result.column,)
            assert sweep.task_s == pytest.approx(sum(r["task_s"] for r in sweep.flight_records))
        processes = [channel.proc for channel in pool._channels or ()]
        assert seeded.setup_s > 0.0
        assert blocked_seeds.setup_s == 0.0 and all(s.setup_s == 0.0 for s in sweeps)
        identities = [{(r["flight_id"], r["worker"], r["pid"]) for r in sweep.flight_records}
                      for sweep in sweeps]
        assert all(identity == seed_identity for identity in identities)
        assert len({pid for _fid, _worker, pid in identities[0]}) == 2
        assert os.getpid() not in {pid for _fid, _worker, pid in identities[0]}
    assert all(not process.is_alive() for process in processes)
    assert returned_early is early_return
    assert proved_no_improvement is (early_return and not capped)


@pytest.mark.parametrize("started,compiled_loop", [
    (False, False), (True, False), (False, True), (True, True),
], ids=["before-start", "after-start", "compiled-before-start", "compiled-after-start"])
@pytest.mark.parametrize("phase", ["seed", "price"])
def test_zeta_expired_sweep_cannot_certify_or_leave_workers_running(started, compiled_loop, phase):
    """Deadlines preserve received results without certifying or abandoning missing work."""
    from freespace_sim.planner.colgen.zeta_pool import ZetaPricingPool, build_seed_columns

    if compiled_loop:
        pytest.importorskip("Cython")
    graphs, params = _zeta_pool_case(compiled_loop=compiled_loop)
    with ZetaPricingPool(graphs, params) as pool:
        if started:
            pool.start()
        processes = [channel.proc for channel in pool._channels or ()]
        expected = ()
        if started:
            # Stop one live worker so the other returns before the in-flight deadline.
            # Process teardown must also reap a child that cannot read its stop message.
            os.kill(processes[0].pid, signal.SIGSTOP)
            fid = next(fid for fid, owner in pool.worker_of.items() if owner == 1)
            expected = ((fid, build_seed_columns(graphs[fid], params) if phase == "seed"
                         else price_zeta(graphs[fid], {}, 0.0, params)),)
        deadline = time.monotonic() + (3.0 if started else -1.0)
        expired = (pool.run_seed_sweep({}, deadline=deadline) if phase == "seed" else
                   pool.run_sweep({}, {3: 0.0, 9: 0.0}, {}, deadline=deadline))
        assert not expired.complete and expired.failure == "deadline"
        if phase == "seed":
            assert expired.results == expected
        else:
            # Parent and worker elapsed times differ; every result and work counter must match.
            timers = dict(wall_s=0.0, bound_build_s=0.0,
                          topology_setup_s=0.0, incumbent_setup_s=0.0)
            assert tuple((fid, replace(result, **timers)) for fid, result in expired.results) == tuple(
                (fid, replace(result, **timers)) for fid, result in expected
            )
        assert all(not process.is_alive() for process in processes)
        with pytest.raises(RuntimeError):
            pool.run_sweep({}, {3: 0.0, 9: 0.0}, {})
    pool.close()


@pytest.mark.parametrize("phase", ["seed", "price"])
def test_zeta_worker_death_returns_incomplete_and_cleans_up_without_waiting_for_deadline(phase):
    """A missing worker must never become a complete pricing certificate or hang a solve."""
    from freespace_sim.planner.colgen.zeta_pool import ZetaPricingPool

    graphs, params = _zeta_pool_case()
    with ZetaPricingPool(graphs, params) as pool:
        pool.start()
        processes = [channel.proc for channel in pool._channels]
        processes[0].kill()
        processes[0].join(timeout=5.0)
        started = time.monotonic()
        result = (pool.run_seed_sweep({}, deadline=started + 30.0) if phase == "seed" else
                  pool.run_sweep({}, {3: 0.0, 9: 0.0}, {}, deadline=started + 30.0))
        assert time.monotonic() - started < 10.0
        assert not result.complete and result.failure is not None
        assert all(not process.is_alive() for process in processes)
        with pytest.raises(RuntimeError):
            pool.run_sweep({}, {3: 0.0, 9: 0.0}, {})


@pytest.mark.parametrize("connection_mode,radius", [
    ("hex12", 480.0), ("hex12_long", 480.0), ("full", 240.0),
])
def test_zeta_supplied_full_seeds_require_explicit_authority_without_widening_pricing(
    connection_mode, radius,
):
    """A shared fixed seed keeps its full claims but never expands restricted search scope."""
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.zeta.geometry import reservation_claims
    from freespace_sim.planner.colgen.zeta_pool import build_seed_columns

    cfg = _cfg(max_ground_delay_s=0.0, dt_s=4.0, nominal_speed_mps=30.0)
    params = _params(seed_nominal_routes=False)
    request = _request(13, (0, 0), (4, 0), cfg)
    fg = replace(build_flight_graph(request, cfg, [], params),
                 corridor_cells=frozenset({(0, 0), (4, 0)}), max_air_hops=4)
    row_index = RowIndex()
    options = ZetaOptions(connection_mode=connection_mode, connection_radius_m=radius,
                          arrival_snap_s=1.0)
    graph = ZetaGraph(fg, cfg, row_index, options)
    full = ZetaGraph(fg, cfg, row_index, replace(options, connection_mode="full",
                                              connection_radius_m=480.0))
    raw = Column(request.flight_id, 0, 0, None, None, ((0, 0), (4, 0)), -100.0)
    expected = full.canonical_column(raw, cost_model(cfg, params))
    assert expected is not None
    assert expected.claims == reservation_claims(full.intent(expected).volumes, cfg,
                                                row_index=row_index)
    assert build_seed_columns(graph, params, (raw,)).columns == ()
    enabled = ZetaGraph(fg, cfg, row_index, replace(options, seed_validation_radius_m=480.0))
    seeds = build_seed_columns(enabled, params, (raw, raw))
    assert seeds.complete and seeds.columns == (expected,) and seeds.provided_primary == expected
    assert build_seed_columns(enabled, params, (raw,), frozenset({next(iter(expected.claims))})).columns == ()
    assert enabled.canonical_column(raw, cost_model(cfg, params)) is None
    assert build_seed_columns(enabled, replace(params, seed_nominal_routes=True)).columns == ()
    priced = price_zeta(enabled, {}, 0.0, params, known_columns=seeds.columns)
    assert priced.complete and priced.column is None and priced.reason == "infeasible"
    for invalid in (options.connection_radius_m - 1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="seed_validation_radius_m"):
            replace(options, seed_validation_radius_m=invalid)
