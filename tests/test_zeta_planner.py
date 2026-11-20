"""Standalone Zeta files flights in order and reads the current reservation ledger."""

from dataclasses import replace
import subprocess
import sys
import time

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.route_domain import RowIndex
from freespace_sim.planner.zeta.geometry import reservation_claims
from freespace_sim.planner.zeta import planner as zeta_planner
from freespace_sim.planner.zeta.planner import ZetaPlannerParams
from freespace_sim.planner.zeta.search import ZetaOptions
from freespace_sim.sim import run
from freespace_sim.types import DenialReason, FlightRequest, Terminal, vec
from freespace_sim.volumes import Volume4D, hover_reservation


def _config(**overrides) -> SimConfig:
    """Build a short single-level world for standalone planning."""
    cfg = SimConfig(
        planner="zeta", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        max_ground_delay_s=80.0,
    )
    return replace(cfg, **overrides)


def _params() -> ZetaPlannerParams:
    """Keep the routes small and use the reference backend."""
    return ZetaPlannerParams(
        max_air_overrun_hops=0, options=ZetaOptions(connection_radius_m=480.0, lazy_successors=True),
    )


@pytest.mark.parametrize(
    "domain", ["full480", "radius240", "radius240-native", "hex12", "forward12-native", "default"],
)
def test_zeta_fcfs_files_buffered_routes_in_request_order(domain: str) -> None:
    first = FlightRequest(1, vec(0, 0, 0), vec(480, 0, 0), 0.0)
    second = replace(first, flight_id=2)
    cfg = _config()
    params = _params()
    if domain == "default":
        # The shipped defaults (tuned pruning on the native loop) must plan the same flights.
        pytest.importorskip("numba")
        pytest.importorskip("Cython")
        params = ZetaPlannerParams(max_air_overrun_hops=0)
    elif domain != "full480":
        native = domain.endswith("-native")
        radius240 = domain.startswith("radius240")
        if native:
            pytest.importorskip("numba")
            pytest.importorskip("Cython")
        params = replace(params, options=replace(
            params.options, connection_radius_m=240.0 if radius240 else 480.0,
            connection_mode="full" if radius240 else "hex12",
            backend="compiled" if native else "reference",
            forward_cruise=native and not radius240,
            traversal="forward_dp" if native and not radius240 else "best_first",
            compiled_loop=native, native_state_index=native, early_return=True,
        ))
    result = run(cfg, requests=[first, second], planner_params=params)

    assert result.verified
    assert [intent.request.flight_id for intent in result.accepted] == [1, 2]
    earlier, later = result.accepted
    assert earlier.planner == later.planner == "zeta"
    assert earlier.ground_delay_s == 0.0 < later.ground_delay_s
    assert result.ledger.n_volumes == len(earlier.volumes) + len(later.volumes)
    row_index = RowIndex()
    assert reservation_claims(earlier.volumes, cfg, row_index=row_index).isdisjoint(
        reservation_claims(later.volumes, cfg, row_index=row_index),
    )


def test_zeta_counts_distinct_live_flights_against_terminal_capacity() -> None:
    cfg = _config(terminal_airspace_always_active=True)
    ledger = ReservationLedger(cfg)
    hub = vec(0, 0, 0)
    terminal = Terminal("hub", capacity=2, radius=180.0)
    ledger.register_static_terminal(hub, terminal)
    dwell = hover_reservation(hub, 0.0, cfg, terminal_id=terminal.id, radius=terminal.radius)
    request = FlightRequest(3, hub, vec(720, 0, 0), 0.0, origin_terminal=terminal)
    planner = get_planner("zeta", _params())

    ledger.commit(1, [dwell])
    ledger.commit(1, [dwell])
    has_room = planner.plan(request, ledger, cfg)
    assert has_room.accepted and has_room.request is request
    assert has_room.ground_delay_s == 0.0
    assert has_room.volumes[0].terminal_id == terminal.id
    assert not ledger.any_conflict(has_room.volumes)

    ledger.commit(2, [dwell])
    full = planner.plan(request, ledger, cfg)
    assert full.accepted and full.ground_delay_s > 0.0
    assert full.volumes[0].t_start >= dwell.t_end
    assert ledger.n_volumes == 3

    ledger.release(1)
    restored = planner.plan(request, ledger, cfg)
    assert restored.accepted and restored.ground_delay_s == 0.0
    assert ledger.n_volumes == 1


def test_zeta_routes_around_foreign_walls_registered_in_the_ledger() -> None:
    cfg = _config(terminal_airspace_always_active=True, max_ground_delay_s=0.0)
    request = FlightRequest(1, vec(0, 0, 0), vec(960, 0, 0), 0.0)
    params = replace(_params(), max_air_overrun_hops=4)
    planner = get_planner("zeta", params)
    ledger = ReservationLedger(cfg)
    direct = planner.plan(request, ledger, cfg)
    assert direct.accepted
    ledger.register_static_terminal(vec(480, 0, 0), Terminal("foreign", radius=90.0))
    assert ledger.any_conflict(direct.volumes)

    detour = planner.plan(request, ledger, cfg)
    assert detour.accepted
    assert detour.air_detour_m > direct.air_detour_m
    assert not ledger.any_conflict(detour.volumes)
    assert ledger.n_volumes == 0


def test_zeta_reroutes_a_round_trip_when_its_customer_pad_is_blocked() -> None:
    cfg = _config()
    req = FlightRequest(
        1, vec(0, 0, 0), vec(480, 0, 0), 0.0, return_to_origin=True, turnaround_s=32.0,
    )
    ledger = ReservationLedger(cfg)
    planner = get_planner("zeta", _params())
    clean = planner.plan(req, ledger, cfg)
    assert clean.accepted and len(clean.leg_starts) == 1
    hold = next(
        v for v in clean.volumes
        if isinstance(v.shape, CylinderSpec)
        and v.shape.z_hi == cfg.ground_level_m + cfg.ground_box_height_m
    )
    mid = (hold.t_start + hold.t_end) / 2.0
    ledger.commit(2, [Volume4D(hold.shape, mid - 1.0, mid + 1.0)])
    assert ledger.any_conflict([hold])

    shifted = planner.plan(req, ledger, cfg)
    assert shifted.accepted and len(shifted.leg_starts) == 1
    assert shifted.ground_delay_s > clean.ground_delay_s
    assert not ledger.any_conflict(shifted.volumes)
    assert ledger.n_volumes == 1


def test_zeta_rejects_a_return_that_was_already_blocked_before_takeoff() -> None:
    cfg = _config(max_ground_delay_s=0.0)
    ledger = ReservationLedger(cfg)
    req = FlightRequest(
        1, vec(0, 0, 0), vec(480, 0, 0), 0.0, t_departure=40.0, parked_since_s=0.0,
    )
    ledger.commit(2, [hover_reservation(req.origin, 4.0, cfg)])
    planner = get_planner("zeta", _params())

    unparked = planner.plan(replace(req, parked_since_s=None), ledger, cfg)
    parked = planner.plan(req, ledger, cfg)
    assert unparked.accepted
    assert parked.denial_reason is DenialReason.BUDGET_EXCEEDED
    assert not parked.accepted
    assert ledger.n_volumes == 1


def test_standalone_zeta_runs_without_importing_column_generation() -> None:
    """FCFS route planning must work when the CG package is unavailable."""
    code = """
import importlib.abc
import sys
class NoColgen(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('freespace_sim.planner.colgen'):
            raise ImportError('column generation unavailable')
sys.meta_path.insert(0, NoColgen())
from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.zeta import ZetaPlannerParams, tuned_options
from freespace_sim.types import FlightRequest, vec
from freespace_sim.planner.zeta.fcfs import run_fcfs
cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0, max_ground_delay_s=0.0)
planner = get_planner('zeta', ZetaPlannerParams(max_air_overrun_hops=0, options=tuned_options(native=False)))
intent = planner.plan(FlightRequest(1, vec(0,0,0), vec(480,0,0), 0.0), ReservationLedger(cfg), cfg)
assert intent.accepted
assert planner.inner.last_result.complete
routes, intents, stats = run_fcfs(cfg, [intent.request], (), planner.inner.params)
assert len(routes) == len(intents) == 1 and stats['violated_rows'] == 0
record, = stats['search_records']
searched = planner.inner.last_result
assert record['route_cost_lower_bound'] == searched.lower_bound
assert record['merge_error_bound'] == searched.merge_error_bound
assert not any(name.startswith('freespace_sim.planner.colgen') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_zeta_rejects_unwrapped_itinerary_and_reports_preparation_deadline() -> None:
    """The direct single-leg adapter refuses itinerary loss and cannot certify a timeout."""
    from freespace_sim.planner.zeta import ZetaPlanner

    cfg = _config()
    ledger = ReservationLedger(cfg)
    req = FlightRequest(1, vec(0, 0, 0), vec(480, 0, 0), 0.0)
    planner = ZetaPlanner(replace(_params(), time_limit_s=1e-12))
    with pytest.raises((ValueError, NotImplementedError), match="(?i)(itinerary|return|round)"):
        planner.plan(replace(req, return_to_origin=True), ledger, cfg)
    intent = planner.plan(req, ledger, cfg)
    assert not intent.accepted and intent.denial_reason is DenialReason.SEARCH_EXHAUSTED
    assert planner.last_result is not None and not planner.last_result.complete
    assert planner.last_result.reason == "deadline"
    assert planner.last_result.lower_bound == -float("inf")
    assert ledger.n_volumes == 0


@pytest.mark.parametrize("weight", [-1.0, 0.0, float("inf"), float("nan")])
def test_shared_zeta_rejects_weights_that_invalidate_distance_bounds(weight: float) -> None:
    """The public objective cannot admit costs for which its route bounds are unsound."""
    from freespace_sim.planner.route_cost import CostModel
    from freespace_sim.planner.zeta import SearchSettings

    for model in (CostModel(weight, 1.0), CostModel(1.0, weight)):
        with pytest.raises(ValueError, match="finite and positive"):
            SearchSettings(model=model)


def test_zeta_reads_only_reservations_its_routes_could_touch(monkeypatch) -> None:
    """Far-away and long-finished filings change no claimable row, plan or preparation work."""
    from freespace_sim.planner.route_cost import cost_model
    from freespace_sim.planner.route_domain import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.planner.zeta import SearchSettings, ZetaGraph, search
    from freespace_sim.planner.zeta import planner as zeta_planner

    cfg = _config()
    params = _params()
    planner = get_planner("zeta", params)
    request = FlightRequest(9, vec(0, 0, 0), vec(480, 0, 0), 4000.0)

    def file(ledger: ReservationLedger, flight_id: int, dx: float, t_departure: float) -> None:
        """Plan and commit one leg parallel to ``request``, shifted by ``dx`` metres."""
        leg = FlightRequest(flight_id, vec(dx, 0, 0), vec(dx + 480, 0, 0), t_departure)
        intent = planner.plan(leg, ledger, cfg)
        assert intent.accepted
        ledger.commit(flight_id, intent.volumes)

    clean, noisy = ReservationLedger(cfg), ReservationLedger(cfg)
    for ledger in (clean, noisy):
        file(ledger, 1, 0.0, request.t_departure)   # occupies the leg's own corridor
    for k in range(12):
        file(noisy, 100 + k, 20_000.0 + 960.0 * k, request.t_departure)   # far away
        file(noisy, 200 + k, 0.0, 200.0 * k)                              # long finished

    catalog = StaticTerminalCatalog(noisy.static_terminals(), cfg)
    row_index = RowIndex()
    graph = ZetaGraph(build_flight_graph(request, cfg, catalog, params), cfg, row_index,
                      options=params.options)
    full = zeta_planner._saturated_rows(noisy, cfg, catalog, row_index)
    scoped = zeta_planner._saturated_rows(noisy, cfg, catalog, row_index,
                                          scope=zeta_planner._claim_scope(graph))
    assert scoped <= full

    # Record every row the search tests against the forbidden set.
    checked: set = set()
    rows = graph.rows
    graph.rows = lambda volume: checked.update(found := rows(volume)) or found
    settings = SearchSettings(model=cost_model(cfg, params))
    by_full = search(graph, {}, settings, forbidden_rows=full)
    by_scope = search(graph, {}, settings, forbidden_rows=scoped)
    checked |= by_full.route.claims
    assert {row for row in checked if row in full} == {row for row in checked if row in scoped}
    assert any(row in full for row in checked)
    assert (by_scope.route, by_scope.priced_cost, by_scope.expanded, by_scope.generated) == (
        by_full.route, by_full.priced_cost, by_full.expanded, by_full.generated)

    conversions = []
    convert = zeta_planner.volume_claims
    monkeypatch.setattr(zeta_planner, "volume_claims",
                        lambda volume, *args, **kwargs: conversions.append(volume)
                        or convert(volume, *args, **kwargs))
    plans = []
    for ledger in (clean, noisy):
        conversions.clear()
        intent = planner.plan(request, ledger, cfg)
        plans.append((len(conversions), planner.inner.last_result.route, intent.cost,
                      [(point.tolist(), stamp) for point, stamp in intent.centerline]))
    assert plans[0] == plans[1]
    assert plans[1][1].departure_step > graph.fg.base_step
    assert noisy.n_volumes > 4 * plans[1][0]


@pytest.mark.parametrize("setting", [
    "flight_levels_m", "terminal_airspace_always_active", "fixed_exit_lanes",
])
def test_zeta_refuses_unsupported_setups_before_any_work(setting: str, monkeypatch) -> None:
    """Multi-level worlds, and hub legs without permanent walls or fixed lanes, fail fast."""
    from experiments.run import _execute, parse_args
    from freespace_sim.planner.zeta import planner as zeta_planner

    overrides = {"terminal_airspace_always_active": True, "fixed_exit_lanes": True}
    overrides.update({"flight_levels_m": (85.0, 100.0)} if setting == "flight_levels_m"
                     else {setting: False})
    cfg = _config(**overrides)
    hub = Terminal("hub", capacity=2, radius=180.0)
    request = FlightRequest(1, vec(0, 0, 0), vec(720, 0, 0), 0.0, origin_terminal=hub)

    def no_work(*args, **kwargs):
        raise AssertionError("validation must precede graph construction")

    monkeypatch.setattr(zeta_planner, "build_flight_graph", no_work)
    with pytest.raises(NotImplementedError, match=f"zeta planner.*{setting}"):
        get_planner("zeta", _params()).plan(request, ReservationLedger(cfg), cfg)
    if setting == "flight_levels_m":
        with pytest.raises(NotImplementedError, match="zeta planner.*flight_levels_m"):
            _execute(parse_args(["--scenario", "metro_uniform", "--planner", "zeta"]))


def test_zeta_first_native_build_does_not_spend_a_short_leg_budget(monkeypatch) -> None:
    """A first native build that outlasts the leg budget must not deny the leg."""
    pytest.importorskip("numba")
    pytest.importorskip("Cython")
    from freespace_sim.planner.zeta import native

    cfg = _config()
    request = FlightRequest(1, vec(0, 0, 0), vec(480, 0, 0), 0.0)
    params = ZetaPlannerParams(max_air_overrun_hops=0)
    # Warm everything else first, so only the native build stands between the leg and its budget.
    assert get_planner("zeta", params).plan(request, ReservationLedger(cfg), cfg).accepted
    built = native._LOADED

    def slow_build(deadline):
        """Stand in for a cold compile: slower than the budget, with the real deadline check."""
        time.sleep(0.6)
        native._remaining(deadline)
        return built

    monkeypatch.setattr(native, "_LOADED", None)
    monkeypatch.setattr(native, "_load", slow_build)
    planner = get_planner("zeta", replace(params, time_limit_s=0.5))
    assert planner.plan(request, ReservationLedger(cfg), cfg).accepted


def test_zeta_default_names_the_native_extra_when_cython_is_missing(monkeypatch) -> None:
    """The native default fails at construction with an install hint; the opt-out still works."""
    real_find_spec = zeta_planner.importlib.util.find_spec
    monkeypatch.setattr(zeta_planner.importlib.util, "find_spec",
                        lambda name, *a: None if name == "Cython" else real_find_spec(name, *a))
    with pytest.raises(RuntimeError, match="zeta-native"):
        get_planner("zeta", ZetaPlannerParams())
    # Opting out keeps the same pruning without the native build.
    get_planner("zeta", ZetaPlannerParams(options=zeta_planner.tuned_options(native=False)))
