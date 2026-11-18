"""Standalone Zeta files flights in order and reads the current reservation ledger."""

from dataclasses import replace
import io
import pickle
import subprocess
import sys

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.route_domain import RowIndex
from freespace_sim.planner.zeta.geometry import reservation_claims
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


@pytest.mark.parametrize("domain", ["full480", "radius240", "radius240-native", "hex12", "forward12-native"])
def test_zeta_fcfs_files_buffered_routes_in_request_order(domain: str) -> None:
    first = FlightRequest(1, vec(0, 0, 0), vec(480, 0, 0), 0.0)
    second = replace(first, flight_id=2)
    cfg = _config()
    params = _params()
    if domain != "full480":
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
from freespace_sim.planner.zeta import ZetaPlannerParams
from freespace_sim.types import FlightRequest, vec
from experiments.zeta_prototype import _run_fcfs
cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0, max_ground_delay_s=0.0)
planner = get_planner('zeta', ZetaPlannerParams(max_air_overrun_hops=0))
intent = planner.plan(FlightRequest(1, vec(0,0,0), vec(480,0,0), 0.0), ReservationLedger(cfg), cfg)
assert intent.accepted
assert planner.inner.last_result.complete
routes, intents, stats = _run_fcfs(cfg, [intent.request], (), planner.inner.params)
assert len(routes) == len(intents) == 1 and stats['violated_rows'] == 0
assert not any(name.startswith('freespace_sim.planner.colgen') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_zeta_archive_migration_preserves_graph_routes_and_claims() -> None:
    """Old saved graphs require explicit migration and retain their canonical route model."""
    from experiments.zeta_migrate import load_archive
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.zeta import ZetaGraph, price_zeta
    from freespace_sim.planner.route_domain import build_flight_graph

    cfg = _config(max_ground_delay_s=0.0)
    params = ColGenParams(max_air_overrun_hops=0)
    req = FlightRequest(1, vec(0, 0, 0), vec(480, 0, 0), 0.0)
    graph = ZetaGraph(build_flight_graph(req, cfg, (), params), cfg, RowIndex())
    seeds = graph.seed_columns(params)
    original = price_zeta(graph, {}, 0.0, params, known_columns=seeds)
    current = pickle.dumps((graph, params, seeds), protocol=2)
    modern = b"freespace_sim.planner.zeta.route_cache\n"
    legacy = b"freespace_sim.planner.colgen.zeta_route_cache\n"
    assert modern in current
    archived = current.replace(modern, legacy)
    with pytest.raises(ModuleNotFoundError, match="zeta_route_cache"):
        pickle.loads(archived)
    restored, restored_params, restored_seeds = load_archive(io.BytesIO(archived))
    assert restored.seed_columns(restored_params) == seeds == restored_seeds
    migrated = pickle.dumps((restored, restored_params, restored_seeds), protocol=2)
    assert legacy not in migrated
    restored, restored_params, restored_seeds = pickle.loads(migrated)
    result = price_zeta(restored, {}, 0.0, restored_params, known_columns=restored_seeds)
    assert result.complete and result.column == original.column
    assert result.column.claims == original.column.claims
    assert result.reduced_cost == original.reduced_cost


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
