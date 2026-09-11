"""Internal pricing receipts cannot bless foreign objects or unsafe shifted cutoffs."""
from dataclasses import replace
import pickle
import gc
import weakref
import math

import pytest

from freespace_sim.planner.colgen import pricing, pricing_pool
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.translate import column_to_intent
from freespace_sim.config import SimConfig
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.translate import Column
from freespace_sim.volumes import column_dwell_s


def _cfg(**overrides):
    values = dict(planner="colgen", flight_levels_m=(100.,), airspace_ceiling_m=125.,
                  region_size_m=(20_000., 20_000.), terminal_airspace_always_active=True,
                  max_ground_delay_s=48., max_detour_factor=10.)
    values.update(overrides)
    return SimConfig(**values)


def _point(cell, cfg):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _graph(cfg, *, overrun):
    request = FlightRequest(11, _point((0, 0), cfg), _point((4, -1), cfg), 0., 0.)
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun)
    return build_flight_graph(request, cfg, (), params), params


def _context(**overrides):
    cfg = _cfg(**overrides)
    graph, params = _graph(cfg, overrun=2)
    catalog = StaticTerminalCatalog((), cfg)
    seed = pricing.seed_column(graph, cfg, model=cost_model(cfg, params))
    return cfg, params, graph, catalog, seed


def _sweep(columns, *, extras=(), certified_ids=None):
    ids = tuple(c.flight_id for c in columns)
    if certified_ids is None:
        certified_ids = set(ids)
    return pricing_pool.SweepResult(
        ids, tuple(1.0 for _ in ids), tuple(columns), None,
        flight_records=tuple(dict(flight_id=fid, priced=True,
                                  _canonical_pricing_result=fid in certified_ids) for fid in ids),
        extra_columns=tuple(extras),
    )


def test_receipt_binds_owner_objects_and_context_with_extra_columns():
    cfg, params, graph, catalog, seed = _context()
    model = cost_model(cfg, params)
    extra = pricing.certify_column(replace(seed, departure_step=seed.departure_step + 1), graph, cfg, model)
    graphs = {seed.flight_id: graph}
    sweep = _sweep([seed], extras=[extra])
    assert pricing_pool.certified_sweep_columns(sweep, graphs, cfg, params, catalog) == ()
    pricing_pool._receipt(sweep, [graph.request], graphs, cfg, params, catalog)
    assert pricing_pool.certified_sweep_columns(sweep, graphs, cfg, params, catalog) == (seed, extra)
    assert pricing_pool.certified_sweep_columns(replace(sweep), graphs, cfg, params, catalog) == ()
    assert pricing_pool.certified_sweep_columns(sweep, graphs, cfg, replace(params, columns_per_flight=2), catalog) == ()
    rebuilt = build_flight_graph(graph.request, cfg, catalog, params)
    assert pricing_pool.certified_sweep_columns(sweep, {seed.flight_id: rebuilt}, cfg, params, catalog) == ()
    object.__setattr__(sweep, 'extra_columns', (replace(extra, claims=frozenset()),))
    assert pricing_pool.certified_sweep_columns(sweep, graphs, cfg, params, catalog) == ()


def test_uncertified_flight_does_not_disable_other_receipts():
    cfg, params, graph, catalog, seed = _context()
    other_request = replace(graph.request, flight_id=12)
    other_graph = build_flight_graph(other_request, cfg, catalog, params)
    other_seed = pricing.seed_column(other_graph, cfg, model=cost_model(cfg, params))
    graphs = {seed.flight_id: graph, other_seed.flight_id: other_graph}
    sweep = _sweep([seed, other_seed], certified_ids={seed.flight_id})
    pricing_pool._receipt(sweep, [graph.request, other_request], graphs, cfg, params, catalog)
    assert pricing_pool.certified_sweep_columns(sweep, graphs, cfg, params, catalog) == (seed,)


def test_receipt_rejects_primary_columns_swapped_between_flights():
    cfg, params, graph, catalog, seed = _context()
    other_request = replace(graph.request, flight_id=12)
    other_graph = build_flight_graph(other_request, cfg, catalog, params)
    other_seed = pricing.seed_column(other_graph, cfg, model=cost_model(cfg, params))
    graphs = {seed.flight_id: graph, other_seed.flight_id: other_graph}
    sweep = _sweep([seed, other_seed])
    swapped = replace(sweep, columns=(other_seed, seed))

    pricing_pool._receipt(
        swapped, [graph.request, other_request], graphs, cfg, params, catalog
    )

    assert pricing_pool.certified_sweep_columns(swapped, graphs, cfg, params, catalog) == ()


def test_mock_pricing_and_foreign_columns_do_not_acquire_internal_provenance(monkeypatch):
    cfg, params, graph, _catalog, seed = _context()
    assert pricing_pool._certified_result(seed, (), graph, params, cfg)
    assert not pricing_pool._certified_result(replace(seed), (), graph, params, cfg)
    monkeypatch.setattr(pricing_pool, 'price_flight', lambda *a, **k: (1., seed))
    assert not pricing_pool._certified_result(seed, (), graph, params, cfg)


def test_mutable_request_cannot_change_pool_payload_or_acquire_receipt():
    cfg, params, graph, catalog, seed = _context()
    request = pickle.loads(pickle.dumps(graph.request))
    # Build a mutable public request, preserving the same field values.
    request = FlightRequest(request.flight_id, request.origin.copy(), request.dest.copy(),
                            request.t_request, request.t_departure, paired_outbound_id=19)
    pool = pricing_pool.PricingPool([request], cfg, replace(params, n_pricing_workers=1), catalog)
    original = pool._request_definitions
    request.origin[0] += 10.
    assert tuple(pricing_pool._request_definition(r) for r in pool._requests) == original
    assert pricing_pool._request_definition(request) != original[0]
    sweep = _sweep([seed])
    pricing_pool._receipt(sweep, [request], {seed.flight_id: graph}, cfg, params, catalog)
    assert pricing_pool.certified_sweep_columns(sweep, {seed.flight_id: graph}, cfg, params, catalog) == ()
    paired_graph = build_flight_graph(pool._requests[0], cfg, catalog, params)
    assert paired_graph.request.paired_outbound_id == 19
    assert pickle.loads(pickle.dumps(paired_graph.request)).paired_outbound_id == 19


def test_certification_repairs_claims_and_cost_and_checks_config():
    cfg, params, graph, _catalog, seed = _context()
    model = cost_model(cfg, params)
    imported = replace(seed, claims=frozenset(), delay_s=-1000.)
    repaired = pricing.certify_column(imported, graph, cfg, model)
    assert repaired == seed
    assert pricing.is_certified_column(repaired, graph, model)
    with pytest.raises(ValueError, match='SimConfig'):
        pricing.certify_column(repaired, graph, replace(cfg, dt_s=cfg.dt_s * 2), model)


def test_certification_registry_keeps_live_winner_without_retaining_temporary_columns():
    cfg, params, graph, _catalog, seed = _context()
    model = cost_model(cfg, params)
    winner = pricing.certify_column(replace(seed), graph, cfg, model)
    temporaries = [pricing.certify_column(replace(seed), graph, cfg, model) for _ in range(20)]
    assert len(graph._search_cache.certified_columns) >= 21
    assert pricing.is_certified_column(winner, graph, model)
    references = [weakref.ref(column) for column in temporaries]
    del temporaries
    gc.collect()
    assert all(reference() is None for reference in references)
    assert pricing.is_certified_column(winner, graph, model)
    winner_id = id(winner)
    del winner
    gc.collect()
    assert winner_id not in graph._search_cache.certified_columns
    assert len(graph._search_cache.certified_columns) == 1  # cached seed only


def test_final_tied_winner_remains_certified_after_more_than_eight_losers():
    cfg, params, graph, _catalog, seed = _context()
    model = cost_model(cfg, params)
    view = pricing.DualView({}, cfg)
    label = pricing._Label(0., seed.departure_step, seed.origin_lane_idx, seed.cell_path, frozenset())
    candidate = pricing._Candidate(1000. - seed.delay_s, seed.delay_s, label, seed.dest_lane_idx)
    result = pricing._certify_candidates([candidate] * 20, graph, view, 0., cfg,
                                         1000., frozenset(), model)
    assert result is not None
    assert pricing.is_certified_column(result[1], graph, model)
    assert pricing_pool._certified_result(result[1], (), graph, params, cfg)


@pytest.mark.parametrize('dt,clock', [(4., 0.), (.1, 0.), (.1, 1_000_000.)])
def test_shifted_incumbent_matches_canonical_claims_and_cost(dt, clock):
    cfg, params, original, _catalog, _seed = _context(dt_s=dt, nominal_speed_mps=120. / dt, max_ground_delay_s=8.)
    request = replace(original.request, t_request=clock, t_departure=clock)
    graph = build_flight_graph(request, cfg, (), params)
    model = cost_model(cfg, params)
    seed = pricing.seed_column(graph, cfg, model=model)
    result = pricing._shifted_seed_incumbent(
        seed, graph, pricing.DualView({}, cfg), 0., cfg, 1000., frozenset(), None, model=model)
    assert result is not None
    rc, column = result
    intent = column_to_intent(column, graph.request, cfg)
    fresh = build_flight_graph(graph.request, cfg, (), params)
    assert column.claims == column_claims(column, fresh, cfg, _intent=intent)
    assert column.delay_s.hex() == model.intent_cost(intent, cfg).hex()
    assert rc.hex() == model.reduced_cost(benefit=1000., cost=column.delay_s, dual_cost=0., pi_f=0.).hex()


def test_shifted_missing_terminal_row_cannot_be_used_as_cutoff():
    """A dwell exactly on a .7-s boundary at 1e12 reproduces the known lost terminal row."""
    cfg = _cfg(dt_s=.7, nominal_speed_mps=120. / .7, max_ground_delay_s=64 * .7,
               max_detour_factor=100., time_buffer_s=0.)
    request = FlightRequest(601, _point((0, 0), cfg), _point((8, 0), cfg), 1e12, 1e12,
                            origin_terminal=Terminal("origin", 1, radius=90.),
                            dest_terminal=Terminal("dest", 1, radius=90.))
    dwell = column_dwell_s(request.origin, request.origin_terminal, cfg, cfg.flight_levels_m[0])
    cfg = replace(cfg, hover_time_s=(math.ceil(dwell / cfg.dt_s) + 2) * cfg.dt_s - dwell)
    params = ColGenParams(solver="highs", max_air_overrun_hops=3)
    graph = build_flight_graph(request, cfg, (), params)
    start, end = min(((a.cell, b.cell) for a in graph.origin_lanes for b in graph.dest_lanes),
                     key=lambda pair: hg.hex_distance(*pair))
    path = [start]
    while path[-1] != end:
        path.append(min((c for c in hg.hex_neighbors(*path[-1])
                         if hg.hex_distance(c, end) < hg.hex_distance(path[-1], end))))
    raw = Column(request.flight_id, graph.base_step, 0,
                 next(i for i, lane in enumerate(graph.origin_lanes) if lane.cell == start),
                 next(i for i, lane in enumerate(graph.dest_lanes) if lane.cell == end),
                 tuple(path), 0.)
    model = cost_model(cfg, params)
    previous = pricing.certify_column(raw, graph, cfg, model)
    found = None
    # The origin dwell was deliberately placed on a binary-inexact terminal boundary.
    # Inspect its fixed 32-step clock cycle; do not sample rows from unordered claims.
    for departure in range(graph.base_step + 1, graph.base_step + 33):
        actual = pricing.certify_column(replace(previous, departure_step=departure), graph, cfg, model)
        missing = actual.claims - pricing._shift_claims(previous.claims, 1)
        terminal_missing = [row for row in missing if row.kind == "term"]
        if terminal_missing:
            found = previous, actual, frozenset([min(terminal_missing, key=repr)])
            break
        previous = actual
    assert found is not None, 'the documented terminal clock boundary must omit a shifted row'
    previous, actual, forbidden = found
    bounded = replace(graph, latest_departure_step=actual.departure_step)
    result = pricing._shifted_seed_incumbent(
        previous, bounded, pricing.DualView({}, cfg), 0., cfg, 1000., forbidden, None, model=model)
    assert result is None


def test_default_shift_reuses_geometry_even_after_claim_cache_eviction(monkeypatch):
    cfg, params, graph, _catalog, seed = _context()
    model = cost_model(cfg, params)
    seed = pricing.certify_column(seed, graph, cfg, model)
    graph._search_cache.certified_claims.clear()
    def unexpected_translation(*args, **kwargs):
        raise AssertionError("default exact clock shift rebuilt reservation geometry")
    monkeypatch.setattr(pricing, 'column_to_intent', unexpected_translation)
    result = pricing._shifted_seed_incumbent(
        seed, graph, pricing.DualView({}, cfg), 0., cfg, 1000., frozenset(), None, model=model)
    assert result is not None
    assert pricing.is_certified_column(result[1], graph, model)


def test_public_known_column_is_repaired_before_becoming_a_cutoff():
    cfg, params, graph, _catalog, seed = _context()
    params = replace(params, bootstrap_roots=0)
    row = min(seed.claims, key=repr)
    view = pricing.DualView({row: 10.}, cfg)
    expected = pricing.price_flight(graph, view, 0., cfg, params, known_column=seed)
    imported = replace(seed, claims=frozenset(), delay_s=-1_000_000.)
    actual = pricing.price_flight(graph, view, 0., cfg, params, known_column=imported)
    assert actual == expected
