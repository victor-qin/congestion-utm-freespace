"""Native goal search must reproduce the bounded Python oracle, including its failures."""
from __future__ import annotations

import numpy as np
import pytest

from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.objective import cost_model
from tests.test_colgen_dp_kernel import GRAPH_SHAPES, _cfg, _random_duals

kernel = pytest.importorskip('freespace_sim.planner.colgen.bootstrap_kernel')


def _case(shape, dual_case, *, forbidden=False):
    """Use existing customer/terminal fixtures with real endpoint windows and role masks."""
    cfg = _cfg(max_ground_delay_s=8.0)
    graph, params = GRAPH_SHAPES[shape](cfg, overrun=2)
    model = cost_model(cfg, params)
    raw = {} if dual_case == 'ties' else _random_duals(graph, cfg, 606)
    if dual_case == 'tiny':
        raw = {row: (-1e-13 if i % 2 else value) for i, (row, value) in enumerate(raw.items())}
    view = pricing.DualView(raw, cfg)
    excluded = frozenset()
    if forbidden:
        seed = pricing.seed_column(graph, cfg, model=model)
        cells = sorted((r for r in seed.claims if r.kind == 'cell'), key=lambda r: r.step)
        excluded = frozenset(cells[len(cells) // 2:len(cells) // 2 + 1])
    topology, rows = dp_prepare.prepared_for(graph, cfg)
    return graph, view, cfg, model, excluded, topology, rows


def _run(case, *, native, budget, order_count=3, diagnostics=None, deadline=None,
         incumbent=None, stronger_envelope=False):
    """Build independent mutable envelopes and call the production oracle or native twin."""
    graph, view, cfg, model, excluded, topology, rows = case
    envelopes = dp_prepare.CompletionEnvelopes(
        graph, cfg, view, benefit=1000., pi_f=0., model=model, forbidden_rows=excluded,
        incumbent=incumbent, deadline=None,
    )
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, rows, benefit=1000., model=model,
        forbidden_rows=excluded, envelopes=envelopes,
    )
    order = np.arange(min(order_count, variants.n_variants), dtype=np.int64)
    assert len(order)
    if stronger_envelope:
        envelopes = dp_prepare.CompletionEnvelopes(
            graph, cfg, view, benefit=1000., pi_f=0., model=model, forbidden_rows=excluded,
            incumbent=incumbent, deadline=None,
        )
        envelopes.set_incumbent((incumbent[0] + 1000., incumbent[1]))
        if stronger_envelope == 'rewound':
            for root in order:
                lane = int(variants.lane_idx[root])
                envelopes._delay_envelope(int(variants.departure_step[root]), None if lane < 0 else lane)
            envelopes.set_incumbent(incumbent)
            assert envelopes.incumbent is incumbent
    args = (graph, view, 0., cfg, 1000., excluded, model, variants, order, topology, envelopes)
    kwargs = dict(incumbent=incumbent, max_labels=budget, deadline=deadline)
    if native:
        return kernel.goal_directed_bootstrap(*args, **kwargs, prepared_rows=rows,
                                              diagnostics=diagnostics)
    oracle = getattr(pricing, '_goal_directed_bootstrap_python', pricing._goal_directed_bootstrap)
    result = oracle(*args, **kwargs)
    return result, pricing._LAST_SEARCH['n_labels']


@pytest.mark.parametrize('shape', sorted(GRAPH_SHAPES))
@pytest.mark.parametrize('dual_case,budget,forbidden', [
    ('ties', 20000, False), ('random', 20000, True), ('tiny', 1, False),
    ('ties', 0, False), ('ties', 4, False),
])
def test_goal_kernel_preserves_certified_column_and_expansion_cap(shape, dual_case, budget, forbidden):
    """Real certification, forbidden visits, tied queues and budget failure match exactly."""
    case = _case(shape, dual_case, forbidden=forbidden)
    expected = _run(case, native=False, budget=budget)
    diagnostics = {}
    actual = _run(case, native=True, budget=budget, diagnostics=diagnostics)
    assert actual == expected
    assert actual[1] <= budget
    if actual[0] is not None:
        assert actual[0][0].hex() == expected[0][0].hex()
        assert actual[0][1].delay_s.hex() == expected[0][1].delay_s.hex()
    assert diagnostics == {'native_used': True, 'expanded': actual[1], 'decline_reason': None,
                           'sinks_asked': diagnostics['sinks_asked'], 'sinks_skipped': 0}
    assert diagnostics['sinks_asked'] >= int(actual[0] is not None)
    if budget == 20000 and not forbidden:
        assert actual[0] is not None


@pytest.mark.parametrize('shape', sorted(GRAPH_SHAPES))
@pytest.mark.parametrize('dual_case,reject', [('ties', 1), ('ties', None), ('random', None), ('tiny', None)])
def test_goal_kernel_resumes_invalid_sinks_and_preserves_all_root_order(shape, dual_case, reject, monkeypatch):
    """Invalid goals resume the same heap; rejection of every goal visits later roots."""
    case = _case(shape, dual_case)
    factory = pricing._sink_certifier
    captures = []

    def certifier(*args, **kwargs):
        real = factory(*args, **kwargs)
        seen = []
        captures.append(seen)

        def certify(incumbent, departure, lane, dest_lane, step, path):
            seen.append((departure, lane, dest_lane, step, path))
            if reject is None or len(seen) <= reject:
                return None
            return real(incumbent, departure, lane, dest_lane, step, path)
        return certify

    monkeypatch.setattr(pricing, '_sink_certifier', certifier)
    expected = _run(case, native=False, budget=20000)
    actual = _run(case, native=True, budget=20000)
    assert actual == expected
    assert captures[0] == captures[1]
    assert len(captures[0]) > 1
    if reject is None:
        assert len({(x[0], x[1]) for x in captures[0]}) > 1
        assert actual[0] is None


def test_goal_kernel_deadline_propagates():
    """An expired caller deadline must not be converted into an unsupported fallback."""
    case = _case('plain', 'ties')
    with pytest.raises(pricing.PricingTimeout):
        _run(case, native=True, budget=20000, deadline=0.)


def test_goal_kernel_decline_is_explicit_and_programming_errors_propagate(monkeypatch):
    """Unsupported inputs allow fallback; unexpected bugs must remain visible."""
    diagnostics = {}
    bad = dp_prepare.PreparedTopology(unsupported_reason="test unsupported topology")
    result = kernel.goal_directed_bootstrap(
        None, None, 0., None, 1., frozenset(), None, None, None, bad, None,
        incumbent=None, max_labels=1, deadline=None, diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics == {'native_used': False, 'decline_reason': 'test unsupported topology'}

    def broken(*args):
        raise RuntimeError("unexpected native bug")

    monkeypatch.setattr(kernel, '_advance', broken)
    with pytest.raises(RuntimeError, match="unexpected native bug"):
        _run(_case('plain', 'ties'), native=True, budget=100)


@pytest.mark.parametrize('mode', ['native', 'declined', 'unavailable', 'error'])
def test_goal_dispatch_preserves_shared_inputs_and_explicit_fallback(mode, monkeypatch):
    """Dispatch reports successful native work, falls back only on decline, and raises bugs."""
    from types import SimpleNamespace

    seen = []
    sentinel = object()
    packed_rows, packed_duals, packed_forbidden = object(), object(), object()
    preparation = SimpleNamespace(
        check=lambda *args: seen.append('checked'), topology_rows=(None, packed_rows),
        duals=packed_duals, forbidden=packed_forbidden,
    )

    def native(*args, **kwargs):
        seen.append('native')
        assert kwargs['prepared_rows'] is packed_rows
        assert kwargs['prepared_duals'] is packed_duals
        assert kwargs['prepared_forbidden'] is packed_forbidden
        if mode == 'error':
            raise RuntimeError('native programming error')
        if mode == 'declined':
            kwargs['diagnostics'].update(native_used=False, decline_reason='unsupported test input')
            return None
        kwargs['diagnostics'].update(sinks_asked=11, sinks_skipped=23)
        return sentinel, 7

    def oracle(*args, **kwargs):
        seen.append('oracle')
        pricing._LAST_SEARCH['n_labels'] = 3
        return sentinel

    monkeypatch.setattr(kernel, 'goal_directed_bootstrap', native)
    monkeypatch.setattr(pricing, '_goal_directed_bootstrap_python', oracle)
    monkeypatch.setattr(pricing, '_dp_kernel', lambda: None if mode == 'unavailable' else object())
    args = (None, None, 0., None, 1., frozenset(), None, None, None, None, None)
    kwargs = dict(incumbent=None, max_labels=20, deadline=None, preparation=preparation)
    if mode == 'error':
        with pytest.raises(RuntimeError, match='native programming error'):
            pricing._goal_directed_bootstrap(*args, **kwargs)
        assert seen == ['checked', 'native']
        return
    assert pricing._goal_directed_bootstrap(*args, **kwargs) is sentinel
    assert pricing._LAST_SEARCH['bootstrap_goal_native'] is (mode == 'native')
    assert pricing._LAST_SEARCH['n_labels'] == (7 if mode == 'native' else 3)
    if mode == 'native':
        assert pricing._LAST_SEARCH['bootstrap_goal_sinks_asked'] == 11
        assert pricing._LAST_SEARCH['bootstrap_goal_sinks_skipped'] == 23
    assert seen == ({'native': ['checked', 'native'],
                     'declined': ['checked', 'native', 'oracle'],
                     'unavailable': ['oracle']}[mode])
    assert pricing._LAST_SEARCH['bootstrap_goal_decline_reason'] == {
        'native': None, 'declined': 'unsupported test input', 'unavailable': 'numba unavailable',
    }[mode]


@pytest.mark.parametrize('shape', sorted(GRAPH_SHAPES))
@pytest.mark.parametrize('dual_case,cutoff_slack', [('ties', 0.), ('ties', 3e-9), ('tiny', 0.)])
def test_native_sink_bound_preserves_results_and_expansions(shape, dual_case, cutoff_slack, monkeypatch):
    """Screen only oracle-rejected sinks; keep surviving callbacks in the same order."""
    case = _case(shape, dual_case)
    graph, view, cfg, model, _excluded, _topology, _rows = case
    seed = pricing.seed_column(graph, cfg, model=model)
    rc = model.reduced_cost(benefit=1000., cost=seed.delay_s,
                           dual_cost=view.claim_cost(seed.claims), pi_f=0.)
    incumbent = (rc - cutoff_slack, seed)
    factory = pricing._sink_certifier
    traces = []

    def recording(*args, **kwargs):
        real = factory(*args, **kwargs)
        trace = []
        traces.append(trace)

        def certify(*sink_args):
            outcome = real(*sink_args)
            trace.append((sink_args[1:], outcome))
            return outcome
        return certify

    monkeypatch.setattr(pricing, '_sink_certifier', recording)
    expected = _run(case, native=False, budget=20000, incumbent=incumbent)
    diagnostics = {}
    actual = _run(case, native=True, budget=20000, incumbent=incumbent, diagnostics=diagnostics)
    assert actual == expected
    assert actual[0][0].hex() == expected[0][0].hex()
    cursor = 0
    for event, outcome in traces[0]:
        if cursor < len(traces[1]) and traces[1][cursor][0] == event:
            assert traces[1][cursor][1] == outcome
            cursor += 1
        else:
            assert outcome is None, 'native screen removed an improving oracle sink'
    assert cursor == len(traces[1])
    assert diagnostics['sinks_asked'] <= len(traces[1])
    if shape == 'plain' and dual_case == 'ties' and cutoff_slack == 0.:
        assert diagnostics['sinks_skipped'] > 0
    if cutoff_slack:
        assert diagnostics['sinks_asked'] > 0
        if shape == 'plain':
            assert actual[0][0] > incumbent[0]


@pytest.mark.parametrize('envelope_mode', [True, 'rewound'])
def test_native_sink_screen_declines_envelopes_from_a_stronger_stage(envelope_mode):
    """A stronger stage's shortened envelope cannot suppress the current improvement."""
    case = _case('plain', 'ties')
    graph, view, cfg, model, *_ = case
    seed = pricing.seed_column(graph, cfg, model=model)
    rc = model.reduced_cost(benefit=1000., cost=seed.delay_s,
                           dual_cost=view.claim_cost(seed.claims), pi_f=0.)
    incumbent = (rc - 1., seed)
    expected = _run(case, native=False, budget=20000, incumbent=incumbent, stronger_envelope=envelope_mode)
    diagnostics = {}
    actual = _run(case, native=True, budget=20000, incumbent=incumbent,
                  stronger_envelope=envelope_mode, diagnostics=diagnostics)
    assert actual == expected
    assert actual[0][0] > incumbent[0]
    assert diagnostics['sinks_skipped'] == 0
    assert diagnostics['sinks_asked'] > 0
