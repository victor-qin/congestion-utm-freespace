"""Legacy CG entry point for the shared compact Zeta search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet

from ..zeta import compact as _implementation
from .network import RowKey
from .params import ColGenParams
from .translate import Column
from .zeta import ZetaGraph, ZetaPricingResult, _price_with


def __getattr__(name: str) -> object:
    """Retain archived helper imports without replacing the legacy pricing adapter."""
    return getattr(_implementation, name)


def price_zeta_compact(
    graph: ZetaGraph, duals: Mapping[RowKey, float], pi_f: float, params: ColGenParams,
    *, known_columns: Sequence[Column] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None, _native_loop: bool = False,
    _compiled_history_type: type | None = None,
    _compiled_coverage_type: type | None = None,
    _compiled_state_type: type | None = None,
) -> ZetaPricingResult:
    """Keep the archived compact pricing API while using the shared route engine.

    Parameters
    ------------
    - graph (ZetaGraph): reusable physical route graph.
    - duals (Mapping[RowKey, float]): frozen signed reservation prices.
    - pi_f (float): finite flight dual.
    - params (ColGenParams): CG objective and improvement tolerance.
    - known_columns (Sequence[Column]): incumbent columns revalidated by search.
    - forbidden_rows (AbstractSet[RowKey]): unavailable capacity rows.
    - deadline (float | None): absolute search cutoff.
    - _native_loop (bool): legacy direct native-buffer option.
    - _compiled_history_type (type | None): prepared native history helper.
    - _compiled_coverage_type (type | None): prepared native route-coverage helper.
    - _compiled_state_type (type | None): prepared native state-index helper.

    Return
    --------
    - output (ZetaPricingResult): ordinary CG column, certificate and work counters.
    """
    return _price_with(
        _implementation.search_compact, graph, duals, pi_f, params,
        known_columns=known_columns, forbidden_rows=forbidden_rows, deadline=deadline,
        _native_loop=_native_loop, _compiled_history_type=_compiled_history_type,
        _compiled_coverage_type=_compiled_coverage_type, _compiled_state_type=_compiled_state_type,
    )
