"""Legacy CG entry point for shared cheap-then-full Zeta pricing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet

from ..zeta import hybrid as _implementation
from .network import RowKey
from .params import ColGenParams
from .translate import Column
from .zeta import ZetaGraph, ZetaPricingResult, _price_with


def __getattr__(name: str) -> object:
    """Retain archived helper imports without exposing the new pricing signature."""
    return getattr(_implementation, name)


def price_hybrid(
    graph: ZetaGraph, duals: Mapping[RowKey, float], pi_f: float, params: ColGenParams,
    *, known_columns: Sequence[Column] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> ZetaPricingResult:
    """Translate the archived hybrid CG call without duplicating either search stage.

    Parameters
    ------------
    - graph (ZetaGraph): full-domain graph retaining caches across calls.
    - duals (Mapping[RowKey, float]): frozen signed reservation prices.
    - pi_f (float): finite flight dual.
    - params (ColGenParams): CG objective and improvement tolerance.
    - known_columns (Sequence[Column]): incumbent columns excluded from early discovery.
    - forbidden_rows (AbstractSet[RowKey]): unavailable rows in both stages.
    - deadline (float | None): shared absolute cutoff.

    Return
    --------
    - output (ZetaPricingResult): aggregated work and a full-domain CG certificate.
    """
    return _price_with(_implementation.price_hybrid, graph, duals, pi_f, params,
                       known_columns=known_columns, forbidden_rows=forbidden_rows, deadline=deadline)
