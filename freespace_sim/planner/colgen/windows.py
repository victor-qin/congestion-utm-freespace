"""Compatibility imports for shared reservation-window geometry."""

from ..reservation_windows import (
    __all__ as __all__,
    AxialCell as AxialCell,
    CellWindow as CellWindow,
    _MIN_ENDPOINT_TIME_PAD_S as _MIN_ENDPOINT_TIME_PAD_S,
    _endpoint_time_pad_s as _endpoint_time_pad_s,
    _periods_overlapping as _periods_overlapping,
    visit_rows as visit_rows,
    _point as _point,
    _shift_volume as _shift_volume,
    _template_arcs as _template_arcs,
    _cross_check_conflicts as _cross_check_conflicts,
    validate_edge_locality as validate_edge_locality,
    _cells_in_ring as _cells_in_ring,
    derive_cell_window as derive_cell_window,
    _distance_to_hex as _distance_to_hex,
    endpoint_claim_cells as endpoint_claim_cells,
    endpoint_claim_steps as endpoint_claim_steps,
    hop_box_stays_in_its_cells as hop_box_stays_in_its_cells,
    terminal_claim_steps as terminal_claim_steps,
)
