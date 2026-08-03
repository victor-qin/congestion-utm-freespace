"""Whole-schedule column-generation planner.

The batch planner lands in PR A Phase 3.  Keeping these exports lazy lets the
Phase-1 geometry package be imported without prematurely depending on that
integration module.
"""

from __future__ import annotations

from importlib import import_module

from .params import ColGenParams

# The Phase-2 solver stays lazy so importing the Phase-1 geometry surface does
# not eagerly import SciPy.  Phase 3 extends this list with the batch planner.
__all__ = ["ColGenParams", "ColGenResult", "ColGenSolver"]


def __getattr__(name: str):
    if name in {"ColGenResult", "ColGenSolver"}:
        solver = import_module(f"{__name__}.solver")
        return getattr(solver, name)
    if name in {"ColumnGenerationPlanner", "run_batch"}:
        try:
            batch = import_module(f"{__name__}.batch")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.batch":
                raise AttributeError(
                    f"{name} is added with colgen batch integration in PR A Phase 3"
                ) from None
            raise

        return getattr(batch, name)
    raise AttributeError(name)
