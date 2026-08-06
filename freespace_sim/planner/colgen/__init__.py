"""Whole-schedule column-generation planner.

The batch planner lands in PR A Phase 3.  Keeping these exports lazy lets the
Phase-1 geometry package be imported without prematurely depending on that
integration module.
"""

from __future__ import annotations

from importlib import import_module

from .params import ColGenParams

# The solver and batch integration stay lazy so importing the geometry surface
# does not eagerly import SciPy.
__all__ = [
    "ColGenParams",
    "ColGenResult",
    "ColGenSolver",
    "ColumnGenerationPlanner",
    "run_batch",
]


def __getattr__(name: str):
    if name in {"ColGenResult", "ColGenSolver"}:
        solver = import_module(f"{__name__}.solver")
        return getattr(solver, name)
    if name in {"ColumnGenerationPlanner", "run_batch"}:
        batch = import_module(f"{__name__}.batch")
        return getattr(batch, name)
    raise AttributeError(name)
