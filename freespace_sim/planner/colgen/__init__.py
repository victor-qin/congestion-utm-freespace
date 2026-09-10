"""Whole-schedule column-generation planner.

The solver and batch exports stay lazy so that importing the geometry surface
does not drag in SciPy or the simulator's filing path with it.
"""

from __future__ import annotations

from importlib import import_module

from .params import ColGenParams

# Names ``__getattr__`` resolves lazily are still listed here, so the public surface (``dir``,
# ``from ... import *``) is complete without importing the solver/batch modules -- and thus
# SciPy -- at import time.
__all__ = [
    "ColGenParams",
    "ColGenResult",
    "ColGenSolver",
    "ColumnGenerationPlanner",
    "run_batch",
]


def __getattr__(name: str):
    """Import the solver/batch module on first access and return the requested export."""
    if name in {"ColGenResult", "ColGenSolver"}:
        solver = import_module(f"{__name__}.solver")
        return getattr(solver, name)
    if name in {"ColumnGenerationPlanner", "run_batch"}:
        batch = import_module(f"{__name__}.batch")
        return getattr(batch, name)
    raise AttributeError(name)
