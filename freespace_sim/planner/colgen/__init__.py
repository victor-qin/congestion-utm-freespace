"""Whole-schedule column-generation planner.

The batch planner lands in PR A Phase 3.  Keeping these exports lazy lets the
Phase-1 geometry package be imported without prematurely depending on that
integration module.
"""

from __future__ import annotations

from importlib import import_module

from .params import ColGenParams

# Phase 3 extends this public list when the batch module lands.  Keeping absent
# names out for now makes ``from ...colgen import *`` safe during Phase 1 while
# direct attribute access remains forward-compatible through ``__getattr__``.
__all__ = ["ColGenParams"]


def __getattr__(name: str):
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
