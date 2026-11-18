"""Compatibility import for archived Zeta experiments; implementation lives in planner.zeta."""
import sys
from ..zeta import completion_geometry as _implementation
sys.modules[__name__] = _implementation
