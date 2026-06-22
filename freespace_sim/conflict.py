"""The conflict predicate — the heart of ASTM strategic deconfliction (§3.2.8).

Two 4D volumes conflict iff their time windows overlap **and** their 3D shapes intersect. We
prune on time first (a cheap scalar test) and only then pay for the exact FCL narrowphase.
"""

from __future__ import annotations

import fcl

from .volumes import Volume4D


def volumes_conflict(a: Volume4D, b: Volume4D, *, security_margin: float = 0.0) -> bool:
    """True iff a and b overlap in time AND in 3D space (ASTM §3.2.8).

    ``security_margin`` (metres) optionally requires the shapes to be that far *apart* rather than
    merely non-overlapping — a clean place to encode safety beyond the corridor buffer.

    **Shared-terminal exemption:** two volumes carrying the same non-None ``terminal_id`` belong to
    the same vertiport terminal and are *mutually transparent* — a multi-pad hub's own flights share
    its airspace column (pad capacity is bounded tactically, not here). Everything else, including a
    cruise corridor (``terminal_id=None``) versus a terminal, conflicts as usual, so the column still
    blocks overflight. This is the one documented exception to the strict no-overlap invariant.
    """
    if a.terminal_id is not None and a.terminal_id == b.terminal_id:
        return False
    if not (a.t_start < b.t_end and b.t_start < a.t_end):   # 1) time-window overlap
        return False
    req = fcl.CollisionRequest()
    res = fcl.CollisionResult()
    n = fcl.collide(a.to_fcl(), b.to_fcl(), req, res)        # 2) exact 3D narrowphase
    if n > 0:
        return True
    if security_margin > 0.0:                               # optional separation buffer
        dreq = fcl.DistanceRequest()
        dres = fcl.DistanceResult()
        dist = fcl.distance(a.to_fcl(), b.to_fcl(), dreq, dres)
        return dist < security_margin
    return False
