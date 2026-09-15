"""The conflict predicate — the heart of ASTM strategic deconfliction (§3.2.8).

Two 4D volumes conflict iff their time windows overlap and their 3D shapes intersect. We
prune on time first (a cheap scalar test) and only then pay for the exact FCL narrowphase.
"""

from __future__ import annotations

import fcl

from .geometry import CylinderSpec
from .volumes import Volume4D


def volumes_conflict(a: Volume4D, b: Volume4D, *, security_margin: float = 0.0) -> bool:
    """True iff a and b overlap in time and in 3D space (ASTM §3.2.8).

    Time is pruned first (a cheap scalar test); only overlapping windows pay for the exact FCL
    narrowphase. Two volumes at the same vertiport (same non-None ``terminal_id``) are mutually
    transparent only when a column is involved — at least one is a hover cylinder
    (``CylinderSpec``) — so a flight's exit-lane corridor through its own hub column does not
    conflict, but two same-hub corridor boxes (``BoxSpec`` vs ``BoxSpec``) still contend. A cruise
    corridor (``terminal_id=None``) or a different hub conflicts as usual, so a busy column still
    blocks overflight. Column vs corridor is derived from shape, not a stored kind.

    Parameters
    ------------
    - a (Volume4D): one 4D operational-intent volume.
    - b (Volume4D): the other 4D volume to test against.
    - security_margin (float): metres of required separation; when > 0, the shapes must be that far
      apart rather than merely non-overlapping, encoding safety beyond the corridor buffer.

    Return
    --------
    - output (bool): True iff the volumes conflict under the shared-terminal exemption above.
    """
    if (a.terminal_id is not None and a.terminal_id == b.terminal_id
            and (isinstance(a.shape, CylinderSpec) or isinstance(b.shape, CylinderSpec))):
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
