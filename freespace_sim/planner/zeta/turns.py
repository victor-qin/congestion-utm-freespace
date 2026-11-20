"""Heading classes for the shared exact turn rule in Zeta's finite connection graphs."""

from __future__ import annotations

from collections.abc import Sequence

from ..turn_rule import Cell, turn_allowed


class TurnClasses:
    """Group incoming offsets whose legal next offsets are identical.

    Class 0 means no constraint: a root before the first cruise link, or a destination
    sink label that has no continuation. Two incoming offsets share a positive class only
    when they allow exactly the same members of the graph's offset set, so merging or
    dominating labels within one class never changes their legal futures.

    Attributes
    ------------
    - offsets (tuple[Cell, ...]): the graph's ordered connection offsets.
    - classes (dict[Cell, int]): incoming offset to positive heading class.
    - allowed (tuple[frozenset[Cell], ...]): legal outgoing offsets per class; index 0 is all.
    - count (int): number of classes including the unconstrained class 0.
    """

    def __init__(self, offsets: Sequence[Cell]) -> None:
        """Precompute exact admissibility masks for every incoming connection offset."""
        self.offsets = tuple((int(q), int(r)) for q, r in offsets)
        masks: dict[tuple[bool, ...], int] = {}
        allowed: list[frozenset[Cell]] = [frozenset(self.offsets)]
        self.classes: dict[Cell, int] = {}
        for incoming in self.offsets:
            mask = tuple(turn_allowed(incoming, outgoing) for outgoing in self.offsets)
            if mask not in masks:
                masks[mask] = len(allowed)
                allowed.append(frozenset(
                    outgoing for outgoing, legal in zip(self.offsets, mask) if legal))
            self.classes[incoming] = masks[mask]
        self.allowed = tuple(allowed)
        self.count = len(allowed)

    def heading(self, source: Cell, target: Cell) -> int:
        """Return the class reached by one link; links outside the offset set raise KeyError."""
        return self.classes[(target[0] - source[0], target[1] - source[1])]
