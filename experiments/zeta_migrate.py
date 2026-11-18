"""Migrate a trusted prototype pickle to the shared Zeta package without changing its source."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import BinaryIO


_MOVED = frozenset({
    "arrival_bound", "bound", "compact", "completion", "completion_geometry",
    "crossing_bound", "forward", "geometry", "history", "history_guard", "hybrid",
    "kernel", "lattice_bound", "leg_cache", "native", "resource_bound", "route_cache",
    "selected_bound", "state_index", "timing",
})


class _ZetaUnpickler(pickle.Unpickler):
    """Resolve prototype helper classes in their current shared modules.

    Attributes
    ------------
    - inherited pickle state: the original trusted stream; CG graph loading still
      performs its normal obsolete-column-cache migration.
    """

    def find_class(self, module: str, name: str) -> object:
        """Translate only the removed helper modules; preserve every other pickle reference."""
        prefix = "freespace_sim.planner.colgen.zeta_"
        if module.startswith(prefix) and module[len(prefix):] in _MOVED:
            module = "freespace_sim.planner.zeta." + module[len(prefix):]
        return super().find_class(module, name)


def load_archive(stream: BinaryIO) -> object:
    """Read a trusted old capture with explicit module migration.

    Parameters
    ------------
    - stream (BinaryIO): opened prototype pickle, left open and unchanged.

    Return
    --------
    - output (object): original payload with helper classes resolved in planner.zeta.
      No global import aliases are installed; ordinary current imports stay clean.
    """
    return _ZetaUnpickler(stream).load()


def main() -> None:
    """Write a migrated copy that ordinary pickle.load can read without legacy modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        payload = load_archive(stream)
    with args.destination.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Migrated {args.source} -> {args.destination}")


if __name__ == "__main__":
    main()
