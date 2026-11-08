"""Explicit device-buffer ownership shared by pricing runtime and retained data."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeviceBuffer:
    """A typed device allocation belonging to exactly one runtime, with no host mirror."""

    owner: object
    native: object
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One ordered kernel invocation whose buffer references remain live until completion."""

    name: str
    count: int
    buffers: tuple[DeviceBuffer, ...]
