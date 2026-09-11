"""Shared deterministic serialization helpers for column-generation analyses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def encode(value: Any) -> Any:
    """Convert experiment data to a deterministic, exact JSON representation."""
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(
            (encode(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def digest(value: Any) -> str:
    """Hash the deterministic representation of an experiment value."""
    payload = json.dumps(encode(value), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of raw artifact bytes."""
    return hashlib.sha256(data).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    """Hash every Python file below a selected source tree's package."""
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted((root / "freespace_sim").rglob("*.py"))
    }


def write_json(path: Path, value: Any) -> None:
    """Write one readable JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
