"""Frozen, fingerprinted study reports on disk."""

import os
import tempfile
from pathlib import Path
from ..errors import Conflict, InputError


def atomic_study_write(destination, payload):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise Conflict("study report destination already exists")
    if not destination.parent.is_dir():
        raise InputError("study report parent directory does not exist")
    fd, temporary = tempfile.mkstemp(prefix=".metrology-study-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Conflict("study destination appeared concurrently") from exc
        return {"path": str(destination), "bytes": len(payload)}
    finally:
        Path(temporary).unlink(missing_ok=True)
