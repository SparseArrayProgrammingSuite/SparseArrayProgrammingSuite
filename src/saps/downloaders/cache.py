"""Shared locations and inter-process locks for source downloads."""

import os
from pathlib import Path

from filelock import FileLock

from saps.storage import DEFAULT_CACHE_DIR


def source_cache_dir(source: str) -> Path:
    return (
        Path(os.environ.get("SAPS_CACHE_DIR") or DEFAULT_CACHE_DIR).expanduser()
        / source
    )


def download_lock(destination: Path) -> FileLock:
    """Lock beside the destination, including while checking and extracting it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(destination.with_name(destination.name + ".lock"))
