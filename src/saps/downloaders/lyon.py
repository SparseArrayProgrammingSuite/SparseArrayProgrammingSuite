"""Pedestrian trajectories from the 2022 Festival of Lights in Lyon."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

from saps.downloaders.cache import download_lock, source_cache_dir

LYON_RECORD_URL = "https://zenodo.org/records/13830435"
_ZIP_URL = "https://zenodo.org/api/records/13830435/files/Data_Madras.zip/content"
_ZIP_MD5 = "2ad3436fb8c6b25ebc72d51eacfa1abb"

# The top-view sequences (30 fps), and the wide CCTV view whose timestamps are
# steady (10 fps). LargeView_zoom_A's jitter within frames and it repeats 90
# rows.
KNOWN_SEQUENCES = (
    *(f"TopView_{s}" for s in ("1A", "1B", "1C", "2A", "2B", "2C", "2D", "2E", "2F")),
    "LargeView_zoom_O",
)

# Velocities are taken over the 0.4 s around the snapshot (EWAP's annotation
# step), which smooths out the tracking jitter between single frames.
VELOCITY_WINDOW = 0.4


def download_lyon_dataset(
    sequence: str,
    *,
    data_dir: str | Path | None = None,
    num_steps: int = 50,
) -> tuple[list[BinsparseTensor], dict[str, Any]]:
    """Return the crowd at a sequence's most crowded frame as particle initial
    conditions, shaped like ``download_ewap_dataset``'s.

    Positions are the sequence's x/m and y/m columns, in meters, without
    rescaling. z and vz are 0, since the z column is a person's estimated
    height rather than a position.
    """
    if sequence not in KNOWN_SEQUENCES:
        raise ValueError(f"Unknown Lyon sequence {sequence!r}")
    view = sequence.split("_", 1)[0]
    zip_path = _ensure_downloaded(data_dir)
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(f"Data_Madras/{view}_trajectories/{sequence}.txt")
    frame, x, y, vx, vy = parse_crowd_snapshot(text.decode("utf-8"))
    zeros = np.zeros_like(x)
    return (
        [
            from_numpy(x),
            from_numpy(y),
            from_numpy(zeros),
            from_numpy(vx),
            from_numpy(vy),
            from_numpy(zeros.copy()),
            from_numpy(np.asarray(0.01, dtype=np.float64)),
        ],
        {
            "size": float(max(x.max() - x.min(), y.max() - y.min(), 1e-9)),
            "steps": num_steps,
            "source_sequence": sequence,
            "source_frame": frame,
            "source_url": LYON_RECORD_URL,
            "n_particles": len(x),
            "source_dimensions": 2,
            "simulation_dimensions": 3,
            "source_x_min": float(x.min()),
            "source_x_max": float(x.max()),
            "source_y_min": float(y.min()),
            "source_y_max": float(y.max()),
        },
    )


def parse_crowd_snapshot(
    text: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the first frame with the most pedestrians, and each of their
    x, y and velocity over the VELOCITY_WINDOW around it, clipped to their
    trajectory.

    Columns are read by name from the ``# id frame x/m y/m ... t/s`` header.
    """
    header = next(line for line in text.splitlines() if line.startswith("# id "))
    column = {name: index for index, name in enumerate(header[1:].split())}
    rows = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    rows = rows[np.lexsort((rows[:, column["frame"]], rows[:, column["id"]]))]
    # LargeView_zoom_O repeats one pedestrian's row at frame 0, 2 mm apart.
    # Keep the first row of each pedestrian and frame.
    key = rows[:, [column["id"], column["frame"]]]
    rows = rows[np.r_[True, np.any(key[1:] != key[:-1], axis=1)]]
    ids, frame, x, y, t = (
        rows[:, column[name]] for name in ("id", "frame", "x/m", "y/m", "t/s")
    )
    frames, counts = np.unique(frame, return_counts=True)
    snapshot = frames[np.argmax(counts)]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    states = []
    for start, end in zip(starts, np.r_[starts[1:], len(ids)], strict=True):
        hit = np.flatnonzero(frame[start:end] == snapshot)
        if hit.size == 0:
            continue
        i = start + hit[0]
        times = t[start:end]
        half = VELOCITY_WINDOW / 2 + 1e-6
        lo = start + np.searchsorted(times, t[i] - half)
        hi = start + np.searchsorted(times, t[i] + half, side="right") - 1
        if hi == lo:
            raise ValueError(f"Pedestrian {ids[i]:g} has one observation near the snapshot")
        dt = t[hi] - t[lo]
        states.append((x[i], y[i], (x[hi] - x[lo]) / dt, (y[hi] - y[lo]) / dt))
    x0, y0, vx, vy = (np.array(values) for values in zip(*states, strict=True))
    return int(snapshot), x0, y0, vx, vy


def _ensure_downloaded(data_dir: str | Path | None) -> Path:
    root = Path(data_dir) if data_dir is not None else source_cache_dir("lyon")
    dest_path = root / "Data_Madras.zip"
    with download_lock(dest_path):
        if dest_path.exists():
            return dest_path
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        try:
            urllib.request.urlretrieve(_ZIP_URL, tmp_path)  # noqa: S310
            md5 = hashlib.md5(tmp_path.read_bytes(), usedforsecurity=False)
            if md5.hexdigest() != _ZIP_MD5:
                raise ValueError(
                    f"Data_Madras.zip does not match Zenodo's MD5 {_ZIP_MD5}"
                )
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest_path
