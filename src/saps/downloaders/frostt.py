"""Downloader for tensors from FROSTT (the Formidable Repository of Open Sparse
Tensors and Tools, frostt.io)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

_BASE_URL = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data"


def _default_data_dir() -> Path:
    # src/saps/downloaders/frostt.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "frostt"


def download_frostt_tensor(path: str, *, data_dir: str | Path | None = None) -> Path:
    """Download (if needed) a FROSTT `.tns.gz` tensor file, returning its local path.

    *path* is the tensor's location under FROSTT's S3 bucket, e.g.
    ``"matrix-multiplication/matmul_3-3-3.tns.gz"`` or
    ``"chicago-crime/comm/chicago-crime-comm.tns.gz"``. Files are cached under
    ``data/frostt/`` (like the SuiteSparse/SNAP/G-CARE downloaders cache under
    ``data/suitesparse``, ``data/snap``, ``data/gcare``) unless *data_dir*
    overrides the location.
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dest_path = root / path
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE_URL}/{path}"
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest_path


def _parse_tns(
    path: Path,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[int, ...]]:
    """Parse a (optionally gzipped) `.tns` coordinate-list tensor file.

    Each line is ``i_1 i_2 ... i_n value``, 1-indexed. Returns 0-indexed index
    arrays (one per mode), the values array, and the dense shape inferred as
    the maximum index seen per mode.
    """
    data = np.loadtxt(path, ndmin=2)
    order = data.shape[1] - 1
    if order < 1:
        raise ValueError(f"Malformed FROSTT tensor file {path}: no value column found")
    indices = tuple(data[:, mode].astype(np.int64) - 1 for mode in range(order))
    values = data[:, order].astype(np.float64)
    shape = tuple(int(idx.max()) + 1 for idx in indices)
    return indices, values, shape


def load_frostt_tensor(
    path: str, *, data_dir: str | Path | None = None
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, Any]]:
    """Download (if needed) and parse a FROSTT tensor into COO index/value arrays."""
    local_path = download_frostt_tensor(path, data_dir=data_dir)
    indices, values, shape = _parse_tns(local_path)
    meta = {
        "dataset_name": path,
        "order": len(shape),
        "shape": shape,
        "nnz": len(values),
    }
    return indices, values, meta
