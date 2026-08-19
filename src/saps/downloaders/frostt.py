"""Downloader for tensors from FROSTT (the Formidable Repository of Open Sparse
Tensors and Tools, frostt.io)."""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd

_BASE_URL = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data"

_RHS_DTYPES = {
    "matrix-multiplication/matmul_2-2-2.tns.gz": np.bool_,
    "matrix-multiplication/matmul_3-3-3.tns.gz": np.bool_,
    "matrix-multiplication/matmul_4-3-2.tns.gz": np.bool_,
    "matrix-multiplication/matmul_4-4-3.tns.gz": np.bool_,
    "matrix-multiplication/matmul_4-4-4.tns.gz": np.bool_,
    "matrix-multiplication/matmul_5-5-5.tns.gz": np.bool_,
    "matrix-multiplication/matmul_6-3-3.tns.gz": np.bool_,
    "nell/nell-2.tns.gz": np.bool_,
    "chicago-crime/comm/chicago-crime-comm.tns.gz": np.int64,
    "lbnl-network/lbnl-network.tns.gz": np.int64,
    "toy/toy.tns.gz": np.float64,
    "nips/nips.tns.gz": np.int64,
    "uber-pickups/uber.tns.gz": np.int64,
    "chicago-crime/geo/chicago-crime-geo.tns.gz": np.int64,
    "vast-2015-mc1/vast-2015-mc1-3d.tns.gz": np.bool_,
    "nell/nell-1.tns.gz": np.bool_,
    "vast-2015-mc1/vast-2015-mc1-5d.tns.gz": np.bool_,
    "enron/enron.tns.gz": np.int64,
    "flickr/flickr-3d.tns.gz": np.bool_,
    "flickr/flickr-4d.tns.gz": np.bool_,
    "delicious/delicious-3d.tns.gz": np.bool_,
    "delicious/delicious-4d.tns.gz": np.bool_,
    "amazon/amazon-reviews.tns.gz": np.int64,
    "patents/patents.tns.gz": np.float64,
    "reddit-2015/reddit-2015.tns.gz": np.int64,
    "fb-m/fb-m.tns.gz": np.bool_,
    "darpa/1998darpa.tns.gz": np.int64,
    "lanl2/lanl2.tns.gz": np.int64,
}


def _default_data_dir() -> Path:
    # src/saps/downloaders/frostt.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "frostt"


def download_frostt_tensor(
    path: str, *, url: str | None = None, data_dir: str | Path | None = None
) -> Path:
    """Download (if needed) a FROSTT `.tns.gz` tensor file, returning its local path.

    *path* is the tensor's location under FROSTT's main S3 bucket, e.g.
    ``"matrix-multiplication/matmul_3-3-3.tns.gz"`` or
    ``"chicago-crime/comm/chicago-crime-comm.tns.gz"``, and also determines
    where the file is cached under ``data/frostt/`` (like the
    SuiteSparse/SNAP/G-CARE downloaders cache under ``data/suitesparse``,
    ``data/snap``, ``data/gcare``) unless *data_dir* overrides the location.

    A few tensors (fb-m, darpa, lanl2) live in a different bucket entirely;
    for those, pass the full download *url* and *path* is only used for local
    caching.
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dest_path = root / path
    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    download_url = url if url is not None else f"{_BASE_URL}/{path}"
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")
    try:
        urllib.request.urlretrieve(download_url, tmp_path)  # noqa: S310
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest_path


def _extract_tns_source(path: Path) -> Path | io.BytesIO:
    """Return a readable source for the `.tns` text, un-wrapping a tar archive
    if present. A few tensors (fb-m, darpa, lanl2) are uploaded as a `.tar.gz`
    (despite the `.tns.gz` name) containing the real data file alongside a
    macOS AppleDouble sidecar file (`._<name>.tns`); most tensors are just a
    plain gzip-compressed `.tns` file, which is not a valid tar and falls back
    to being read directly.
    """
    try:
        with tarfile.open(path, "r:gz") as tf:
            members = [
                member
                for member in tf.getmembers()
                if member.isfile() and not Path(member.name).name.startswith("._")
            ]
            if len(members) != 1:
                names = [member.name for member in members]
                raise ValueError(
                    f"Expected exactly one data file in tar archive {path}, found"
                    f" {names}"
                )
            extracted = tf.extractfile(members[0])
            assert extracted is not None
            return io.BytesIO(extracted.read())
    except tarfile.ReadError:
        return path


def _parse_tns(
    path: Path,
    rhs_dtype: np.dtype | type,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[int, ...]]:
    """Parse a (optionally gzipped, optionally tar-wrapped) `.tns` coordinate-list
    tensor file.

    Each line is ``i_1 i_2 ... i_n value``, 1-indexed. Returns 0-indexed index
    arrays (one per mode), the values array, and the dense shape inferred as
    the maximum index seen per mode.
    """
    source = _extract_tns_source(path)
    preview = pd.read_csv(source, sep=r"\s+", header=None, comment="#", nrows=1)
    order = preview.shape[1] - 1
    if order < 1:
        raise ValueError(f"Malformed FROSTT tensor file {path}: no value column found")
    dtypes = {mode: np.int64 for mode in range(order)}
    dtypes[order] = rhs_dtype
    if isinstance(source, io.BytesIO):
        source.seek(0)
    df = pd.read_csv(source, sep=r"\s+", header=None, comment="#", dtype=dtypes)
    indices = tuple(df.iloc[:, mode].to_numpy() - 1 for mode in range(order))
    values = df.iloc[:, order].to_numpy()
    shape = tuple(int(idx.max()) + 1 for idx in indices)
    if all(dim <= np.iinfo(np.int32).max for dim in shape):
        indices = tuple(idx.astype(np.int32) for idx in indices)
    return indices, values, shape


def load_frostt_tensor(
    path: str,
    *,
    url: str | None = None,
    data_dir: str | Path | None = None,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, Any]]:
    """Download (if needed) and parse a FROSTT tensor into COO index/value arrays."""
    local_path = download_frostt_tensor(path, url=url, data_dir=data_dir)
    indices, values, shape = _parse_tns(local_path, _RHS_DTYPES[path])
    meta = {
        "dataset_name": path,
        "order": len(shape),
        "shape": shape,
        "nnz": len(values),
    }
    return indices, values, meta
