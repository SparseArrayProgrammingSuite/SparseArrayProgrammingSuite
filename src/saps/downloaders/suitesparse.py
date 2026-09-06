"""Downloader for matrices from the SuiteSparse Matrix Collection (via ssgetpy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _default_data_dir() -> Path:
    # src/saps/downloaders/suitesparse.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "suitesparse"


def download_suitesparse_matrix(
    name: str, *, data_dir: str | Path | None = None
) -> tuple[Path, Any]:
    """Search SuiteSparse for *name* and download/extract it if not already cached.

    Returns ``(matrix_dir, matrix)``, where *matrix* is the ``ssgetpy`` search
    result. Matrices are cached under ``data/suitesparse/`` (like the SNAP and
    G-CARE downloaders cache under ``data/snap`` and ``data/gcare``) unless
    *data_dir* overrides the location.
    """
    import ssgetpy

    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    root.mkdir(parents=True, exist_ok=True)

    matches = ssgetpy.search(name=name)
    if not matches:
        raise ValueError(f"No matrix found with name '{name}'")
    exact = [match for match in matches if match.name.lower() == name.lower()]
    matrix = exact[0] if exact else matches[0]
    path, _archive = matrix.download(destpath=str(root), extract=True)
    return Path(path), matrix


def _download_and_read_matrix(
    name: str, data_dir: str | Path | None
) -> tuple[Path, Any, Any]:
    from scipy.io import mmread

    matrix_dir, matrix = download_suitesparse_matrix(name, data_dir=data_dir)
    matrix_path = matrix_dir / f"{matrix.name}.mtx"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    A = mmread(matrix_path).tocoo(copy=False)
    A.sum_duplicates()
    return matrix_dir, matrix, A


def load_suitesparse_matrix(
    name: str, *, data_dir: str | Path | None = None
) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    """Download (if needed) and parse a SuiteSparse matrix into a SciPy COO matrix.

    Returns ``(A, b, meta)``. ``b`` is the matrix's real right-hand-side vector
    (``<name>_b.mtx``) when the SuiteSparse collection entry ships one, else
    ``None``. There's no way to know this ahead of a download -- the SuiteSparse
    index doesn't expose it -- but checking costs nothing extra since the whole
    archive is already downloaded and extracted to read the matrix itself.
    """
    matrix_dir, matrix, A = _download_and_read_matrix(name, data_dir)
    rhs_path = matrix_dir / f"{matrix.name}_b.mtx"
    b = load_suitesparse_rhs(matrix_dir, matrix.name) if rhs_path.exists() else None
    meta = {
        "dataset_name": name,
        "matrix_group": matrix.group,
        "n": A.shape[0],
        "nnz": A.nnz,
        "shape": A.shape,
        "has_b_file": b is not None,
    }
    return A, b, meta


def load_suitesparse_rhs(matrix_dir: str | Path, matrix_name: str) -> np.ndarray:
    """Load the ``<matrix_name>_b.mtx`` right-hand-side vector from *matrix_dir*."""
    from scipy.io import mmread

    rhs_path = Path(matrix_dir) / f"{matrix_name}_b.mtx"
    if not rhs_path.exists():
        raise FileNotFoundError(f"Matrix file not found at {rhs_path}")
    b = mmread(rhs_path)
    if not isinstance(b, np.ndarray):
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
    return b.flatten()


def random_rhs_for_matrix(A: Any, *, seed: int = 0, density: float = 0.1) -> np.ndarray:
    """Synthesize a deterministic RHS ``b = A @ x`` for a random sparse ``x``."""
    from scipy.sparse import random as sp_random

    rng = np.random.default_rng(seed)
    x = sp_random(
        A.shape[1], 1, density=density, format="coo", dtype=np.float64, random_state=rng
    )
    b = A @ x
    return b.toarray().flatten()
