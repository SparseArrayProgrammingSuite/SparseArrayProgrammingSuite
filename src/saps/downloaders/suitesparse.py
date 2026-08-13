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
    matrix = matches[0]
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
    return matrix_dir, matrix, mmread(matrix_path).tocoo()


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

def load_lpnetlib_problem(
    name: str, *, data_dir: str | Path | None = None
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Download (if needed) and parse an LPnetlib linear program.

    The LPnetlib group stores each Netlib LP as ``minimize c'x`` subject to
    ``A x = b`` and ``lo <= x <= hi``, shipping the objective vector, the two
    bound vectors, and the objective offset ``z0`` as separate Matrix Market
    files beside the matrix. 

    Returns ``(A, b, c, lo, hi, meta)``, with bounds normalized so that an
    unbounded variable reads as an IEEE infinity. Bounds default to ``lo = 0``
    and ``hi = inf`` when an entry omits the files entirely. ``z0`` is reported
    in *meta* rather than returned, since it shifts the objective value without
    moving the optimum.
    """
    matrix_dir, matrix, A = _download_and_read_matrix(name, data_dir)
    if matrix.group != "LPnetlib":
        raise ValueError(
            f"Matrix '{name}' belongs to group '{matrix.group}', not LPnetlib"
        )

    rows, cols = A.shape
    c = _read_vector(matrix_dir / f"{matrix.name}_c.mtx")
    b = load_suitesparse_rhs(matrix_dir, matrix.name)

    lo_path = matrix_dir / f"{matrix.name}_lo.mtx"
    hi_path = matrix_dir / f"{matrix.name}_hi.mtx"
    lo = _read_vector(lo_path) if lo_path.exists() else np.zeros(cols)
    hi = _read_vector(hi_path) if hi_path.exists() else np.full(cols, np.inf)
    lo = np.where(lo <= -_INFINITE_BOUND, -np.inf, lo)
    hi = np.where(hi >= _INFINITE_BOUND, np.inf, hi)

    z0_path = matrix_dir / f"{matrix.name}_z0.mtx"
    z0 = float(_read_vector(z0_path)[0]) if z0_path.exists() else 0.0

    for label, vector, expected in (
        ("b", b, rows),
        ("c", c, cols),
        ("lo", lo, cols),
        ("hi", hi, cols),
    ):
        if vector.shape[0] != expected:
            raise ValueError(
                f"LPnetlib problem '{name}' has a {label} vector of length"
                f" {vector.shape[0]}, expected {expected}"
            )

    meta = {
        "dataset_name": name,
        "matrix_group": matrix.group,
        "shape": A.shape,
        "nnz": A.nnz,
        "z0": z0,
        "has_lo_file": lo_path.exists(),
        "has_hi_file": hi_path.exists(),
    }
    return A, b, c, lo, hi, meta

def random_rhs_for_matrix(A: Any, *, seed: int = 0, density: float = 0.1) -> np.ndarray:
    """Synthesize a deterministic RHS ``b = A @ x`` for a random sparse ``x``."""
    from scipy.sparse import random as sp_random

    rng = np.random.default_rng(seed)
    x = sp_random(
        A.shape[1], 1, density=density, format="coo", dtype=np.float64, random_state=rng
    )
    b = A @ x
    return b.toarray().flatten()
