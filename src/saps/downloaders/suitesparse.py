"""Downloader for matrices from the SuiteSparse Matrix Collection (via ssgetpy)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from filelock import FileLock

from saps.downloaders.cache import source_cache_dir


def download_suitesparse_matrix(
    source_name: str, *, data_dir: str | Path | None = None
) -> tuple[Path, Any]:
    """Download/extract a SuiteSparse matrix identified by ``group/name``.

    Returns ``(matrix_dir, matrix)``, where *matrix* is the ``ssgetpy`` search
    result. Matrices are cached under ``$SAPS_CACHE_DIR/suitesparse/group/name``
    unless *data_dir* overrides the SuiteSparse cache root.
    """
    import ssgetpy

    matrix = _find_suitesparse_matrix(ssgetpy, source_name)
    root = Path(data_dir) if data_dir is not None else source_cache_dir("suitesparse")
    parent = root / matrix.group
    parent.mkdir(parents=True, exist_ok=True)
    matrix_dir = parent / matrix.name
    with FileLock(parent / f"{matrix.name}.lock"):
        if not matrix_dir.exists():
            with tempfile.TemporaryDirectory(prefix=".saps-", dir=parent) as staging:
                path, _archive = matrix.download(destpath=staging, extract=True)
                Path(path).replace(matrix_dir)
    return matrix_dir, matrix


def _split_suitesparse_source_name(source_name: str) -> tuple[str, str]:
    group, separator, matrix_name = source_name.partition("/")
    if not separator or not group or not matrix_name:
        raise ValueError(
            f"SuiteSparse source names must use 'group/name', got '{source_name}'"
        )
    return group, matrix_name


def _find_suitesparse_matrix(ssgetpy: Any, source_name: str) -> Any:
    group, matrix_name = _split_suitesparse_source_name(source_name)
    matches = [
        matrix
        for matrix in ssgetpy.search(group=group, limit=-1)
        if matrix.group == group and matrix.name == matrix_name
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Multiple SuiteSparse matrices found for '{source_name}'")
    raise ValueError(f"No SuiteSparse matrix found for '{source_name}'")


def _download_and_read_matrix(
    source_name: str, data_dir: str | Path | None
) -> tuple[Path, Any, Any]:
    from scipy.io import mmread

    matrix_dir, matrix = download_suitesparse_matrix(source_name, data_dir=data_dir)
    matrix_path = matrix_dir / f"{matrix.name}.mtx"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    A = mmread(matrix_path).tocoo(copy=False)
    A.sum_duplicates()
    return matrix_dir, matrix, A


def load_suitesparse_matrix(
    source_name: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    """Download (if needed) and parse a SuiteSparse matrix into a SciPy COO matrix.

    Returns ``(A, b, meta)``. ``b`` contains all real right-hand sides from
    ``<name>_b.mtx``: a vector for a single RHS, or a matrix with one RHS per
    column. It is ``None`` when no compatible RHS file is available. The whole
    archive is downloaded and extracted together, so loading all RHS vectors
    requires no additional download.
    """
    matrix_dir, matrix, A = _download_and_read_matrix(source_name, data_dir)
    rhs_path = matrix_dir / f"{matrix.name}_b.mtx"
    b = None
    rhs_error = None
    if rhs_path.exists():
        try:
            b = load_suitesparse_rhs(
                matrix_dir,
                matrix.name,
                expected_length=A.shape[0],
            )
        except ValueError as exc:
            rhs_error = str(exc)
    meta = {
        "dataset_name": source_name,
        "matrix_group": matrix.group,
        "n": A.shape[0],
        "nnz": A.nnz,
        "shape": A.shape,
        "has_b_file": b is not None,
    }
    if rhs_error is not None:
        meta["ignored_b_file"] = True
        meta["rhs_error"] = rhs_error
    return A, b, meta


def load_suitesparse_rhs(
    matrix_dir: str | Path,
    matrix_name: str,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Load all RHS vectors, orienting multiple RHSs as columns when length is known."""
    from scipy.io import mmread

    rhs_path = Path(matrix_dir) / f"{matrix_name}_b.mtx"
    if not rhs_path.exists():
        raise FileNotFoundError(f"Matrix file not found at {rhs_path}")
    b = mmread(rhs_path)
    if not isinstance(b, np.ndarray):
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
    b = np.asarray(b)
    if expected_length is not None:
        if (
            b.ndim == 2
            and b.shape[0] != expected_length
            and b.shape[1] == expected_length
        ):
            b = b.T
        if b.ndim not in (1, 2) or b.shape[0] != expected_length:
            raise ValueError(
                f"SuiteSparse RHS file {rhs_path} has shape {b.shape}, "
                f"expected RHS vectors of length {expected_length}"
            )
    if b.ndim == 2 and (
        b.shape[1] == 1 or (expected_length is None and b.shape[0] == 1)
    ):
        return b.flatten()
    return b


def read_vector(path: Path) -> np.ndarray:
    """Read a Matrix Market file holding a single column into a flat array."""
    from scipy.io import mmread

    values = mmread(path)
    if not isinstance(values, np.ndarray):
        values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    return values.flatten()


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
    c = read_vector(matrix_dir / f"{matrix.name}_c.mtx")
    b = read_vector(matrix_dir / f"{matrix.name}_b.mtx")

    infinite_bound = 1e30
    lo_path = matrix_dir / f"{matrix.name}_lo.mtx"
    hi_path = matrix_dir / f"{matrix.name}_hi.mtx"
    lo = read_vector(lo_path) if lo_path.exists() else np.zeros(cols)
    hi = read_vector(hi_path) if hi_path.exists() else np.full(cols, np.inf)
    lo = np.where(lo <= -infinite_bound, -np.inf, lo)
    hi = np.where(hi >= infinite_bound, np.inf, hi)

    z0_path = matrix_dir / f"{matrix.name}_z0.mtx"
    z0 = float(read_vector(z0_path)[0]) if z0_path.exists() else 0.0

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
