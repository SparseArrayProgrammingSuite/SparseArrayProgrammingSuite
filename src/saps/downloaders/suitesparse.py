"""Downloader for matrices from the SuiteSparse Matrix Collection (via ssgetpy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _default_data_dir() -> Path:
    # src/saps/downloaders/suitesparse.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "suitesparse"


def download_suitesparse_matrix(
    source_name: str, *, data_dir: str | Path | None = None
) -> tuple[Path, Any]:
    """Download/extract a SuiteSparse matrix identified by ``group/name``.

    Returns ``(matrix_dir, matrix)``, where *matrix* is the ``ssgetpy`` search
    result. Matrices are cached under ``data/suitesparse/`` (like the SNAP and
    G-CARE downloaders cache under ``data/snap`` and ``data/gcare``) unless
    *data_dir* overrides the location.
    """
    import ssgetpy

    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    root.mkdir(parents=True, exist_ok=True)

    matrix = _find_suitesparse_matrix(ssgetpy, source_name)
    path, _archive = matrix.download(destpath=str(root), extract=True)
    return Path(path), matrix


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
    rhs_index: int | None = None,
) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    """Download (if needed) and parse a SuiteSparse matrix into a SciPy COO matrix.

    Returns ``(A, b, meta)``. ``b`` is the matrix's real right-hand-side vector
    (``<name>_b.mtx``) when the SuiteSparse collection entry ships one
    unambiguous RHS vector, or when *rhs_index* selects one RHS from a multi-RHS
    file. Otherwise ``b`` is ``None``. There's no way to know this ahead of a
    download -- the SuiteSparse index doesn't expose it -- but checking costs
    nothing extra since the whole archive is already downloaded and extracted to
    read the matrix itself.
    """
    matrix_dir, matrix, A = _download_and_read_matrix(source_name, data_dir)
    rhs_path = matrix_dir / f"{matrix.name}_b.mtx"
    b = None
    rhs_error = None
    if not rhs_path.exists():
        if rhs_index is not None:
            raise ValueError(f"SuiteSparse matrix '{source_name}' has no RHS file")
    else:
        try:
            b = load_suitesparse_rhs(
                matrix_dir,
                matrix.name,
                expected_length=A.shape[0],
                rhs_index=rhs_index,
            )
        except ValueError as exc:
            if rhs_index is not None:
                raise
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
    rhs_index: int | None = None,
) -> np.ndarray:
    """Load the ``<matrix_name>_b.mtx`` right-hand-side vector from *matrix_dir*."""
    from scipy.io import mmread

    rhs_path = Path(matrix_dir) / f"{matrix_name}_b.mtx"
    if not rhs_path.exists():
        raise FileNotFoundError(f"Matrix file not found at {rhs_path}")
    b = mmread(rhs_path)
    if not isinstance(b, np.ndarray):
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
    b = np.asarray(b)
    if expected_length is None:
        if rhs_index is not None:
            raise ValueError("rhs_index requires expected_length")
        return b.flatten()
    return _coerce_rhs_vector(b, expected_length, rhs_path, rhs_index=rhs_index)


def _coerce_rhs_vector(
    rhs: np.ndarray,
    expected_length: int,
    rhs_path: Path,
    *,
    rhs_index: int | None = None,
) -> np.ndarray:
    if rhs_index is not None and rhs_index < 0:
        raise ValueError(f"rhs_index must be nonnegative, got {rhs_index}")

    if rhs.ndim == 1 and rhs.shape[0] == expected_length:
        if rhs_index not in (None, 0):
            raise ValueError(
                f"SuiteSparse RHS file {rhs_path} contains 1 RHS vector, "
                f"got rhs_index={rhs_index}"
            )
        return rhs

    if rhs.ndim == 2:
        if rhs.shape == (expected_length, 1):
            if rhs_index not in (None, 0):
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains 1 RHS vector, "
                    f"got rhs_index={rhs_index}"
                )
            return rhs[:, 0]
        if rhs.shape == (1, expected_length):
            if rhs_index not in (None, 0):
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains 1 RHS vector, "
                    f"got rhs_index={rhs_index}"
                )
            return rhs[0, :]
        if rhs.shape[0] == expected_length:
            rhs_count = rhs.shape[1]
            if rhs_index is None:
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains {rhs_count} RHS "
                    "vectors; select one with rhs_index"
                )
            if rhs_index >= rhs_count:
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains {rhs_count} RHS "
                    f"vectors, got rhs_index={rhs_index}"
                )
            return rhs[:, rhs_index]
        if rhs.shape[1] == expected_length:
            rhs_count = rhs.shape[0]
            if rhs_index is None:
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains {rhs_count} RHS "
                    "vectors; select one with rhs_index"
                )
            if rhs_index >= rhs_count:
                raise ValueError(
                    f"SuiteSparse RHS file {rhs_path} contains {rhs_count} RHS "
                    f"vectors, got rhs_index={rhs_index}"
                )
            return rhs[rhs_index, :]

    flat = rhs.flatten()
    if flat.shape[0] == expected_length:
        if rhs_index not in (None, 0):
            raise ValueError(
                f"SuiteSparse RHS file {rhs_path} contains 1 RHS vector, "
                f"got rhs_index={rhs_index}"
            )
        return flat

    raise ValueError(
        f"SuiteSparse RHS file {rhs_path} has shape {rhs.shape}, "
        f"expected a vector of length {expected_length}"
    )


def random_rhs_for_matrix(A: Any, *, seed: int = 0, density: float = 0.1) -> np.ndarray:
    """Synthesize a deterministic RHS ``b = A @ x`` for a random sparse ``x``."""
    from scipy.sparse import random as sp_random

    rng = np.random.default_rng(seed)
    x = sp_random(
        A.shape[1], 1, density=density, format="coo", dtype=np.float64, random_state=rng
    )
    b = A @ x
    return b.toarray().flatten()
