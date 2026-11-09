#!/usr/bin/env python3
"""Probe SuiteSparse matrices by running SAPS solver benchmarks on them.

This script intentionally calls the SAPS benchmark implementations instead of
estimating spectral radii or condition numbers. Each output line is a JSON
record for one matrix with per-solver convergence results.

Matrices and the ssgetpy index live in the SAPS cache (by default this
checkout's `.saps/outputs/cache`, as in `bin/run_benchmark.py`), and solvers run
on this checkout's SciPy framework unless `SAPS_FRAMEWORK` names another.

    pixi run python scripts/scrape_matrices.py --solver amg --max-nnz 1000000
    pixi run python scripts/scrape_matrices.py --solver cg --matrix HB/gr_30_30
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
from scipy.io import mminfo

from binsparse.conversions import from_scipy

import saps.benchmarks.cg as saps_cg
import saps.benchmarks.cg_amg as saps_amg
import saps.benchmarks.cg_block_jacobi as saps_cg_block_jacobi
import saps.benchmarks.cg_jacobi as saps_cg_jacobi
import saps.benchmarks.gmres as saps_gmres
import saps.benchmarks.jacobi as saps_jacobi
import saps.benchmarks.lsqr as saps_lsqr
import saps.downloaders.suitesparse as saps_suitesparse_downloader
from saps.benchmarks.cg import CGBenchmark, CGDataset, CGSuiteSparseGenerator
from saps.benchmarks.cg_amg import (
    CGAMGBenchmark,
    CGAMGDataset,
    CGAMGSuiteSparseGenerator,
)
from saps.benchmarks.cg_block_jacobi import (
    CGBlockJacobiBenchmark,
    CGBlockJacobiDataset,
    CGBlockJacobiTestGenerator,
)
from saps.benchmarks.cg_jacobi import (
    CGJacobiBenchmark,
    CGJacobiDataset,
    CGJacobiSuiteSparseGenerator,
)
from saps.benchmarks.gmres import (
    GMRESBenchmark,
    GMRESDataset,
    GMRESSuiteSparseGenerator,
)
from saps.benchmarks.jacobi import (
    JacobiBenchmark,
    JacobiDataset,
    JacobiSuiteSparseGenerator,
)
from saps.benchmarks.lsqr import LSQRBenchmark, LSQRDataset, LSQRSuiteSparseGenerator
from saps.framework import load_framework
from saps.storage import DEFAULT_CACHE_DIR

REPO_DIR = Path(__file__).resolve().parent.parent


SOLVER_NAMES = (
    "jacobi",
    "cg",
    "jacobi_cg",
    "block_jacobi_cg",
    "lsqr",
    "gmres",
    "amg",
)

LSQR_MATRIX_KINDS = frozenset({"least squares problem"})

MAX_SUITESPARSE_RHS_PER_MATRIX = 100

ACCEPTED_MATRIX_KINDS = frozenset(
    {
        "tomography problem",
        "thermal problem",
        "theoretical/quantum chemistry problem sequence",
        "theoretical/quantum chemistry problem",
        "subsequent theoretical/quantum chemistry problem",
        "subsequent structural problem",
        "subsequent semiconductor device problem",
        "subsequent power network problem",
        "subsequent optimization problem",
        "subsequent computational fluid dynamics problem",
        "subsequent circuit simulation problem",
        "subsequent 2d/3d problem",
        "structural problem sequence",
        "structural problem",
        "semiconductor process problem",
        "semiconductor device problem sequence",
        "semiconductor device problem",
        "robotics problem",
        "random 2d/3d problem",
        "power network problem sequence",
        "power network problem",
        "optimization problem sequence",
        "optimization problem",
        "optimal control problem",
        "model reduction problem",
        "materials problem",
        "frequency domain circuit simulation problem",
        "electromagnetics problem",
        "eigenvalue/model reduction problem",
        "economic problem",
        "data analytics problem",
        "computer vision problem",
        "computer graphics/vision problem",
        "computational fluid dynamics problem sequence",
        "computational fluid dynamics problem",
        "computational fluid dynamics",
        "computational chemistry problem",
        "circuit simulation problem sequence",
        "circuit simulation problem",
        "chemical process simulation problem sequence",
        "chemical process simulation problem",
        "chemical oceanography problem",
        "acoustics problem",
        "2d/3d problem sequence",
        "2d/3d problem",
    }
)


def _matrix_kind(matrix: Any) -> str:
    return _normalize_matrix_kind(getattr(matrix, "kind", ""))


def _normalize_matrix_kind(matrix_kind: Any) -> str:
    return str(matrix_kind).strip().lower()


@dataclass(frozen=True)
class RHSVariant:
    kind: str
    index: int | None
    count: int
    shape: tuple[int, int] | None = None
    error: str | None = None


def _matrix_source_name(matrix: Any) -> str:
    return f"{matrix.group}/{matrix.name}"


def _record_key(matrix_group: str, matrix_name: str, rhs_index: int | None) -> tuple:
    return (matrix_group, matrix_name, rhs_index)


def _record_key_from_record(record: dict[str, Any]) -> tuple:
    return _record_key(
        str(record.get("matrix_group", "")),
        str(record.get("matrix_name", "")),
        record.get("rhs_index"),
    )


def _record_label(matrix: Any, variant: RHSVariant) -> str:
    label = _matrix_source_name(matrix)
    if variant.kind == "suitesparse":
        return f"{label} rhs{variant.index}"
    return label


def _rhs_variants_from_shape(
    rhs_shape: tuple[int, int],
    expected_length: int,
) -> list[RHSVariant]:
    rows, cols = rhs_shape
    if rows == expected_length:
        return [
            RHSVariant("suitesparse", index, cols, rhs_shape)
            for index in range(min(cols, MAX_SUITESPARSE_RHS_PER_MATRIX))
        ]
    if cols == expected_length:
        return [
            RHSVariant("suitesparse", index, rows, rhs_shape)
            for index in range(min(rows, MAX_SUITESPARSE_RHS_PER_MATRIX))
        ]
    if rows * cols == expected_length:
        return [RHSVariant("suitesparse", 0, 1, rhs_shape)]
    return [
        RHSVariant(
            "synthetic",
            None,
            0,
            rhs_shape,
            (
                f"SuiteSparse RHS shape {rhs_shape} does not match matrix "
                f"row count {expected_length}"
            ),
        )
    ]


def _solver_will_run(solver_name: str, matrix: Any) -> bool:
    return SOLVERS[solver_name].skip_reason(matrix) is None


def _rhs_variants(
    matrix: Any,
    solvers: Iterable[str],
    suitesparse_data_dir: Path | None,
) -> list[RHSVariant]:
    rows = int(matrix.rows)
    if not any(_solver_will_run(solver, matrix) for solver in solvers):
        return [RHSVariant("synthetic", None, 0)]

    matrix_dir, downloaded_matrix = (
        saps_suitesparse_downloader.download_suitesparse_matrix(
            _matrix_source_name(matrix),
            data_dir=suitesparse_data_dir,
        )
    )
    rhs_path = matrix_dir / f"{downloaded_matrix.name}_b.mtx"
    if not rhs_path.exists():
        return [RHSVariant("synthetic", None, 0)]

    try:
        rhs_rows, rhs_cols = (int(value) for value in mminfo(rhs_path)[:2])
    except Exception as exc:  # noqa: BLE001
        return [RHSVariant("synthetic", None, 0, error=str(exc))]

    return _rhs_variants_from_shape((rhs_rows, rhs_cols), rows)


@dataclass
class SolverSpec:
    benchmark: Any
    generator: Any
    dataset_cls: type[Any]
    square_required: bool = True
    residual_kind: str = "linear"
    rel_tol: float = 1e-6
    max_iter: int = 100
    accepted_kinds: frozenset[str] = ACCEPTED_MATRIX_KINDS
    spd_required: bool = False
    discretization_required: bool = False
    restart_limit: int | None = None

    def accepts_matrix_kind(self, matrix_kind: str) -> bool:
        return matrix_kind in self.accepted_kinds

    def make_dataset(
        self,
        source_name: str,
        rhs_index: int | None,
    ) -> Any:
        kwargs = {
            "rhs_index": rhs_index,
            "max_iter": self.max_iter,
            "rel_tol": self.rel_tol,
        }
        return self.dataset_cls(source_name, **kwargs)

    def benchmark_meta(self, A: sps.spmatrix) -> dict[str, Any]:
        meta = {
            "rel_tol": self.rel_tol,
            "max_iter": self.max_iter,
        }
        if self.restart_limit is not None:
            meta["restart"] = min(self.restart_limit, A.shape[0])
        return meta

    def result_settings(self) -> dict[str, Any]:
        return {
            "rel_tol": self.rel_tol,
            "max_iter": self.max_iter,
        }

    def skip_reason(self, matrix: Any) -> str | None:
        matrix_kind = _matrix_kind(matrix)
        if not self.accepts_matrix_kind(matrix_kind):
            return f"{self.benchmark.name} does not accept matrix kind {matrix_kind!r}"
        if self.square_required and int(matrix.rows) != int(matrix.cols):
            return f"{self.benchmark.name} requires a square matrix"
        if self.spd_required and not matrix.isspd:
            return (
                f"{self.benchmark.name} requires a symmetric positive definite matrix"
            )
        if self.discretization_required and not matrix.is2d3d:
            return f"{self.benchmark.name} requires a 2D/3D discretization"
        return None


def _as_real_sparse_matrix(matrix: Any) -> sps.spmatrix:
    if not sps.issparse(matrix):
        raise TypeError("Expected SAPS to provide a SciPy sparse matrix")
    A = matrix
    if np.iscomplexobj(A.data):
        raise ValueError("SAPS solver benchmarks expect real-valued matrices")
    A = A.astype(np.float64).tocoo(copy=False)
    A.sum_duplicates()
    return A.tocsr()


@contextlib.contextmanager
def _saps_suitesparse_context(
    suitesparse_data_dir: Path | None,
    seed: int,
) -> Iterator[None]:
    modules = (
        saps_amg,
        saps_cg,
        saps_cg_block_jacobi,
        saps_cg_jacobi,
        saps_gmres,
        saps_jacobi,
        saps_lsqr,
    )
    original_fetches = {
        module: module.fetch_suitesparse_linear_system for module in modules
    }

    # Like saps.benchmarks.suitesparse.fetch_suitesparse_linear_system, but for
    # any SuiteSparse matrix, not only those listed in SuiteSparseMatrixGenerator.
    def _fetch_suitesparse_linear_system(
        source_name: str,
        *,
        rhs_index: int | None = None,
    ):
        A, rhs, _meta = saps_suitesparse_downloader.load_suitesparse_matrix(
            source_name,
            data_dir=suitesparse_data_dir,
        )
        if rhs is not None and (rhs_index is not None or rhs.ndim == 1):
            b = rhs if rhs.ndim == 1 else rhs[:, rhs_index]
            return from_scipy(A), b, True
        b = saps_suitesparse_downloader.random_rhs_for_matrix(A.tocoo(), seed=seed)
        return from_scipy(A), b, False

    for module in original_fetches:
        module.fetch_suitesparse_linear_system = _fetch_suitesparse_linear_system

    try:
        yield
    finally:
        for module, fetch in original_fetches.items():
            module.fetch_suitesparse_linear_system = fetch


SOLVERS = {
    "jacobi": SolverSpec(
        benchmark=JacobiBenchmark(),
        generator=JacobiSuiteSparseGenerator(),
        dataset_cls=JacobiDataset,
        max_iter=1000,
    ),
    "cg": SolverSpec(
        benchmark=CGBenchmark(),
        generator=CGSuiteSparseGenerator(),
        dataset_cls=CGDataset,
    ),
    "jacobi_cg": SolverSpec(
        benchmark=CGJacobiBenchmark(),
        generator=CGJacobiSuiteSparseGenerator(),
        dataset_cls=CGJacobiDataset,
    ),
    "block_jacobi_cg": SolverSpec(
        benchmark=CGBlockJacobiBenchmark(),
        # The test generator factors the preconditioner directly; the SuiteSparse
        # generator caches it in the SAPS cache, which needs a listed dataset.
        generator=CGBlockJacobiTestGenerator(),
        dataset_cls=CGBlockJacobiDataset,
    ),
    "lsqr": SolverSpec(
        benchmark=LSQRBenchmark(),
        generator=LSQRSuiteSparseGenerator(),
        dataset_cls=LSQRDataset,
        square_required=False,
        residual_kind="least_squares",
        max_iter=1000,
        accepted_kinds=LSQR_MATRIX_KINDS,
    ),
    "gmres": SolverSpec(
        benchmark=GMRESBenchmark(),
        generator=GMRESSuiteSparseGenerator(),
        dataset_cls=GMRESDataset,
        restart_limit=50,
    ),
    # Smoothed aggregation targets symmetric positive definite systems from
    # discretized elliptic PDEs, which SuiteSparse flags as 2D/3D problems.
    "amg": SolverSpec(
        benchmark=CGAMGBenchmark(),
        generator=CGAMGSuiteSparseGenerator(),
        dataset_cls=CGAMGDataset,
        spd_required=True,
        discretization_required=True,
    ),
}


def _safe_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _linear_residual(
    A: sps.spmatrix,
    b: np.ndarray,
    x: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    residual = b - A @ x
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    relative_residual = residual_norm / max(b_norm, 1e-300)
    return {
        "residual_norm": _safe_float(residual_norm),
        "relative_residual": _safe_float(relative_residual),
        "converged": bool(relative_residual < tolerance),
    }


def _least_squares_residual(
    A: sps.spmatrix,
    b: np.ndarray,
    x: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    residual = b - A @ x
    gradient = A.T @ residual
    residual_norm = np.linalg.norm(residual)
    gradient_norm = np.linalg.norm(gradient)
    b_norm = np.linalg.norm(b)
    atb_norm = np.linalg.norm(A.T @ b)
    A_norm = np.linalg.norm(A.data)
    x_norm = np.linalg.norm(x)
    relative_residual = residual_norm / max(b_norm, 1e-300)
    relative_gradient = gradient_norm / max(atb_norm, 1e-300)
    lsqr_residual_threshold = (
        tolerance * A_norm * x_norm / max(b_norm, 1e-300) + tolerance
    )
    lsqr_gradient_ratio = gradient_norm / max(A_norm * residual_norm, 1e-300)
    residual_converged = relative_residual <= lsqr_residual_threshold
    gradient_converged = lsqr_gradient_ratio <= tolerance
    return {
        "residual_norm": _safe_float(residual_norm),
        "relative_residual": _safe_float(relative_residual),
        "gradient_norm": _safe_float(gradient_norm),
        "relative_gradient": _safe_float(relative_gradient),
        "lsqr_residual_threshold": _safe_float(lsqr_residual_threshold),
        "lsqr_gradient_ratio": _safe_float(lsqr_gradient_ratio),
        "residual_converged": bool(residual_converged),
        "gradient_converged": bool(gradient_converged),
        "converged": bool(residual_converged or gradient_converged),
    }


def _run_solver(
    solver_name: str,
    matrix: Any,
    rhs_index: int | None,
    xp: Any,
) -> dict[str, Any]:
    spec = SOLVERS[solver_name]
    skip_reason = spec.skip_reason(matrix)
    if skip_reason is not None:
        return {
            "status": "skipped",
            "converged": False,
            "reason": skip_reason,
            **spec.result_settings(),
        }

    start = time.perf_counter()
    try:
        dataset = spec.make_dataset(_matrix_source_name(matrix), rhs_index)
        problem = spec.generator.generate(dataset)
        data = [xp.from_binsparse(value) for value in problem.inputs]
        meta = dict(problem.meta)
        run_A = _as_real_sparse_matrix(data[0])
        run_b = np.asarray(data[1], dtype=np.float64).reshape(-1)
        meta = {**meta, **spec.benchmark_meta(run_A)}
        output = spec.benchmark.benchmark(xp, meta, *data)
        elapsed = time.perf_counter() - start
        if isinstance(output, tuple):
            output = output[0]
        if sps.issparse(output):
            output = output.toarray()
        x = np.asarray(output, dtype=np.float64).reshape(-1)
        residual = (
            _least_squares_residual(run_A, run_b, x, spec.rel_tol)
            if spec.residual_kind == "least_squares"
            else _linear_residual(run_A, run_b, x, spec.rel_tol)
        )
        return {
            "status": "ok",
            "seconds": _safe_float(elapsed),
            **spec.result_settings(),
            **residual,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "status": "error",
            "converged": False,
            "seconds": _safe_float(elapsed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **spec.result_settings(),
        }


def _done_record_keys(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()

    keys = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                keys.add(_record_key_from_record(json.loads(line)))
            except (KeyError, json.JSONDecodeError):
                continue
    return keys


def _chunk_items(items: list[Any], chunk_count: int, chunk_index: int) -> list[Any]:
    chunk = []
    for item_index, item in enumerate(items):
        if item_index % chunk_count == chunk_index:
            chunk.append(item)
    return chunk


def _search_matrices(args: argparse.Namespace) -> list[Any]:
    # The index lives in the SAPS cache, where the downloader looks for it.
    ssgetpy = saps_suitesparse_downloader._import_ssgetpy()
    matrices = list(ssgetpy.search(limit=-1))
    found_count = len(matrices)
    if args.matrix_kind:
        accepted_kinds = {_normalize_matrix_kind(kind) for kind in args.matrix_kind}
        matrices = [
            matrix for matrix in matrices if _matrix_kind(matrix) in accepted_kinds
        ]
    if args.matrix:
        names = set(args.matrix)
        matrices = [m for m in matrices if _matrix_source_name(m) in names]
    if args.max_nnz is not None:
        matrices = [m for m in matrices if int(m.nnz) <= args.max_nnz]
    matrices.sort(key=lambda matrix: (matrix.group, matrix.name))
    if args.shuffle:
        random.Random(args.seed).shuffle(matrices)
    chunk = _chunk_items(matrices, args.chunk_count, args.chunk_index)
    print(
        "Selected "
        f"{len(chunk)} of {found_count} SuiteSparse search results "
        f"for chunk {args.chunk_index + 1}/{args.chunk_count}.",
        flush=True,
    )
    return chunk


def _matrix_record(
    matrix: Any,
    variant: RHSVariant,
    solvers: Iterable[str],
    xp: Any,
) -> dict[str, Any]:
    rows = int(matrix.rows)
    cols = int(matrix.cols)
    matrix_kind = _matrix_kind(matrix)
    nnz = int(matrix.nnz)

    results = {}
    source_name = _matrix_source_name(matrix)
    for solver_name in solvers:
        results[solver_name] = _run_solver(solver_name, matrix, variant.index, xp)

    record = {
        "matrix_name": matrix.name,
        "matrix_group": matrix.group,
        "source_name": source_name,
        "matrix_kind": matrix_kind,
        "shape": [rows, cols],
        "n": cols,
        "nnz": nnz,
        "rhs_kind": variant.kind,
        "rhs_index": variant.index,
        "rhs_count": variant.count,
        "rhs_shape": list(variant.shape) if variant.shape is not None else None,
        "results": results,
    }
    if variant.error is not None:
        record["rhs_error"] = variant.error
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download SuiteSparse matrices and call SAPS solver benchmarks to "
            "record whether they converge."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scrape_matrices.jsonl"),
        help=(
            "JSONL output path. Existing matrix names are skipped unless --force "
            "is set."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Root of group/name matrix directories, such as ~/.ssgetpy/MM. "
            "Defaults to $SAPS_CACHE_DIR/suitesparse."
        ),
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=SOLVER_NAMES,
        help="Solver to run. Repeat to run multiple solvers. Defaults to all solvers.",
    )
    parser.add_argument(
        "--matrix-kind",
        action="append",
        help=(
            "Only inspect SuiteSparse entries with this matrix kind. Repeat to "
            "include multiple kinds."
        ),
    )
    parser.add_argument(
        "--matrix",
        action="append",
        help="Only inspect this group/name SuiteSparse entry. Repeat for more.",
    )
    parser.add_argument(
        "--max-nnz",
        type=int,
        help="Only inspect SuiteSparse entries with at most this many nonzeros.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()

    if args.chunk_count < 1:
        parser.error("--chunk-count must be at least 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
        parser.error("--chunk-index must be between 0 and --chunk-count - 1")

    # Like bin/run_benchmark.py, default to this checkout's cache, not the cwd's.
    os.environ.setdefault("SAPS_CACHE_DIR", str(REPO_DIR / DEFAULT_CACHE_DIR))
    xp = load_framework(
        os.environ.get("SAPS_FRAMEWORK", REPO_DIR / "frameworks" / "saps_scipy.py")
    )
    solvers = args.solver if args.solver is not None else list(SOLVER_NAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = set() if args.force else _done_record_keys(args.output)
    matrices = _search_matrices(args)

    with (
        _saps_suitesparse_context(args.data_dir, args.seed),
        args.output.open(
            "a",
            encoding="utf-8",
        ) as output,
    ):
        for matrix in matrices:
            try:
                variants = _rhs_variants(matrix, solvers, args.data_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to inspect {_matrix_source_name(matrix)}: {exc}")
                record = {
                    "matrix_name": matrix.name,
                    "matrix_group": matrix.group,
                    "source_name": _matrix_source_name(matrix),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                continue

            for variant in variants:
                key = _record_key(matrix.group, matrix.name, variant.index)
                label = _record_label(matrix, variant)
                if key in completed:
                    print(f"Skipping {label}; already present in {args.output}")
                    continue

                print(f"Checking {label}", flush=True)
                try:
                    record = _matrix_record(matrix, variant, solvers, xp)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to check {label}: {exc}")
                    record = {
                        "matrix_name": matrix.name,
                        "matrix_group": matrix.group,
                        "source_name": _matrix_source_name(matrix),
                        "rhs_kind": variant.kind,
                        "rhs_index": variant.index,
                        "rhs_count": variant.count,
                        "rhs_shape": (
                            list(variant.shape) if variant.shape is not None else None
                        ),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    if variant.error is not None:
                        record["rhs_error"] = variant.error

                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
