#!/usr/bin/env python3
"""
Script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import argparse
import json
import numpy as np
import scipy.sparse as sparse

import ssgetpy
from frameworks.saps_scipy import SciPyFramework
from saps.benchmarks.cg import CGBenchmark, CGDataset, CGGenerator
from saps.benchmarks.jacobi import JacobiBenchmark, JacobiDataset, JacobiGenerator
from saps.benchmarks.lsqr import LSQRBenchmark, LSQRDataset, LSQRGenerator
from saps.benchmarks.preconditioned_cg import (
    BlockJacobiCGGenerator,
    JacobiCGGenerator,
    JacobiPreconditionedCGBenchmark,
    PreconditionedCGDataset,
    PreconditionedCGBenchmark,
)


def append_to_json(
    filename,
    matrix_name,
    matrix_group,
    convergence_metric,
    m,
    n,
    nnz,
    solver,
):
    """Append matrix name and benchmark convergence metric to JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    metric_key = f"{solver} convergence metric"
    if any(
        entry["matrix_name"] == matrix_name and metric_key in entry
        for entry in data
    ):
        return False

    data = [
        entry
        for entry in data
        if metric_key in entry and entry["matrix_name"] != matrix_name
    ]
    data.append(
        {
            "matrix_name": matrix_name,
            "matrix_group": matrix_group,
            metric_key: convergence_metric,
            "m": m,
            "n": n,
            "nnz": nnz,
        }
    )

    data.sort(key=lambda x: x[metric_key])

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return True


def already_in_json(filename, matrix_name, solver):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
            metric_key = f"{solver} convergence metric"
            return any(
                entry["matrix_name"] == matrix_name and metric_key in entry
                for entry in data
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return False


DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-20
DEFAULT_MAX_ITERS = 1000

SOLVER_STATUS_KEYS = ("saved", "skipped", "already", "error")
SCIPY_XP = SciPyFramework()

SOLVER_CONFIGS = {
    "jacobi": (JacobiDataset, JacobiGenerator, JacobiBenchmark),
    "cg": (CGDataset, CGGenerator, CGBenchmark),
    "jacobi_cg": (
        PreconditionedCGDataset,
        JacobiCGGenerator,
        JacobiPreconditionedCGBenchmark,
    ),
    "block_jacobi_cg": (
        PreconditionedCGDataset,
        BlockJacobiCGGenerator,
        PreconditionedCGBenchmark,
    ),
    "lsqr": (LSQRDataset, LSQRGenerator, LSQRBenchmark),
}
SOLVERS = tuple(SOLVER_CONFIGS)


def make_dataset(solver, matrix):
    dataset_cls, _generator_cls, _benchmark_cls = SOLVER_CONFIGS[solver]
    common_kwargs = {
        "rel_tol": DEFAULT_REL_TOL,
        "abs_tol": DEFAULT_ABS_TOL,
        "max_iters": DEFAULT_MAX_ITERS,
    }
    if dataset_cls is PreconditionedCGDataset:
        return dataset_cls(matrix.name, "", **common_kwargs)
    return dataset_cls(matrix.name, nnz=matrix.nnz, **common_kwargs)


def make_generator(solver):
    _dataset_cls, generator_cls, _benchmark_cls = SOLVER_CONFIGS[solver]
    return generator_cls()


def make_benchmark(solver):
    _dataset_cls, _generator_cls, benchmark_cls = SOLVER_CONFIGS[solver]
    return benchmark_cls()


def generate_solver_data(solver, matrix):
    problem = make_generator(solver).generate(make_dataset(solver, matrix))
    data = [SCIPY_XP.from_binsparse(item) for item in problem.inputs]
    return data, problem.meta


def norm(x):
    return np.linalg.norm(np.asarray(x).ravel())


def frobenius_norm(A):
    if sparse.issparse(A):
        return np.sqrt(np.sum(A.data * A.data))
    return np.linalg.norm(np.asarray(A), ord="fro")


def linear_system_convergence_metric(A, b, x, meta):
    b = np.asarray(b).ravel()
    x = np.asarray(x).ravel()
    residual = b - A @ x
    tolerance = max(meta["rel_tol"] * norm(b), meta["abs_tol"])
    return norm(residual) / tolerance


def lsqr_convergence_metric(A, b, x, meta):
    b = np.asarray(b).ravel()
    x = np.asarray(x).ravel()
    residual = b - A @ x
    rnorm = norm(residual)
    if rnorm <= meta["abs_tol"]:
        return 0.0

    bnorm = norm(b)
    if bnorm == 0:
        return 0.0 if rnorm == 0 else np.inf

    anorm = frobenius_norm(A)
    xnorm = norm(x)
    residual_threshold = meta["rel_tol"] * anorm * xnorm + meta["rel_tol"] * bnorm
    residual_metric = rnorm / max(residual_threshold, meta["abs_tol"])

    gradient = A.T @ residual
    gradient_threshold = meta["rel_tol"] * anorm * rnorm
    gradient_metric = norm(gradient) / gradient_threshold if gradient_threshold else np.inf
    return min(residual_metric, gradient_metric)


def benchmark_convergence_metric(solver, data, meta):
    output = make_benchmark(solver).benchmark(SCIPY_XP, data, meta)
    x = output[0]
    if solver == "lsqr":
        A, b = data
        return lsqr_convergence_metric(A, b, x, meta)

    A, b = data[:2]
    return linear_system_convergence_metric(A, b, x, meta)


def record_solver_status(status_counts, solver, status, total_problems):
    status_counts[solver][status] += 1
    processed = sum(status_counts[solver].values())
    counts = ", ".join(
        f"{key}={status_counts[solver][key]}" for key in SOLVER_STATUS_KEYS
    )
    print(f"Status {solver}: {processed}/{total_problems} problems ({counts})")


def record_all_solver_status(status_counts, status, total_problems):
    for solver in SOLVERS:
        record_solver_status(status_counts, solver, status, total_problems)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape matrices from SuiteSparse Matrix Collection"
    )
    parser.add_argument(
        "--maxsize",
        type=float,
        default=None,
        help="Maximum matrix nnz to retrieve",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="matrices.json",
        help="Output JSON file for matrices and benchmark convergence metrics",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of disjoint matrix batches to split the search results into",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Zero-based batch index to process",
    )
    args = parser.parse_args()
    if args.num_batches < 1:
        parser.error("--num-batches must be at least 1")
    if args.batch_index < 0 or args.batch_index >= args.num_batches:
        parser.error("--batch-index must satisfy 0 <= batch-index < num-batches")
    search_params = {"limit": -1}
    if args.maxsize is not None:
        search_params["nzbounds"] = (0, args.maxsize)
    all_matrices = list(ssgetpy.search(**search_params))
    total_matrices = len(all_matrices)
    matrices = [
        matrix
        for matrix_index, matrix in enumerate(all_matrices)
        if matrix_index % args.num_batches == args.batch_index
    ]
    print(
        f"Processing batch {args.batch_index + 1}/{args.num_batches}: "
        f"{len(matrices)} of {total_matrices} matrices"
    )
    status_counts = {
        solver: {key: 0 for key in SOLVER_STATUS_KEYS} for solver in SOLVERS
    }

    for problem_index, matrix in enumerate(matrices, start=1):
        print(
            f"Starting problem: {matrix.name} "
            f"({problem_index}/{len(matrices)}, kind={matrix.kind!r})"
        )
        m = matrix.rows
        n = matrix.cols
        if m <= 1 or n <= 1:
            print(f"Skipping matrix {matrix.name} with 1 dimensions")
            record_all_solver_status(status_counts, "skipped", len(matrices))
            continue

        matrix_kind = (matrix.kind or "").lower()
        for solver in SOLVERS:
            if args.num_batches == 1:
                output_file = f"{solver}_{args.output}"
            else:
                output_file = f"{solver}_batch_{args.batch_index}_{args.output}"
            if already_in_json(output_file, matrix.name, solver):
                print(f"Skipping {matrix.name}, already in {output_file}")
                record_solver_status(status_counts, solver, "already", len(matrices))
                continue
            if solver == "lsqr" and matrix_kind != "least squares problem":
                print(
                    f"Skipping matrix {matrix.name} of kind {matrix.kind!r} for {solver}"
                )
                record_solver_status(status_counts, solver, "skipped", len(matrices))
                continue

            if solver != "lsqr" and m != n:
                print(
                    f"Skipping non-square matrix {matrix.name}"
                    f" of shape {m}x{n} for {solver}"
                )
                record_solver_status(status_counts, solver, "skipped", len(matrices))
                continue
            if solver != "lsqr" and not matrix.isspd:
                print(f"Skipping non-SPD matrix {matrix.name} for {solver}")
                record_solver_status(status_counts, solver, "skipped", len(matrices))
                continue
            status = calculate_and_save_solver_result(
                output_file,
                matrix,
                m,
                n,
                solver,
            )
            record_solver_status(status_counts, solver, status, len(matrices))


def calculate_and_save_solver_result(output_file, matrix, m, n, solver):
    try:
        data, meta = generate_solver_data(solver, matrix)
        convergence_metric = benchmark_convergence_metric(solver, data, meta)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"Error computing {solver} convergence for {matrix.name}: {e}")
        return "error"

    if not np.isfinite(convergence_metric) or convergence_metric > 1:
        print(
            f"Skipping {matrix.name} for {solver}: benchmark convergence metric "
            f"{convergence_metric} exceeds 1 within {DEFAULT_MAX_ITERS} iterations"
        )
        return "skipped"

    saved = append_to_json(
        output_file,
        matrix.name,
        matrix.group,
        float(convergence_metric),
        m,
        n,
        matrix.nnz,
        solver,
    )
    if saved:
        print(
            f"Saved {matrix.name} {solver} convergence metric "
            f"{convergence_metric} to {output_file}"
        )
        return "saved"
    else:
        print(f"Skipping {matrix.name}, already in {output_file}")
        return "already"


if __name__ == "__main__":
    main()
