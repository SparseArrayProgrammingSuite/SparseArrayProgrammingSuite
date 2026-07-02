#!/usr/bin/env python3
"""
Script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import ssgetpy
from frameworks.saps_scipy import SciPyFramework
from saps.benchmarks.cg import CGBenchmark, CGDataset
from saps.benchmarks.jacobi import JacobiBenchmark, JacobiDataset
from saps.benchmarks.lsqr import LSQRBenchmark, LSQRDataset
from saps.benchmarks.preconditioned_cg import (
    JacobiPreconditionedCGBenchmark,
    PreconditionedCGDataset,
    PreconditionedCGBenchmark,
)


@dataclass(frozen=True)
class SolverRunStats:
    iterations: int
    tolerance: float
    converged: bool


def iterations_key(solver):
    return f"{solver} iterations"


def tolerance_key(solver):
    return f"{solver} tolerance"


def converged_key(solver):
    return f"{solver} converged"


def append_to_json(
    filename,
    matrix_name,
    matrix_group,
    stats,
    m,
    n,
    nnz,
    solver,
):
    """Append matrix name and solver iteration stats to JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    iter_key = iterations_key(solver)
    tol_key = tolerance_key(solver)
    conv_key = converged_key(solver)
    if any(
        entry["matrix_name"] == matrix_name and iter_key in entry
        for entry in data
    ):
        return False

    data = [
        entry
        for entry in data
        if iter_key in entry and entry["matrix_name"] != matrix_name
    ]
    data.append(
        {
            "matrix_name": matrix_name,
            "matrix_group": matrix_group,
            iter_key: int(stats.iterations),
            tol_key: float(stats.tolerance),
            conv_key: bool(stats.converged),
            "m": m,
            "n": n,
            "nnz": nnz,
        }
    )

    data.sort(key=lambda x: (x[iter_key], x["matrix_name"]))

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return True


def already_in_json(filename, matrix_name, solver):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
            iter_key = iterations_key(solver)
            return any(
                entry["matrix_name"] == matrix_name and iter_key in entry
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
    "jacobi": (JacobiDataset, JacobiBenchmark),
    "cg": (CGDataset, CGBenchmark),
    "jacobi_cg": (PreconditionedCGDataset, JacobiPreconditionedCGBenchmark),
    "block_jacobi_cg": (PreconditionedCGDataset, PreconditionedCGBenchmark),
    "lsqr": (LSQRDataset, LSQRBenchmark),
}
SOLVERS = tuple(SOLVER_CONFIGS)


def make_dataset(solver, matrix):
    dataset_cls, _benchmark_cls = SOLVER_CONFIGS[solver]
    common_kwargs = {
        "rel_tol": DEFAULT_REL_TOL,
        "abs_tol": DEFAULT_ABS_TOL,
        "max_iters": DEFAULT_MAX_ITERS,
    }
    if dataset_cls is PreconditionedCGDataset:
        return dataset_cls(matrix.name, "", **common_kwargs)
    return dataset_cls(matrix.name, nnz=matrix.nnz, **common_kwargs)


def make_benchmark(solver):
    _dataset_cls, benchmark_cls = SOLVER_CONFIGS[solver]
    return benchmark_cls()


def suite_generator(benchmark):
    for generator in benchmark.generators:
        if "test" not in generator.suites:
            return generator
    raise ValueError(f"No SuiteSparse generator found for {benchmark.name}")


def generate_data_with_existing_generator(solver, matrix):
    benchmark = make_benchmark(solver)
    generator = suite_generator(benchmark)
    dataset = make_dataset(solver, matrix)
    print(f"Generating {matrix.name} for {solver} with {generator.name}")
    problem = generator.generate(dataset)
    data = [SCIPY_XP.from_binsparse(item) for item in problem.inputs]
    return data, problem.meta


def vector_norm(x):
    return float(np.linalg.norm(np.asarray(x).ravel()))


def vector_dot(x, y):
    return float(np.dot(np.asarray(x).ravel(), np.asarray(y).ravel()))


def linear_system_tolerance(b, meta):
    b = np.asarray(b).ravel()
    return float(max(meta["rel_tol"] * vector_norm(b), meta["abs_tol"]))


def jacobi_run_stats(data, meta):
    A, b, x = data
    b = np.asarray(b).ravel()
    x = np.asarray(x).ravel()
    tolerance = linear_system_tolerance(b, meta)
    d = np.asarray(A.diagonal()).ravel()
    if np.any(d == 0):
        raise ValueError("Jacobi requires nonzero diagonal entries.")

    r = np.asarray(b - A @ x).ravel()
    rnorm = vector_norm(r)
    iterations = 0

    while rnorm >= tolerance and iterations < meta["max_iters"]:
        x = x + r / d
        r = np.asarray(b - A @ x).ravel()
        rnorm = vector_norm(r)
        iterations += 1

    return SolverRunStats(
        iterations=iterations,
        tolerance=tolerance,
        converged=bool(np.isfinite(rnorm) and rnorm < tolerance),
    )


def cg_run_stats(data, meta):
    A, b, x = data
    b = np.asarray(b).ravel()
    x = np.asarray(x).ravel()
    tolerance = linear_system_tolerance(b, meta)
    tol_sq = tolerance * tolerance

    r = np.asarray(b - A @ x).ravel()
    p = r
    rr = vector_dot(r, r)
    iterations = 0

    if rr >= tol_sq:
        while iterations < meta["max_iters"]:
            Ap = np.asarray(A @ p).ravel()
            denominator = vector_dot(p, Ap)
            if denominator == 0:
                raise ValueError("CG encountered a zero search-direction denominator.")
            alpha = rr / denominator
            x = x + alpha * p
            r = r - alpha * Ap

            old_rr = rr
            rr = vector_dot(r, r)
            iterations += 1

            if rr < tol_sq:
                break

            beta = rr / old_rr
            p = r + beta * p

    return SolverRunStats(
        iterations=iterations,
        tolerance=tolerance,
        converged=bool(np.isfinite(rr) and rr < tol_sq),
    )


def preconditioned_cg_run_stats(solver, data, meta):
    A, b, x, M = data
    benchmark = make_benchmark(solver)
    b = np.asarray(b).ravel()
    x = np.asarray(x).ravel()
    tolerance = linear_system_tolerance(b, meta)
    tol_sq = tolerance * tolerance

    r = np.asarray(b - A @ x).ravel()
    z = benchmark._solve_cg(SCIPY_XP, M, r)
    rho = vector_dot(r, z)
    p = z
    rr = vector_dot(r, r)
    iterations = 0

    if rr >= tol_sq:
        while iterations < meta["max_iters"]:
            Ap = np.asarray(A @ p).ravel()
            denominator = vector_dot(p, Ap)
            if denominator == 0:
                raise ValueError(
                    "Preconditioned CG encountered a zero search-direction "
                    "denominator."
                )
            alpha = rho / denominator
            x = x + alpha * p
            r = r - alpha * Ap

            rr = vector_dot(r, r)
            iterations += 1

            if rr < tol_sq:
                break

            z = benchmark._solve_cg(SCIPY_XP, M, r)
            new_rho = vector_dot(r, z)
            if rho == 0:
                raise ValueError("Preconditioned CG encountered a zero rho value.")
            beta = new_rho / rho
            p = z + beta * p
            rho = new_rho

    return SolverRunStats(
        iterations=iterations,
        tolerance=tolerance,
        converged=bool(np.isfinite(rr) and rr < tol_sq),
    )


def lsqr_run_stats(data, meta):
    A, b = data
    b = np.asarray(b).ravel()
    rel_tol = meta["rel_tol"]
    abs_tol = meta["abs_tol"]
    atol = meta.get("atol", rel_tol)
    btol = meta.get("btol", rel_tol)
    conlim = meta.get("conlim", 1.0e8)
    max_iters = meta["max_iters"]

    u = b
    beta = vector_norm(u)
    if beta == 0:
        return SolverRunStats(iterations=0, tolerance=float(abs_tol), converged=True)
    u = u / beta

    v = np.asarray(A.T @ u).ravel()
    alpha = vector_norm(v)
    if alpha == 0:
        return SolverRunStats(iterations=0, tolerance=float(abs_tol), converged=True)
    v = v / alpha

    solution_is_zero = False
    bnorm = beta
    ctol = 1 / conlim

    Arnorm = alpha * beta
    if Arnorm == 0:
        solution_is_zero = True

    w = v
    phi_bar = beta
    rho_bar = alpha
    iterations = 0

    Anorm_sq = beta**2
    xnorm_sq = 0
    dnorm_sq = 0
    Acond = 0
    exit_reason = 0
    tolerance = float(rel_tol)

    while iterations < max_iters and not solution_is_zero:
        iterations += 1

        u = np.asarray(A @ v - alpha * u).ravel()

        beta = vector_norm(u)
        if beta == 0:
            exit_reason = 1
            break
        u = u / beta

        v = np.asarray(A.T @ u - beta * v).ravel()
        alpha = vector_norm(v)
        if alpha == 0:
            exit_reason = 2
            break
        v = v / alpha

        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar *= s
        step = phi / rho

        dk = 1.0 / rho * w
        dnorm_sq += np.sum(np.multiply(dk, dk))

        w = v - (theta / rho) * w

        rnorm = abs(phi_bar)
        Arnorm = alpha * abs(phi_bar * c)

        Anorm_sq += alpha**2 + beta**2
        Anorm = np.sqrt(Anorm_sq)

        xnorm_sq += step**2
        xnorm = np.sqrt(xnorm_sq)

        Acond = Anorm * np.sqrt(dnorm_sq)

        test1 = rnorm / bnorm
        test2 = Arnorm / (Anorm * rnorm)
        test3 = 1 / Acond

        tolerance = float(atol * Anorm * xnorm / bnorm + btol)

        if test3 <= ctol:
            exit_reason = 3
        if test2 <= atol:
            exit_reason = 2
        if test1 <= tolerance or rnorm <= abs_tol:
            exit_reason = 1

        if exit_reason > 0:
            break

    return SolverRunStats(
        iterations=iterations,
        tolerance=tolerance,
        converged=exit_reason in (1, 2),
    )


def solver_run_stats(solver, data, meta):
    if solver == "jacobi":
        return jacobi_run_stats(data, meta)
    if solver == "cg":
        return cg_run_stats(data, meta)
    if solver in ("jacobi_cg", "block_jacobi_cg"):
        return preconditioned_cg_run_stats(solver, data, meta)
    if solver == "lsqr":
        return lsqr_run_stats(data, meta)
    raise ValueError(f"Unknown solver: {solver}")


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
        help="Output JSON file for matrices and solver iteration stats",
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
        data, meta = generate_data_with_existing_generator(solver, matrix)
        stats = solver_run_stats(solver, data, meta)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"Error computing {solver} iteration stats for {matrix.name}: {e}")
        return "error"

    saved = append_to_json(
        output_file,
        matrix.name,
        matrix.group,
        stats,
        m,
        n,
        matrix.nnz,
        solver,
    )
    if saved:
        print(
            f"Saved {matrix.name} {solver} iterations={stats.iterations}, "
            f"tolerance={stats.tolerance}, converged={stats.converged} "
            f"to {output_file}"
        )
        return "saved"
    else:
        print(f"Skipping {matrix.name}, already in {output_file}")
        return "already"


if __name__ == "__main__":
    main()
