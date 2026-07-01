#!/usr/bin/env python3
"""
Script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import scipy as sp
from scipy.io import mmread
from scipy.sparse.linalg._eigen.arpack import ArpackError

import ssgetpy


def append_to_json(
    filename,
    matrix_name,
    matrix_group,
    convergence_value,
    m,
    n,
    nnz,
    solver,
):
    """Append matrix name and normalized convergence criteria to JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    if any(entry["matrix_name"] == matrix_name for entry in data):
        return False

    data.append(
        {
            "matrix_name": matrix_name,
            "matrix_group": matrix_group,
            f"{solver} convergence criteria": convergence_value,
            "m": m,
            "n": n,
            "nnz": nnz,
        }
    )

    data.sort(key=lambda x: x[f"{solver} convergence criteria"])

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return True


def already_in_json(filename, matrix_name):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
            return any(entry["matrix_name"] == matrix_name for entry in data)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def check_jacobi_normalized_convergence(A, tol=1e-3):
    d = A.diagonal()
    D = sp.sparse.diags(1 / d, format="csr")
    M = -(D @ A - sp.sparse.eye(A.shape[0]))

    vals = sp.sparse.linalg.eigs(M, k=1, return_eigenvectors=False, tol=tol)
    sr_value = abs(np.max(vals[0]))
    print(f"Normalized Jacobi convergence factor: {sr_value}")
    return sr_value


def check_cg_normalized_convergence(A, M=None, tol=1e-3):
    max_eig = sp.sparse.linalg.eigsh(
        A, M=M, k=1, return_eigenvectors=False, tol=tol / (tol + 2)
    )[0]
    min_eig = sp.sparse.linalg.eigsh(
        A, M=M, k=1, sigma=0, return_eigenvectors=False, tol=tol / (tol + 2)
    )[0]

    condition_num = max_eig / min_eig
    if condition_num < 1:
        convergence_value = np.inf
    elif condition_num == 1:
        convergence_value = 0
    else:
        sqrt_kappa = math.sqrt(condition_num)
        convergence_value = (sqrt_kappa - 1) / (sqrt_kappa + 1)
    print(f"Condition number of A: {condition_num}")
    print(f"Normalized CG convergence factor: {convergence_value}")
    return convergence_value


def check_jacobi_cg_normalized_convergence(A, tol=1e-3):
    M = sp.sparse.diags(A.diagonal())
    return check_cg_normalized_convergence(A, M, tol=tol)


def check_block_jacobi_cg_normalized_convergence(A, tol=1e-3):
    A_csr = A.tocsr()
    n = A_csr.shape[0]
    p = min(10, n)
    block_size = n // p
    blocks = []
    i = 0
    while i < n:
        j = min(i + block_size, n)
        A_ii = A_csr[i:j, i:j].toarray()
        blocks.append(A_ii)
        i = j
    M = sp.sparse.block_diag(blocks)
    return check_cg_normalized_convergence(A, M, tol=tol)


def check_lsqr_normalized_convergence(A, tol=1e-3):
    try:
        # Compute the largest singular value
        max_s = sp.sparse.linalg.svds(
            A,
            k=1,
            which="LM",
            return_singular_vectors=False,
            tol=math.sqrt(tol / (tol + 2)),
        )[0]
    except ArpackError:
        print("Could not compute largest singular value for matrix.")
        return np.inf

    try:
        # Compute the smallest singular value
        min_s = sp.sparse.linalg.svds(
            A,
            k=1,
            which="SM",
            return_singular_vectors=False,
            tol=math.sqrt(tol / (tol + 2)),
        )[0]
    except ArpackError:
        print("Could not compute smallest singular value for matrix.")
        return np.inf

    condition_num = max_s / min_s if min_s != 0 else np.inf
    if condition_num < 1:
        convergence_value = np.inf
    elif condition_num == 1:
        convergence_value = 0
    else:
        convergence_value = (condition_num - 1) / (condition_num + 1)
    print(f"Condition number of A (LSQR): {condition_num}")
    print(f"Normalized LSQR convergence factor: {convergence_value}")
    return convergence_value


SOLVER_DICT = {
    "jacobi": check_jacobi_normalized_convergence,
    "cg": check_cg_normalized_convergence,
    "jacobi_cg": check_jacobi_cg_normalized_convergence,
    "block_jacobi_cg": check_block_jacobi_cg_normalized_convergence,
    "lsqr": check_lsqr_normalized_convergence,
}

TOLERANCE_DICT = {
    "jacobi": 1e-6,
    "cg": 1e-6,
    "jacobi_cg": 1e-6,
    "block_jacobi_cg": 1e-6,
    "lsqr": 1e-6,
}

MAXIT_DICT = {
    "jacobi": 1000,
    "cg": 1000,
    "jacobi_cg": 1000,
    "block_jacobi_cg": 1000,
    "lsqr": 1000,
}


def convergence_threshold(solver):
    """Return the largest factor that proves convergence within max iterations."""
    tolerance = TOLERANCE_DICT[solver]
    max_iterations = MAXIT_DICT[solver]
    if tolerance <= 0:
        raise ValueError("solver tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("solver max iterations must be positive")
    return math.pow(tolerance, 1 / max_iterations)


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
        help="Output JSON file for matrices and convergence criteria",
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
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Tolerance to use for eigensolver-based convergence estimates",
    )
    args = parser.parse_args()
    if args.num_batches < 1:
        parser.error("--num-batches must be at least 1")
    if args.batch_index < 0 or args.batch_index >= args.num_batches:
        parser.error("--batch-index must satisfy 0 <= batch-index < num-batches")
    if args.tol < 0:
        parser.error("--tol must be non-negative")
    search_params = {"limit": -1}
    if args.maxsize is not None:
        search_params["nzbounds"] = (0, args.maxsize)
    matrices = list(ssgetpy.search(**search_params))
    total_matrices = len(matrices)
    matrices = matrices[args.batch_index :: args.num_batches]
    print(
        f"Processing batch {args.batch_index + 1}/{args.num_batches}: "
        f"{len(matrices)} of {total_matrices} matrices"
    )

    for matrix in matrices:
        path, _archive = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        print(f"Matrix: {matrix.name}, Path: {matrix_path}")
        if not matrix_path or not os.path.exists(matrix_path):
            continue

        A = mmread(matrix_path)  # This is the full sparse matrix
        m, n = A.shape
        if A.shape[0] <= 1 or A.shape[1] <= 1:
            continue

        # Convert to CSR format if needed for better diagonal access
        if not hasattr(A, "diagonal"):
            A = A.tocsr()

        for solver in SOLVER_DICT:
            if args.num_batches == 1:
                output_file = f"{solver}_{args.output}"
            else:
                output_file = f"{solver}_batch_{args.batch_index}_{args.output}"
            if already_in_json(output_file, matrix.name):
                print(f"Skipping {matrix.name}, already in {output_file}")
                continue
            if solver != "lsqr" and m != n:
                print(
                    f"Skipping non-square matrix {matrix.name}"
                    f" of shape {A.shape} for {solver}"
                )
                continue
            if solver != "lsqr" and not matrix.isspd:
                print(f"Skipping non-SPD matrix {matrix.name} for {solver}")
                continue
            calculate_and_save_solver_result(
                output_file,
                matrix,
                A,
                m,
                n,
                solver,
                args.tol,
            )


def calculate_and_save_solver_result(output_file, matrix, A, m, n, solver, tol=1e-3):
    threshold = convergence_threshold(solver)

    for tol in [0.05, tol]:
        try:
            convergence_value = SOLVER_DICT[solver](A, tol=tol)
        except (ArpackError, RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            print(f"Error computing {solver} convergence for {matrix.name}: {e}")
            return

        if np.isinf(convergence_value) or np.isnan(convergence_value):
            convergence_value = sys.float_info.max

        if convergence_value/(1+tol) >= threshold:
            print(
                f"Skipping {matrix.name} for {solver}: normalized convergence factor "
                f"{convergence_value} cannot converge within "
                f"{MAXIT_DICT[solver]} iterations to the tolerance {TOLERANCE_DICT[solver]} "
                f"(requires < {threshold}) (tol={tol})"
            )
            return

    if convergence_value*(1+tol) >= threshold:
        print(
            f"Skipping {matrix.name} for {solver}: normalized convergence factor "
            f"{convergence_value} does not converge within "
            f"{MAXIT_DICT[solver]} iterations to the tolerance {TOLERANCE_DICT[solver]} "
            f"(requires < {threshold}) (tol={tol})"
        )
        return

    # Write to JSON file
    saved = append_to_json(
        output_file,
        matrix.name,
        matrix.group,
        float(convergence_value),
        m,
        n,
        A.nnz,
        solver,
    )
    if saved:
        print(f"Saved {matrix.name} {solver} convergence criteria to {output_file}")
    else:
        print(f"Skipping {matrix.name}, already in {output_file}")


if __name__ == "__main__":
    main()
