#!/usr/bin/env python3
"""
Script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import argparse
import json
import math
import os
import random
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
    estimated_iterations,
    iteration_tolerance,
    n,
    nnz,
    solver,
):
    """Append matrix name and normalized convergence criteria to JSON file."""
    # Try to load existing data, or create empty list if file doesn't exist
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Append new entry
    data.append(
        {
            "matrix_name": matrix_name,
            "matrix_group": matrix_group,
            f"{solver} convergence criteria": convergence_value,
            f"{solver} estimated iterations": estimated_iterations,
            "estimated iteration tolerance": iteration_tolerance,
            "n": n,
            "nnz": nnz,
        }
    )

    data.sort(key=lambda x: x[f"{solver} convergence criteria"])

    # Write back to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def already_in_json(filename, matrix_name):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
            return any(entry["matrix_name"] == matrix_name for entry in data)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def estimate_iterations(convergence_value, tolerance):
    if convergence_value == 0:
        return 1
    if not 0 < convergence_value < 1:
        return None
    return max(1, math.ceil(math.log(tolerance) / math.log(convergence_value)))


def check_jacobi_normalized_convergence(A):
    d = A.diagonal()
    D = sp.sparse.diags(1 / d, format="csr")
    M = -(D @ A - sp.sparse.eye(A.shape[0]))

    vals = sp.sparse.linalg.eigsh(M, k=1, return_eigenvectors=False, tol=0.001)
    sr_value = abs(np.max(vals[0]))
    print(f"Normalized Jacobi convergence factor: {sr_value}")
    return sr_value


def check_cg_normalized_convergence(A, M=None):
    max_eig = sp.sparse.linalg.eigsh(A, M=M, k=1, return_eigenvectors=False, tol=0.001)[
        0
    ]
    min_eig = sp.sparse.linalg.eigsh(
        A, M=M, k=1, sigma=0, return_eigenvectors=False, tol=0.001
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


def check_jacobi_cg_normalized_convergence(A):
    M = sp.sparse.diags(A.diagonal())
    return check_cg_normalized_convergence(A, M)


def check_block_jacobi_cg_normalized_convergence(A):
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
    return check_cg_normalized_convergence(A, M)


def check_lsqr_normalized_convergence(A):
    try:
        # Compute the largest singular value
        max_s = sp.sparse.linalg.svds(
            A, k=1, which="LM", return_singular_vectors=False
        )[0]
    except ArpackError:
        print("Could not compute largest singular value for matrix.")
        return np.inf

    try:
        # Compute the smallest singular value
        min_s = sp.sparse.linalg.svds(
            A, k=1, which="SM", return_singular_vectors=False
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


def main():
    parser = argparse.ArgumentParser(
        description="Scrape matrices from SuiteSparse Matrix Collection"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Maximum number of matrices to retrieve",
    )
    parser.add_argument(
        "--maxsize", type=int, default=100000, help="Maximum matrix nnz to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="matrices.json",
        help="Output JSON file for matrices and convergence criteria",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="jacobi",
        choices=["jacobi", "cg", "jacobi_cg", "block_jacobi_cg", "lsqr"],
        help="Solver to check convergence for",
    )
    parser.add_argument(
        "--iteration-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance used when estimating iteration counts",
    )
    args = parser.parse_args()
    if not 0 < args.iteration_tolerance < 1:
        raise ValueError("--iteration-tolerance must be between 0 and 1.")

    search_params = {"nzbounds": (0, args.maxsize), "limit": args.limit}
    if args.solver == "cg" or args.solver == "jacobi":
        search_params["isspd"] = True
    matrices = ssgetpy.search(**search_params)

    # Take a random permutation
    matrices = random.sample(list(matrices), len(matrices))
    output_file = f"{args.solver}_{args.output}"
    for matrix in matrices:
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        print(f"Matrix: {matrix.name}, Path: {matrix_path}")
        if matrix_path and os.path.exists(matrix_path):
            if already_in_json(output_file, matrix.name):
                print(f"Skipping {matrix.name}, already in {output_file}")
                continue
            A = mmread(matrix_path)  # This is the full sparse matrix
            (m, n) = A.shape
            if args.solver != "lsqr" and m != n:
                print(
                    f"Skipping non-square matrix {matrix.name}"
                    f" of shape {A.shape} for {args.solver}"
                )
                continue
            # Convert to CSR format if needed for better diagonal access
            if not hasattr(A, "diagonal"):
                A = A.tocsr()

            # Calculate the convergence criteria
            try:
                if A.shape[0] > 1 and A.shape[1] > 1:
                    convergence_value = SOLVER_DICT[args.solver](A)
                    if np.isinf(convergence_value) or np.isnan(convergence_value):
                        convergence_value = sys.float_info.max
                    estimated_iterations = estimate_iterations(
                        convergence_value, args.iteration_tolerance
                    )

                    # Write to JSON file
                    append_to_json(
                        output_file,
                        matrix.name,
                        matrix.group,
                        float(convergence_value),
                        estimated_iterations,
                        args.iteration_tolerance,
                        n,
                        A.nnz,
                        args.solver,
                    )
                    print(f"Saved {matrix.name} convergence criteria to {output_file}")

            except ArpackError as e:
                print(f"Error computing convergence criteria for {matrix.name}: {e}")
                continue


if __name__ == "__main__":
    main()
