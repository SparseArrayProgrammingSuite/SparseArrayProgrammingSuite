#!/usr/bin/env python3
"""
Script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import argparse
import json
import os
import random

import numpy as np
import scipy as sp
from scipy.io import mmread
from scipy.sparse.linalg._eigen.arpack import ArpackError

import ssgetpy


def append_to_json(
    filename, matrix_name, matrix_group, convergence_value, n, nnz, solver
):
    """Append matrix name and convergence criteria to JSON file."""
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
            "n": n,
            "nnz": nnz,
        }
    )

    data.sort(key=lambda x: x[f"{solver} convergence criteria"])

    # Write back to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def already_in_json(filename, matrix_name):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)
            return any(entry["matrix_name"] == matrix_name for entry in data)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def check_jacobi_iteration_matrix_convergence(A):
    d = A.diagonal()
    D = sp.sparse.diags(1 / d, format="csr")
    M = -(D @ A - sp.sparse.eye(A.shape[0]))

    vals = sp.sparse.linalg.eigsh(M, k=1, return_eigenvectors=False, tol=0.001)
    sr_value = np.max(vals[0])
    print(f"SR of (A-D)/D: {sr_value}")
    return sr_value


def check_cg_iteration_matrix_convergence_speed(A, M=None):
    max_eig = sp.sparse.linalg.eigsh(A, M=M, k=1, return_eigenvectors=False, tol=0.001)[
        0
    ]
    min_eig = sp.sparse.linalg.eigsh(
        A, M=M, k=1, sigma=0, return_eigenvectors=False, tol=0.001
    )[0]

    condition_num = max_eig / min_eig
    print(f"Condition number of A: {condition_num}")
    return condition_num


def check_jacobi_cg_iteration_matrix_convergence_speed(A):
    M = sp.sparse.diags(A.diagonal())
    return check_cg_iteration_matrix_convergence_speed(A, M)


def check_block_jacobi_cg_iteration_matrix_convergence_speed(A):
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
    return check_cg_iteration_matrix_convergence_speed(A, M)


SOLVER_DICT = {
    "jacobi": check_jacobi_iteration_matrix_convergence,
    "cg": check_cg_iteration_matrix_convergence_speed,
    "jacobi_cg": check_jacobi_cg_iteration_matrix_convergence_speed,
    "block_jacobi_cg": check_block_jacobi_cg_iteration_matrix_convergence_speed,
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
        choices=["jacobi", "cg", "jacobi_cg", "block_jacobi_cg"],
        help="Solver to check convergence for",
    )
    args = parser.parse_args()

    # Get all matrices (use a large limit to get the full collection)
    matrices = ssgetpy.search(isspd=True, nzbounds=(0, args.maxsize), limit=args.limit)

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
            if m != n:
                print(f"Skipping non-square matrix {matrix.name} of shape {A.shape}")
                continue
            # Convert to CSR format if needed for better diagonal access
            if not hasattr(A, "diagonal"):
                A = A.tocsr()

            # Calculate the convergence criteria
            try:
                convergence_value = SOLVER_DICT[args.solver](A)

                # Write to JSON file
                append_to_json(
                    output_file,
                    matrix.name,
                    matrix.group,
                    float(convergence_value),
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
