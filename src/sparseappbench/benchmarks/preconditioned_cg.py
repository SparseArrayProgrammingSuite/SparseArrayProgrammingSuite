import os

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
Name: Preconditioned Conjugate Gradient (Block Jacobi)
Author: Benjamin Berol
Email: bberol3@gatech.edu
Motivation:

Role of Sparsity:

Implementation:
Hand-written code modelling the algorithm structure outlined in:
https://www.netlib.org/templates/templates.pdf Page 13
Data Generation:
Data collected from SuiteSparse Matrix Collection consisting of symmetric
positive definite matrices, particularly those with a low convergence criteria.
Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


def benchmark_cg(
    xp, A_bench, M_bench, b_bench, x_bench, rel_tol=1e-8, abs_tol=1e-20, max_iters=10000
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    M = xp.lazy(xp.from_benchmark(M_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    x = xp.lazy(xp.from_benchmark(x_bench))

    tolerance = max(
        xp.compute(xp.lazy(rel_tol) * xp.sqrt(xp.vecdot(b, b)))[()], abs_tol
    )
    # tol_sq used to avoid having to sqrt dot products when checking tolerance
    tol_sq = tolerance * tolerance

    r = b - A @ x
    z = solve_block_jacobi_cg(M, r)
    rho = xp.vecdot(r, z)
    p = z
    it = 0
    rr = xp.compute(xp.vecdot(r, r))[()]

    if rr >= tol_sq:
        while it < max_iters:
            x = xp.lazy(x)
            r = xp.lazy(r)
            p = xp.lazy(p)

            Ap = A @ p
            alpha = rho / xp.vecdot(p, Ap)
            x += alpha * p
            r -= alpha * Ap

            x = xp.compute(x)
            r = xp.compute(r)
            p = xp.compute(p)
            new_rr = xp.compute(xp.vecdot(r, r))[()]

            it += 1

            if new_rr < tol_sq:
                break

            z = solve_block_jacobi_cg(M, r)
            new_rho = xp.vecdot(r, z)
            beta = new_rho / rho
            p = z + beta * p
            rho = new_rho
            rr = new_rr

    x_solution = xp.compute(x)
    return xp.to_benchmark(x_solution)


def solve_block_jacobi_cg():
    return


def generate_cg_data(source, has_b_file=False):
    matrices = ssgetpy.search(name=source)
    if not matrices:
        raise ValueError(f"No matrix found with name '{source}'")
    matrix = matrices[0]
    (path, archive) = matrix.download(extract=True)
    matrix_path = os.path.join(path, matrix.name + ".mtx")
    if matrix_path and os.path.exists(matrix_path):
        A = mmread(matrix_path)
    else:
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    rng = np.random.default_rng(0)
    A = A.tocoo()

    if has_b_file:
        matrix_path = os.path.join(path, matrix.name + "_b.mtx")
        if matrix_path and os.path.exists(matrix_path):
            b = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        if not isinstance(b, np.ndarray):
            b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        b = b.flatten()
    else:
        x = random(
            A.shape[1], 1, density=0.1, format="coo", dtype=np.float64, random_state=rng
        )
        b = A @ x
        b = b.toarray().flatten()
    x = np.zeros(A.shape[1])

    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    b_bin = BinsparseFormat.from_numpy(b)
    x_bin = BinsparseFormat.from_numpy(x)
    return (A_bin, b_bin, x_bin)


def dg_cg_sparse_1():
    return generate_cg_data("mesh3em5")


def dg_cg_sparse_2():
    return generate_cg_data("bcsstm02")


def dg_cg_sparse_3():
    return generate_cg_data("fv1")


def dg_cg_sparse_4():
    return generate_cg_data("Muu")


def dg_cg_sparse_5():
    return generate_cg_data("Chem97ZtZ")


def dg_cg_sparse_6():
    return generate_cg_data("Dubcova1")


def dg_cg_sparse_7():
    return generate_cg_data("t3dl_e")


def dg_cg_sparse_8():
    return generate_cg_data("bcsstk09")
