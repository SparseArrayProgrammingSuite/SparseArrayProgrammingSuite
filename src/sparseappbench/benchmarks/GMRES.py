import os

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
Name: GMRES (Generalized Minimal Residual Method)

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do:
This code is implements the GMRES algorithm for solving indefinite
and non-symmetric linear systems. The algorithm follows the Arnoldi
iteration process where a Krylov matrix is maintained at each iteration.
Starting with an initial guess and the residual for that guess, the matrix
A is dot producted with the previous residual to obtain the next basis vector.
This algorithm also uses a similar method to Gram-Schmidt to ensure that
the Kyrlov matrix is orthogonal. I also maintain an upper Hessenberg matrix
which keeps track of the dot products between different basis vectors and
the norm of the new basis vector. The Hessenberg matrix follows the property:
Q_n * A = Q_(n+1) * H_n where Q is the Krylov matrix. This matrix allows
for a simplified least squares problem to be solved at each iteration so that
the residual is minimized at each step. My implementation restarts the Kyrlov
matrix every 50 iterations and will end when the current residual / initial residual
is less than the tolerance level.

Citation for reference implementation:
https://github.com/SparseApplicationBenchmark/SparseApplicationBenchmark/pull/45/files#diff-ba19ac630cf7b27852173e387c91d502f769e88858efa82bda51b1d1e8861a59

Motivation: GMRES is the most widely used and effective method for solving linear
systems that are indefinite, non-symmetric, and are sparse in nature.
https://www.netlib.org/templates/templates.pdf
https://www.netlib.org/utk/people/JackDongarra/PAPERS/sparse-bench.pdf

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. Generative AI might have been used to construct tests.
This statement is written by hand.
"""


def gmres(
    xp, A_binsparse, b_binsparse, x0_binsparse, restart=50, tol=1e-8, max_iter=1000
):
    A = xp.lazy(xp.from_benchmark(A_binsparse))
    b = xp.lazy(xp.from_benchmark(b_binsparse))
    x0 = xp.lazy(xp.from_benchmark(x0_binsparse))

    itcount = 0
    r0 = b - A @ x0
    initial_beta = xp.compute(xp.linalg.norm(r0))[()]
    if initial_beta < tol:
        return xp.to_benchmark(xp.compute(x0))

    rcurr = r0 / initial_beta
    beta = initial_beta

    while itcount < max_iter:
        Q = xp.zeros((A.shape[0], restart + 1), dtype=float)
        H = xp.zeros((restart + 1, restart), dtype=float)
        Q[:, 0] = rcurr

        x_cycle_start = x0
        for i in range(restart):
            x0 = xp.lazy((x0, rcurr))
            rcurr = A @ Q[:, i]

            H[: i + 1, i] = xp.compute(xp.vecdot(Q[:, : i + 1].T, rcurr))
            rcurr = rcurr - Q[:, : i + 1] @ H[: i + 1, i]
            H[i + 1, i] = xp.compute(xp.linalg.norm(rcurr))[()]
            Q[:, i + 1] = rcurr / H[i + 1, i]

            e1 = xp.zeros((i + 2,), dtype=float)
            e1[0] = beta

            H_reduced = H[: i + 2, : i + 1]
            coeffs, _, _, _ = xp.linalg.lstsq(H_reduced, e1, rcond=None)
            x0 = x_cycle_start + Q[:, : i + 1] @ coeffs

            r0 = b - A @ x0
            r0_norm = xp.compute(xp.linalg.norm(r0))[()]
            rcurr = r0 / r0_norm
            if r0_norm / initial_beta < tol:
                return xp.to_benchmark(xp.compute(x0))

            itcount += 1
            if itcount >= max_iter:
                break

        beta = r0_norm

    xsol = xp.compute(x0)
    return xp.to_benchmark(xsol)


def generate_gmres_data(source, has_b_file=False):
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


def dg_gmres_sparse_1():
    return generate_gmres_data("mesh3em5")


def dg_gmres_sparse_2():
    return generate_gmres_data("bcsstm02")


def dg_gmres_sparse_3():
    return generate_gmres_data("fv1")


def dg_gmres_sparse_4():
    return generate_gmres_data("Muu")


def dg_gmres_sparse_5():
    return generate_gmres_data("Chem97ZtZ")


def dg_gmres_sparse_6():
    return generate_gmres_data("Dubcova1")


def dg_gmres_sparse_7():
    return generate_gmres_data("t3dl_e")


def dg_gmres_sparse_8():
    return generate_gmres_data("bcsstk09")
