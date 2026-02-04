import os

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
Name: LSQR Iterative Solver
Author: Benjamin Berol
Email: bberol3@gatech.edu
Motivation:

Role of Sparsity:

Implementation:

Data Generation:

Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


def benchmark_lsqr(
    xp, A_bench, b_bench, atol=1e-9, btol=1e-9, conlim=1.0e8, max_iters=1000
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    exit = 0

    u = b
    beta = normof2(xp, u, u)
    u = u / beta

    v = A.T @ u
    alpha = normof2(xp, v, v)
    v = v / alpha

    solution_is_zero = False
    bnorm = beta
    ctol = 1 / conlim

    Arnorm = alpha * beta
    if Arnorm == 0:
        solution_is_zero = True

    w = v
    x = xp.zeros(A.shape[1])
    phi_bar = beta
    rho_bar = alpha
    it = 0

    # An approximation of the Frobenius norm of A squared using an
    # iterative update by summing the squares of the scalars alpha and beta
    Anorm_sq = beta**2

    # An approximation of the vector norm of x squared based on the
    # step size contributing each iteration
    xnorm_sq = 0

    # The Fronbenius norm squared of the matrix of search directions
    # updated by adding the squared norm of each search direction
    dnorm_sq = 0

    # An approximation of the condition number of A found by multiplying
    # Anorm by sqrt(ddnorm)
    Acond = 0

    while it < max_iters and not solution_is_zero:
        it += 1

        u = A @ v - alpha * u

        beta = normof2(xp, u, u)
        u = u / beta

        v = A.T @ u - beta * v
        alpha = normof2(xp, v, v)
        v = v / alpha

        rho = xp.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar *= s
        step = phi / rho

        x += step * w

        dk = 1.0 / rho * w
        dnorm_sq += xp.sum(xp.multiply(dk, dk))

        w = v - (theta / rho) * w

        # Estimate for the size of the residual r = b - Ax
        rnorm = abs(phi_bar)

        # Estimate of the norm of the gradient ATr
        Arnorm = alpha * abs(phi_bar * c)
        print(Arnorm)

        Anorm_sq += alpha**2 + beta**2
        Anorm = xp.sqrt(Anorm_sq)

        xnorm_sq += step**2
        xnorm = xp.sqrt(xnorm_sq)

        Acond = Anorm * xp.sqrt(dnorm_sq)

        test1 = rnorm / bnorm
        test2 = Arnorm / (Anorm * rnorm)
        test3 = 1 / Acond

        reltol = btol + atol * Anorm * xnorm / bnorm

        if test3 <= ctol:
            exit = 3
        if test2 <= atol:
            exit = 2
        if test1 <= reltol:
            exit = 1

        if exit > 0:
            break
    return x, exit, it


def normof2(xp, x, y):
    return xp.sqrt(xp.sum(xp.multiply(x, y)))


def generate_lsqr_data(source, has_b_file=False):
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
        x_true = random(
            A.shape[1], 1, density=0.1, format="coo", dtype=np.float64, random_state=rng
        )
        b = A @ x_true
        b = b.toarray().flatten()

        noise_level = 0.1 * np.linalg.norm(b)
        noise = rng.standard_normal(b.shape) * noise_level
        b += noise

    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    b_bin = BinsparseFormat.from_numpy(b)
    return (A_bin, b_bin)


def dg_lsqr_sparse_1():
    return generate_lsqr_data("abb313")
