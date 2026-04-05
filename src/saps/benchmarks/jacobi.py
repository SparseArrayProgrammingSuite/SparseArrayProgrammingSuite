import os

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
Name: Jacobi Iterative Solver
Author: Benjamin Berol
Email: bberol3@gatech.edu
Motivation:
"There exist a wide variety of iterative methods, with the two main classes being the
stationary iterative methods (Young, 1971), such as Richardson's method and Jacobi's
method, and Krylov subspace methods (Liesen and Strakos, 2012) such as the conjugate
gradient method (CG; Hestenes and Stiefel, 1952)."
J. Cockayne, I. C. F. Ipsen, C. J. Oates, and T. W. Reid,
“Probabilistic iterative methods for linear systems,”
J. Mach. Learn. Res., vol. 22, p. 1-34, 2021.
[Online]. Available: https://www.jmlr.org/papers/volume22/21-0031/21-0031.pdf
Role of Sparsity:
Sparsity makes the Jacobi method efficient because each update only needs to
access the nonzero entries, reducing complexity from O(mn) to O(nnz)
Implementation:
Hand-written code modelling the algorithm structure outlined in:
https://www.cs.princeton.edu/~appel/papers/jacobi.pdf and
https://courses.grainger.illinois.edu/cs357/su2014/lectures/lecture10.pdf
Data Generation:
Data collected from SuiteSparse Matrix Collection consisting of symmetric positive
definite matrices whose Jacobi iteration matrices have spectral radius < 1.
Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


def benchmark_jacobi(
    xp, A_bench, b_bench, x_bench, rel_tol=1e-6, abs_tol=1e-20, max_iters=1000
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    x = xp.lazy(xp.from_benchmark(x_bench))

    tolerance = max(rel_tol * xp.compute(norm(xp, b))[()], abs_tol)
    d = xp.with_fill_value(xp.diagonal(A), 1)
    if xp.compute(xp.any(d == 0)):
        raise ValueError("Jacobi requires nonzero diagonal entries.")

    r = b - A @ x
    it = 0

    while xp.compute(norm(xp, r))[()] >= tolerance and it < max_iters:
        x = x + r / d
        x = xp.lazy(xp.compute(x))

        r = b - A @ x
        r = xp.lazy(xp.compute(r))
        it += 1
    if it >= max_iters:
        raise RuntimeError(
            "Jacobi did not converge within the maximum number of iterations"
        )
    x_solution = xp.compute(x)
    return xp.to_benchmark(x_solution)


def norm(xp, v):
    return xp.sqrt(xp.sum(xp.multiply(v, v)))


def generate_jacobi_data(source, has_b_file=False):
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


def dg_jacobi_sparse_1():
    return generate_jacobi_data("mesh3em5")  # nnz = 1,889


def dg_jacobi_sparse_2():
    return generate_jacobi_data("Trefethen_200")  # nnz = 2,873


def dg_jacobi_sparse_3():
    return generate_jacobi_data("Chem97ZtZ")  # nnz = 7,361


def dg_jacobi_sparse_4():
    return generate_jacobi_data("Trefethen_500")  # nnz = 8,478


def dg_jacobi_sparse_5():
    return generate_jacobi_data("Trefethen_700")  # nnz = 12,654


def dg_jacobi_sparse_6():
    return generate_jacobi_data("fv1")  # nnz = 85,264


def dg_jacobi_sparse_7():
    return generate_jacobi_data("fv2")  # nnz = 87,025


def dg_jacobi_sparse_8():
    return generate_jacobi_data("Trefethen_20000")  # nnz = 554,466


# Matrices below run extremely slowly on numpy framework (>1 minutes per convergence):

# def dg_jacobi_sparse_9():
#     return generate_jacobi_data("obstclae")  # nnz = 197,608

# def dg_jacobi_sparse_10():
#     return generate_jacobi_data("minsurfo")  # nnz = 203,622

# def dg_jacobi_sparse_11():
#     return generate_jacobi_data("jnlbrng1") #nnz = 199,200

# def dg_jacobi_sparse_12():
#     return generate_jacobi_data("shallow_water1") #nnz = 327,680

# def dg_jacobi_sparse_13():
#     return generate_jacobi_data("shallow_water2") #nnz = 327,680

# def dg_jacobi_sparse_14():
#     return generate_jacobi_data("finan512") #nnz = 596,992
