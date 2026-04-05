"""
Name: Matrix Multiplication
Author: Kyle Deeds
Email: kdeeds@cs.washington.edu
Motivation (Importance of problem with citation):
"Sparse-Sparse matrix multiply is a fundamental and expensive computational
kernel in numerous scientific computing applications and graph algorithms"
J. Gao et al., “A Systematic Survey of General Sparse Matrix-matrix
Multiplication,” ACM Comput. Surv., vol. 55, no. 12, p. 244:1-244:36, Mar. 2023,
doi: 10.1145/3571157.
Role of sparsity (How sparsity is used in the problem):
The inputs to the matrix multiply are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""

import numpy as np
import scipy.sparse as sp

from ..binsparse_format import BinsparseFormat

r"""
    benchmark_matmul(xp, A_bench, B_bench)

Computes $C_ij = \sum_k A_ik * B_kj$.

Args:
----
    xp: The array API module to use
    A_bench: The first matrix in binsparse format
    B_bench: The second matrix in binsparse format
Returns:
-------
    The result of the matrix multiplication in binsparse format
"""


def benchmark_matmul(xp, A_bench, B_bench):
    A_lazy = xp.lazy(xp.from_benchmark(A_bench))
    B_lazy = xp.lazy(xp.from_benchmark(B_bench))
    C_lazy = xp.matmul(A_lazy, B_lazy)
    C_eager = xp.compute(C_lazy)
    return xp.to_benchmark(C_eager)


def dg_matmul_dense_small():
    rng = np.random.default_rng(0)
    A = rng.random((32, 32), dtype=np.float32)
    B = rng.random((32, 32), dtype=np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)


def dg_matmul_dense_large():
    rng = np.random.default_rng(0)
    A = rng.random((4096, 4096), dtype=np.float32)
    B = rng.random((4096, 4096), dtype=np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)


def dg_matmul_sparse_small():
    rng = np.random.default_rng(0)
    A = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng)
    B = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng)
    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    B_bin = BinsparseFormat.from_coo((B.row, B.col), B.data, B.shape)
    return (A_bin, B_bin)


def dg_matmul_sparse_large():
    rng = np.random.default_rng(0)
    A = sp.random(
        4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng
    )
    B = sp.random(
        4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng
    )
    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    B_bin = BinsparseFormat.from_coo((B.row, B.col), B.data, B.shape)
    return (A_bin, B_bin)
