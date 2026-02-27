"""
Name: High-Order SVD (Tucker Decomposition) for 5D Tensors

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do: This code implements the Tucker Decomposition or
HOSVD algorithm for decomposing high-order tensors into a core-tensor that
can be projected onto factor matrices along each mode. A typical 3D
tensor will have 3 modes (row, column, and frontal) and thus 3 factor matrices.
The algorithm starts by finding the initial factor matrices by performing SVD
on matrix unfoldings along each mode. Certain columns in these factor matrices
are selected based on the ranks parameter. Then, the algorithm iteratively
updates each factor matrix by projecting the original tensor onto other factor
matrices. The iteration continues until max iterations is reached or the change
in factor matrices becomes insignificant. The resulting factor matrices and the
core tensor are returned by the benchmark function.

Citation for reference implementation:
https://epubs.siam.org/doi/10.1137/07070111X

Motivation: Tensor decomposition are essential for efficiently analyzing
multi-dimensional data and can cut out noise during preprocessing. Tensor
decomposition has applications in signal processing, computer vision, numerical
linear algebra, and many other fields. HOSVD or Tucker Decomposition is one
of the most widely used methods for tensor decomposition on high-level tensors,
which are tensors with 3 or more dimensions.
DOI: 10.36227/techrxiv.174417403.38431928/v1
https://epubs.siam.org/doi/10.1137/07070111X

Data Generation: The data for this benchmark was created by randomly generating
factor matrices that were both sparse and dense. These factor matrices were
used to construct a factorizable matrix.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. Generative AI might have been used to construct tests.
This statement is written by hand.
"""

import numpy as np

from ..binsparse_format import BinsparseFormat


def benchmark_hosvd(xp, X_bench, ranks_bench, max_iter=50, tolerance=1e-8):
    X = xp.lazy(xp.from_benchmark(X_bench))
    ranks = xp.lazy(xp.from_benchmark(ranks_bench))

    dimensions = X.shape
    num_modes = len(dimensions)

    # initial HOSVD by performing SVD on matrix unfoldings along each mode
    initial_factors = [None] * num_modes
    for mode in range(num_modes):
        perm = [mode] + list(range(mode)) + list(range(mode + 1, num_modes))
        unfold = xp.reshape(xp.transpose(X, perm), (dimensions[mode], -1))

        U, S, Vt = xp.linalg.svd(unfold, full_matrices=False)
        initial_factors[mode] = U[:, : ranks[mode]]

    # iteration to update each factor matrix by projecting the original
    # tensor onto other factor matrices
    for _iteration in range(max_iter):
        prev_factors = initial_factors[:]
        for mode in range(num_modes):
            initial_factors[mode] = xp.lazy(initial_factors[mode])

            if mode == 0:
                update = xp.einsum(
                    "Y[i, r1, r2, r3, r4] += X[i, j, k, l, m] * B[j, r1]"
                    "* C[k, r2] * D[l, r3] * E[m, r4]",
                    X=X,
                    B=initial_factors[1],
                    C=initial_factors[2],
                    D=initial_factors[3],
                    E=initial_factors[4],
                )
            elif mode == 1:
                update = xp.einsum(
                    "Y[r0, j, r2, r3, r4] += X[i, j, k, l, m] * A[i, r0]"
                    "* C[k, r2] * D[l, r3] * E[m, r4]",
                    X=X,
                    A=initial_factors[0],
                    C=initial_factors[2],
                    D=initial_factors[3],
                    E=initial_factors[4],
                )
            elif mode == 2:
                update = xp.einsum(
                    "Y[r0, r1, k, r3, r4] += X[i, j, k, l, m] * A[i, r0]"
                    "* B[j, r1] * D[l, r3] * E[m, r4]",
                    X=X,
                    A=initial_factors[0],
                    B=initial_factors[1],
                    D=initial_factors[3],
                    E=initial_factors[4],
                )
            elif mode == 3:
                update = xp.einsum(
                    "Y[r0, r1, r2, l, r4] += X[i, j, k, l, m] * A[i, r0]"
                    "* B[j, r1] * C[k, r2] * E[m, r4]",
                    X=X,
                    A=initial_factors[0],
                    B=initial_factors[1],
                    C=initial_factors[2],
                    E=initial_factors[4],
                )
            elif mode == 4:
                update = xp.einsum(
                    "Y[r0, r1, r2, r3, m] += X[i, j, k, l, m] * A[i, r0]"
                    "* B[j, r1] * C[k, r2] * D[l, r3]",
                    X=X,
                    A=initial_factors[0],
                    B=initial_factors[1],
                    C=initial_factors[2],
                    D=initial_factors[3],
                )

            perm = [mode] + list(range(mode)) + list(range(mode + 1, num_modes))
            unfold_update = xp.reshape(
                xp.transpose(update, perm), (dimensions[mode], -1)
            )

            U, S, Vt = xp.linalg.svd(unfold_update, full_matrices=False)
            initial_factors[mode] = U[:, : ranks[mode]]

            initial_factors[mode] = xp.compute(initial_factors[mode])

        # stop iterations when solutions stop changing significantly
        change = (
            xp.linalg.norm(initial_factors[0] - prev_factors[0])
            + xp.linalg.norm(initial_factors[1] - prev_factors[1])
            + xp.linalg.norm(initial_factors[2] - prev_factors[2])
            + xp.linalg.norm(initial_factors[3] - prev_factors[3])
            + xp.linalg.norm(initial_factors[4] - prev_factors[4])
        )
        if xp.compute(change)[()] < tolerance:
            break

    core_tensor = xp.einsum(
        "G[p, q, r, s, t] += X[i, j, k, l, m] * A[i, p]"
        "* B[j, q] * C[k, r] * D[l, s] * E[m, t]",
        X=X,
        A=initial_factors[0],
        B=initial_factors[1],
        C=initial_factors[2],
        D=initial_factors[3],
        E=initial_factors[4],
    )
    core_tensor = xp.compute(core_tensor)
    core_bench = xp.to_benchmark(core_tensor)
    factors_bench = [
        xp.to_benchmark(initial_factors[0]),
        xp.to_benchmark(initial_factors[1]),
        xp.to_benchmark(initial_factors[2]),
        xp.to_benchmark(initial_factors[3]),
        xp.to_benchmark(initial_factors[4]),
    ]
    return core_bench, factors_bench


def dg_hosvd_random_small():
    """
    Generate a dense low-rank 5D tensor using random factor matrices.
    """
    dim1, dim2, dim3, dim4, dim5 = 10, 10, 10, 10, 10
    ranks = (3, 3, 3, 3, 3)
    rng = np.random.default_rng(42)

    G = rng.random(ranks).astype(np.float64)
    A = rng.random((dim1, ranks[0])).astype(np.float64)
    B = rng.random((dim2, ranks[1])).astype(np.float64)
    C = rng.random((dim3, ranks[2])).astype(np.float64)
    D = rng.random((dim4, ranks[3])).astype(np.float64)
    E = rng.random((dim5, ranks[4])).astype(np.float64)

    X_dense = np.einsum("pqrst,ip,jq,kr,ls,mt->ijklm", G, A, B, C, D, E)

    indices = np.nonzero(np.ones_like(X_dense))
    values = X_dense[indices]
    X_bin = BinsparseFormat.from_coo(indices, values, (dim1, dim2, dim3, dim4, dim5))

    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))
    return (X_bin, ranks_bin)


def dg_hosvd_sparse_small():
    """
    Generate a sparse low-rank 5D tensor using random factor matrices.
    """
    dim1, dim2, dim3, dim4, dim5 = 20, 20, 20, 20, 20
    ranks = (3, 3, 3, 3, 3)
    rng = np.random.default_rng(42)

    def get_sparse_factor(rows, cols, density=0.2):
        nnz = int(rows * cols * density)
        if nnz < 1:
            nnz = 1
        indices = rng.choice(rows * cols, nnz, replace=False)
        mat = np.zeros(rows * cols)
        mat[indices] = rng.random(nnz)
        return mat.reshape((rows, cols)).astype(np.float64)

    G = get_sparse_factor(
        ranks[0], ranks[1] * ranks[2] * ranks[3] * ranks[4], density=0.5
    ).reshape(ranks)
    A = get_sparse_factor(dim1, ranks[0], density=0.2)
    B = get_sparse_factor(dim2, ranks[1], density=0.2)
    C = get_sparse_factor(dim3, ranks[2], density=0.2)
    D = get_sparse_factor(dim4, ranks[3], density=0.2)
    E = get_sparse_factor(dim5, ranks[4], density=0.2)

    X_dense = np.einsum("pqrst,ip,jq,kr,ls,mt->ijklm", G, A, B, C, D, E)

    indices = np.nonzero(X_dense)
    values = X_dense[indices]

    X_bin = BinsparseFormat.from_coo(indices, values, (dim1, dim2, dim3, dim4, dim5))

    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))
    return (X_bin, ranks_bin)
