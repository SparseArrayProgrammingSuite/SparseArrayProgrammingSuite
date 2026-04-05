"""
Name: CP-ALS for Tensor Decomposition
Author: Grace Wang
Email: gwang426@gatech.edu
Motivation:
"The Alternating Least Squares (ALS) algorithm for CANDECOMP/PARAFAC (CP) plays a
critical role in tensor decomposition, which has applications in various fields such as
signal processing, pscyhometrics, neuroscience, and graph analysis. Within the ALS
algorithm, the Matricized-Tensor Times Khatri-Rao Product (MTTKRP) operation is a
computationally intensive step that often dominates the overall runtime. Efficiently
implementing MTTKRP is crucial for the performance of the ALS algorithm."
T. G Kolda and B. W. Bader, "Tensor Decompositions and Applications," SIAM Review,
vol. 51, no. 3, p. 455-500, 2009, doi: 10.1137/07070111X.
Role of sparsity:
The input tensor is sparse, and the ALS algorithm takes advantage of this sparsity
through its MTTKRP kernel, which process only the non-zero elements of the tensor.
For sparse tensors with nnz << I * J * K, the complexity reduces from O(I * J * K * R)
to O(nnz * R), where nnz is the number of non-zero elements in the tensor and R is the
decomposition rank. This makes it practical to work with large-scale applications.
Implementation:
Handwritten code based on the standard CP-ALS algorithm from:
T. G Kolda and B. W. Bader, "Tensor Decompositions and Applications," SIAM Review,
vol. 51, no. 3, p. 455-500, 2009, doi: 10.1137/07070111X.
Reference implementations used and translated:
https://github.com/willow-ahrens/18.335FinalProject/blob/submit/TFGDCANDECOMP.py
https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_cp.py
Data Generation:
Sparse tensors with controlled rank and sparsity were generated using random tensor
creation methods from Tensortools combined with sparsification techniques from TensorLy.
Statement on the use of Generative AI:
No generative AI was used to construct the benchmark function itself. Generative AI was
used to debug some parts of the code. This statement was written by hand.
"""

import numpy as np

from ..binsparse_format import BinsparseFormat

"""
benchmark_cp_als(xp, X_bench, rank, max_iter)

Computes the CP decomposition using Alternating Least Squares (ALS).
Factorizes a 3rd-order tensor X into factor matrices A, B, C such that:
$X \approx \\sum_{r=1}^{R} \\lambda_r \\cdot a_r \\circ b_r \\circ c_r$
where $\\circ$ denotes the outeer product, R is the rank, and $\\lambda$
are the weights.

Args:
----
xp: The array API module to use
X_bench: The input 3rd-order sparse tensor in binsparse format
rank: Number of components (rank) for the decomposition
max_iter: Maximum number of ALS iterations

Returns:
-------
Tuple of (A_bench, B_bench, C_bench, lambda_bench) in binsparse format where:
- A, B, and C are the normalized factor matrices
- lambda are the component weights
"""


def benchmark_cp_als(xp, X_bench, rank, max_iter=50):
    X_eager = xp.from_benchmark(X_bench)
    X = xp.lazy(X_eager)

    dim1, dim2, dim3 = X_bench.data["shape"]
    dtype = X_bench.data["values"].dtype

    A = xp.from_benchmark(
        BinsparseFormat.from_numpy(
            np.random.default_rng(0).random((dim1, rank)).astype(dtype)
        )
    )
    B = xp.from_benchmark(
        BinsparseFormat.from_numpy(
            np.random.default_rng(0).random((dim2, rank)).astype(dtype)
        )
    )
    C = xp.from_benchmark(
        BinsparseFormat.from_numpy(
            np.random.default_rng(0).random((dim3, rank)).astype(dtype)
        )
    )

    for _iteration in range(max_iter):
        (A, B, C) = xp.lazy((A, B, C))
        # Update A
        mttkrp_result = xp.einsum(
            "mttkrp_result[i, r] += X[i, j, k] * B[j, r] * C[k, r]", X=X, B=B, C=C
        )
        CtC = xp.einsum("CtC[r, s] += C[k, r] * C[k, s]", C=C)
        BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B=B)
        G = xp.multiply(CtC, BtB)
        G_pinv = xp.linalg.pinv(G)
        A = xp.matmul(mttkrp_result, G_pinv)

        # Update B
        mttkrp_result = xp.einsum(
            "mttkrp_result[j, r] += X[i, j, k] * A[i, r] * C[k, r]", X=X, A=A, C=C
        )
        AtA = xp.einsum("AtA[r, s] += A[i, r] * A[i, s]", A=A)
        G = xp.multiply(CtC, AtA)
        G_pinv = xp.linalg.pinv(G)
        B = xp.matmul(mttkrp_result, G_pinv)

        # Update C
        mttkrp_result = xp.einsum(
            "mttkrp_result[k, r] += X[i, j, k] * A[i, r] * B[j, r]", X=X, A=A, B=B
        )
        BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B=B)
        G = xp.multiply(BtB, AtA)
        G_pinv = xp.linalg.pinv(G)
        C = xp.matmul(mttkrp_result, G_pinv)

        A, B, C = xp.compute((A, B, C))

    (A, B, C) = xp.lazy((A, B, C))

    # Normalizing factors
    A_norms_sq = xp.einsum("norms[r] += A[i, r] * A[i, r]", A=A)
    B_norms_sq = xp.einsum("norms[r] += B[j, r] * B[j, r]", B=B)
    C_norms_sq = xp.einsum("norms[r] += C[k, r] *C[k, r]", C=C)

    A_norms = xp.sqrt(A_norms_sq)
    B_norms = xp.sqrt(B_norms_sq)
    C_norms = xp.sqrt(C_norms_sq)

    # Computing lambda
    lambda_vals = xp.multiply(xp.multiply(A_norms, B_norms), C_norms)

    A_norms_2d = xp.expand_dims(A_norms, 0)
    B_norms_2d = xp.expand_dims(B_norms, 0)
    C_norms_2d = xp.expand_dims(C_norms, 0)

    # Case: avoiding division by zero
    eps = 1e-10
    A_norms_safe = xp.maximum(A_norms_2d, eps)
    B_norms_safe = xp.maximum(B_norms_2d, eps)
    C_norms_safe = xp.maximum(C_norms_2d, eps)

    A = xp.divide(A, A_norms_safe)
    B = xp.divide(B, B_norms_safe)
    C = xp.divide(C, C_norms_safe)

    # Now compute everything at once

    (A, B, C, lambda_vals) = xp.compute((A, B, C, lambda_vals))

    # Convert to binsparse format
    A_bench_out = xp.to_benchmark(A)
    B_bench_out = xp.to_benchmark(B)
    C_bench_out = xp.to_benchmark(C)
    lambda_bench_out = xp.to_benchmark(lambda_vals)

    return (A_bench_out, B_bench_out, C_bench_out, lambda_bench_out)


# Data generators
def dg_cp_als_sparse_small():
    dim1, dim2, dim3 = 20, 20, 20
    rank = 3
    nnz = int(0.01 * dim1 * dim2 * dim3)

    all_indices = np.random.default_rng(0).choice(
        dim1 * dim2 * dim3, size=nnz, replace=False
    )
    i_idx, j_idx, k_idx = np.unravel_index(all_indices, (dim1, dim2, dim3))

    values = np.random.default_rng(0).random(nnz).astype(np.float32)
    X_bin = BinsparseFormat.from_coo((i_idx, j_idx, k_idx), values, (dim1, dim2, dim3))
    max_iter = 50

    return (X_bin, rank, max_iter)


def dg_cp_als_factorizable_small():
    """
    Generating a small factorizable tensor by creating random factor matrices
    and reconstructing the tensor from them (tensor should decompose easily
    with low reconstruction error).
    """
    dim1, dim2, dim3 = 20, 20, 20
    rank = 3

    rng = np.random.default_rng(0)

    # Create simple factor matrices for easier decomposition
    A = rng.random((dim1, rank)).astype(np.float32)
    B = rng.random((dim2, rank)).astype(np.float32)
    C = rng.random((dim3, rank)).astype(np.float32)

    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    C = C / np.linalg.norm(C, axis=0, keepdims=True)
    lambdas = np.array([100.0, 50.0, 25.0], dtype=np.float32)
    A = A * lambdas

    X_dense = np.einsum("ir,jr,kr->ijk", A, B, C)

    indices = np.nonzero(np.ones_like(X_dense))
    values = X_dense[indices]
    X_bin = BinsparseFormat.from_coo(indices, values, (dim1, dim2, dim3))
    max_iter = 100

    return (X_bin, rank, max_iter)
