import pytest

from sparseappbench.benchmarks.cp_als_4d import (
    benchmark_cp_als,
    dg_cp_als_factorizable_small,
    dg_cp_als_sparse_small,
)
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_cp_als_basic(xp):
    """Testing that CP-ALS in 4D runs without
    errors and produces correct output shapes"""
    X_bin, rank, max_iter = dg_cp_als_sparse_small()

    A_bin, B_bin, C_bin, D_bin, lambda_bin = benchmark_cp_als(
        xp, X_bin, rank, max_iter=max_iter
    )

    # Checking output shapes
    dim1, dim2, dim3, dim4 = X_bin.data["shape"]
    assert A_bin.data["shape"] == (dim1, rank)
    assert B_bin.data["shape"] == (dim2, rank)
    assert C_bin.data["shape"] == (dim3, rank)
    assert D_bin.data["shape"] == (dim4, rank)
    assert lambda_bin.data["shape"] == (rank,)

    print(f" CP-ALS test passed with {xp.__class__.__name__}")


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_reconstruction_error(xp):
    """Tests that CP-ALS in 4D produces low
    reconstruction error on a factorizable tensor"""
    X_bin, rank, max_iter = dg_cp_als_factorizable_small()

    A_bin, B_bin, C_bin, D_bin, lambda_bin = benchmark_cp_als(
        xp, X_bin, rank, max_iter=max_iter
    )

    # Converting everything to numpy for direct computation
    import numpy as np

    X_values = X_bin.data["values"]
    X_indices = (
        X_bin.data["indices_0"],
        X_bin.data["indices_1"],
        X_bin.data["indices_2"],
        X_bin.data["indices_3"],
    )

    A_vals = A_bin.data["values"].reshape(A_bin.data["shape"])
    B_vals = B_bin.data["values"].reshape(B_bin.data["shape"])
    C_vals = C_bin.data["values"].reshape(C_bin.data["shape"])
    D_vals = D_bin.data["values"].reshape(D_bin.data["shape"])
    lambda_vals = lambda_bin.data["values"]

    # Y[i,j,k] = sum_r lambda_r * A[i,r] * B[j,r] * C[k,r]
    i_idx, j_idx, k_idx, l_idx = X_indices
    Y_reconstructed = np.zeros(len(X_values), dtype=np.float32)
    for r in range(rank):
        Y_reconstructed += (
            lambda_vals[r]
            * A_vals[i_idx, r]
            * B_vals[j_idx, r]
            * C_vals[k_idx, r]
            * D_vals[l_idx, r]
        )

    X_norm = np.linalg.norm(X_values)
    diff = Y_reconstructed - X_values
    diff_norm = np.linalg.norm(diff)
    rel_error = diff_norm / X_norm

    assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.6f}"


@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_cp_als_factorizable_basic(xp):
    """Test CP-ALS on factorizable tensor (basic shape check)"""
    X_bin, rank, max_iter = dg_cp_als_factorizable_small()

    A_bin, B_bin, C_bin, D_bin, lambda_bin = benchmark_cp_als(
        xp, X_bin, rank, max_iter=max_iter
    )
    dim1, dim2, dim3, dim4 = X_bin.data["shape"]
    assert A_bin.data["shape"] == (dim1, rank)
    assert B_bin.data["shape"] == (dim2, rank)
    assert C_bin.data["shape"] == (dim3, rank)
    assert D_bin.data["shape"] == (dim4, rank)
    assert lambda_bin.data["shape"] == (rank,)
    print(f"CP-ALS factorizable test passed with {xp.__class__.__name__}")
