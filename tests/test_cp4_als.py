import pytest

import numpy as np

import saps.benchmarks.cp4_als as cp4_als
from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.cp4_als import (
    CP4_ALS,
    CP4FactorizeableGenerator,
)


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_reconstruction_error(xp):
    """Tests that CP-ALS produces low reconstruction error on a factorizable tensor"""
    gen = CP4FactorizeableGenerator()
    problem = gen.generate(gen.datasets[0])
    (X_bin,) = problem.inputs
    rank = problem.meta["rank"]
    max_iter = problem.meta["max_iter"]

    cp4_als.xp = NumpyFramework()

    X = cp4_als.xp.from_binsparse(X_bin)

    A, B, C, D, L = CP4_ALS().benchmark((X,), {"rank": rank, "max_iter": max_iter})

    Y = cp4_als.xp.einsum(
        "Y[i,j,k,l] += L[r] * A[i,r] * B[j,r] * C[k,r] * D[l, r]",
        L=L,
        A=A,
        B=B,
        C=C,
        D=D,
    )
    X_norm = np.linalg.norm(X)
    diff = Y - X
    diff_norm = np.linalg.norm(diff)
    rel_error = diff_norm / X_norm

    assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.6f}"


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_factorizable_basic(xp):
    """Test CP-ALS on factorizable tensor (basic shape check)"""
    gen = CP4FactorizeableGenerator()
    problem = gen.generate(gen.datasets[0])
    (X_bin,) = problem.inputs
    rank = problem.meta["rank"]
    max_iter = problem.meta["max_iter"]

    cp4_als.xp = NumpyFramework()

    X = cp4_als.xp.from_binsparse(X_bin)

    A, B, C, D, lambda_vals = CP4_ALS().benchmark(
        (X,), {"rank": rank, "max_iter": max_iter}
    )
    dim1, dim2, dim3, dim4 = X_bin.data["shape"]
    assert A.shape == (dim1, rank)
    assert B.shape == (dim2, rank)
    assert C.shape == (dim3, rank)
    assert D.shape == (dim4, rank)
    assert lambda_vals.shape == (rank,)
    print(f"CP-ALS factorizable test passed with {xp.__class__.__name__}")
