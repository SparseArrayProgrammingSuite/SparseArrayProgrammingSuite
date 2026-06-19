import pytest

import numpy as np

import saps.benchmarks.cp5_als as cp5_als
from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.cp5_als import (
    CP5_ALS,
    CP5FactorizeableGenerator,
)


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_reconstruction_error(xp):
    """Tests that CP-ALS produces low reconstruction error on a factorizable tensor"""
    gen = CP5FactorizeableGenerator()
    X_bin, rank, max_iter = gen.generate(gen.datasets[0])

    cp5_als.xp = NumpyFramework()

    X = cp5_als.xp.from_binsparse(X_bin)

    A, B, C, D, E, L = CP5_ALS().benchmark((X,), {"rank": rank, "max_iter": max_iter})

    Y = cp5_als.xp.einsum(
        "Y[i,j,k,l,m] += L[r] * A[i,r] * B[j,r] * C[k,r] * D[l, r] * E[m, r]",
        L=L,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
    )
    X_norm = np.linalg.norm(X)
    diff = Y - X
    diff_norm = np.linalg.norm(diff)
    rel_error = diff_norm / X_norm

    assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.6f}"


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_factorizable_basic(xp):
    """Test CP-ALS on factorizable tensor (basic shape check)"""
    gen = CP5FactorizeableGenerator()
    X_bin, rank, max_iter = gen.generate(gen.datasets[0])

    cp5_als.xp = NumpyFramework()

    X = cp5_als.xp.from_binsparse(X_bin)

    A, B, C, D, E, lambda_vals = CP5_ALS().benchmark(
        (X,), {"rank": rank, "max_iter": max_iter}
    )

    dim1, dim2, dim3, dim4, dim5 = X_bin.data["shape"]
    assert A.shape == (dim1, rank)
    assert B.shape == (dim2, rank)
    assert C.shape == (dim3, rank)
    assert D.shape == (dim4, rank)
    assert E.shape == (dim5, rank)
    assert lambda_vals.shape == (rank,)
    print(f"CP-ALS factorizable test passed with {xp.__class__.__name__}")
