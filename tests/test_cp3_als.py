import pytest

from saps.benchmarks.cp3_als import (
    CP3_ALS,
    CP3FactorizeableGenerator,
)
import saps.benchmarks.cp3_als as cp3_als
from frameworks.saps_numpy import NumpyFramework
import numpy as np


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_reconstruction_error(xp):
    """Tests that CP-ALS produces low reconstruction error on a factorizable tensor"""
    gen = CP3FactorizeableGenerator()
    X_bin, rank, max_iter = gen.generate(gen.datasets[0])

    cp3_als.xp = NumpyFramework()

    X = cp3_als.xp.from_binsparse(X_bin)

    A_bin, B_bin, C_bin, lambda_bin = CP3_ALS().benchmark(
        (X,), {"rank":rank, "max_iter":max_iter}
    )

    A = cp3_als.xp.from_binsparse(A_bin)
    B = cp3_als.xp.from_binsparse(B_bin)
    C = cp3_als.xp.from_binsparse(C_bin)
    l = cp3_als.xp.from_binsparse(lambda_bin)

    Y = cp3_als.xp.einsum(
        "Y[i,j,k] += l[r] * A[i,r] * B[j,r] * C[k,r]", l=l, A=A, B=B, C=C)
    X_norm = np.linalg.norm(X)
    diff = Y - X
    diff_norm = np.linalg.norm(diff)
    rel_error = diff_norm / X_norm

    assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.6f}"


@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_factorizable_basic(xp):
    """Test CP-ALS on factorizable tensor (basic shape check)"""
    gen = CP3FactorizeableGenerator()
    X_bin, rank, max_iter = gen.generate(gen.datasets[0])

    cp3_als.xp = NumpyFramework()

    X = cp3_als.xp.from_binsparse(X_bin)

    A_bin, B_bin, C_bin, lambda_bin = CP3_ALS().benchmark(
        (X,), {"rank":rank, "max_iter":max_iter}
    )
    dim1, dim2, dim3 = X_bin.data["shape"]
    assert A_bin.data["shape"] == (dim1, rank)
    assert B_bin.data["shape"] == (dim2, rank)
    assert C_bin.data["shape"] == (dim3, rank)
    assert lambda_bin.data["shape"] == (rank,)
    print(f"CP-ALS factorizable test passed with {xp.__class__.__name__}")
