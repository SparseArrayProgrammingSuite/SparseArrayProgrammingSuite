"""
Tests for SUMMA and SpSUMMA benchmark functions.

Run with plain pytest (P=1) for correctness on a single process:
    pytest test_summa.py -v

Run with mpirun for distributed correctness (P must be a perfect square):
    mpirun -n 4 pytest test_summa.py -v
    mpirun -n 16 pytest test_summa.py -v
"""

import numpy as np
import pytest
from mpi4py import MPI

from summa_benchmark import (
    benchmark_summa_dense,
    benchmark_summa_sparse,
    dg_summa_dense_small,
    dg_summa_sparse_small,
)

_rank = MPI.COMM_WORLD.Get_rank()
_P = MPI.COMM_WORLD.Get_size()


# ---------------------------------------------------------------------------
# Sparse tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["standard", "improved"])
def test_summa_sparse_output_shape(variant):
    """SpSUMMA returns a matrix with the correct shape."""
    A, B = dg_summa_sparse_small()
    C = benchmark_summa_sparse(A, B, variant=variant)

    if _rank == 0:
        assert C.shape == (A.shape[0], B.shape[1]), (
            f"Expected shape {(A.shape[0], B.shape[1])}, got {C.shape}"
        )

    if _rank == 0:
        print(f"  SpSUMMA ({variant}) shape test passed with P={_P}")


@pytest.mark.parametrize("variant", ["standard", "improved"])
def test_summa_sparse_correctness(variant):
    """SpSUMMA result matches scipy sparse reference A @ B."""
    A, B = dg_summa_sparse_small()
    C = benchmark_summa_sparse(A, B, variant=variant)

    if _rank == 0:
        C_ref = (A @ B).toarray()
        np.testing.assert_allclose(
            C.toarray(), C_ref, atol=1e-10,
            err_msg=f"SpSUMMA ({variant}) result differs from reference",
        )

    if _rank == 0:
        print(f"  SpSUMMA ({variant}) correctness test passed with P={_P}")


def test_summa_sparse_nnz_reasonable():
    """nnz(C) is in a plausible range for ER(d=5) matrices of size n=500."""
    A, B = dg_summa_sparse_small()
    C = benchmark_summa_sparse(A, B, variant="standard")

    if _rank == 0:
        n, d = 500, 5
        expected_nnz = d ** 2 * n  # = 12500
        # Allow 5x slack (sparse random variance is high for small n)
        assert C.nnz < 5 * expected_nnz, (
            f"nnz(C)={C.nnz} unexpectedly large (expected ~{expected_nnz})"
        )
        assert C.nnz > 0, "Result matrix is all zeros"


# ---------------------------------------------------------------------------
# Dense tests
# ---------------------------------------------------------------------------

def test_summa_dense_output_shape():
    """Dense SUMMA returns a matrix with the correct shape."""
    A, B = dg_summa_dense_small()
    C = benchmark_summa_dense(A, B)

    if _rank == 0:
        assert C.shape == (A.shape[0], B.shape[1]), (
            f"Expected shape {(A.shape[0], B.shape[1])}, got {C.shape}"
        )

    if _rank == 0:
        print(f"  Dense SUMMA shape test passed with P={_P}")


def test_summa_dense_correctness():
    """Dense SUMMA result matches numpy reference A @ B."""
    A, B = dg_summa_dense_small()
    C = benchmark_summa_dense(A, B)

    if _rank == 0:
        C_ref = A @ B
        np.testing.assert_allclose(
            C, C_ref, rtol=1e-5, atol=1e-8,
            err_msg="Dense SUMMA result differs from numpy reference",
        )

    if _rank == 0:
        print(f"  Dense SUMMA correctness test passed with P={_P}")


def test_summa_dense_dtype():
    """Dense SUMMA result is float64."""
    A, B = dg_summa_dense_small()
    C = benchmark_summa_dense(A, B)

    if _rank == 0:
        assert C.dtype == np.float64, f"Expected float64, got {C.dtype}"
