from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.amg import (
    AMGBenchmark,
    _build_tentative_prolongator,
    _tentative_prolongator,
)

_TRIDIAG_3 = np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]])
_TRIDIAG_4 = np.array(
    [
        [8.0, -1.0, 0.0, 0.0],
        [-1.0, 8.0, -1.0, 0.0],
        [0.0, -1.0, 8.0, -1.0],
        [0.0, 0.0, -1.0, 8.0],
    ]
)
_TRIDIAG_5 = np.array(
    [
        [4.0, -1.0, 0.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 4.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, 0.0, -1.0, 4.0],
    ]
)
_SPARSE_5 = np.array(
    [
        [15.0, -2.0, 0.0, 0.0, -1.0],
        [-2.0, 14.0, -3.0, 0.0, 0.0],
        [0.0, -3.0, 16.0, -2.0, 0.0],
        [0.0, 0.0, -2.0, 15.0, -3.0],
        [-1.0, 0.0, 0.0, -3.0, 17.0],
    ]
)


def _solve(A, b, *, rel_tol=1e-8):
    """Run the AMG benchmark function on a dense SPD ``A`` and RHS ``b``."""
    xp = NumpyFramework()
    A_csr = sps.csr_matrix(A)
    tentative = _build_tentative_prolongator(A_csr, aggregate_size=3)
    benchmark = AMGBenchmark()
    meta = {"rel_tol": rel_tol, "max_iter": 200, "num_sweeps": 2}
    A_in = xp.from_binsparse(from_scipy(A_csr.tocoo()))
    b_in = xp.from_binsparse(from_numpy(b))
    x0_in = xp.from_binsparse(from_numpy(np.zeros(A.shape[1])))
    tentative_in = xp.from_binsparse(from_scipy(tentative.tocoo()))
    return np.asarray(benchmark.benchmark(xp, meta, A_in, b_in, x0_in, tentative_in))


@pytest.mark.parametrize(
    "A",
    [
        pytest.param(_TRIDIAG_3, id="3x3_tridiagonal"),
        pytest.param(_TRIDIAG_4, id="4x4_tridiagonal"),
        pytest.param(_TRIDIAG_5, id="5x5_tridiagonal"),
        pytest.param(_SPARSE_5, id="5x5_sparse"),
    ],
)
def test_amg_matches_direct_solve(A):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(A.shape[0])
    expected = np.linalg.solve(A, b)
    x = _solve(A, b)
    residual = np.linalg.norm(b - A @ x)
    assert residual <= 1e-6 * np.linalg.norm(b)
    assert np.allclose(x, expected, rtol=1e-4, atol=1e-6)


def test_galerkin_triple_product_shapes():
    """R is short/wide, P is tall/narrow, A and the coarse operator square."""
    A = sps.csr_matrix(_SPARSE_5)
    P = _build_tentative_prolongator(A, aggregate_size=3)
    R = P.T
    n_fine, n_coarse = P.shape
    assert A.shape[0] == A.shape[1], "A must be square"
    assert n_fine == A.shape[0]
    assert n_coarse < n_fine, "P must be tall and narrow (coarsening)"
    assert R.shape == (n_coarse, n_fine), "R must be short and wide"
    coarse = (R @ A @ P).tocsr()
    assert coarse.shape == (n_coarse, n_coarse), "R @ A @ P must be square"


def test_tentative_prolongator_is_a_partition():
    agg = np.array([0, 0, 1, 1, 2])
    P = _tentative_prolongator(agg).toarray()
    assert P.shape == (5, 3)
    # Each row selects exactly one aggregate.
    assert np.array_equal(P.sum(axis=1), np.ones(5))
    assert np.array_equal(P.argmax(axis=1), agg)


def _run_check(x, *, A=None, b=None, meta=None, ref_meta=None):
    A = np.eye(3) if A is None else A
    b = np.ones(3) if b is None else b
    P = sps.coo_array(_tentative_prolongator(np.array([0, 0, 1])))
    benchmark = AMGBenchmark()
    benchmark._input = [
        from_scipy(sps.coo_array(A)),
        from_numpy(b),
        from_numpy(np.zeros(A.shape[1])),
        from_scipy(P),
    ]
    benchmark._output = [from_numpy(x)]
    benchmark._meta = meta or {}
    benchmark._ref_outputs = [from_scipy(sps.coo_array(np.abs(A)))]
    benchmark._ref_meta = {**(ref_meta or {}), "bnorm": float(np.linalg.norm(b))}
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="amg_check")))


def test_check_accepts_converged_solution():
    _run_check(np.ones(3))


def test_check_rejects_wrong_solution():
    with pytest.raises(AssertionError):
        _run_check(np.ones(3) + 0.5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_check_rejects_nonfinite_solution(bad_value):
    with np.errstate(invalid="ignore"), pytest.raises(AssertionError):
        _run_check(np.full(3, bad_value))


def test_check_respects_relative_tolerance():
    meta = {"rel_tol": 1e-4}
    _run_check(np.ones(3) - 1e-7, meta=meta)
    with pytest.raises(AssertionError):
        _run_check(np.ones(3) - 2e-4, meta=meta)
