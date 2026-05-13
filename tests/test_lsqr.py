import pytest

import numpy as np

import saps.benchmarks.lsqr as lsqr
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import (
    PyDataSparseFramework,
)
from saps_framework import BinsparseFormat


def as_dense(array):
    if hasattr(array, "todense"):
        return np.asarray(array.todense())
    return np.asarray(array)


def run_lsqr_benchmark(xp, A, b):
    benchmark = lsqr.LSQRBenchmark()
    prev_xp = getattr(lsqr, "xp", None)
    lsqr.xp = xp
    try:
        (x_sol,) = benchmark.benchmark([A, b], {})
    finally:
        lsqr.xp = prev_xp
    return x_sol


@pytest.mark.parametrize(
    "xp, A, b, convergence",
    [
        (
            PyDataSparseFramework(),  # Underdetermined
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0]]),
            np.array([4.1, 10.1]),  # b = A @ [1, 2, 3] + noise
            "residual",
        ),
        (
            PyDataSparseFramework(),  # Overdetermined
            np.array(
                [[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0], [4.0, -3.0, 1.0]]
            ),
            np.array([13.2, -3.3, 8.1, 12.4]),  # b = A @ [2, -1, 1] + noise
            "gradient",
        ),
        (
            PyDataSparseFramework(),  # Exact Solution
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
            np.array([4.0, 8.0, 16.0]),  # b = A @ [1, 2, 3]
            "residual",
        ),
        (
            NumpyFramework(),  # Underdetermined
            np.array(
                [
                    [8.0, -1.0, 0.0, 0.0],
                    [-1.0, 8.0, -1.0, 0.0],
                    [0.0, -1.0, 8.0, -1.0],
                ]
            ),
            np.array([8.1, -2.2, 6.3]),  # b = A @ [1, 0, 1, 2] + noise
            "residual",
        ),
        (
            NumpyFramework(),  # Overdetermined
            np.array(
                [[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0], [5.0, 1.0, 2.0]]
            ),
            np.array([40.1, 10.2, -18.3, 15.4]),  # b = A @ [3, 1, -2] + noise
            "gradient",
        ),
        (
            NumpyFramework(),  # Exact Solution
            np.array(
                [
                    [8.0, -1.0, 0.0, 0.0],
                    [-1.0, 8.0, -1.0, 0.0],
                    [0.0, -1.0, 8.0, -1.0],
                    [0.0, 0.0, -1.0, 8.0],
                ]
            ),
            np.array([8.0, -2.0, 6.0, 15.0]),  # b = A @ [1, 0, 1, 2]
            "residual",
        ),
        (
            NumpyFramework(),  # Underdetermined
            np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0]]),
            np.array([118.1, 116.1]),  # b = A @ [1, 1, 1] + noise
            "residual",
        ),
        (
            NumpyFramework(),  # Overdetermined
            np.array(
                [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0], [2.0, 1.0, 3.0]]
            ),
            np.array([5.1, 7.2, 11.3, 12.4]),  # b = A @ [1, 2, 3] + noise
            "gradient",
        ),
        (
            NumpyFramework(),  # Exact Solution
            np.array(
                [
                    [15.0, -2.0, 0.0, 0.0, -1.0],
                    [-2.0, 14.0, -3.0, 0.0, 0.0],
                    [0.0, -3.0, 16.0, -2.0, 0.0],
                    [0.0, 0.0, -2.0, 15.0, -3.0],
                    [-1.0, 0.0, 0.0, -3.0, 17.0],
                ]
            ),
            np.array([27.0, -1.0, -18.0, 8.0, 46.0]),  # b = A @ [2, 0, -1, 1, 3]
            "residual",
        ),
    ],
)
def test_lsqr_solver(xp, A, b, convergence):
    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)
    A_input = xp.from_binsparse(A_bin)
    b_input = xp.from_binsparse(b_bin)

    x_sol = as_dense(run_lsqr_benchmark(xp, A_input, b_input))

    residual = b - A @ x_sol

    if convergence == "residual":
        assert np.linalg.norm(residual) < 1e-5 * np.linalg.norm(b) + 1e-5
    elif convergence == "gradient":
        assert np.linalg.norm(A.T @ residual) < 1e-5 * np.linalg.norm(A.T @ b) + 1e-5
