import pytest

import numpy as np

from sparseappbench.benchmarks.lsqr import benchmark_lsqr
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import (
    PyDataSparseFramework,
)


@pytest.mark.parametrize(
    "xp, A, b, expected_exit_code",
    [
        (
            PyDataSparseFramework(),  # Underdetermined
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0]]),
            np.array([4.1, 10.1]),  # b = A @ [1, 2, 3] + noise
            1,
        ),
        (
            PyDataSparseFramework(),  # Overdetermined
            np.array(
                [[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0], [4.0, -3.0, 1.0]]
            ),
            np.array([13.2, -3.3, 8.1, 12.4]),  # b = A @ [2, -1, 1] + noise
            2,
        ),
        (
            PyDataSparseFramework(),  # Exact Solution
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
            np.array([4.0, 8.0, 16.0]),  # b = A @ [1, 2, 3]
            1,
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
            1,
        ),
        (
            NumpyFramework(),  # Overdetermined
            np.array(
                [[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0], [5.0, 1.0, 2.0]]
            ),
            np.array([40.1, 10.2, -18.3, 15.4]),  # b = A @ [3, 1, -2] + noise
            2,
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
            1,
        ),
        (
            CheckerFramework(),  # Underdetermined
            np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0]]),
            np.array([118.1, 116.1]),  # b = A @ [1, 1, 1] + noise
            1,
        ),
        (
            CheckerFramework(),  # Overdetermined
            np.array(
                [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0], [2.0, 1.0, 3.0]]
            ),
            np.array([5.1, 7.2, 11.3, 12.4]),  # b = A @ [1, 2, 3] + noise
            2,
        ),
        (
            CheckerFramework(),  # Exact Solution
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
            1,
        ),
    ],
)
def test_lsqr_solver(xp, A, b, expected_exit_code):
    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)

    results = benchmark_lsqr(xp, A_bin, b_bin)
    x_sol = xp.from_benchmark(results[0])
    actual_exit_code = results[1]

    residual = b - A @ x_sol

    assert actual_exit_code == expected_exit_code

    if expected_exit_code == 1:  # Converged due to small residual
        assert np.linalg.norm(residual) < 1e-5 * np.linalg.norm(b) + 1e-5
    elif expected_exit_code == 2:  # Converged due to small gradient
        assert np.linalg.norm(A.T @ residual) < 1e-5 * np.linalg.norm(A.T @ b) + 1e-5
