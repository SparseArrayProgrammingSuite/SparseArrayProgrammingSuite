import pytest

import numpy as np

from sparseappbench.benchmarks.matmul import benchmark_matmul
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "xp,A,B",
    [
        (
            NumpyFramework(),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ),
        (
            NumpyFramework(),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            NumpyFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
        ),
        (
            CheckerFramework(),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ),
        (
            CheckerFramework(),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            CheckerFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
        ),
    ],
)
def test_benchmark_matmul(xp, A, B):
    C_ref = A @ B

    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)

    C_bin = benchmark_matmul(xp, A_bin, B_bin)
    C_bin = BinsparseFormat.to_coo(C_bin)

    assert C_bin == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(C_ref))
