import pytest

import numpy as np

import saps.benchmarks.jacobi as jacobi
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import (
    PyDataSparseFramework,
)
from saps_framework import BinsparseFormat


@pytest.mark.parametrize(
    "xp, A, b, x",
    [
        (
            PyDataSparseFramework(),
            np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]),
            np.array([5.0, 8.0, 8.0]),
            np.zeros((3,)),
        ),
        (
            SciPyFramework(),
            np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]),
            np.array([5.0, 8.0, 8.0]),
            np.zeros((3,)),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [10.0, 1.0, 0.0, 2.0],
                    [1.0, 8.0, 1.0, 0.0],
                    [0.0, 2.0, 9.0, 1.0],
                    [1.0, 0.0, 1.0, 7.0],
                ]
            ),
            np.array([16.0, 18.0, 15.0, 16.0]),
            np.zeros((4,)),
        ),
        (
            NumpyFramework(),
            np.array([[20.0, 3.0, 1.0], [2.0, 15.0, 4.0], [1.0, 2.0, 18.0]]),
            np.array([24.0, 21.0, 21.0]),
            np.zeros((3,)),
        ),
    ],
)
def test_jacobi_solver(xp, A, b, x):
    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)
    x_bin = BinsparseFormat.from_numpy(x)

    jacobi.xp = xp
    benchmark = jacobi.JacobiBenchmark()
    x_sol = benchmark.benchmark(
        [
            xp.from_binsparse(A_bin),
            xp.from_binsparse(b_bin),
            xp.from_binsparse(x_bin),
        ],
        {},
    )[0]
    x_sol = np.round(x_sol, decimals=4)

    b_coo = BinsparseFormat.to_coo(b_bin)
    assert b_coo == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(A @ x_sol))
