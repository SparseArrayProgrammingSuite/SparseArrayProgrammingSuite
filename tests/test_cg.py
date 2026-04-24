import pytest

import numpy as np

import sparseappbench.benchmarks.cg as cg
from saps_framework import BinsparseFormat 
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import PyDataSparseFramework


@pytest.mark.parametrize(
    "xp, A, b, x",
    [
        (
            PyDataSparseFramework(),
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
            np.array([4.0, 8.0, 16.0]),  # b = A @ [1, 2, 3]
            np.zeros((3,)),
        ),
        (
            NumpyFramework(),
            np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
            np.array([13.0, -3.0, 8.0]),  # b = A @ [2, -1, 1]
            np.zeros((3,)),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [8.0, -1.0, 0.0, 0.0],
                    [-1.0, 8.0, -1.0, 0.0],
                    [0.0, -1.0, 8.0, -1.0],
                    [0.0, 0.0, -1.0, 8.0],
                ]
            ),
            np.array([8.0, -2.0, 6.0, 15.0]),  # b = A @ [1, 0, 1, 2]
            np.zeros((4,)),
        ),
        (
            NumpyFramework(),
            np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
            np.array([40.0, 10.0, -18.0]),  # b = A @ [3, 1, -2]
            np.zeros((3,)),
        ),
        (
            SciPyFramework(),
            np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]),
            np.array([118.0, 116.0, 118.0]),  # b = A @ [1, 1, 1]
            np.zeros((3,)),
        ),
        (
            SciPyFramework(),
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
            np.zeros((5,)),
        ),
    ],
)
def test_jacobi_solver(xp, A, b, x):
    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)
    x_bin = BinsparseFormat.from_numpy(x)

    cg.xp = xp
    benchmark = cg.CGBenchmark()
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
