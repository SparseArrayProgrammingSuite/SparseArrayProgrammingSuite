import pytest

import numpy as np
import scipy.sparse as sp

from sparseappbench.benchmarks.preconditioned_cg import (
    generate_block_jacobi_M,
    preconditioned_cg,
    solve_block_jacobi_cg,
    solve_jacobi_cg,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import (
    PyDataSparseFramework,
)

A0 = np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]])
A1 = np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]])
A2 = np.array(
    [
        [8.0, -1.0, 0.0, 0.0],
        [-1.0, 8.0, -1.0, 0.0],
        [0.0, -1.0, 8.0, -1.0],
        [0.0, 0.0, -1.0, 8.0],
    ]
)
A3 = np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]])
A4 = np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]])
A5 = np.array(
    [
        [15.0, -2.0, 0.0, 0.0, -1.0],
        [-2.0, 14.0, -3.0, 0.0, 0.0],
        [0.0, -3.0, 16.0, -2.0, 0.0],
        [0.0, 0.0, -2.0, 15.0, -3.0],
        [-1.0, 0.0, 0.0, -3.0, 17.0],
    ]
)


@pytest.mark.parametrize(
    "xp, A, b, x, M, solve",
    [
        (
            PyDataSparseFramework(),
            A0,
            np.array([17.0, 1.0, 11.0]),  # b = A @ [3, 1, 2]
            np.zeros((3,)),
            generate_block_jacobi_M(sp.coo_matrix(A0)),
            solve_block_jacobi_cg,
        ),
        (
            PyDataSparseFramework(),
            A1,
            np.array([10.0, 7.0, 5.0]),  # b = A @ [1, 1, 1]
            np.zeros((3,)),
            sp.coo_matrix(A1).diagonal(),
            solve_jacobi_cg,
        ),
        (
            NumpyFramework(),
            A2,
            np.array([17.0, -10.0, 0.0, 8.0]),  # b = A @ [2, -1, 0, 1]
            np.zeros((4,)),
            generate_block_jacobi_M(sp.coo_matrix(A2)),
            solve_block_jacobi_cg,
        ),
        (
            NumpyFramework(),
            A3,
            np.array([-12.0, 30.0, 43.0]),  # b = A @ [-1, 2, 4]
            np.zeros((3,)),
            sp.coo_matrix(A3).diagonal(),
            solve_jacobi_cg,
        ),
        (
            CheckerFramework(),
            A4,
            np.array([590.0, 580.0, 590.0]),  # b = Ad @ [5, 5, 5]
            np.zeros((3,)),
            generate_block_jacobi_M(sp.coo_matrix(A4)),
            solve_block_jacobi_cg,
        ),
        (
            CheckerFramework(),
            A5,
            np.array([6.0, 17.0, 34.0, 39.0, 72.0]),  # b = A @ [1, 2, 3, 4, 5]
            np.zeros((5,)),
            sp.coo_matrix(A5).diagonal(),
            solve_jacobi_cg,
        ),
    ],
)
def test_preconditioned_cg(xp, A, b, x, M, solve):
    if sp.issparse(M):
        M_bin = BinsparseFormat.from_coo((M.row, M.col), M.data, M.shape)
    else:
        M_bin = BinsparseFormat.from_numpy(M)

    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)
    x_bin = BinsparseFormat.from_numpy(x)

    x_sol = preconditioned_cg(xp, A_bin, b_bin, x_bin, M_bin, solve)
    x_sol = xp.from_benchmark(x_sol)
    x_sol = np.round(x_sol, decimals=4)

    b_coo = BinsparseFormat.to_coo(b_bin)
    assert b_coo == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(A @ x_sol))
