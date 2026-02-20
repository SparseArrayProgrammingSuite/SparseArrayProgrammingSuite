import pytest

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from sparseappbench.benchmarks.GMRES import (
    dg_gmres_sparse_1,
    dg_gmres_sparse_2,
    dg_gmres_sparse_3,
    dg_gmres_sparse_4,
    dg_gmres_sparse_5,
    dg_gmres_sparse_6,
    dg_gmres_sparse_7,
    dg_gmres_sparse_8,
    gmres,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def get_framework():
    return NumpyFramework()


@pytest.mark.parametrize("seed", [42, 123])
def scipy_gmres_test(seed):
    rng = np.random.default_rng(seed)
    N = 50
    A = scipy.sparse.random(N, N, density=0.1, random_state=rng)
    A = A + scipy.sparse.eye(N) * N
    x_true = rng.standard_normal(N)
    b = A @ x_true
    x0 = np.zeros(N)

    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    b_bin = BinsparseFormat.from_numpy(b)
    x0_bin = BinsparseFormat.from_numpy(x0)

    xp = get_framework()

    x_bench_bin = gmres(xp, A_bin, b_bin, x0_bin, restart=20, tol=1e-8, max_iter=1000)
    x_bench = xp.from_benchmark(x_bench_bin)

    x_scipy, info = scipy.sparse.linalg.gmres(
        A, b, x0=x0, restart=20, tol=1e-8, atol=0, maxiter=1000
    )
    assert info == 0, "Scipy GMRES failed to converge"

    res_bench = np.linalg.norm(b - A @ x_bench)

    assert res_bench < 1e-5, f"Benchmark GMRES did not converge well: {res_bench}"
    assert np.allclose(x_bench, x_scipy, atol=1e-4, rtol=1e-4), (
        "Solutions differ significantly from Scipy"
    )


@pytest.mark.parametrize(
    "A_dense, b, x0",
    [
        (np.array([[2.0, 0.0], [0.0, 3.0]]), np.array([4.0, 9.0]), np.zeros(2)),
        (
            np.array([[10.0, 2.0, 1.0], [1.0, 20.0, 1.0], [1.0, 2.0, 10.0]]),
            np.array([13.0, 22.0, 13.0]),
            np.zeros(3),
        ),
        (
            np.array(
                [
                    [4.0, -1.0, 0.0, 0.0],
                    [-1.0, 4.0, -1.0, 0.0],
                    [0.0, -1.0, 4.0, -1.0],
                    [0.0, 0.0, -1.0, 3.0],
                ]
            ),
            np.array([3.0, 2.0, 2.0, 2.0]),
            np.zeros(4),
        ),
    ],
)
def test_gmres_sample_examples(A_dense, b, x0):
    xp = get_framework()

    A_coo = scipy.sparse.coo_matrix(A_dense)
    A_bin = BinsparseFormat.from_coo((A_coo.row, A_coo.col), A_coo.data, A_coo.shape)
    b_bin = BinsparseFormat.from_numpy(b)
    x0_bin = BinsparseFormat.from_numpy(x0)

    x_bench_bin = gmres(
        xp, A_bin, b_bin, x0_bin, restart=A_dense.shape[0], tol=1e-8, max_iter=100
    )
    x_bench = xp.from_benchmark(x_bench_bin)

    residual = np.linalg.norm(b - A_dense @ x_bench)
    assert residual < 1e-6, f"Residual too high: {residual}"


@pytest.mark.parametrize(
    "generator",
    [
        dg_gmres_sparse_1,
        dg_gmres_sparse_2,
        dg_gmres_sparse_3,
        dg_gmres_sparse_4,
        dg_gmres_sparse_5,
        dg_gmres_sparse_6,
        dg_gmres_sparse_7,
        dg_gmres_sparse_8,
    ],
)
def test_gmres_sparse_generators(generator):
    xp = get_framework()
    try:
        A_bin, b_bin, x0_bin = generator()
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"Failed to download/load data: {e}")

    x_bench_bin = gmres(xp, A_bin, b_bin, x0_bin, restart=100, tol=1e-5, max_iter=3000)
    x_bench = xp.from_benchmark(x_bench_bin)

    A = xp.from_benchmark(A_bin)
    b = xp.from_benchmark(b_bin)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        assert np.linalg.norm(x_bench) < 1e-12
    else:
        res_norm = np.linalg.norm(b - A @ x_bench)
        rel_resid = res_norm / b_norm

        print(f"Generator {generator.__name__} Relative Residual: {rel_resid}")

        assert rel_resid < 1e-4, f"Relative residual too high: {rel_resid}"
