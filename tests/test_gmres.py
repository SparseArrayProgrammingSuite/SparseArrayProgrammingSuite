import pytest

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import saps.benchmarks.GMRES as gmres
from frameworks.saps_numpy import NumpyFramework


def get_framework():
    return NumpyFramework()


def run_gmres_benchmark(A, b, x0, restart=50, tol=1e-8, max_iter=1000):
    benchmark = gmres.GMRESBenchmark()
    prev_xp = getattr(gmres, "xp", None)
    gmres.xp = get_framework()
    try:
        (x_bench,) = benchmark.benchmark(
            [A, b, x0],
            {
                "restart": restart,
                "tol": tol,
                "max_iter": max_iter,
            },
        )
    finally:
        gmres.xp = prev_xp
    return x_bench


@pytest.mark.parametrize("seed", [42, 123])
def test_scipy_gmres(seed):
    rng = np.random.default_rng(seed)
    N = 50
    A = scipy.sparse.random(N, N, density=0.1, random_state=rng)
    A = A + scipy.sparse.eye(N) * N
    x_true = rng.standard_normal(N)
    b = A @ x_true
    x0 = np.zeros(N)

    x_bench = run_gmres_benchmark(
        A.toarray(), b, x0, restart=20, tol=1e-8, max_iter=1000
    )

    x_scipy, info = scipy.sparse.linalg.gmres(
        A, b, x0=x0, restart=20, rtol=1e-8, atol=0, maxiter=1000
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
    x_bench = run_gmres_benchmark(
        A_dense, b, x0, restart=A_dense.shape[0], tol=1e-8, max_iter=100
    )

    residual = np.linalg.norm(b - A_dense @ x_bench)
    assert residual < 1e-6, f"Residual too high: {residual}"


@pytest.mark.parametrize(
    "dataset",
    gmres.GMRESGenerator().datasets,
)
def test_gmres_sparse_generators(dataset):
    xp = get_framework()
    try:
        data = gmres.GMRESGenerator().generate(dataset).inputs
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"Failed to download/load data: {e}")

    A, b, x0 = [xp.from_binsparse(d) for d in data]
    x_bench = run_gmres_benchmark(A, b, x0, restart=100, tol=1e-5, max_iter=3000)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-12:
        assert np.linalg.norm(x_bench) < 1e-12
    else:
        res_norm = np.linalg.norm(b - A @ x_bench)
        rel_resid = res_norm / b_norm

        print(f"Dataset {dataset.name} Relative Residual: {rel_resid}")

        assert rel_resid < 1e-4, f"Relative residual too high: {rel_resid}"
