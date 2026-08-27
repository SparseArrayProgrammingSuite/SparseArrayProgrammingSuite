import numpy as np
import pytest

from sparseappbench.benchmarks.approx_nn import (
    benchmark_johnson_lindenstrauss_nn,
    dg_approx_nn_cifar10,
    dg_approx_nn_mnist,
    dg_approx_nn_netflix,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_jl_preserves_distance(rng):
    xp = NumpyFramework()
    n_samples = 20
    n_features = 10
    n_queries = 4
    k = 3
    eps = 0.01

    data_np = rng.standard_normal((n_samples, n_features))
    query_np = rng.standard_normal((n_queries, n_features))

    nearest_ind, _ = benchmark_johnson_lindenstrauss_nn(
        xp,
        BinsparseFormat.from_numpy(data_np),
        BinsparseFormat.from_numpy(query_np),
        k=k,
        eps=eps,
        seed=13,
    )

    nearest_ind = xp.from_benchmark(nearest_ind)

    diff = xp.einsum("X[i, j, k] = Q[i, k] - D[j, k]", Q=query_np, D=data_np)
    orig_distances = np.sqrt(np.sum(diff**2, axis=-1))

    true_nearest = np.min(orig_distances, axis=1)
    approx_nearest = orig_distances[xp.arange(n_queries), nearest_ind[:, 0]]
    assert np.all(approx_nearest <= (1 + eps) * true_nearest)


def test_dg_approx_nn_mnist():
    xp = NumpyFramework()
    k = 5
    eps = 0.3

    data_set, query_set = dg_approx_nn_mnist()

    data_np = xp.from_benchmark(data_set)
    query_np = xp.from_benchmark(query_set)

    nearest_ind, nearest_dist = benchmark_johnson_lindenstrauss_nn(
        xp, data_set, query_set, k=k, eps=eps
    )

    nearest_ind = xp.from_benchmark(nearest_ind)
    nearest_dist = xp.from_benchmark(nearest_dist)

    #number of rows and columns match images and neighbors
    assert nearest_ind.shape == (len(query_np), k)
    assert nearest_dist.shape == (len(query_np), k)

    #valid distances
    assert np.all(nearest_ind >= 0) and np.all(nearest_ind < len(data_np))
    assert np.all(nearest_dist >= 0)

    #sorted correctly
    assert np.all(nearest_dist[:, :-1] <= nearest_dist[:, 1:])


def test_dg_approx_nn_cifar10():
    xp = NumpyFramework()
    k = 5
    eps = 0.3

    try:
        data_set, query_set = dg_approx_nn_cifar10()
    except Exception as e:
        pytest.skip(f"Failed to download/load CIFAR-10 data: {e}")

    data_np = xp.from_benchmark(data_set)
    query_np = xp.from_benchmark(query_set)

    nearest_ind, nearest_dist = benchmark_johnson_lindenstrauss_nn(
        xp, data_set, query_set, k=k, eps=eps
    )

    nearest_ind = xp.from_benchmark(nearest_ind)
    nearest_dist = xp.from_benchmark(nearest_dist)

    assert nearest_ind.shape == (len(query_np), k)
    assert nearest_dist.shape == (len(query_np), k)
    assert np.all(nearest_ind >= 0) and np.all(nearest_ind < len(data_np))
    assert np.all(nearest_dist >= 0)
    assert np.all(nearest_dist[:, :-1] <= nearest_dist[:, 1:])


def test_dg_approx_nn_netflix():
    xp = NumpyFramework()
    k = 5
    eps = 0.3

    try:
        data_set, query_set = dg_approx_nn_netflix()
    except Exception as e:
        pytest.skip(f"Failed to download/load Netflix data: {e}")

    data_np = xp.from_benchmark(data_set)
    query_np = xp.from_benchmark(query_set)

    nearest_ind, nearest_dist = benchmark_johnson_lindenstrauss_nn(
        xp, data_set, query_set, k=k, eps=eps
    )

    nearest_ind = xp.from_benchmark(nearest_ind)
    nearest_dist = xp.from_benchmark(nearest_dist)

    assert nearest_ind.shape == (len(query_np), k)
    assert nearest_dist.shape == (len(query_np), k)
    assert np.all(nearest_ind >= 0) and np.all(nearest_ind < len(data_np))
    assert np.all(nearest_dist >= 0)
    assert np.all(nearest_dist[:, :-1] <= nearest_dist[:, 1:])
