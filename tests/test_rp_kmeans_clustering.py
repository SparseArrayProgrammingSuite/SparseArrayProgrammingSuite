import numpy as np
import pytest

from sparseappbench.benchmarks.rp_kmeans_clustering import (
    dg_kmeans_cifar10,
    dg_kmeans_mnist,
    dg_kmeans_netflix,
    rp_kmeans_clustering,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_rp_kmeans_sanity_check():
    xp = NumpyFramework()
    points = xp.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [5.0, 5.0, 5.0],
            [5.1, 5.0, 5.2],
            [-10.0, -10.1, -10.2],
            [-9.9, -9.8, -9.7],
        ],
        dtype=np.float32,
    )
    A_bin = BinsparseFormat.from_numpy(points)

    result = rp_kmeans_clustering(xp, A_bin, k=3, eps=0.3, c=0.5, max_iter=5)
    labels = xp.from_benchmark(result).tolist()

    assert (
        labels[0] == labels[1]
        and labels[2] == labels[3]
        and labels[4] == labels[5]
        and len(set(labels)) == 3
    )


def test_rp_kmeans_two_clusters():
    xp = NumpyFramework()
    points = xp.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [-0.2, 0.0, 0.1],
            [0.3, 0.2, 0.1],
            [-20.0, -40.0, -60.0],
        ],
        dtype=np.float32,
    )
    A_bin = BinsparseFormat.from_numpy(points)

    result = rp_kmeans_clustering(xp, A_bin, k=2, eps=0.2, c=1, max_iter=5)
    labels = xp.from_benchmark(result).tolist()

    assert labels[0] == labels[1] == labels[2] == labels[3] and labels[0] != labels[4]


def test_dg_kmeans_mnist():
    xp = NumpyFramework()

    data_set, k, eps = dg_kmeans_mnist()

    data_np = xp.from_benchmark(data_set)
    result = rp_kmeans_clustering(xp, data_set, k=k, eps=eps)
    labels = xp.from_benchmark(result)

    assert labels.shape == (len(data_np),)
    assert np.all(labels >= 0) and np.all(labels < k)

    #correct number of clusters (10)
    assert len(np.unique(labels)) == k


def test_dg_kmeans_cifar10():
    xp = NumpyFramework()

    try:
        data_set, k, eps = dg_kmeans_cifar10()
    except Exception as e:
        pytest.skip(f"Failed to download/load CIFAR-10 data: {e}")

    data_np = xp.from_benchmark(data_set)
    result = rp_kmeans_clustering(xp, data_set, k=k, eps=eps)
    labels = xp.from_benchmark(result)

    assert labels.shape == (len(data_np),)
    assert np.all(labels >= 0) and np.all(labels < k)
    assert len(np.unique(labels)) == k


def test_dg_kmeans_netflix():
    xp = NumpyFramework()

    try:
        data_set, k, eps = dg_kmeans_netflix()
    except Exception as e:
        pytest.skip(f"Failed to download/load Netflix data: {e}")

    data_np = xp.from_benchmark(data_set)
    result = rp_kmeans_clustering(xp, data_set, k=k, eps=eps)
    labels = xp.from_benchmark(result)

    assert labels.shape == (len(data_np),)
    assert np.all(labels >= 0) and np.all(labels < k)
