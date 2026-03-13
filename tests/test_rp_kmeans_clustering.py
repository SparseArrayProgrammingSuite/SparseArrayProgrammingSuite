import numpy as np

from sparseappbench.benchmarks.rp_kmeans_clustering import (
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

    labels = rp_kmeans_clustering(xp, A_bin, k=3, eps=0.3, c=0.5, max_iter=5).tolist()

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

    labels = rp_kmeans_clustering(xp, A_bin, k=2, eps=0.2, c=1, max_iter=5).tolist()

    assert labels[0] == labels[1] == labels[2] == labels[3] and labels[0] != labels[4]
