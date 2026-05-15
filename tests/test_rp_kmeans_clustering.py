import numpy as np

import saps.benchmarks.rp_kmeans_clustering as rp_kmeans
from frameworks.saps_numpy import NumpyFramework
from saps_framework import BinsparseFormat


def run_rp_kmeans(xp, A, k, eps, c=1, max_iter=100):
    benchmark = rp_kmeans.RPKMeansBenchmark()
    prev_xp = getattr(rp_kmeans, "xp", None)
    rp_kmeans.xp = xp
    try:
        (labels,) = benchmark.benchmark(
            [A], {"k": k, "eps": eps, "c": c, "max_iter": max_iter}
        )
    finally:
        rp_kmeans.xp = prev_xp
    return labels


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
    A_input = xp.from_binsparse(A_bin)

    labels = run_rp_kmeans(xp, A_input, k=3, eps=0.3, c=0.5, max_iter=5).tolist()

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
    A_input = xp.from_binsparse(A_bin)

    labels = run_rp_kmeans(xp, A_input, k=2, eps=0.2, c=1, max_iter=5).tolist()

    assert labels[0] == labels[1] == labels[2] == labels[3] and labels[0] != labels[4]
