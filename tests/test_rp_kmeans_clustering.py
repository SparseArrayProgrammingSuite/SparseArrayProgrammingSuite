import numpy as np

import sparseappbench.benchmarks.rp_kmeans_clustering as rp_kmeans_clustering
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_rp_two_clusters():
    xp = NumpyFramework()
    rp_kmeans_clustering.xp = xp

    dataset = rp_kmeans_clustering.RPKMeansDataset(
        name="test",
        pretty_name="test kmeans two cluster",
        description="test kmeans with sparse random projections",
        tags=["test", "rnla", "sparse"],
        k=2,
        eps=0.2,
        c=1,
        A=(
            xp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, -0.1, 0.0],
                    [-0.2, 0.0, 0.1],
                    [0.3, 0.2, 0.1],
                    [-20.0, -40.0, -60.0],
                ],
                dtype=np.float32,
            )
        ),
        max_iter=5,
    )
    benchmark = rp_kmeans_clustering.RPKmeansClustering()

    labels = benchmark.benchmark(dataset)

    assert labels[0] == labels[1] == labels[2] == labels[3] and labels[0] != labels[4]
