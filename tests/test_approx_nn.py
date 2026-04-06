import numpy as np

import saps.benchmarks.approx_nn as approx_nn
from saps.frameworks.numpy_framework import NumpyFramework


def test_jl_preserves_distance(rng):
    xp = NumpyFramework()
    approx_nn.xp = xp

    dataset = approx_nn.JLApproxNNDataset(
        name="test",
        pretty_name="test JL ANN",
        description="test dense data and query matrices with sparse random projection.",
        tags=["test", "rnla", "sparse"],
        n_samples=20,
        n_features=10,
        n_queries=4,
        k=3,
        eps=0.01,
        seed=42,
    )
    ((data, query, projection_matrix), meta) = approx_nn.JLApproxNNGenerator().generate(
        dataset
    )
    data = xp.from_binsparse(data)
    query = xp.from_binsparse(query)
    projection_matrix = xp.from_binsparse(projection_matrix)

    benchmark = approx_nn.JLApproxNearestNeighbor()
    nearest_ind, _ = benchmark.benchmark([data, query, projection_matrix], meta)

    # True distances
    diff = xp.einsum("X[i, j, k] = Q[i, k] - D[j, k]", Q=query, D=data)
    orig_distances = np.sqrt(np.sum(diff**2, axis=-1))

    # Checks if the returned nearest neighbors are a similar distance as the
    # true nearest neighbors
    true_nearest = np.min(orig_distances, axis=1)
    approx_nearest = orig_distances[xp.arange(dataset.n_queries), nearest_ind[:, 0]]
    assert np.all(approx_nearest <= (1 + dataset.eps) * true_nearest)
