import pytest

import numpy as np

from sparseappbench.benchmarks.approx_nn import (
    benchmark_johnson_lindenstrauss_nn,
    data_knn_rla_generator,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_jl_preserves_distance(rng):
    xp = NumpyFramework()
    n_samples = 20
    n_features = 10
    k = 3
    eps = 0.01

    data_bench = rng.standard_normal((n_samples, n_features))
    query_bench = rng.standard_normal((4, n_features))

    projection_matrix = data_knn_rla_generator(xp, data_bench, seed=13, eps=eps)

    nearest_ind, _ = benchmark_johnson_lindenstrauss_nn(
        xp, data_bench, query_bench, projection_matrix, k=k, eps=eps
    )

    # Convert benchmark objects back into framework arrays
    nearest_ind = xp.from_benchmark(nearest_ind)

    # True distances
    diff = xp.einsum("X[i, j, k] = Q[i, k] - D[j, k]", Q=query_bench, D=data_bench)
    orig_distances = np.sqrt(np.sum(diff**2, axis=-1))
    print("Original distances:", orig_distances)
    print("Nearest indices:", nearest_ind)


    # Checks if the returned nearest neighbors are a similar distance as the
    # true nearest neighbors
    true_nearest = np.min(orig_distances, axis=1)
    approx_nearest = orig_distances[xp.arange(4),nearest_ind[:,1]]
    print("True nearest distances:", true_nearest)
    print("Approx nearest distances:", approx_nearest)
    assert np.all(approx_nearest <= (1 + eps) * true_nearest)




def test_data_knn_rla_generator_shape_and_scale(rng):
    xp = NumpyFramework()
    n_samples = 10
    n_features = 8
    eps = 0.2

    data_bench = rng.standard_normal((n_samples, n_features))
    U = data_knn_rla_generator(xp, data_bench, seed=13, eps=eps)

    # Convert benchmark object back into framework array

    target_dim = int(np.log(n_samples) / (eps * eps))
    if target_dim > n_features:
        target_dim = n_features

    assert U.shape == (n_features, target_dim)
    assert np.count_nonzero(U) > 0
