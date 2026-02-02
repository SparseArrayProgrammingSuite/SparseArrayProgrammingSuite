import numpy as np
import scipy as sp

"""
Name: Random Numerical Linear Algenra
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to create python tests that are for RLA methods.
Specifically, I will first show the application of the JL Lemma for NN.
My goal is to write benchmarks on applications of RNLA,
for graph algorithms, PDEs, and Scientific Machine Learning

Murray, R., Demmel, J., Mahoney, M. W., Erichson, N. B.,
Melnichenko, M., Malik, O. A., ... & Dongarra, J. (2023).
Randomized numerical linear algebra: A perspective on the field with an eye to software.
arXiv preprint arXiv:2302.11474.
Role of sparsity (How sparsity is used in the problem):
The inputs to the matrix multiply are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""


def benchmark_johnson_lindenstrauss_nn(
    xp, data_bench, query_bench, projection_matrix, k=5, eps=0.1
):
    data = xp.lazy(data_bench)
    query = xp.lazy(query_bench)
    P = xp.lazy(projection_matrix)

    n_samples, n_features = data.shape
    #  Johnson Lindenstrauss Theorem Lemmna.
    # The eps represents the disortion of distance by epsilon,
    # between the the original space and the reduced subspace
    target_dim = np.log(n_samples) / (eps * eps)
    if target_dim > n_features:
        target_dim = n_features

    # Project to lower subspace
    projected_data = xp.matmul(data, P)
    projected_query = xp.matmul(query, P)

    # -----K Nearest Neighbour from here on out--------

    # Euclidean distances
    diff = projected_data - projected_query
    distances = xp.sqrt(xp.sum(diff**2, axis=1))

    # Get nearest k neighbors.
    sorted_indices = xp.argsort(distances)

    # Get nearest indices and associated distances.
    nearest_indices = xp.take(sorted_indices, xp.arange(k))
    nearest_distances = xp.take(xp.sort(distances), xp.arange(k))

    nearest_indices = xp.compute(nearest_indices)
    nearest_distances = xp.compute(nearest_distances)

    return xp.to_benchmark(nearest_indices), xp.to_benchmark(nearest_distances)


def data_knn_rla_generator(xp, data_bench, seed=40, eps=0.1):
    data = xp.lazy(data_bench)
    n_samples, n_features = data.shape
    #  Johnson Lindenstrauss Theorem Lemmna.
    # The eps represents the disortion of distance by epsilon,
    # between the the original space and the reduced subspace
    target_dim = np.log(n_samples) / (eps * eps)
    if target_dim > n_features:
        target_dim = n_features

    s = np.sqrt(n_features)  # s = 1/density
    density = 1.0 / s  # probability of a nonzero entry = density.
    density_half = density / 2.0  # probability for + or -
    scale = np.sqrt(s / target_dim)  # scale = sqrt(s / n_components)

    rng = np.random.default_rng(seed)

    U_Neg = sp.sparse.random(
        n_features,
        target_dim,
        density_half,
        data_rvs=lambda k: np.full(
            k, -scale, dtype=float
        ),  # specified dtype to see of that made a difference
        random_state=rng,
    )
    U_Pos = sp.sparse.random(
        n_features,
        target_dim,
        density_half,
        data_rvs=lambda k: np.full(
            k, scale, dtype=float
        ),  # specified dtype to see of that made a difference
        random_state=rng,
    )
    U_dense = (U_Neg + U_Pos).toarray()
    return xp.lazy(U_dense)
