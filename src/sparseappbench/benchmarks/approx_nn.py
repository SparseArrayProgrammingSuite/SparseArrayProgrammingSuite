import numpy as np
import scipy as sp
from sklearn.datasets import fetch_openml

from ..binsparse_format import BinsparseFormat

"""
Name: Random Numerical Linear Algenra
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to create python tests that are for RLA methods.
Specifically, I will first show the application of the JL Lemma for NN.
My goal is to write benchmarks on applications of RNLA,
for graph algorithms, PDEs, and Scientific Machine Learning

https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615

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
    xp, data_bench, query_bench, k=5, eps=0.1, seed=40
):
    data = xp.lazy(xp.from_benchmark(data_bench))
    query = xp.lazy(xp.from_benchmark(query_bench))
    projection = xp.lazy(_generate_rla_projection(xp.from_benchmark(data_bench), seed=seed, eps=eps))

    # Project to lower subspace
    projected_data = xp.matmul(data, projection)
    projected_query = xp.matmul(query, projection)

    # -----K Nearest Neighbour from here on out--------

    # Euclidean distances
    diff = xp.einsum(
        "X[i, j, k] = Q[i, k] - D[j, k]", Q=projected_query, D=projected_data
    )
    distances = xp.sqrt(xp.sum(diff**2, axis=-1))

    # Get nearest k neighbors.
    sorted_indices = xp.argsort(distances)

    # Get nearest indices and associated distances.
    nearest_indices = xp.take(sorted_indices, xp.arange(k), axis=1)
    nearest_distances = xp.take(xp.sort(distances), xp.arange(k), axis=1)

    nearest_indices = xp.compute(nearest_indices)
    nearest_distances = xp.compute(nearest_distances)

    return xp.to_benchmark(nearest_indices), xp.to_benchmark(nearest_distances)


def _generate_rla_projection(data, seed=40, eps=0.1):
    n_samples, n_features = data.shape
    #  Johnson Lindenstrauss Theorem Lemmna.
    # The eps represents the disortion of distance by epsilon,
    # between the the original space and the reduced subspace
    target_dim = np.ceil(np.log(n_samples) / (eps * eps)).astype(int)

    rng = np.random.default_rng(seed)

    s = np.sqrt(n_features)  # s = 1/density
    density = 1.0 / s  # probability of a nonzero entry = density.
    density_half = density / 2.0  # probability for + or -
    scale = np.sqrt(s / target_dim)  # scale = sqrt(s / n_components)

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
    return (U_Neg + U_Pos).toarray()


def _load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    return X[:60000], X[60000:]

def dg_approx_nn_mnist():
    training, testing = _load_mnist()

    #sampling 2000/60000 for train, 200/60000 for test
    random = np.random.default_rng(50)
    training = training[random.choice(len(training), size=2000, replace=False)]
    testing = testing[random.choice(len(testing), size=200, replace=False)]

    return (
        BinsparseFormat.from_numpy(training),
        BinsparseFormat.from_numpy(testing),
    )
