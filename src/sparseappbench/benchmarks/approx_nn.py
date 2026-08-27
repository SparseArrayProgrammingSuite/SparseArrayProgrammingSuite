import os
import pickle
import tarfile
import urllib.request

import numpy as np
import scipy as sp
from sklearn.datasets import fetch_openml

from ..binsparse_format import BinsparseFormat
import kagglehub

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
    projection = xp.lazy(rla_projection(xp.from_benchmark(data_bench), seed=seed, eps=eps))

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


def rla_projection(data, seed=40, eps=0.1):
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


def generate_mnist_data(n_train=None, n_test=None, seed=0):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    training, testing = X[:60000], X[60000:]

    rng = np.random.default_rng(seed)
    if n_train is not None:
        training = training[rng.choice(len(training), size=n_train, replace=False)]
    if n_test is not None:
        testing = testing[rng.choice(len(testing), size=n_test, replace=False)]

    train_bin = BinsparseFormat.from_numpy(training)
    test_bin = BinsparseFormat.from_numpy(testing)
    return (train_bin, test_bin)


def dg_approx_nn_mnist():
    return generate_mnist_data(n_train=2000, n_test=200, seed=50)


def generate_cifar10_data(n_train=None, n_test=None, seed=0):
    
    #Seeing if dataset is already cached
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sparseappbench", "cifar10")
    extracted_dir = os.path.join(cache_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        os.makedirs(cache_dir, exist_ok=True)
        tar_path = os.path.join(cache_dir, "cifar-10-python.tar.gz")
        urllib.request.urlretrieve(
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path
        )
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)

    train_arrays = []

    #CIFAR splits 50,000 training images across 5 pickle files
    for i in range(1, 6):
        with open(os.path.join(extracted_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        train_arrays.append(batch[b"data"])
    training = np.concatenate(train_arrays, axis=0).astype(np.float32) / 255.0

    with open(os.path.join(extracted_dir, "test_batch"), "rb") as f:
        testing = pickle.load(f, encoding="bytes")[b"data"].astype(np.float32) / 255.0

    rng = np.random.default_rng(seed)
    if n_train is not None:
        training = training[rng.choice(len(training), size=n_train, replace=False)]
    if n_test is not None:
        testing = testing[rng.choice(len(testing), size=n_test, replace=False)]

    return BinsparseFormat.from_numpy(training), BinsparseFormat.from_numpy(testing)


def dg_approx_nn_cifar10():
    return generate_cifar10_data(n_train=2000, n_test=200, seed=0)


def generate_netflix_data(n_train=None, n_test=None, seed=0):

    cache_path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")

    row_list = []
    col_list = []
    val_list = []
    user_map = {}
    current_movie = 0
    data_files = sorted(f for f in os.listdir(cache_path) if f.startswith("combined_data") and f.endswith(".txt"))
    for fname in data_files:
        with open(os.path.join(cache_path, fname), "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    current_movie = int(line[:-1]) - 1
                else:
                    uid_str, rating_str, _ = line.split(",", 2)
                    uid = int(uid_str)
                    if uid not in user_map:
                        user_map[uid] = len(user_map)
                    row_list.append(user_map[uid])
                    col_list.append(current_movie)
                    val_list.append(float(rating_str))

    n_users = len(user_map)
    n_movies = 17770
    X = sp.sparse.csr_matrix(
        (np.array(val_list, dtype=np.float32),
         (np.array(row_list, dtype=np.int32), np.array(col_list, dtype=np.int32))),
        shape=(n_users, n_movies),
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_users)
    n_train_actual = n_train if n_train is not None else int(0.8 * n_users)
    n_test_actual = n_test if n_test is not None else n_users - n_train_actual

    train_coo = X[perm[:n_train_actual]].tocoo()
    test_coo = X[perm[n_train_actual:n_train_actual + n_test_actual]].tocoo()

    return (
        BinsparseFormat.from_coo((train_coo.row, train_coo.col), train_coo.data, train_coo.shape),
        BinsparseFormat.from_coo((test_coo.row, test_coo.col), test_coo.data, test_coo.shape),
    )


def dg_approx_nn_netflix():
    return generate_netflix_data(n_train=2000, n_test=200, seed=0)
