"""
Name: Random Projections for k-means Clustering
Author: Maksim Krylykov
Email: mkrylykov3@gatech.edu

Motivation:
Random Projections reduce dimensionality for k-means clustering.

Role of sparsity:
Input points can be high-dimensional and sparse,
which are then projected on a random matrix.

Implementation based on:
C. Boutsidis, A. Zouzias, P. Drineas.
"Random Projections for k-means Clustering".
arXiv:1011.4632 (2010).

Data Generation:
Test points for this benchmark were created manually.

Generative AI:
No generative AI was used to implement benchmark functions.
"""

import math
import os
import pickle
import tarfile
import urllib.request

import numpy as np
import scipy.sparse
from sklearn.datasets import fetch_openml

from ..binsparse_format import BinsparseFormat
import kagglehub

"""
Labels points into k clusters.

Args:
----
xp : array_api
    The array API module to utilize
A_benchmark : BinsparseFormat
    Sparse input matrix
k : int
    Number of clusters
eps : float
    Error parameter in (0, 1/3)
c : float
    Constant factor for new dimensionality t
max_iter : int
    Number of iterations for k-means

Returns:
-------
Returns xp.array of size n: labels of input points.
"""


def rp_kmeans_clustering(xp, A_benchmark, k, eps, c=1, max_iter=100):
    assert c > 0
    assert eps > 0 and eps < 1 / 3
    assert k > 0
    A = xp.from_benchmark(A_benchmark)
    A = xp.lazy(A)
    n, d = A.shape
    t = int(c * math.ceil(k / eps**2))
    value = 1 / (t**0.5)
    rng = np.random.default_rng(0)
    R = np.where(rng.random((d, t)) < 0.5, value, -value)
    A = xp.lazy(A)
    R = xp.lazy(R)
    A_prime = xp.compute(xp.matmul(A, R))
    labels = kmeans(xp, A_prime, k, max_iter)
    return xp.to_benchmark(xp.compute(labels))


def kmeans(xp, A, k, max_iter=100):
    n, t = A.shape
    centroids = A[:k, :]
    labels = xp.zeros((n,), dtype=xp.int64)
    ks = xp.arange(k, dtype=xp.int64)
    one = xp.asarray(1, dtype=A.dtype)
    for _ in range(max_iter):
        old_labels = labels
        dists = xp.sum((A[:, None] - centroids[None, :]) ** 2, axis=2)
        labels = xp.argmin(dists, axis=1)
        H = xp.equal(labels[:, None], ks[None, :]).astype(A.dtype)
        counts = xp.sum(H, axis=0)
        sums = xp.matmul(xp.transpose(H), A)
        counts_nonz = xp.where(counts > 0, counts, one)
        new_centroids = sums / counts_nonz[:, None]
        new_centroids = xp.where((counts == 0)[:, None], centroids, new_centroids)
        centroids = new_centroids
        if xp.all(labels == old_labels).item():
            break

    return labels


def generate_mnist_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    training, testing = X[:60000], X[60000:]

    train_bin = BinsparseFormat.from_numpy(training)
    test_bin = BinsparseFormat.from_numpy(testing)
    return (train_bin, test_bin)
 

def dg_kmeans_mnist():
    train_bin, _ = generate_mnist_data()
    return (train_bin, 10, 0.3)


def generate_cifar10_data():
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
    for i in range(1, 6):
        with open(os.path.join(extracted_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        train_arrays.append(batch[b"data"])
    training = np.concatenate(train_arrays, axis=0).astype(np.float32) / 255.0

    return BinsparseFormat.from_numpy(training)


def dg_kmeans_cifar10():
    training = generate_cifar10_data()
    return (training, 10, 0.3)


def generate_netflix_data():

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
                    #1-based to 0-based
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
    X = scipy.sparse.csr_matrix(
        (np.array(val_list, dtype=np.float32),
         (np.array(row_list, dtype=np.int32), np.array(col_list, dtype=np.int32))),
        shape=(n_users, n_movies),
    )

    rng = np.random.default_rng(0)
    train_coo = X[rng.permutation(n_users)[:5000]].tocoo()

    return BinsparseFormat.from_coo((train_coo.row, train_coo.col), train_coo.data, train_coo.shape)

def dg_kmeans_netflix():
    training = generate_netflix_data()
    return (training, 10, 0.3)