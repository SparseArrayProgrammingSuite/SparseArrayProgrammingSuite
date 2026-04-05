"""
wha: Graph Convolutional Network Inference
Author: Tarun Devi
Email: tdevi3@gatech.edu
Motivation: "Graphs are widely used for abstracting  systems of interacting objects,
 such as social networks (Easley et al., 2010), knowledge graphs (Nickel et al., 2015),
molecular graphs (Wu et al., 2018), and biological networks (Barabasi & Oltvai, 2004),
as well as for modeling 3D objects (Simonovsky & Komodakis, 2017),
manifolds (Bronstein et al., 2017), and source code (Allamanis et al.,
2017). Machine learning (ML), especially deep learning,
on graphs is an emerging field (Hamilton et al., 2017b; Bronstein et al., 2017)."
W. Hu et al., “Open Graph Benchmark: Datasets for Machine Learning on Graphs,”
arXiv, vol. 2005.00687, pp. 1–15, Feb. 2021, doi: 10.48550/arXiv.2005.00687.
Role of Sparsity:
To represent a graph, an adjaceny matrix is used, which is inherently sparse.
Implementation Details:
Implmentation is hand-written, based on the code observable in this repository:
https://anonymous.4open.science/r/scorch/README.md
Data Generation:
Data generators have not been implemented yet - using random weights for the matrix
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""

import os

import numpy as np
from scipy.io import mmread

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
    benchmark_gcn(xp, adjacency_bench, features_bench, weights1_bench,
      bias1_bench, weights2_bench, bias2_bench)

Computes a 2-layer Graph Convolutional Network forward pass:
    h1 = ReLU(adjacency @ features @ weights1 + bias1)
    output = adjacency @ h1 @ weights2 + bias2

Args:
----
xp : array_api
    Array API module (e.g. numpy, cupy, torch)
adjacency_bench : BinsparseFormat
    Sparse adjacency matrix of the graph
features_bench : BinsparseFormat
    Node feature matrix
weights1_bench : BinsparseFormat
    Weights for first GCN layer
bias1_bench : BinsparseFormat
    Bias for first GCN layer
weights2_bench : BinsparseFormat
    Weights for second GCN layer
bias2_bench : BinsparseFormat
    Bias for second GCN layer

Returns:
-------
BinsparseFormat
    Output node embeddings after 2-layer GCN
"""


def benchmark_gcn(
    xp,
    adjacency_bench,
    features_bench,
    weights1_bench,
    bias1_bench,
    weights2_bench,
    bias2_bench,
):
    adjacency = xp.lazy(xp.from_benchmark(adjacency_bench))
    features = xp.lazy(xp.from_benchmark(features_bench))
    weights1 = xp.lazy(xp.from_benchmark(weights1_bench))
    bias1 = xp.lazy(xp.from_benchmark(bias1_bench))
    weights2 = xp.lazy(xp.from_benchmark(weights2_bench))
    bias2 = xp.lazy(xp.from_benchmark(bias2_bench))

    # Layer 1: adjacency @ features -> linear transform -> ReLU
    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = xp.maximum(h1, 0)  # ReLU activation

    # Layer 2: adjacency @ h1 -> linear transform
    h2 = adjacency @ h1
    output = h2 @ weights2 + bias2

    solution = xp.compute(output)
    return xp.to_benchmark(solution)


def gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2):
    """Reference NumPy implementation of the 2-layer GCN used for tests.

    Inputs are dense NumPy arrays; adjacency is treated as a dense matrix for
    simplicity in tests (small graphs).
    """

    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = np.maximum(h1, 0)

    h2 = adjacency @ h1
    return h2 @ weights2 + bias2


def generate_gcn_data(
    source: str,
    feature_dim: int = 16,
    hidden_dim: int = 8,
    out_dim: int = 1,
    seed: int | None = None,
):
    matrices = ssgetpy.search(name=source)
    if not matrices:
        raise ValueError(f"No matrix found with name '{source}'")
    matrix = matrices[0]
    (path, archive) = matrix.download(extract=True)
    matrix_path = os.path.join(path, matrix.name + ".mtx")
    if matrix_path and os.path.exists(matrix_path):
        A = mmread(matrix_path)
    else:
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    rng = np.random.default_rng(0)
    A = A.tocoo()

    # Create feature/weight arrays using the RNG (deterministic)
    n = A.shape[0]
    features = rng.standard_normal((n, feature_dim))
    weights1 = rng.standard_normal((feature_dim, hidden_dim))
    bias1 = np.zeros((hidden_dim,))
    weights2 = rng.standard_normal((hidden_dim, out_dim))
    bias2 = np.zeros((out_dim,))

    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)
    return (A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b)


# Data generators for different graph types


def dg_gcn_social_1():
    """Small social network graph."""
    # Zachary's karate club - classic small social network
    return generate_gcn_data(
        "karate", feature_dim=16, hidden_dim=8, out_dim=1
    )  # 34 nodes, 78 edges


def dg_gcn_social_2():
    """Medium social network graph."""
    # Dolphins social network
    return generate_gcn_data(
        "dolphins", feature_dim=16, hidden_dim=8, out_dim=1
    )  # 62 nodes, 159 edges


def dg_gcn_social_3():
    """Larger social network graph."""
    # ca-GrQc: General Relativity collaboration network
    return generate_gcn_data(
        "ca-GrQc", feature_dim=32, hidden_dim=16, out_dim=1
    )  # ~5K nodes, ~14K edges


def dg_gcn_road_1():
    """Small road network graph."""
    # Chesapeake road network
    return generate_gcn_data(
        "chesapeake", feature_dim=8, hidden_dim=4, out_dim=1
    )  # ~39 nodes, ~170 edges


def dg_gcn_road_2():
    """Medium road network graph."""
    # Central road network
    return generate_gcn_data(
        "road_central", feature_dim=16, hidden_dim=8, out_dim=1
    )  # ~14K nodes


def dg_gcn_molecular_1():
    """Small molecular graph - Email network."""
    # Email communication network
    return generate_gcn_data(
        "email", feature_dim=16, hidden_dim=8, out_dim=1
    )  # 1.1K nodes, 5.5K edges


def dg_gcn_molecular_2():
    """Medium molecular graph - PDDB protein structure."""
    # PDDB protein structure
    return generate_gcn_data(
        "Chebyshev3", feature_dim=24, hidden_dim=12, out_dim=1
    )  # 4.1K nodes


def dg_gcn_large_1():
    """Large citation network (AIDS-like size)."""
    # ca-HepPh: Citation network
    return generate_gcn_data(
        "ca-HepPh", feature_dim=64, hidden_dim=32, out_dim=1
    )  # ~12K nodes, ~237K edges


def dg_gcn_large_2():
    """Very large road network."""
    # USA road network
    return generate_gcn_data(
        "road_usa", feature_dim=64, hidden_dim=32, out_dim=1
    )  # ~24M nodes, ~58M edges


# Keep original for backward compatibility
def dg_gcn_bcsstk01():
    """Original small structural engineering matrix (for backward compatibility)."""
    return generate_gcn_data(
        "bcsstk01", feature_dim=16, hidden_dim=8, out_dim=1
    )  # 48 nodes, 186 edges
