import os

import numpy as np
from scipy.io import mmread

import ssgetpy

from ..binsparse_format import BinsparseFormat

"""
Name: Floyd-Warshall algorithm
Co-Authors: Aarav Joglekar, Joel Mathew Cherian
Email: ajoglekar32@gatech.edu

Motivation:
The Floyd-Warshall algorithm computes the shortest paths between every pair of vertices
in a weighted directed graph.

Role of sparsity:
Sparse graphs reduce unnecessary computation, as most entries in the adjacency
matrix represent non-edges and begin as inifinity. Efficient sparse representations
allow the backend framework to skip work and minimize memory movement during the
relaxation steps of the algorithm.

Implementation Reference:
J. Kepner and J. Gilbert (eds.), “Graph Algorithms in the Language of Linear Algebra,”
Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2011.

Data Sources Used for Testing:
Unit tests use real-world networks, including the Chesapeake road network
and soc-tribes network.
from the Network Repository:
    @inproceedings{nr,
        title={The Network Data Repository with Interactive Graph Analytics
        and Visualization},
        author={Ryan A. Rossi and Nesreen K. Ahmed},
        booktitle={AAAI},
        url={https://networkrepository.com},
        year={2015}
    }

Data Generation:
Data is collected from the SuiteSparse Matrix Collection and standard benchmark graph
datasets, with sparse adjacency matrices converted into unweighted all-pairs shortest
path inputs.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. Generative AI might have been used to construct
tests. This statement was written by hand.
"""


def floyd_warshall(xp, edges_binsparse):
    """
    Returns the all pair shortest path i.e. A[i,j] is the shortest
    path from i to j
    """
    edges = xp.from_benchmark(edges_binsparse)
    n, m = edges.shape
    assert n == m
    G = edges
    for _ in range(n):
        G = xp.einsum("G[i,j] min= G[i,k] + G[k,j]", G=G)
    return xp.to_benchmark(G)


def generate_floyd_warshall_data(source, symmetrize=False):
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

    A = A.tocoo()
    n, m = A.shape
    if n != m:
        raise ValueError(f"Floyd-Warshall requires a square matrix, got {A.shape}")

    G = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(G, 0.0)
    if A.nnz > 0:
        G[A.row, A.col] = A.data.astype(np.float64)

    if symmetrize:
        G = np.minimum(G, G.T)

    G_bin = BinsparseFormat.from_numpy(G)
    return (G_bin,)


def dg_fw_sparse_1():
    return generate_floyd_warshall_data("bcspwr01", symmetrize=True)


def dg_fw_sparse_2():
    return generate_floyd_warshall_data("bcspwr02", symmetrize=True)


def dg_fw_sparse_3():
    return generate_floyd_warshall_data("bcspwr03", symmetrize=True)


def dg_fw_sparse_4():
    return generate_floyd_warshall_data("chesapeake", symmetrize=True)


def dg_fw_sparse_5():
    return generate_floyd_warshall_data("ash85")


def dg_fw_sparse_6():
    return generate_floyd_warshall_data("arc130")


def dg_fw_sparse_7():
    return generate_floyd_warshall_data("bcspwr04", symmetrize=True)


def dg_fw_sparse_8():
    return generate_floyd_warshall_data("ash292")
