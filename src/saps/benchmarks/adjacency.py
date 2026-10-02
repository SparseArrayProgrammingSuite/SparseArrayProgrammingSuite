"""Shared adjacency conversions for graph benchmarks."""

import numpy as np
from scipy.sparse import coo_array

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_scipy, from_sparse, to_numpy, to_scipy


def zero_one_adjacency(adjacency: BinsparseTensor, dtype=bool) -> BinsparseTensor:
    """0-1 adjacency with a one wherever ``adjacency`` stores a nonzero.

    Connectivity kernels should not see SNAP signs, timestamps, or interaction
    counts, or GAP edge weights. Explicit zeros are dropped and repeated
    coordinates are merged with logical or, so opposite-signed entries cannot
    cancel. Self-loops are kept.
    """
    edges = to_scipy(adjacency).tocoo()
    pattern = coo_array(
        (edges.data != 0, (edges.row, edges.col)), shape=edges.shape
    ).tocsr()
    pattern.eliminate_zeros()
    return from_scipy(pattern.tocoo().astype(np.dtype(dtype)))


def distance_matrix(
    adjacency: BinsparseTensor, *, keep_weights: bool = False, symmetrize: bool = False
) -> BinsparseTensor:
    """Shortest-path input: edge lengths, a zero diagonal, and infinite fill.

    Explicit zeros and self-loops are not edges, and repeated entries are summed
    as when reading any sparse matrix. Without ``keep_weights`` every edge has
    length 1. With ``symmetrize`` each edge also runs backwards, keeping the
    shorter direction.
    """
    try:
        edges = to_scipy(adjacency).tocoo()
    except TypeError:
        edges = coo_array(to_numpy(adjacency))
    # The constructor sums repeated entries. COO.from_scipy_sparse would not,
    # since binsparse marks its scipy matrices canonical even when they repeat.
    A = sparse.COO(np.stack([edges.row, edges.col]), edges.data, shape=edges.shape)
    G = sparse.where(A != 0, A if keep_weights else 1.0, np.inf)
    G = sparse.where(sparse.eye(A.shape[0], dtype=bool), 0.0, G)
    if symmetrize:
        G = np.minimum(G, G.T)
    return from_sparse(G)
