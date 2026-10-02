"""Shared adjacency conversions and squaring limits for graph benchmarks."""

import numpy as np
from scipy.sparse import coo_array

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_scipy, from_sparse, to_numpy, to_scipy

DEFAULT_MAX_DENSITY = 0.01


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
        A = sparse.COO.from_scipy_sparse(to_scipy(adjacency))
    except TypeError:
        A = sparse.COO.from_numpy(to_numpy(adjacency))
    G = sparse.where(A != 0, A if keep_weights else 1.0, np.inf)
    G = sparse.where(sparse.eye(A.shape[0], dtype=bool), 0.0, G)
    if symmetrize:
        G = np.minimum(G, G.T)
    return from_sparse(G)


def squaring_count(
    n: int, max_degree: int, max_density: float = DEFAULT_MAX_DENSITY
) -> int:
    """Square the row-support bound (degree + diagonal), stopping at the limit.

    Zero steps leaves the initialized graph, even if it already exceeds the
    density limit. Cap the budget at enough squarings to cover n - 1 hops.
    """
    row_bound, steps = min(n, max_degree + 1), 0
    while (
        steps < max(0, n - 2).bit_length() and min(n, row_bound**2) <= max_density * n
    ):
        row_bound, steps = min(n, row_bound**2), steps + 1
    return steps
