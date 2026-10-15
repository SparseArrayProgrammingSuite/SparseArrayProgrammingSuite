"""Shared adjacency conversions and squaring limits for graph benchmarks."""

import numpy as np
import scipy.sparse as scipy_sparse
from scipy.sparse import coo_array, triu

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_scipy, from_sparse, to_numpy, to_scipy

from saps_framework.binsparse_utils import tensor_data

DEFAULT_MAX_DENSITY = 0.01


def zero_one_adjacency(
    adjacency: BinsparseTensor,
    dtype=bool,
    *,
    simple_undirected: bool = False,
    upper_triangle: bool = False,
) -> BinsparseTensor:
    """0-1 adjacency with a one wherever ``adjacency`` stores a nonzero.

    Connectivity kernels should not see SNAP signs, timestamps, or interaction
    counts, or GAP edge weights. Explicit zeros are dropped and repeated
    coordinates are merged with logical or, so opposite-signed entries cannot
    cancel. Self-loops are kept unless ``simple_undirected`` is requested, which
    also adds the reverse of every edge for published undirected clique counts.
    ``upper_triangle`` then keeps only entries with row < column, so clique
    counters visit each set of vertices once.
    """
    edges = to_scipy(adjacency).tocoo()
    pattern = coo_array(
        (edges.data != 0, (edges.row, edges.col)), shape=edges.shape
    ).tocsr()
    if simple_undirected:
        pattern = pattern.maximum(pattern.T)
        pattern.setdiag(False)
    pattern.eliminate_zeros()
    if upper_triangle:
        pattern = triu(pattern, k=1)
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


def graph_csr(
    tensor: BinsparseTensor, *, distances: bool = False, boolean: bool = False
):
    """Keep absent edges sparse, including infinity-filled distance inputs.

    Explicit zero-length edges in distance matrices must survive the conversion.
    """
    data = tensor_data(tensor)
    if scipy_sparse.issparse(data):
        result = scipy_sparse.csr_array(data, copy=True)
    elif isinstance(data, sparse.SparseArray):
        coo = data.asformat("coo")
        keep = np.isfinite(coo.data) if distances else coo.data != 0
        result = scipy_sparse.coo_array(
            (coo.data[keep], tuple(coo.coords[:, keep])), shape=coo.shape
        ).tocsr()
    else:
        keep = np.isfinite(data) if distances else data != 0
        row, col = np.nonzero(keep)
        result = scipy_sparse.coo_array(
            (data[row, col], (row, col)), shape=data.shape
        ).tocsr()
    if not distances:
        result.eliminate_zeros()
    # csgraph requires C-int indices on several supported SciPy versions.
    if max(*result.shape, result.nnz) <= np.iinfo(np.int32).max:
        result.indices = result.indices.astype(np.int32, copy=False)
        result.indptr = result.indptr.astype(np.int32, copy=False)
    return result.astype(bool, copy=False) if boolean else result
