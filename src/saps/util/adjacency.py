"""Shared adjacency conversions and squaring limits for graph benchmarks."""

import numpy as np
import scipy.sparse as scipy_sparse
from scipy.sparse import coo_array, triu

import sparse
from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_scipy, to_numpy, to_scipy

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


def _row_major(row, col, n):
    """Row-major positions in an n-column matrix, without full-size temporaries."""
    linear = row.astype(np.int64)
    linear *= n
    linear += col
    return linear


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
    shape, n = edges.shape, edges.shape[0]
    # Rebuild the COO so tocsr sums repeated entries and sorts each row, since
    # binsparse marks its scipy matrices canonical even when they repeat. Large
    # GAP graphs only fit if no step makes more than a few copies of the edges.
    edges = coo_array((edges.data, (edges.row, edges.col)), shape=edges.shape)
    edges = edges.tocsr().tocoo()
    keep = (edges.data != 0) & (edges.row != edges.col)
    row, col = edges.row[keep], edges.col[keep]
    if keep_weights:
        length = edges.data[keep].astype(np.result_type(edges.data, 1.0), copy=False)
    else:
        length = np.ones(row.size)
    del edges, keep
    if symmetrize:
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        order = np.argsort(_row_major(row, col, n), kind="stable")
        row, col, length = row[order], col[order], np.concatenate([length, length])
        first = np.flatnonzero(np.diff(_row_major(row, col, n), prepend=-1))
        length = np.minimum.reduceat(length[order], first) if first.size else length
        row, col = row[first], col[first]
        del order, first
    diagonal = np.arange(n, dtype=row.dtype)
    at = np.searchsorted(_row_major(row, col, n), _row_major(diagonal, diagonal, n))
    return COORMatrix(
        shape,
        row.size + n,
        fill=True,
        fill_value=np.inf,
        indices_0=np.insert(row, at, diagonal),
        indices_1=np.insert(col, at, diagonal),
        values=np.insert(length, at, 0),
    )


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
