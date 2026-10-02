"""Shared adjacency conversions for graph benchmarks."""

import numpy as np
from scipy.sparse import coo_array

from binsparse import BinsparseTensor
from binsparse.conversions import from_scipy, to_scipy


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
