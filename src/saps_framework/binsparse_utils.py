"""Small SAPS integration helpers for the reference ``binsparse`` package."""

import numpy as np
import scipy.sparse as scipy_sparse

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import to_numpy, to_scipy, to_sparse


def tensor_data(tensor: BinsparseTensor):
    for convert in (to_scipy, to_sparse, to_numpy):
        try:
            return convert(tensor)
        except TypeError:
            pass
    raise TypeError(f"Cannot convert {type(tensor).__name__} to a supported array")


def binsparse_equal(left: BinsparseTensor, right: BinsparseTensor) -> bool:
    left_data, right_data = tensor_data(left), tensor_data(right)
    if left_data.shape != right_data.shape:
        return False

    if scipy_sparse.issparse(left_data) and scipy_sparse.issparse(right_data):
        return bool((left_data != right_data).nnz == 0)

    if isinstance(left_data, sparse.SparseArray) and isinstance(
        right_data, sparse.SparseArray
    ):
        return bool((left_data != right_data).nnz == 0)

    return bool(np.array_equal(left_data, right_data))
