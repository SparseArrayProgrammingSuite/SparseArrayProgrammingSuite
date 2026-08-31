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
    if left.shape != right.shape:
        return False

    left_data, right_data = tensor_data(left), tensor_data(right)
    left_scipy = scipy_sparse.issparse(left_data)
    right_scipy = scipy_sparse.issparse(right_data)
    left_sparse = isinstance(left_data, sparse.SparseArray)
    right_sparse = isinstance(right_data, sparse.SparseArray)

    if left_scipy and right_scipy:
        return bool((left_data != right_data).nnz == 0)
    if left_sparse and right_sparse:
        return bool((left_data != right_data).nnz == 0)
    if left_scipy and right_sparse:
        left_data = sparse.COO.from_scipy_sparse(left_data)
        return bool((left_data != right_data).nnz == 0)
    if left_sparse and right_scipy:
        right_data = sparse.COO.from_scipy_sparse(right_data)
        return bool((left_data != right_data).nnz == 0)

    if left_scipy:
        left_data = left_data.toarray()
    elif left_sparse:
        left_data = left_data.todense()
    if right_scipy:
        right_data = right_data.toarray()
    elif right_sparse:
        right_data = right_data.todense()
    return bool(np.array_equal(left_data, right_data))
