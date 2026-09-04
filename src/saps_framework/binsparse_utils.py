from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

import sparse
from binsparse import BinsparseTensor
from binsparse.conversions import to_numpy, to_scipy, to_sparse


def to_canonical_coo(tensor: BinsparseTensor) -> Any:
    """Return `tensor` as a SciPy COO array that is genuinely canonical:
    lexicographically sorted, duplicate-free, and without explicitly stored zeros.

    `to_scipy` cannot represent dense or vector tensors, so those are routed
    through NumPy instead.
    """
    try:
        coo = to_scipy(tensor).tocoo()
    except TypeError:
        return scipy_sparse.coo_array(to_numpy(tensor))
    # Rebuild rather than trusting `has_canonical_format`: `to_scipy` force-sets
    # that flag without verifying it, so a tensor built directly from a
    # COORMatrix can arrive unsorted or carrying duplicate coordinates and any
    # later `sum_duplicates()` would silently no-op.
    coo = scipy_sparse.coo_array((coo.data, coo.coords), shape=coo.shape)
    coo.sum_duplicates()
    # Frameworks disagree on whether an operation retains explicit zeros, so
    # drop them to keep sparsity patterns comparable across implementations.
    coo.eliminate_zeros()
    return coo


def assert_coo_allclose(
    expected: BinsparseTensor,
    actual: BinsparseTensor,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> None:
    """Assert two binsparse tensors hold the same shape, sparsity pattern, and
    (within tolerance) the same values.
    """
    expected_coo = to_canonical_coo(expected)
    actual_coo = to_canonical_coo(actual)
    assert expected_coo.shape == actual_coo.shape, (
        f"Shape mismatch: expected {expected_coo.shape}, got {actual_coo.shape}"
    )
    assert expected_coo.nnz == actual_coo.nnz, (
        f"Stored-value count mismatch: expected {expected_coo.nnz}, "
        f"got {actual_coo.nnz}"
    )
    for axis, (expected_idx, actual_idx) in enumerate(
        zip(expected_coo.coords, actual_coo.coords, strict=True)
    ):
        assert np.array_equal(expected_idx, actual_idx), (
            f"Sparsity pattern mismatch along axis {axis}"
        )
    assert np.allclose(expected_coo.data, actual_coo.data, rtol=rtol, atol=atol), (
        "Values differ beyond tolerance"
    )


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
