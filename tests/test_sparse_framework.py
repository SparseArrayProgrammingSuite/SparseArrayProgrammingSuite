import pytest

import numpy as np

import array_api_compat.numpy as compat_np
import sparse as sp

from frameworks.saps_sparse import PyDataSparseFramework


@pytest.mark.parametrize(
    "sparse_inputs", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("axis", [0, 1])
def test_stack_dense_and_sparse_arrays(sparse_inputs, axis):
    dense = [np.array([1.0, 0.0, 2.0]), np.array([0.0, 3.0, 0.0])]
    arrays = [
        sp.asarray(array) if use_sparse else array
        for array, use_sparse in zip(dense, sparse_inputs, strict=True)
    ]
    actual = PyDataSparseFramework().stack(arrays, axis=axis)
    if any(sparse_inputs):
        assert isinstance(actual, sp.SparseArray)
        actual = actual.todense()
    else:
        assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, np.stack(dense, axis=axis))


@pytest.mark.parametrize("format", ["dense", "coo", "gcxs", "dok"])
@pytest.mark.parametrize(
    "options", [{"axis": 0, "stable": False}, {"descending": True, "stable": True}]
)
def test_argsort_returns_dense_indices(format, options):
    dense = np.array([[3.0, 0.0, 3.0], [0.0, -1.0, 0.0]])
    array = dense if format == "dense" else sp.asarray(dense).asformat(format)

    actual = PyDataSparseFramework().argsort(array, **options)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, compat_np.argsort(dense, **options))


def test_argsort_preserves_nonzero_fill_and_stable_ties():
    dense = np.array([[np.inf, 2.0, 0.0, np.inf], [1.0, np.inf, -1.0, 1.0]])
    array = sp.COO.from_numpy(dense, fill_value=np.inf)

    actual = PyDataSparseFramework().argsort(array)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, [[2, 1, 0, 3], [2, 0, 3, 1]])
    np.testing.assert_array_equal(array.todense(), dense)


def test_zeros_scatter_does_not_materialize_large_hash_axis():
    xp = PyDataSparseFramework()
    array = xp.zeros((100_000, 2**31), dtype=xp.bool)
    array[
        np.array([0, 99_999], dtype=np.uint64),
        np.array([2**31 - 1, 17], dtype=np.uint32),
    ] = True

    assert isinstance(array, sp.COO)
    assert array.nnz == 2
    assert array.coords.dtype == np.uint64
    np.testing.assert_array_equal(array.coords, [[0, 99_999], [2**31 - 1, 17]])
    np.testing.assert_array_equal(array.data, [1, 1])


def test_sparse_scatter_updates_and_matrix_product():
    xp = PyDataSparseFramework()
    array = xp.zeros((3, 4), dtype=xp.int64)
    rows = sp.asarray([0, 1, 2])
    columns = sp.asarray([1, 1, 3])
    array[rows, columns] = sp.asarray([1, 2, 3])
    array[1, 1] = 0
    array[0, 1] = 4

    expected = np.array([[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]])
    np.testing.assert_array_equal(array.todense(), expected)
    product = xp.matmul(array, xp.matrix_transpose(array))
    np.testing.assert_array_equal(product.todense(), expected @ expected.T)


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("sparse_indices", [False, True])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_take_along_axis_matches_numpy(sparse_input, sparse_indices, axis):
    dense = np.array([[3.0, np.inf, 1.0], [0.0, 4.0, np.inf]])
    indices = np.argsort(dense, axis=axis)
    indices = np.take(indices, [0], axis=axis)
    array = sp.COO.from_numpy(dense, fill_value=np.inf) if sparse_input else dense
    index_array = sp.asarray(indices) if sparse_indices else indices

    result = PyDataSparseFramework().take_along_axis(array, index_array, axis=axis)

    if sparse_input:
        assert isinstance(result, sp.SparseArray)
        result = result.todense()
    np.testing.assert_array_equal(result, np.take_along_axis(dense, indices, axis=axis))


def test_take_along_axis_broadcasts_and_handles_negative_indices():
    dense = np.arange(24).reshape(2, 4, 3)
    indices = np.array([[[0], [-1]]])

    result = PyDataSparseFramework().take_along_axis(sp.asarray(dense), indices, axis=1)

    np.testing.assert_array_equal(
        result.todense(), np.take_along_axis(dense, indices, axis=1)
    )


def test_take_along_axis_preserves_hypersparse_input():
    array = sp.COO(
        np.array([[0, 1], [7, 999_999_999]]),
        np.array([2.0, 3.0]),
        shape=(2, 1_000_000_000),
        fill_value=np.inf,
    )

    result = PyDataSparseFramework().take_along_axis(
        array, np.array([[7, 8], [999_999_999, 0]]), axis=1
    )

    assert isinstance(result, sp.SparseArray)
    np.testing.assert_array_equal(result.todense(), [[2.0, np.inf], [3.0, np.inf]])
