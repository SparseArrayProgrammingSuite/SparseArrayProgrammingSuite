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
