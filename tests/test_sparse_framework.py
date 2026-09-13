import pytest

import numpy as np

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
