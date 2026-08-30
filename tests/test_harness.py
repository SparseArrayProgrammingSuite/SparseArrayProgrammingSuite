import numpy as np

import sparse as sp
from binsparse import COORMatrix
from binsparse.conversions import from_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import (
    PyDataSparseFramework,
)
from saps_framework.binsparse_utils import binsparse_equal


def test_numpy_framework():
    framework = NumpyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_binsparse(arr)
    assert binsparse_equal(bsf, bsf_converted), "Dense array to_binsparse failed"

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = COORMatrix(shape, len(data), indices_0=row, indices_1=col, values=data)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = framework.to_binsparse(arr_sparse_converted)
    assert binsparse_equal(bsf_sparse, bsf_sparse_converted), (
        "Sparse array to_binsparse failed"
    )


def test_pydata_sparse_framework():
    framework = PyDataSparseFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, sp.asnumpy(arr_converted)), (
        "Dense array conversion failed"
    )

    bsf_converted = framework.to_binsparse(arr_converted)
    assert binsparse_equal(bsf, bsf_converted), "Dense array to_binsparse failed"

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = COORMatrix(shape, len(data), indices_0=row, indices_1=col, values=data)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, sp.asnumpy(arr_sparse_converted)), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = framework.to_binsparse(arr_sparse_converted)
    assert binsparse_equal(bsf_sparse, bsf_sparse_converted), (
        "Sparse array to_binsparse failed"
    )


def test_scipy_framework():
    framework = SciPyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_binsparse(arr_converted)
    assert binsparse_equal(bsf, bsf_converted), "Dense array to_binsparse failed"

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = COORMatrix(shape, len(data), indices_0=row, indices_1=col, values=data)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted.toarray()), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = framework.to_binsparse(arr_sparse_converted)
    assert binsparse_equal(bsf_sparse, bsf_sparse_converted), (
        "Sparse array to_binsparse failed"
    )
