import numpy as np

import sparse as sp

from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.scipy_framework import SciPyFramework
from sparseappbench.frameworks.sparse_framework import (
    PyDataSparseFramework,
)


def test_numpy_framework():
    framework = NumpyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_binsparse(arr)
    assert BinsparseFormat.to_coo(bsf) == BinsparseFormat.to_coo(bsf_converted), (
        "Dense array to_binsparse failed"
    )

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = BinsparseFormat.to_coo(
        framework.to_binsparse(arr_sparse_converted)
    )
    assert bsf_sparse == bsf_sparse_converted, "Sparse array to_binsparse failed"


def test_pydata_sparse_framework():
    framework = PyDataSparseFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, sp.asnumpy(arr_converted)), (
        "Dense array conversion failed"
    )

    bsf_converted = framework.to_binsparse(arr_converted)
    assert BinsparseFormat.to_coo(bsf) == BinsparseFormat.to_coo(bsf_converted), (
        "Dense array to_binsparse failed"
    )

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, sp.asnumpy(arr_sparse_converted)), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = BinsparseFormat.to_coo(
        framework.to_binsparse(arr_sparse_converted)
    )
    assert bsf_sparse == bsf_sparse_converted, "Sparse array to_binsparse failed"


def test_scipy_framework():
    framework = SciPyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_binsparse(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_binsparse(arr_converted)
    assert BinsparseFormat.to_coo(bsf) == BinsparseFormat.to_coo(bsf_converted), (
        "Dense array to_binsparse failed"
    )

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_binsparse(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted.toarray()), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = BinsparseFormat.to_coo(
        framework.to_binsparse(arr_sparse_converted)
    )
    assert bsf_sparse == bsf_sparse_converted, "Sparse array to_binsparse failed"
