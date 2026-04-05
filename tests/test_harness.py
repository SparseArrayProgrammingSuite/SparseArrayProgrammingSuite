import numpy as np

import sparse as sp

from benchmarks.matmul import (
    benchmark_matmul,
    dg_matmul_dense_small,
)
from saps.binsparse_format import BinsparseFormat
from saps.frameworks.checker_framework import CheckerFramework
from saps.frameworks.numpy_framework import NumpyFramework
from saps.frameworks.sparse_framework import (
    PyDataSparseFramework,
)


def test_numpy_framework():
    framework = NumpyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_benchmark(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_benchmark(arr)
    assert BinsparseFormat.to_coo(bsf) == BinsparseFormat.to_coo(bsf_converted), (
        "Dense array to_benchmark failed"
    )

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_benchmark(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = BinsparseFormat.to_coo(
        framework.to_benchmark(arr_sparse_converted)
    )
    assert bsf_sparse == bsf_sparse_converted, "Sparse array to_benchmark failed"


def test_pydata_sparse_framework():
    framework = PyDataSparseFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_benchmark(bsf)
    assert np.array_equal(arr, sp.asnumpy(arr_converted)), (
        "Dense array conversion failed"
    )

    bsf_converted = framework.to_benchmark(arr_converted)
    assert BinsparseFormat.to_coo(bsf) == BinsparseFormat.to_coo(bsf_converted), (
        "Dense array to_benchmark failed"
    )

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_benchmark(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, sp.asnumpy(arr_sparse_converted)), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = BinsparseFormat.to_coo(
        framework.to_benchmark(arr_sparse_converted)
    )
    assert bsf_sparse == bsf_sparse_converted, "Sparse array to_benchmark failed"


def test_checker_framework():
    framework = CheckerFramework()
    # Test lazy and compute
    assert isinstance(
        benchmark_matmul(framework, *dg_matmul_dense_small()), BinsparseFormat
    ), "MatMul benchmark failed with CheckerFramework"

    def bad_benchmark_no_compute(framework, A_bench, B_bench):
        A_lazy = framework.lazy(framework.from_benchmark(A_bench))
        B_lazy = framework.lazy(framework.from_benchmark(B_bench))
        C_lazy = framework.matmul(A_lazy, B_lazy)
        # Intentionally not calling compute to test error handling
        return framework.to_benchmark(C_lazy)

    try:
        bad_benchmark_no_compute(framework, *dg_matmul_dense_small())
        raise ValueError(
            "Expected error for converting lazy tensor to benchmark format"
        )
    except AssertionError:
        pass  # Expected
