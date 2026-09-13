import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_scipy, to_numpy

from frameworks.saps_scipy import SciPyFramework
from saps.benchmarks.BFS import BreadthFirstSearchBenchmark


@pytest.mark.parametrize(
    "value", [np.int64(7), np.float64(2.5), np.complex128(1j), np.bool_(True), 7, 2.5]
)
def test_scipy_scalar_output_roundtrip(value):
    xp = SciPyFramework()
    result = to_numpy(xp.to_binsparse(value))

    assert result.shape == ()
    assert result.dtype == np.asarray(value).dtype
    assert result == value


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scipy_sparse_einsum_matvec(format, reverse, monkeypatch):
    xp = SciPyFramework()
    matrix = getattr(sps, f"{format}_array")([[1.0, 0, 2], [0, 3, 0]])
    vector = np.array([2.0, 3.0])

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Sparse inputs must not be densified")

    for sparse_type in (sps.csr_array, sps.csc_array, sps.coo_array):
        monkeypatch.setattr(sparse_type, "toarray", forbid_dense)

    product = "v[i] * A[i,j]" if reverse else "A[i,j] * v[i]"
    result = xp.einsum(f"y[j] += {product}", A=matrix, v=vector)

    np.testing.assert_array_equal(result, [2, 9, 4])


def test_scipy_sparse_einsum_transpose_and_scalar_reduction():
    xp = SciPyFramework()
    matrix = sps.csr_array([[1, 0, 2], [0, 3, 0]])

    transpose = xp.einsum("B[j,i] = A[i,j]", A=matrix)
    assert sps.issparse(transpose)
    np.testing.assert_array_equal(transpose.toarray(), matrix.toarray().T)
    total = xp.einsum("s[] += A[i,j]", A=matrix)
    assert to_numpy(xp.to_binsparse(total)) == 6


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_scipy_expand_sparse_vector(axis):
    xp = SciPyFramework()
    vector = sps.coo_array([1, 0, 3])
    result = xp.expand_dims(vector, axis)

    assert sps.issparse(result)
    np.testing.assert_array_equal(result.toarray(), np.expand_dims([1, 0, 3], axis))


def test_scipy_sparse_bfs_checks_all_test_datasets():
    xp = SciPyFramework()
    benchmark = BreadthFirstSearchBenchmark()
    params = [param for param in benchmark.params if "test" in param.dataset.suites]
    assert params
    for param in params:
        benchmark.setup(param, use_cache=False, xp=xp)
        benchmark._input = [
            from_scipy(sps.csr_array(to_numpy(item))) for item in benchmark._input
        ]
        benchmark.run(param)
        benchmark.teardown(param)
