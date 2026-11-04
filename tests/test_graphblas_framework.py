import math

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy

gb = pytest.importorskip("graphblas", reason="python-graphblas is not installed")

from frameworks.saps_graphblas import GraphBLASFramework, Tensor  # noqa: E402
from saps.benchmarks.bfs import BFSBenchmark  # noqa: E402
from saps.benchmarks.matrix_multiplication import (  # noqa: E402
    MatrixMultiplicationBenchmark,
)
from saps_framework.binsparse_utils import tensor_data  # noqa: E402


@pytest.fixture
def xp():
    return GraphBLASFramework()


def vector(indices, values, size, fill=0):
    return Tensor(gb.Vector.from_coo(indices, values, size=size), fill)


def dense(xp, tensor):
    data = tensor_data(xp.to_binsparse(tensor))
    if sps.issparse(data):
        return data.toarray()
    return data.todense() if hasattr(data, "todense") else np.asarray(data)


# --------------------------------------------------------- operator selection


def test_multiply_with_zero_fills_intersects(xp):
    a = vector([0, 1], [2.0, 3.0], 4)
    b = vector([1, 2], [4.0, 5.0], 4)
    result = xp.multiply(a, b)
    assert result.data.nvals == 1
    np.testing.assert_array_equal(dense(xp, result), [0, 12, 0, 0])


def test_add_with_zero_fills_unions(xp):
    a = vector([0, 1], [2.0, 3.0], 4)
    b = vector([1, 2], [4.0, 5.0], 4)
    result = xp.add(a, b)
    assert result.data.nvals == 3
    np.testing.assert_array_equal(dense(xp, result), [2, 7, 5, 0])


def test_multiply_with_unit_fills_uses_union_defaults(xp):
    a = xp.with_fill_value(vector([0, 1], [2.0, 3.0], 4), 1)
    b = xp.with_fill_value(vector([1, 2], [4.0, 5.0], 4), 1)
    result = xp.multiply(a, b)
    assert result.fill_value == 1
    np.testing.assert_array_equal(dense(xp, result), [2, 12, 5, 1])


def test_add_with_infinite_fills_intersects(xp):
    a = xp.with_fill_value(vector([0, 1], [2.0, 3.0], 4), math.inf)
    b = xp.with_fill_value(vector([1, 2], [4.0, 5.0], 4), math.inf)
    result = xp.add(a, b)
    assert result.data.nvals == 1
    np.testing.assert_array_equal(dense(xp, result), [np.inf, 7, np.inf, np.inf])


def test_one_annihilating_fill_masks_by_its_structure(xp):
    frontier = vector([0, 2], [True, True], 4, fill=False)
    not_visited = xp.logical_not(vector([2, 3], [True, True], 4, fill=False))
    result = xp.logical_and(frontier, not_visited)
    assert result.data.nvals <= frontier.data.nvals
    np.testing.assert_array_equal(dense(xp, result), [True, False, False, False])


def test_scalar_operand_binds_with_apply(xp):
    a = vector([0, 2], [1, 5], 4)
    result = xp.add(a, 3)
    assert result.fill_value == 3
    np.testing.assert_array_equal(dense(xp, result), [4, 3, 8, 3])


# ------------------------------------------------------------------ reductions


def test_sum_counts_booleans_as_integers(xp):
    result = xp.sum(vector([0, 1, 3], [True, True, True], 5, fill=False))
    assert result.ndim == 0
    assert result.dtype == np.int64
    assert int(result) == 3


def test_sum_includes_nonzero_fill(xp):
    a = xp.with_fill_value(vector([0], [5.0], 4), 2.0)
    assert float(xp.sum(a)) == 11.0
    m = xp.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(dense(xp, xp.sum(m, axis=0)), [4.0, 6.0])
    np.testing.assert_array_equal(dense(xp, xp.sum(m, axis=1)), [3.0, 7.0])


# ----------------------------------------------------------------- contraction


def test_matmul_matches_numpy(xp):
    left = sps.random(6, 5, density=0.4, random_state=0, format="csr")
    right = sps.random(5, 4, density=0.4, random_state=1, format="csr")
    result = xp.matmul(
        xp.from_binsparse(from_scipy(left)), xp.from_binsparse(from_scipy(right))
    )
    np.testing.assert_allclose(dense(xp, result), (left @ right).toarray())


def test_einsum_matvec_over_transposed_matrix(xp):
    matrix = np.array([[1.0, 0, 2], [0, 3, 0]])
    a = xp.from_binsparse(from_scipy(sps.csr_array(matrix)))
    v = xp.asarray(np.array([2.0, 3.0]))
    result = xp.einsum("y[j] += A[i,j] * v[i]", A=a, v=v)
    np.testing.assert_array_equal(dense(xp, result), [2, 9, 4])


def test_einsum_min_plus_uses_infinite_fill(xp):
    weights = np.array([[np.inf, 1.0, np.inf], [np.inf, np.inf, 2.0], [4, 1, np.inf]])
    stored = np.isfinite(weights)
    a = Tensor(
        gb.Matrix.from_coo(*np.nonzero(stored), weights[stored], nrows=3, ncols=3),
        math.inf,
    )
    result = xp.einsum("C[i,j] min= A[i,k] + A[k,j]", A=a)
    assert result.data.nvals < 9
    expected = np.min(weights[:, :, None] + weights[None, :, :], axis=1)
    np.testing.assert_array_equal(dense(xp, result), expected)


def test_einsum_scales_rows_by_a_vector(xp):
    m = np.array([[1.0, 2.0], [0.0, 3.0]])
    d = np.array([10.0, 100.0])
    result = xp.einsum("C[i,j] = A[i,j] - d[i]", A=xp.asarray(m), d=xp.asarray(d))
    np.testing.assert_array_equal(dense(xp, result), m - d[:, None])


# ----------------------------------------------------------- where / indexing


def test_where_keeps_both_fills(xp):
    cond = vector([1, 2], [True, False], 4, fill=False)
    x1 = xp.with_fill_value(vector([1], [7], 4), 5)
    x2 = vector([0], [9], 4)
    result = xp.where(cond, x1, x2)
    np.testing.assert_array_equal(dense(xp, result), [9, 7, 0, 0])


def test_setitem_and_getitem(xp):
    v = xp.zeros((3,), dtype=bool)
    v[1] = True
    assert bool(v[1]) and not bool(v[0])
    assert v.data.nvals == 1


def test_dense_roundtrip(xp):
    array = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(
        dense(xp, xp.from_binsparse(from_numpy(array))), array
    )


# ------------------------------------------------- end-to-end benchmark checks


def _test_suite_params(benchmark):
    return [
        pytest.param(param, id=f"{param.generator.name}.{param.dataset.name}")
        for param in benchmark.params
        if "test" in param.dataset.suites
        and param.generator.name != "matrix_multiplication_suitesparse"
    ]


def _run_and_check(benchmark, param, xp):
    # teardown runs the benchmark's own correctness check against the reference.
    benchmark.setup(param, use_cache=False, xp=xp)
    benchmark.run(param)
    benchmark.teardown(param)


@pytest.mark.parametrize("param", _test_suite_params(MatrixMultiplicationBenchmark()))
def test_matrix_multiplication_benchmark(xp, param):
    _run_and_check(MatrixMultiplicationBenchmark(), param, xp)


@pytest.mark.parametrize("param", _test_suite_params(BFSBenchmark()))
def test_bfs_benchmark(xp, param):
    _run_and_check(BFSBenchmark(), param, xp)
