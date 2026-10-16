from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

cupy = pytest.importorskip("cupy", reason="CuPy is not installed")
cusp = pytest.importorskip(
    "cupyx.scipy.sparse", reason="cupyx.scipy.sparse is not available"
)

if not cupy.is_available():  # pragma: no cover - depends on the host
    pytest.skip("no CUDA device available", allow_module_level=True)

if not hasattr(cusp, "csr_array"):  # pragma: no cover - depends on the host
    pytest.skip(
        "cupyx.scipy.sparse.csr_array requires CuPy >= 14.2", allow_module_level=True
    )

from frameworks.saps_cupy import CuPyFramework  # noqa: E402
from saps.benchmarks.elementwise_multiplication import (  # noqa: E402
    ElementwiseMultiplicationBenchmark,
)


@pytest.fixture
def xp():
    return CuPyFramework()


def host(array):
    """Bring a framework result back to NumPy for comparison."""
    if cusp.issparse(array):
        return array.get().toarray()
    if isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return np.asarray(array)


def dense_binsparse(tensor):
    """Materialize a binsparse tensor, sparse or dense, as a NumPy array."""
    try:
        return to_numpy(tensor)
    except TypeError:
        return to_scipy(tensor).toarray()


@pytest.fixture
def matrix():
    return sps.csr_array(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))


# --------------------------------------------------------------------- roundtrip


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_sparse_roundtrip_preserves_values(xp, format):
    host_matrix = getattr(sps, f"{format}_array")(
        np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    )
    device = xp.from_binsparse(from_scipy(host_matrix))

    assert cusp.issparse(device)
    np.testing.assert_array_equal(
        dense_binsparse(xp.to_binsparse(device)), host_matrix.toarray()
    )


def test_dense_roundtrip_lands_on_device(xp):
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    device = xp.from_binsparse(from_numpy(array))

    assert isinstance(device, cupy.ndarray)
    np.testing.assert_array_equal(to_numpy(xp.to_binsparse(device)), array)


@pytest.mark.parametrize(
    "value", [np.int64(7), np.float64(2.5), np.complex128(1j), np.bool_(True), 7, 2.5]
)
def test_scalar_output_roundtrip(xp, value):
    result = to_numpy(xp.to_binsparse(value))

    assert result.shape == ()
    assert result == value


def test_integer_sparse_is_widened_not_rejected(xp):
    # cupyx sparse accepts only bool/float/complex, while scipy also takes ints.
    host_matrix = sps.csr_array(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64))
    device = xp.from_binsparse(from_scipy(host_matrix))

    assert device.dtype == np.float64
    np.testing.assert_array_equal(host(device), host_matrix.toarray())


def test_integer_sparse_rejects_lossy_float64_conversion(xp):
    value = 2**53 + 1
    host_matrix = sps.csr_array(np.array([[value]], dtype=np.int64))

    with pytest.raises(ValueError, match="without loss"):
        xp.from_binsparse(from_scipy(host_matrix))


def test_dia_output_converts_back_to_binsparse(xp):
    # cupyx builders such as eye() return DIA, which binsparse cannot read.
    diagonal = cusp.eye(3, format="dia")

    np.testing.assert_array_equal(dense_binsparse(xp.to_binsparse(diagonal)), np.eye(3))


# ------------------------------------------------ semantics cupyx does not share


def test_power_on_sparse_is_elementwise(xp, matrix):
    # `**` is matrix power on an spmatrix; the wrapper must stay on sparse arrays.
    result = xp.power(xp.from_binsparse(from_scipy(matrix)), 2)

    np.testing.assert_array_equal(host(result), matrix.toarray() ** 2)


def test_power_on_sparse_accepts_zero_dimensional_cupy_exponent(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.power(device, cupy.asarray(2))

    np.testing.assert_array_equal(host(result), matrix.toarray() ** 2)


def test_power_on_spmatrix_is_elementwise(xp):
    # xp.tril resolves to cupyx and returns a coo_matrix, where ** is matrix power.
    lower = xp.tril(cupy.asarray([[1.0, 0.0], [2.0, 3.0]]))

    assert isinstance(lower, cusp.spmatrix)
    np.testing.assert_array_equal(host(xp.power(lower, 2)), [[1, 0], [4, 9]])


def test_permute_dims_without_axes_argument(xp, matrix):
    # cupyx rejects transpose(axes=...), which the SciPy wrapper passes through.
    device = xp.from_binsparse(from_scipy(matrix))

    np.testing.assert_array_equal(
        host(xp.permute_dims(device, (1, 0))), matrix.toarray().T
    )
    np.testing.assert_array_equal(
        host(xp.permute_dims(device, (0, 1))), matrix.toarray()
    )


def test_reshape_accepts_array_api_copy_keyword(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.reshape(device, (3, 2), copy=None)

    np.testing.assert_array_equal(host(result), matrix.toarray().reshape(3, 2))


def test_reshape_above_rank_two_densifies(xp, matrix):
    # cupyx sparse is rank <= 2, so a higher-rank result has to go dense.
    result = xp.reshape(xp.from_binsparse(from_scipy(matrix)), (1, 2, 3))

    assert isinstance(result, cupy.ndarray)
    np.testing.assert_array_equal(host(result), matrix.toarray().reshape(1, 2, 3))


def test_expand_dims_above_sparse_rank_densifies(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.expand_dims(device, 0)

    assert isinstance(result, cupy.ndarray)
    np.testing.assert_array_equal(host(result), matrix.toarray()[None, :, :])


def test_eye_returns_a_sparse_array_not_a_matrix(xp):
    result = xp.eye(3)

    assert cusp.issparse(result)
    assert not isinstance(result, cusp.spmatrix)
    np.testing.assert_array_equal(host(result), np.eye(3))


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_eye_with_integer_dtype_is_dense(xp, dtype):
    # cupyx sparse has no integer dtypes.
    result = xp.eye(3, k=1, dtype=dtype)

    assert isinstance(result, cupy.ndarray)
    assert result.dtype == dtype
    np.testing.assert_array_equal(host(result), np.eye(3, k=1, dtype=dtype))


def test_sparse_matmul_applies_dtype(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.matmul(device, xp.permute_dims(device, (1, 0)), dtype=cupy.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(host(result), matrix.toarray() @ matrix.toarray().T)


def test_sparse_matmul_rejects_unsupported_kwargs(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    with pytest.raises(TypeError, match="doesn't support"):
        xp.matmul(device, xp.permute_dims(device, (1, 0)), out=None)


def test_divide_with_one_sparse_operand_matches_dense(xp, matrix):
    # cupyx turns 0 / 0 into 0 for sparse / dense, and rejects dense / sparse.
    device = xp.from_binsparse(from_scipy(matrix))
    other = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 4.0]])

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_left = matrix.toarray() / other
        expected_right = other / matrix.toarray()
    np.testing.assert_array_equal(
        host(xp.divide(device, cupy.asarray(other))), expected_left
    )
    np.testing.assert_array_equal(
        host(xp.divide(cupy.asarray(other), device)), expected_right
    )


# ----------------------------------------- operations CuPy ufuncs reject outright


def test_sum_over_sparse_axis(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    np.testing.assert_array_equal(
        host(xp.sum(device, axis=0)), matrix.toarray().sum(axis=0)
    )
    np.testing.assert_array_equal(host(xp.sum(device)), matrix.toarray().sum())


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_sparse_reductions_return_dense(xp, matrix, reduction):
    # A sparse reduction result would not broadcast against a sparse operand.
    device = xp.from_binsparse(from_scipy(matrix))

    result = getattr(xp, reduction)(device, axis=0)

    assert isinstance(result, cupy.ndarray)
    np.testing.assert_array_equal(
        host(result), getattr(matrix.toarray(), reduction)(axis=0)
    )


@pytest.mark.parametrize("reduction", ["sum", "max", "min", "any", "all"])
def test_sparse_reductions_keepdims_with_no_axis(xp, matrix, reduction):
    device = xp.from_binsparse(from_scipy(matrix))

    result = getattr(xp, reduction)(device, keepdims=True)
    expected = getattr(np, reduction)(matrix.toarray(), keepdims=True)

    assert result.shape == (1, 1)
    np.testing.assert_array_equal(host(result), expected)


@pytest.mark.parametrize("reduction", ["max", "min"])
def test_sparse_extrema_support_tuple_axes(xp, matrix, reduction):
    device = xp.from_binsparse(from_scipy(matrix))

    result = getattr(xp, reduction)(device, axis=(0, 1))

    np.testing.assert_array_equal(
        host(result), getattr(np, reduction)(matrix.toarray(), axis=(0, 1))
    )


@pytest.mark.parametrize("op", ["logical_or", "logical_and"])
def test_logical_ops_accept_sparse(xp, matrix, op):
    other = sps.csr_array(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 5.0]]))
    left = xp.from_binsparse(from_scipy(matrix))
    right = xp.from_binsparse(from_scipy(other))

    result = getattr(xp, op)(left, right)
    expected = getattr(np, op)(matrix.toarray(), other.toarray())

    np.testing.assert_array_equal(host(result).astype(bool), expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_concat_of_sparse_blocks(xp, matrix, axis):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.concat([device, device], axis=axis)

    assert cusp.issparse(result)
    np.testing.assert_array_equal(
        host(result), np.concatenate([matrix.toarray()] * 2, axis=axis)
    )


def test_concat_accepts_negative_axis(xp, matrix):
    device = xp.from_binsparse(from_scipy(matrix))

    result = xp.concat([device, device], axis=-1)

    np.testing.assert_array_equal(
        host(result), np.concatenate([matrix.toarray()] * 2, axis=-1)
    )


def test_any_and_all_over_sparse(xp):
    zeros = xp.from_binsparse(from_scipy(sps.csr_array(np.zeros((2, 2)))))
    full = xp.from_binsparse(from_scipy(sps.csr_array(np.ones((2, 2)))))

    assert not bool(xp.any(zeros))
    assert bool(xp.any(full))
    assert not bool(xp.all(zeros))
    assert bool(xp.all(full))


# ----------------------------------------------------------------------- replace


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (0, 7, [7, 2, np.nan, 2]),
        (2, 0, [0, 0, np.nan, 0]),
        (np.nan, 0, [0, 2, 0, 2]),
        (9, 1, [0, 2, np.nan, 2]),
    ],
)
def test_replace_dense_without_mutating_input(xp, old, new, expected):
    values = np.array([0.0, 2.0, np.nan, 2.0])
    array = xp.from_binsparse(from_numpy(values.copy()))

    result = xp.replace(array, old, new)

    np.testing.assert_array_equal(host(result), expected)
    np.testing.assert_array_equal(host(array), values)


def test_replace_promotes_integer_values(xp):
    array = xp.from_binsparse(from_numpy(np.array([0, 2], dtype=np.int64)))

    np.testing.assert_array_equal(host(xp.replace(array, 2, 1.5)), [0, 1.5])


@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
def test_replace_sparse_formats(xp, format):
    values = np.array([[0.0, 2.0, np.nan], [2.0, 0.0, 4.0]])
    array = getattr(cusp, f"{format}_array")(getattr(sps, f"{format}_array")(values))

    changed = xp.replace(array, 2, 0)

    assert cusp.issparse(changed)
    assert changed.format == format
    np.testing.assert_array_equal(host(changed), [[0, 0, np.nan], [0, 0, 4]])
    np.testing.assert_array_equal(
        host(xp.replace(array, 0, 3)), [[3, 2, np.nan], [2, 3, 4]]
    )
    np.testing.assert_array_equal(
        host(xp.replace(array, np.nan, 0)), [[0, 2, 0], [2, 0, 4]]
    )
    np.testing.assert_array_equal(host(array), values)


def test_replace_combines_duplicate_sparse_coordinates_first(xp):
    array = cusp.coo_array(sps.coo_array(([2.0, 3.0], ([0, 0], [1, 1])), shape=(2, 3)))

    np.testing.assert_array_equal(host(xp.replace(array, 5, 0)), np.zeros((2, 3)))
    assert host(array)[0, 1] == 5


# ------------------------------------------------------------------ integration


def test_linalg_falls_back_to_cupy_namespace(xp):
    array = cupy.asarray(np.array([[3.0, 0.0], [0.0, 2.0]]))

    singular_values = xp.linalg.svd(array, compute_uv=False)

    np.testing.assert_allclose(sorted(host(singular_values)), [2.0, 3.0])


def test_sparse_einsum_matvec(xp):
    matrix = cusp.csr_array(sps.csr_array(np.array([[1.0, 0, 2], [0, 3, 0]])))
    vector = cupy.asarray(np.array([2.0, 3.0]))

    result = xp.einsum("y[j] += A[i,j] * v[i]", A=matrix, v=vector)

    np.testing.assert_array_equal(host(result), [2, 9, 4])


def test_elementwise_benchmark_matches_reference(xp, matrix):
    other = sps.csr_array(np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 5.0]]))
    left_device = xp.from_binsparse(from_scipy(matrix))
    right_device = xp.from_binsparse(from_scipy(other))

    result = ElementwiseMultiplicationBenchmark().benchmark(
        xp, {}, left_device, right_device
    )

    expected = matrix.multiply(other).toarray()
    np.testing.assert_array_equal(dense_binsparse(xp.to_binsparse(result)), expected)


def test_multiply_matches_scipy_on_sparse(xp):
    left = sps.csr_array(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    right = sps.csr_array(np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 5.0]]))

    result = xp.multiply(
        xp.from_binsparse(from_scipy(left)), xp.from_binsparse(from_scipy(right))
    )

    np.testing.assert_array_equal(
        host(result), to_scipy(from_scipy(left.multiply(right))).toarray()
    )
