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
from saps.benchmarks.elementwise import ElementwiseBenchmark  # noqa: E402


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

    result = ElementwiseBenchmark().benchmark(xp, [left_device, right_device], {})[0]

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
