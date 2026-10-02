import pytest

import numpy as np
import scipy.sparse as sps

import sparse
from binsparse.conversions import from_numpy, from_scipy, from_sparse

from frameworks.saps_numpy import NumpyFramework

gl = pytest.importorskip("galley_jl_python")
from frameworks.saps_galley import GalleyFramework  # noqa: E402


@pytest.fixture(scope="module")
def xp():
    return GalleyFramework()


def _dense(binsparse_array):
    return np.asarray(NumpyFramework().from_binsparse(binsparse_array))


MATRIX = sps.random(6, 4, density=0.4, format="csr", random_state=1)
TENSOR = sparse.random((3, 4, 5), density=0.3, random_state=0)


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (from_numpy(np.arange(6.0).reshape(2, 3)), np.arange(6.0).reshape(2, 3)),
        (from_numpy(np.arange(4.0)), np.arange(4.0)),
        (from_numpy(np.asarray(3.0)), np.asarray(3.0)),
        (from_scipy(MATRIX), MATRIX.toarray()),
        (from_scipy(MATRIX.tocoo()), MATRIX.toarray()),
        (from_sparse(TENSOR), TENSOR.todense()),
    ],
)
def test_binsparse_round_trip(xp, array, expected):
    np.testing.assert_allclose(
        _dense(xp.to_binsparse(xp.from_binsparse(array))), expected
    )


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (lambda xp, A: 1.0 - A, lambda A: 1.0 - A),
        (lambda xp, A: A * np.ones((6, 4)), lambda A: A),
        (lambda xp, A: A.T, lambda A: A.T),
        (lambda xp, A: xp.maximum(A, 0.5), lambda A: np.maximum(A, 0.5)),
        (lambda xp, A: xp.minimum(0.5, A), lambda A: np.minimum(A, 0.5)),
        (lambda xp, A: xp.where(A > 0, A, 2.0), lambda A: np.where(A > 0, A, 2.0)),
        (lambda xp, A: xp.vecdot(A, A, axis=0), lambda A: (A * A).sum(axis=0)),
        (
            lambda xp, A: xp.einsum("y[i] += A[i,j] * x[j]", A=A, x=np.ones(4)),
            lambda A: A.sum(axis=1),
        ),
    ],
)
def test_operations(xp, func, expected):
    A = xp.from_binsparse(from_scipy(MATRIX))
    np.testing.assert_allclose(
        _dense(xp.to_binsparse(func(xp, A))), expected(MATRIX.toarray())
    )


def test_with_fill_value(xp):
    A = xp.with_fill_value(xp.from_binsparse(from_scipy(MATRIX)), np.inf)
    assert A.fill_value == np.inf
    expected = np.where(MATRIX.toarray() == 0, np.inf, MATRIX.toarray())
    np.testing.assert_array_equal(_dense(xp.to_binsparse(A)), expected)


def test_numpy_dtypes_are_accepted(xp):
    zeros = xp.zeros((3,), dtype=np.int64)
    assert zeros.dtype == xp.int64
    assert xp.isdtype(xp.int64, "integral")
    assert xp.iinfo(xp.int64).max == np.iinfo(np.int64).max


def test_unsupported_operations_raise(xp):
    A = xp.from_binsparse(from_scipy(MATRIX))
    with pytest.raises(TypeError):
        A[0, 0] = 1.0
    with pytest.raises(AttributeError):
        _ = xp.concat


def _harness_closure(xp, function):
    # Mirrors how saps.benchmark wraps a benchmark function before `xp.compile`.
    def benchmark(meta, *data_args):
        return function(xp, meta, *data_args)

    return benchmark


def _scaled_sum(xp, meta, A):
    B = A * meta["scale"]
    return B + A


def test_compile_fuses_straight_line_benchmark(xp, monkeypatch):
    from galley_jl_python.fused import dataflow

    computed = []
    compute_tensor = dataflow.compute_tensor

    def counting_compute(tensor):
        if not tensor.is_computed():
            computed.append(tensor)
        return compute_tensor(tensor)

    monkeypatch.setattr(dataflow, "compute_tensor", counting_compute)

    def benchmark(xp, meta, A):
        B = A * 2.0
        C = B + A
        return C  # noqa: RET504

    compiled = xp.compile(_harness_closure(xp, benchmark))
    A = xp.from_binsparse(from_scipy(MATRIX))
    result = compiled({}, A)

    np.testing.assert_allclose(_dense(xp.to_binsparse(result)), 3 * MATRIX.toarray())
    assert len(computed) == 1


class _Benchmark:
    def benchmark(self, xp, meta, A):
        total = xp.sum(A)
        if float(total) > 0:
            A = A + 1.0
        return A


def test_compile_bound_benchmark_with_branch_on_a_lazy_value(xp):
    compiled = xp.compile(_harness_closure(xp, _Benchmark().benchmark))
    A = xp.from_binsparse(from_scipy(MATRIX))

    np.testing.assert_allclose(
        _dense(xp.to_binsparse(compiled({}, A))), MATRIX.toarray() + 1.0
    )


def test_compile_falls_back_when_jit_cannot_parse(xp):
    def benchmark(xp, meta, A):
        return [A * 2.0 for _ in range(1)][0]

    closure = _harness_closure(xp, benchmark)
    with pytest.warns(UserWarning, match="runs eagerly"):
        compiled = xp.compile(closure)
    assert compiled is closure
