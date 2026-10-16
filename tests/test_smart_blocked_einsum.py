import pytest

import numpy as np

import sparse as sp

from frameworks import saps_smart
from frameworks.saps_smart import SmartSparseFramework


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("complex_values", [False, True])
def test_blocked_all_modes_match_dense(n, complex_values, monkeypatch):
    rng = np.random.default_rng(24)
    shape = (4, 5, 3, 2, 6)[:n]
    x = rng.normal(size=shape)
    if complex_values:
        x = x + 1j * rng.normal(size=shape)
    x[rng.random(shape) > 0.2] = 0
    factors = [rng.normal(size=(size, 3)) for size in shape]
    labels = "ijklm"[:n]
    xp = SmartSparseFramework()
    tensor = xp.asarray(sp.COO(x))
    monkeypatch.setattr(saps_smart, "_EINSUM_BLOCK_SIZE", 2)

    def fail(*args, **kwargs):
        pytest.fail("blocked contraction must neither densify X nor call sparse.einsum")

    monkeypatch.setattr(sp, "einsum", fail)
    monkeypatch.setattr(sp.COO, "todense", fail)
    for mode in range(n):
        others = [axis for axis in range(n) if axis != mode]
        expression = f"Y[rank, {labels[mode]}] += X[" + ",".join(labels) + "]"
        inputs = {"X": tensor}
        for axis in others:
            expression += f" * F{axis}[{labels[axis]}, rank]"
            inputs[f"F{axis}"] = xp.asarray(factors[axis])
        equation = (
            labels
            + ","
            + ",".join(labels[axis] + "r" for axis in others)
            + "->r"
            + labels[mode]
        )
        expected = np.einsum(equation, x, *[factors[axis] for axis in others])
        np.testing.assert_allclose(
            xp.einsum(expression, **inputs), expected, atol=1e-12
        )


def test_hosvd_projection_with_multiple_free_indices(monkeypatch):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 5, 3))
    x[x < 0.5] = 0
    b, c = rng.random((5, 2)), rng.random((3, 3))
    xp = SmartSparseFramework()
    monkeypatch.setattr(saps_smart, "_EINSUM_BLOCK_SIZE", 3)
    got = xp.einsum(
        "Y[q,i,p] += X[i,j,k] * B[j,p] * C[k,q]",
        X=xp.asarray(sp.COO(x)),
        B=xp.asarray(sp.COO(b)),
        C=xp.asarray(sp.COO(c)),
    )
    np.testing.assert_allclose(got, np.einsum("ijk,jp,kq->qip", x, b, c))


def test_huge_logical_tensor_never_materialized(monkeypatch):
    xp = SmartSparseFramework()
    x = sp.COO(
        [[0, 0, 1, 1], [1, 9000, 1, 8000], [3, 18000, 3, 19000]],
        [2.0, 3.0, 5.0, 7.0],
        shape=(2, 10000, 20000),
    )
    b = np.ones((10000, 2))
    c = np.full((20000, 2), 2.0)
    monkeypatch.setattr(saps_smart, "_EINSUM_BLOCK_SIZE", 2)

    def fail(*args, **kwargs):
        pytest.fail("must not materialize the sparse tensor or broadcast product")

    monkeypatch.setattr(sp.COO, "todense", fail)
    monkeypatch.setattr(sp, "einsum", fail)
    got = xp.einsum(
        "Y[i,r] += X[i,j,k] * B[j,r] * C[k,r]",
        X=xp.asarray(x),
        B=xp.asarray(b),
        C=xp.asarray(c),
    )
    np.testing.assert_array_equal(got, [[10.0, 10.0], [24.0, 24.0]])


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_empty_sparse_contraction(shape):
    xp = SmartSparseFramework()
    a = xp.asarray(sp.COO.from_numpy(np.zeros(shape)))
    got = xp.einsum("Y[i] += A[i,j] * B[j]", A=a, B=xp.asarray(np.ones(3)))
    np.testing.assert_array_equal(got, np.zeros(shape[0]))


def test_dense_budget_keeps_native_fallback(monkeypatch):
    xp = SmartSparseFramework()
    a = xp.asarray(sp.COO(np.eye(3)))
    monkeypatch.setattr(saps_smart, "_EINSUM_DENSE_BYTES", 1)
    native = sp.einsum
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(sp, "einsum", spy)
    got = xp.einsum("Y[i] += A[i,j]", A=a)
    np.testing.assert_array_equal(got, np.ones(3))
    assert len(calls) == 1
