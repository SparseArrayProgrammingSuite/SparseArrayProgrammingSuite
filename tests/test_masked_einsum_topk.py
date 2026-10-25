import pytest

import numpy as np

import sparse as sp

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseKernels


def _mask_and_factors(seed, sparse_factors):
    rng = np.random.default_rng(seed)
    mask = rng.random((30, 40)) < 0.1
    a = rng.standard_normal((30, 8)) * (rng.random((30, 8)) < 0.5)
    b = rng.standard_normal((40, 8)) * (rng.random((40, 8)) < 0.5)
    expected = mask * (a @ b.T)
    if sparse_factors:
        a, b = sp.COO.from_numpy(a), sp.COO.from_numpy(b)
    return sp.COO.from_numpy(mask), a, b, expected


@pytest.mark.parametrize("sparse_factors", [False, True])
@pytest.mark.parametrize(
    "program",
    [
        "S[q,n] += C[q,n] * Q[q,f] * D[n,f]",
        "S[q,n] += Q[q,f] * C[n,q] * D[f,n]",
    ],
)
def test_smart_masked_einsum_computes_only_mask_entries(program, sparse_factors):
    mask, a, b, expected = _mask_and_factors(0, sparse_factors)
    if "C[n,q]" in program:
        mask, b = mask.T, b.T
    kernels = SmartSparseKernels()
    result = kernels.einsum(program, C=mask, Q=a, D=b)
    assert isinstance(result, sp.COO)
    assert result.nnz <= mask.nnz
    np.testing.assert_allclose(result.todense(), expected, atol=1e-12)


def test_smart_masked_einsum_never_forms_the_full_product(monkeypatch):
    mask, a, b, expected = _mask_and_factors(1, sparse_factors=True)
    monkeypatch.setattr(
        SmartSparseKernels, "_blocked_einsum", lambda *args: pytest.fail("fell back")
    )
    result = SmartSparseKernels().einsum(
        "S[q,n] += C[q,n] * Q[q,f] * D[n,f]", C=mask, Q=a, D=b
    )
    np.testing.assert_allclose(result.todense(), expected, atol=1e-12)


@pytest.mark.parametrize("largest", [False, True])
def test_smart_sparse_topk_matches_dense_topk(largest):
    rng = np.random.default_rng(2)
    fill = -np.inf if largest else np.inf
    dense = np.where(rng.random((20, 50)) < 0.3, rng.standard_normal((20, 50)), fill)
    dense[:, :3] = rng.standard_normal((20, 3))  # every row stores at least 3
    dense[0, 10] = dense[0, 11] = -5 if not largest else 5  # a tie, by index
    x = sp.COO.from_numpy(dense, fill_value=fill)
    values, indices = SmartSparseKernels().topk(x, 3, axis=1, largest=largest)
    want_values, want_indices = NumpyFramework().topk(dense, 3, axis=1, largest=largest)
    np.testing.assert_array_equal(indices, want_indices)
    np.testing.assert_array_equal(values, want_values)
    assert list(indices[0, :2]) == [10, 11]


def test_smart_sparse_topk_falls_back_when_a_row_stores_too_few():
    dense = np.full((3, 6), np.inf)
    dense[0, [1, 4]] = [0.5, 0.25]  # row 0 stores only two entries
    dense[1:, :3] = 1.0
    x = sp.COO.from_numpy(dense, fill_value=np.inf)
    values, indices = SmartSparseKernels().topk(x, 3, axis=1, largest=False)
    np.testing.assert_array_equal(indices[0], [4, 1, 0])
    np.testing.assert_array_equal(values[0], [0.25, 0.5, np.inf])


def test_numpy_factors_out_an_output_shaped_mask(monkeypatch):
    rng = np.random.default_rng(3)
    mask = rng.random((30, 40)) < 0.1
    a, b = rng.standard_normal((30, 8)), rng.standard_normal((40, 8))
    monkeypatch.setattr(np, "einsum", lambda *args, **kwargs: pytest.fail("np.einsum"))
    result = NumpyFramework().einsum(
        "S[q,n] += C[q,n] * Q[q,f] * D[n,f]", C=mask, Q=a, D=b
    )
    np.testing.assert_allclose(result, mask * (a @ b.T), atol=1e-12)
