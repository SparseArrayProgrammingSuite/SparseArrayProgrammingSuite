import pytest

import numpy as np

import array_api_compat.numpy as compat_np
import sparse as sp

from frameworks.saps_smart import SmartSparseKernels
from frameworks.saps_sparse import PyDataSparseFramework


@pytest.fixture(params=[SmartSparseKernels, PyDataSparseFramework])
def xp(request):
    return request.param()


@pytest.mark.parametrize(
    "shape, options",
    [
        ((4,), {}),
        ((4, None), {"dtype": None}),
        ((3, 5), {"k": 1, "dtype": np.bool_}),
        ((5, 3), {"k": -2, "dtype": np.float32}),
        ((3, 5), {"k": 5, "dtype": np.int32}),
        ((5, 3), {"k": -5}),
        ((0, 3), {}),
        ((3, 0), {}),
        ((0,), {}),
        ((4,), {"dtype": None, "device": "cpu"}),
    ],
)
def test_eye_is_sparse_and_matches_numpy(xp, shape, options):
    actual = xp.eye(*shape, **options)
    expected = compat_np.eye(*shape, **options)

    assert isinstance(actual, sp.SparseArray)
    assert actual.fill_value == 0
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.todense(), expected)


def test_large_eye_uses_linear_storage_without_dense_construction(xp, monkeypatch):
    def reject_dense_eye(*args, **kwargs):
        pytest.fail("eye must not allocate a dense identity matrix")

    monkeypatch.setattr(np, "eye", reject_dense_eye)
    n = 100_000
    actual = xp.eye(n, dtype=bool)

    assert isinstance(actual, sp.COO)
    assert actual.shape == (n, n)
    assert actual.nnz == n
    assert actual.fill_value == 0
    assert actual.nbytes <= n * (2 * np.dtype(np.intp).itemsize + 1)
    np.testing.assert_array_equal(actual.coords[0], np.arange(n))
    np.testing.assert_array_equal(actual.coords[1], np.arange(n))
    assert np.all(actual.data)
