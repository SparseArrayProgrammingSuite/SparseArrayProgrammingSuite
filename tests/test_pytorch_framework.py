import pytest

import numpy as np

import sparse
from binsparse.conversions import from_sparse, to_numpy

torch = pytest.importorskip("torch")

from frameworks.saps_pytorch import PytorchFramework  # noqa: E402


@pytest.fixture
def xp():
    return PytorchFramework()


def test_nonzero_fill_input_is_dense(xp):
    # Torch sparse tensors can't represent an implicit inf.
    coo = sparse.COO([[0, 1], [1, 0]], [2.0, 3.0], shape=(2, 2), fill_value=np.inf)

    result = xp.from_binsparse(from_sparse(coo))

    assert result.layout == torch.strided
    np.testing.assert_array_equal(result.numpy(), coo.todense())


@pytest.mark.parametrize("value", [np.array([[1.0, 2.0]]), np.float64(2.5), 3])
def test_to_binsparse_accepts_numpy_and_scalars(xp, value):
    np.testing.assert_array_equal(to_numpy(xp.to_binsparse(value)), value)


@pytest.mark.parametrize("name", ["pinv", "svd"])
@pytest.mark.parametrize("make_sparse", [False, True])
def test_linalg_pinv_and_svd(xp, name, make_sparse):
    matrix = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    tensor = torch.asarray(matrix)
    if make_sparse:
        tensor = tensor.to_sparse()

    result = getattr(xp.linalg, name)(tensor)

    if name == "pinv":
        np.testing.assert_allclose(result.numpy(), np.linalg.pinv(matrix))
    else:
        np.testing.assert_allclose(result.S.numpy(), np.linalg.svd(matrix).S)


@pytest.mark.parametrize("axes", [None, (1, 2, 0)])
def test_transpose_follows_numpy(xp, axes):
    array = np.arange(24.0).reshape(2, 3, 4)

    result = xp.transpose(torch.asarray(array), axes)

    np.testing.assert_array_equal(result.numpy(), np.transpose(array, axes))


def test_array_accepts_numpy_dtype(xp):
    # Generated constant functions pass NumPy dtypes.
    result = xp.array([1, 2], dtype=np.float64)

    assert torch.is_tensor(result)
    assert result.dtype == torch.float64
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0])
