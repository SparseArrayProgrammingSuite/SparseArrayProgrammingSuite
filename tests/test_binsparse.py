import pytest

import numpy as np

from saps_framework.binsparse_format import BinsparseFormat


@pytest.mark.parametrize("array", [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])])
def test_binsparse_numpy(array):
    x = BinsparseFormat.from_numpy(array)
    assert x.data["format"] == "dense"
    assert x.data["shape"] == array.shape
    assert BinsparseFormat.deserialize(x.serialize()) == x


@pytest.mark.parametrize(
    "indices, values, shape",
    [((np.array([1, 2, 3]), np.array([0, 1, 2])), np.array([0.1, 0.2, 0.3]), (4, 3))],
)
def test_binsparse_coo(indices, values, shape):
    x = BinsparseFormat.from_coo(indices, values, shape)
    assert x.data["format"] == "COO"
    assert len(x.data["shape"]) == len(indices) == len(shape)
    assert BinsparseFormat.deserialize(x.serialize()) == x
