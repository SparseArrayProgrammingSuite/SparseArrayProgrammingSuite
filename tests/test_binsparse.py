import pytest

import numpy as np

from binsparse import BinsparseTensor

from saps_framework.binsparse_utils import deserialize, equal, serialize


@pytest.mark.parametrize("array", [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])])
def test_binsparse_numpy(array):
    x = BinsparseTensor.from_numpy(array)
    assert x.data["format"] == "dense"
    assert x.data["shape"] == array.shape
    assert equal(deserialize(serialize(x)), x)


@pytest.mark.parametrize(
    "indices, values, shape",
    [((np.array([1, 2, 3]), np.array([0, 1, 2])), np.array([0.1, 0.2, 0.3]), (4, 3))],
)
def test_binsparse_coo(indices, values, shape):
    x = BinsparseTensor.from_coo(indices, values, shape)
    assert x.data["format"] == "COO"
    assert len(x.data["shape"]) == len(indices) == len(shape)
    assert equal(deserialize(serialize(x)), x)
