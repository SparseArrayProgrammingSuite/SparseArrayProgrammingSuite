import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps_framework.binsparse_utils import (
    binsparse_equal,
    deserialize,
    from_coo,
    serialize,
)


@pytest.mark.parametrize("array", [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])])
def test_binsparse_numpy(array):
    x = from_numpy(array)
    assert np.array_equal(to_numpy(x), array)
    assert x.shape == array.shape
    assert binsparse_equal(deserialize(serialize(x)), x)


@pytest.mark.parametrize(
    "indices, values, shape",
    [((np.array([1, 2, 3]), np.array([0, 1, 2])), np.array([0.1, 0.2, 0.3]), (4, 3))],
)
def test_binsparse_coo(indices, values, shape):
    x = from_coo(indices, values, shape)
    assert np.array_equal(x.indices_0, indices[0])
    assert np.array_equal(x.indices_1, indices[1])
    assert len(x.shape) == len(indices) == len(shape)
    assert binsparse_equal(deserialize(serialize(x)), x)
