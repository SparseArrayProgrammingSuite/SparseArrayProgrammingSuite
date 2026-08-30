import pytest

import numpy as np

from binsparse import COORMatrix
from binsparse.conversions import from_numpy, to_numpy

from saps_framework.binsparse_utils import binsparse_equal


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2, 3]),
        np.array([[1, 2], [3, 4]]),
        np.arange(24).reshape(2, 3, 4),
    ],
)
def test_binsparse_numpy(array):
    x = from_numpy(array)
    assert np.array_equal(to_numpy(x), array)
    assert x.shape == array.shape
    assert binsparse_equal(from_numpy(array.copy()), x)


def test_binsparse_numpy_detects_different_values_and_shapes():
    assert not binsparse_equal(
        from_numpy(np.zeros((2, 2))), from_numpy(np.ones((2, 2)))
    )
    assert not binsparse_equal(from_numpy(np.zeros((2, 2))), from_numpy(np.zeros((4,))))


@pytest.mark.parametrize(
    "indices, values, shape",
    [((np.array([1, 2, 3]), np.array([0, 1, 2])), np.array([0.1, 0.2, 0.3]), (4, 3))],
)
def test_binsparse_coo(indices, values, shape):
    x = COORMatrix(
        shape,
        len(values),
        indices_0=indices[0],
        indices_1=indices[1],
        values=values,
    )
    assert np.array_equal(x.indices_0, indices[0])
    assert np.array_equal(x.indices_1, indices[1])
    assert len(x.shape) == len(indices) == len(shape)
    assert binsparse_equal(
        COORMatrix(
            shape,
            len(values),
            indices_0=indices[0].copy(),
            indices_1=indices[1].copy(),
            values=values.copy(),
        ),
        x,
    )
