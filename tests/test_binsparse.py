import pytest

import numpy as np

from binsparse import COORMatrix, DMATRMatrix
from binsparse.conversions import from_numpy, to_numpy

from saps_framework.binsparse_utils import assert_binsparse_allclose, binsparse_equal


@pytest.mark.parametrize("array", [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])])
def test_binsparse_numpy(array):
    x = from_numpy(array)
    assert np.array_equal(to_numpy(x), array)
    assert x.shape == array.shape
    assert binsparse_equal(from_numpy(array.copy()), x)


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


def test_binsparse_equal_compares_sparse_and_dense_formats_by_value():
    sparse_tensor = COORMatrix(
        (3, 3),
        3,
        indices_0=np.array([0, 1, 2]),
        indices_1=np.array([0, 2, 1]),
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    dense_tensor = DMATRMatrix(
        (3, 3),
        9,
        values=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
            dtype=np.float32,
        ),
    )

    assert binsparse_equal(sparse_tensor, dense_tensor)
    assert binsparse_equal(dense_tensor, sparse_tensor)


@pytest.mark.parametrize("replacement", [0.0, -np.inf, np.nan])
def test_sparse_distance_comparison_rejects_wrong_unreachable_entries(replacement):
    expected = COORMatrix(
        (2, 2),
        2,
        fill=True,
        fill_value=np.inf,
        indices_0=np.array([0, 1]),
        indices_1=np.array([0, 1]),
        values=np.array([0.0, 0.0]),
    )
    actual = np.array([[0.0, replacement], [np.inf, 0.0]])
    with pytest.raises(AssertionError):
        assert_binsparse_allclose(expected, from_numpy(actual))
