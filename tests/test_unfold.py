import pytest

import numpy as np
import scipy.sparse as sps

import sparse as sp

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import PyDataSparseFramework


def _expected_unfold(
    x,
    kernel_shape,
    *,
    axes,
    strides=None,
    dilations=None,
    padding=None,
    fill_value=0,
):
    rank = len(kernel_shape)
    strides = (1,) * rank if strides is None else tuple(strides)
    dilations = (1,) * rank if dilations is None else tuple(dilations)
    pad_width = [(0, 0)] * x.ndim
    if padding is not None:
        for axis, pair in zip(axes, padding, strict=True):
            pad_width[axis] = tuple(pair)
    padded = np.pad(x, pad_width, mode="constant", constant_values=fill_value)
    effective = tuple(
        (kernel - 1) * dilation + 1
        for kernel, dilation in zip(kernel_shape, dilations, strict=True)
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        effective,
        axis=axes,
    )
    slices = [slice(None)] * windows.ndim
    for axis, step in zip(axes, strides, strict=True):
        slices[axis] = slice(None, None, step)
    for axis, dilation in enumerate(dilations, start=x.ndim):
        slices[axis] = slice(None, None, dilation)
    return windows[tuple(slices)]


def _as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "todense"):
        return np.asarray(array.todense())
    return np.asarray(array)


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_unfold_matches_numpy_sliding_windows(framework_cls):
    xp = framework_cls()
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    actual = xp.unfold(
        xp.asarray(x),
        (2, 3),
        axes=(1, 2),
        strides=(1, 2),
        dilations=(1, 1),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )
    expected = _expected_unfold(
        x,
        (2, 3),
        axes=(1, 2),
        strides=(1, 2),
        dilations=(1, 1),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )

    np.testing.assert_array_equal(_as_numpy(actual), expected)


def test_unfold_applies_dilation_on_window_axes():
    xp = NumpyFramework()
    x = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)

    actual = xp.unfold(
        x,
        (2, 2),
        axes=(0, 1),
        strides=(1, 1),
        dilations=(2, 3),
    )
    expected = _expected_unfold(
        x,
        (2, 2),
        axes=(0, 1),
        strides=(1, 1),
        dilations=(2, 3),
    )

    np.testing.assert_array_equal(actual, expected)


def test_pytorch_unfold_matches_numpy_for_nd_case():
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    actual = xp.unfold(
        torch.asarray(x),
        (2, 3),
        axes=(1, 2),
        strides=(1, 2),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )
    expected = _expected_unfold(
        x,
        (2, 3),
        axes=(1, 2),
        strides=(1, 2),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )

    np.testing.assert_array_equal(_as_numpy(actual), expected)


def test_pytorch_unfold_uses_conv2d_shaped_layout():
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = np.arange(1 * 2 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 5)

    actual = xp.unfold(
        torch.asarray(x),
        (2, 3),
        axes=(2, 3),
        strides=(1, 2),
        dilations=(1, 1),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )
    expected = _expected_unfold(
        x,
        (2, 3),
        axes=(2, 3),
        strides=(1, 2),
        dilations=(1, 1),
        padding=((1, 0), (0, 1)),
        fill_value=-1,
    )

    np.testing.assert_array_equal(_as_numpy(actual), expected)


def test_scipy_sparse_unfold_uses_sparse_output():
    xp = SciPyFramework()
    x_dense = np.zeros((4, 5), dtype=np.float32)
    x_dense[0, 1] = 2
    x_dense[2, 3] = 5
    x_dense[3, 0] = 7
    x = sps.coo_array(x_dense)

    actual = xp.unfold(
        x,
        (2, 2),
        axes=(0, 1),
        strides=(1, 2),
        padding=((1, 0), (0, 1)),
    )
    expected = _expected_unfold(
        x_dense,
        (2, 2),
        axes=(0, 1),
        strides=(1, 2),
        padding=((1, 0), (0, 1)),
    )

    assert isinstance(actual, sp.SparseArray)
    np.testing.assert_array_equal(_as_numpy(actual), expected)


def test_pydata_sparse_unfold_keeps_sparse_output():
    xp = PyDataSparseFramework()
    x_dense = np.zeros((2, 5, 6), dtype=np.float32)
    x_dense[0, 1, 2] = 2
    x_dense[1, 3, 4] = 5
    x = sp.COO.from_numpy(x_dense)

    actual = xp.unfold(
        x,
        (2, 2),
        axes=(1, 2),
        strides=(1, 2),
        dilations=(2, 1),
    )
    expected = _expected_unfold(
        x_dense,
        (2, 2),
        axes=(1, 2),
        strides=(1, 2),
        dilations=(2, 1),
    )

    assert isinstance(actual, sp.SparseArray)
    np.testing.assert_array_equal(_as_numpy(actual), expected)
