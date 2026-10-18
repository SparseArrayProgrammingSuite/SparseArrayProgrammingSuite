from pathlib import Path

import pytest

import numpy as np
import scipy.sparse as sps

import sparse as sp

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_smart import SmartSparseFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.framework import load_framework
from saps_framework import Framework

PADDING_MODES = ["constant", "reflect", "edge", "wrap"]


def _expected_pad(x, padding, *, axes, padding_mode, fill_value=0):
    pad_width = [(0, 0)] * x.ndim
    for axis, pair in zip(axes, padding, strict=True):
        pad_width[axis] = tuple(pair)
    if padding_mode == "constant":
        return np.pad(x, pad_width, mode="constant", constant_values=fill_value)
    return np.pad(x, pad_width, mode=padding_mode)


def _expected_unfold(
    x,
    kernel_shape,
    *,
    axes,
    strides=None,
    dilations=None,
    padding=None,
    padding_mode="constant",
    fill_value=0,
):
    rank = len(kernel_shape)
    strides = (1,) * rank if strides is None else tuple(strides)
    dilations = (1,) * rank if dilations is None else tuple(dilations)
    padding = ((0, 0),) * rank if padding is None else padding
    padded = _expected_pad(
        x, padding, axes=axes, padding_mode=padding_mode, fill_value=fill_value
    )
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


def _expected_conv(
    x, w, *, strides, dilations, padding, group=1, combine=np.multiply, reduce=np.sum
):
    batch, channels, *_ = x.shape
    out_channels = w.shape[0]
    rank = w.ndim - 2
    patches = _expected_unfold(
        x.reshape(batch, group, channels // group, *x.shape[2:]),
        w.shape[2:],
        axes=tuple(range(3, 3 + rank)),
        strides=strides,
        dilations=dilations,
        padding=padding,
    )
    weights = w.reshape(group, out_channels // group, *w.shape[1:])
    # (n, g, 1, c, *windows, *kernel) against (1, g, m, c, *1, *kernel).
    patches = np.expand_dims(patches, 2)
    weights = weights.reshape(1, *weights.shape[:3], *((1,) * rank), *weights.shape[3:])
    reduced_axes = (3, *range(4 + rank, 4 + 2 * rank))
    y = reduce(combine(patches, weights), axis=reduced_axes)
    return y.reshape(batch, out_channels, *y.shape[3:])


def _as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "todense"):
        return np.asarray(array.todense())
    if hasattr(array, "toarray"):
        return array.toarray()
    return np.asarray(array)


def _sparse_input(framework_cls, x):
    if framework_cls is SciPyFramework:
        return sps.coo_array(x)
    if framework_cls is PyDataSparseFramework:
        return sp.COO.from_numpy(x)
    return x


def _sparse_array(seed, shape):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    x[rng.random(shape) < 0.6] = 0
    return x


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_pad_functions_handle_wide_padding_and_short_axes(framework_cls, padding_mode):
    xp = framework_cls()
    pad = getattr(xp, f"pad_{padding_mode}")
    for size in [1, 2, 3, 4]:
        x = np.arange(1, 2 * size + 1, dtype=np.float32).reshape(2, size)
        for padding in [(0, 0), (1, 2), (3, 0), (9, 7)]:
            actual = pad(_sparse_input(framework_cls, x), [padding], axes=1)
            expected = _expected_pad(x, [padding], axes=(1,), padding_mode=padding_mode)
            np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize("framework_name", ["numpy", "torch", "scipy", "pydata"])
def test_shared_concat_fallback_pads_every_framework(framework_name, padding_mode):
    # Call the base class directly so native overrides don't intercept it.
    fallback = getattr(Framework, f"pad_{padding_mode}")
    if framework_name == "torch":
        torch = pytest.importorskip("torch")
        from frameworks.saps_pytorch import PytorchFramework

        xp = PytorchFramework()
    else:
        xp = {
            "numpy": NumpyFramework,
            "scipy": SciPyFramework,
            "pydata": PyDataSparseFramework,
        }[framework_name]()

    for size in [1, 2, 3, 4]:
        x = np.arange(1, 2 * size + 1, dtype=np.float32).reshape(2, size)
        for padding in [(0, 0), (1, 2), (3, 0), (9, 7)]:
            if framework_name == "torch":
                x_in = torch.asarray(x)
            else:
                x_in = _sparse_input(type(xp), x)
            actual = fallback(xp, x_in, [padding], axes=1)
            expected = _expected_pad(x, [padding], axes=(1,), padding_mode=padding_mode)
            np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize("padding_mode", ["reflect", "edge", "wrap"])
def test_pad_functions_reject_padding_empty_axes(padding_mode):
    xp = NumpyFramework()
    pad = getattr(xp, f"pad_{padding_mode}")
    with pytest.raises(ValueError, match="empty axis"):
        pad(np.zeros((0, 2)), 1, axes=0)
    assert pad(np.zeros((0, 2)), 0, axes=0).shape == (0, 2)
    assert xp.pad_constant(np.zeros((0, 2)), 1, axes=0).shape == (2, 2)


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_pad_functions_match_numpy_pad(framework_cls, padding_mode):
    xp = framework_cls()
    x = _sparse_array(0, (3, 4, 5))
    pad = getattr(xp, f"pad_{padding_mode}")

    actual = pad(_sparse_input(framework_cls, x), ((2, 0), (1, 5)), axes=(0, 2))
    expected = _expected_pad(
        x, ((2, 0), (1, 5)), axes=(0, 2), padding_mode=padding_mode
    )

    if framework_cls is not NumpyFramework:
        assert not isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_pad_constant_pads_every_axis_by_default(framework_cls):
    xp = framework_cls()
    x = _sparse_array(1, (2, 3))

    actual = xp.pad_constant(_sparse_input(framework_cls, x), 1, fill_value=7)

    np.testing.assert_array_equal(_as_numpy(actual), np.pad(x, 1, constant_values=7))


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
def test_pytorch_pad_functions_pad_leading_axes(padding_mode):
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    pad = getattr(xp, f"pad_{padding_mode}")

    # F.pad's non-constant modes can't pad the leading axis of a 3D tensor.
    actual = pad(torch.asarray(x), (4, 1), axes=0)
    expected = _expected_pad(x, [(4, 1)], axes=(0,), padding_mode=padding_mode)

    np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
def test_pytorch_pad_functions_pad_trailing_axes(padding_mode):
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    pad = getattr(xp, f"pad_{padding_mode}")
    # F.pad handles narrow padding of trailing axes; wider padding falls back.
    for size in [1, 2, 3, 4]:
        x = np.arange(2 * 3 * size * 4, dtype=np.float32).reshape(2, 3, size, 4)
        for padding in [(0, 0), (1, 2), (3, 0), (9, 7)]:
            actual = pad(torch.asarray(x), [padding, (1, 1)], axes=(2, 3))
            expected = _expected_pad(
                x, [padding, (1, 1)], axes=(2, 3), padding_mode=padding_mode
            )
            np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_padded_unfold_matches_numpy_sliding_windows(framework_cls):
    xp = framework_cls()
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    actual = xp.padded_unfold(
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


@pytest.mark.parametrize("padding_mode", PADDING_MODES)
@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_padded_unfold_padding_modes(framework_cls, padding_mode):
    xp = framework_cls()
    x = _sparse_array(2, (2, 5, 4))

    actual = xp.padded_unfold(
        _sparse_input(framework_cls, x),
        (2, 3),
        axes=(1, 2),
        strides=(2, 1),
        dilations=(2, 1),
        padding=((3, 1), (0, 2)),
        padding_mode=padding_mode,
    )
    expected = _expected_unfold(
        x,
        (2, 3),
        axes=(1, 2),
        strides=(2, 1),
        dilations=(2, 1),
        padding=((3, 1), (0, 2)),
        padding_mode=padding_mode,
    )

    if framework_cls is not NumpyFramework:
        assert isinstance(actual, sp.SparseArray)
    np.testing.assert_array_equal(_as_numpy(actual), expected)


@pytest.mark.parametrize(
    "framework_cls",
    [SciPyFramework, PyDataSparseFramework],
)
def test_sparse_padded_unfold_with_nonzero_fill(framework_cls):
    xp = framework_cls()
    x = _sparse_array(3, (4, 5))

    actual = xp.padded_unfold(
        _sparse_input(framework_cls, x),
        (2, 2),
        axes=(0, 1),
        padding=1,
        fill_value=-3,
    )
    expected = _expected_unfold(
        x, (2, 2), axes=(0, 1), padding=((1, 1), (1, 1)), fill_value=-3
    )

    np.testing.assert_array_equal(_as_numpy(actual), expected)


def test_padded_unfold_rejects_unknown_padding_mode():
    xp = NumpyFramework()
    with pytest.raises(ValueError, match="padding_mode"):
        xp.padded_unfold(np.zeros((3, 3)), (2, 2), padding_mode="symmetric")


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


def test_pytorch_padded_unfold_matches_numpy_for_nd_case():
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    actual = xp.padded_unfold(
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


def test_pytorch_padded_unfold_uses_conv2d_shaped_layout():
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = np.arange(1 * 2 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 5)

    actual = xp.padded_unfold(
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


def test_scipy_sparse_padded_unfold_uses_sparse_output():
    xp = SciPyFramework()
    x_dense = np.zeros((4, 5), dtype=np.float32)
    x_dense[0, 1] = 2
    x_dense[2, 3] = 5
    x_dense[3, 0] = 7
    x = sps.coo_array(x_dense)

    actual = xp.padded_unfold(
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


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework, SmartSparseFramework],
)
def test_conv_matches_grouped_reference(framework_cls):
    xp = framework_cls()
    x = _sparse_array(4, (2, 4, 7, 6))
    w = _sparse_array(5, (6, 2, 3, 2))
    kwargs = {"strides": (2, 1), "dilations": (1, 2), "padding": ((1, 3), (2, 0))}

    if framework_cls is SmartSparseFramework:
        x_in = xp.wrap(sp.COO.from_numpy(x))
        actual = xp.asnumpy(xp.conv(x_in, xp.asarray(w), group=2, **kwargs))
    else:
        actual = xp.conv(_sparse_input(framework_cls, x), w, group=2, **kwargs)
    expected = _expected_conv(x, w, group=2, **kwargs)

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_pytorch_conv_matches_reference(rank):
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = _sparse_array(6, (2, 4, *((5,) * rank)))
    w = _sparse_array(7, (6, 2, *((2,) * rank)))
    padding = ((1, 0),) * rank

    actual = xp.conv(
        torch.asarray(x), torch.asarray(w), strides=2, padding=padding, group=2
    )
    expected = _expected_conv(
        x, w, strides=(2,) * rank, dilations=(1,) * rank, padding=padding, group=2
    )

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("combine", "reduce", "combine_op", "reduce_op"),
    [
        ("+", "max", np.add, np.max),
        ("max", "min", np.maximum, np.min),
        ("-", "+", np.subtract, np.sum),
    ],
)
def test_conv_with_other_operators(combine, reduce, combine_op, reduce_op):
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    x = _sparse_array(8, (1, 2, 5, 5))
    w = _sparse_array(9, (3, 2, 2, 3))
    kwargs = {"strides": 1, "dilations": 1, "padding": 1}
    expected = _expected_conv(
        x,
        w,
        strides=(1, 1),
        dilations=(1, 1),
        padding=((1, 1), (1, 1)),
        combine=combine_op,
        reduce=reduce_op,
    )

    for xp, x_in, w_in in [
        (NumpyFramework(), x, w),
        (PytorchFramework(), torch.asarray(x), torch.asarray(w)),
    ]:
        actual = xp.conv(x_in, w_in, combine=combine, reduce=reduce, **kwargs)
        np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-5, atol=1e-5)


def test_pytorch_conv_falls_back_for_other_padding_modes():
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    x = _sparse_array(10, (1, 2, 4, 5))
    w = _sparse_array(11, (3, 2, 3, 3))

    actual = PytorchFramework().conv(
        torch.asarray(x), torch.asarray(w), padding=2, padding_mode="wrap"
    )
    expected = NumpyFramework().conv(x, w, padding=2, padding_mode="wrap")

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-5, atol=1e-5)


def test_conv_rejects_mismatched_groups():
    xp = NumpyFramework()
    with pytest.raises(ValueError, match="group=2"):
        xp.conv(np.zeros((1, 3, 4, 4)), np.zeros((2, 1, 2, 2)), group=2)


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework],
)
def test_pool_matches_numpy_reductions(framework_cls):
    xp = framework_cls()
    x = _sparse_array(12, (2, 3, 5, 6))
    kwargs = {"strides": (2, 1), "dilations": (1, 2), "padding": 1}
    windows = _expected_unfold(
        x,
        (2, 2),
        axes=(2, 3),
        strides=(2, 1),
        dilations=(1, 2),
        padding=((1, 1), (1, 1)),
        fill_value=-np.inf,
    )

    actual = xp.pool(
        _sparse_input(framework_cls, x),
        (2, 2),
        reduce="max",
        fill_value=-np.inf,
        **kwargs,
    )
    np.testing.assert_array_equal(_as_numpy(actual), windows.max(axis=(4, 5)))

    actual = xp.pool(_sparse_input(framework_cls, x), (2, 2), reduce="+", **kwargs)
    zero_padded = np.where(np.isinf(windows), 0, windows)
    np.testing.assert_allclose(
        _as_numpy(actual), zero_padded.sum(axis=(4, 5)), rtol=1e-5, atol=1e-6
    )


def test_pytorch_pool_matches_torch_pooling():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from frameworks.saps_pytorch import PytorchFramework

    xp = PytorchFramework()
    x = torch.asarray(_sparse_array(13, (2, 3, 6, 5)))

    actual = xp.pool(x, (3, 2), reduce="max", strides=2, padding=1, fill_value=-np.inf)
    np.testing.assert_array_equal(
        _as_numpy(actual), _as_numpy(F.max_pool2d(x, (3, 2), stride=2, padding=1))
    )

    actual = xp.pool(x, (3, 2), reduce="mean", strides=2, padding=1)
    expected = F.avg_pool2d(x, (3, 2), stride=2, padding=1, count_include_pad=True)
    np.testing.assert_allclose(_as_numpy(actual), _as_numpy(expected), rtol=1e-6)


@pytest.mark.parametrize(
    ("reduce", "dilations", "padding_mode"),
    [
        ("max", (1, 2), "reflect"),
        ("mean", (1, 1), "wrap"),
        ("mean", (2, 1), "edge"),
        ("+", (1, 1), "constant"),
    ],
)
def test_pytorch_pool_matches_numpy_framework(reduce, dilations, padding_mode):
    torch = pytest.importorskip("torch")
    from frameworks.saps_pytorch import PytorchFramework

    x = _sparse_array(16, (2, 3, 6, 5))
    kwargs = {
        "reduce": reduce,
        "strides": (2, 1),
        "dilations": dilations,
        "padding": ((1, 2), (2, 0)),
        "padding_mode": padding_mode,
    }

    actual = PytorchFramework().pool(torch.asarray(x), (2, 3), **kwargs)
    expected = NumpyFramework().pool(x, (2, 3), **kwargs)

    np.testing.assert_allclose(_as_numpy(actual), expected, rtol=1e-5, atol=1e-6)


def test_tagger_records_conv_and_pool_as_single_operations():
    path = Path(__file__).resolve().parents[1] / "frameworks/saps_tagger.py"
    xp = type(load_framework(path))(wrapped=NumpyFramework())
    x = _sparse_array(14, (1, 2, 4, 4))
    w = _sparse_array(15, (3, 2, 2, 2))

    y = xp.conv(xp.asarray(x), xp.asarray(w), padding=1)
    pooled = xp.pool(y, (2, 2), reduce="max", strides=2)

    np.testing.assert_allclose(
        pooled.array,
        NumpyFramework().pool(
            NumpyFramework().conv(x, w, padding=1), (2, 2), reduce="max", strides=2
        ),
    )
    assert {"conv", "pool"} <= set(xp.stats["operators"])
    assert "einsum" not in xp.stats["operators"]
