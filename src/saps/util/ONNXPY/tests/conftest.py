"""Shared test helpers for ONNXPY round-trip tests."""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort
import pytest
from onnx import ModelProto

from onnxpy import compile_to_source


class _NumpyWithConv:
    """Test array namespace implementing conv and pool with NumPy views."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def _windows(x, kernel_shape, strides, dilations, padding, fill_value):
        """Pad x's spatial axes, then view it as (N, C, *windows, *kernel)."""
        rank = len(kernel_shape)
        axes = tuple(range(2, 2 + rank))
        strides = (1,) * rank if strides is None else strides
        dilations = (1,) * rank if dilations is None else dilations
        padding = ((0, 0),) * rank if padding is None else padding
        padded = np.pad(x, [(0, 0), (0, 0), *padding], constant_values=fill_value)
        effective = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_shape, dilations, strict=True)
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded, effective, axis=axes)
        steps = (
            *(slice(None, None, stride) for stride in strides),
            *(slice(None, None, dilation) for dilation in dilations),
        )
        return windows[(slice(None), slice(None), *steps)]

    def conv(self, x, w, *, strides=None, dilations=None, padding=None, group=1):
        rank = w.ndim - 2
        n, c, *_ = x.shape
        m = w.shape[0]
        windows = self._windows(x, w.shape[2:], strides, dilations, padding, 0)
        windows = windows.reshape(n, group, c // group, *windows.shape[2:])
        w = w.reshape(group, m // group, c // group, *w.shape[2:])
        spatial = list(range(4, 4 + rank))
        kernel = list(range(4 + rank, 4 + 2 * rank))
        y = np.einsum(
            windows, [0, 1, 2, *spatial, *kernel], w, [1, 3, 2, *kernel], [0, 1, 3, *spatial]
        )
        return y.reshape(n, m, *y.shape[3:])

    def pool(
        self,
        x,
        kernel_shape,
        *,
        reduce,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        windows = self._windows(x, kernel_shape, strides, dilations, padding, fill_value)
        reduce_op = {"max": np.max, "+": np.sum, "mean": np.mean}[reduce]
        return reduce_op(windows, axis=tuple(range(x.ndim, windows.ndim)))


@pytest.fixture(scope="session")
def array_namespace():
    return _NumpyWithConv()


def _compile_and_run(model: ModelProto, feed: dict[str, np.ndarray], xp):
    source, tensor_inputs = compile_to_source(model)
    scope = {"math": math}
    exec(source, scope)
    args = [feed[vi.name] for vi in model.graph.input if vi.name not in tensor_inputs]
    return scope["model"](xp, *args, *tensor_inputs.values())


def _run_ort(model: ModelProto, feed: dict[str, np.ndarray]):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {k: np.asarray(v) for k, v in feed.items()})
    return res[0] if len(res) == 1 else tuple(res)


@pytest.fixture
def assert_matches_ort(array_namespace):
    """Compile, execute, and compare against onnxruntime within tolerance."""

    def _do(
        model: ModelProto,
        feed: dict[str, np.ndarray],
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        actual = _compile_and_run(model, feed, array_namespace)
        expected = _run_ort(model, feed)
        if isinstance(expected, tuple):
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_allclose(np.asarray(a), e, atol=atol, rtol=rtol)
        else:
            np.testing.assert_allclose(np.asarray(actual), expected, atol=atol, rtol=rtol)

    return _do
