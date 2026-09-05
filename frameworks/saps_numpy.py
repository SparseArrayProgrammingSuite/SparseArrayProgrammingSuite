import numpy as np

from binsparse import (
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, to_numpy, to_sparse

from saps_framework import Framework, einsum, normalize_unfold_args


class NumpyFramework(Framework):
    def __init__(self):
        pass

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return to_numpy(array)
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return to_numpy(array)
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return to_numpy(array)
            case _:
                return np.asarray(to_sparse(array).todense())

    def to_binsparse(self, array):
        return from_numpy(np.asarray(array))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(np, prgm, **kwargs)

    def unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        array = np.asarray(x)
        kernel_t = tuple(int(size) for size in kernel_shape)
        axes_t, strides_t, dilations_t, padding_t = normalize_unfold_args(
            array.ndim,
            kernel_t,
            axes,
            strides,
            dilations,
            padding,
        )
        effective_kernel = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_t, dilations_t, strict=True)
        )

        pad_width = [(0, 0)] * array.ndim
        for axis, pad_pair in zip(axes_t, padding_t, strict=True):
            pad_width[axis] = pad_pair
        if any(pair != (0, 0) for pair in pad_width):
            array = np.pad(
                array,
                pad_width,
                mode="constant",
                constant_values=fill_value,
            )

        windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore[call-overload]
            array,
            effective_kernel,
            axis=axes_t,
        )
        slices: list[slice] = [slice(None)] * windows.ndim
        for axis, step in zip(axes_t, strides_t, strict=True):
            slices[axis] = slice(None, None, step)
        for window_axis, dilation in enumerate(dilations_t, start=array.ndim):
            slices[window_axis] = slice(None, None, dilation)
        return windows[tuple(slices)]

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(np, name)


xp = NumpyFramework()
