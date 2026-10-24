from math import prod

import numpy as np
from scipy import ndimage, signal

from binsparse import (
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, to_numpy, to_scipy, to_sparse

from saps_framework import Framework, normalize_pad_width, normalize_unfold_args
from saps_framework.einsum import (
    Access,
    Call,
    Einsum,
    nary_ops,
    parse_einsum,
    reduction_ops,
)


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
            case CSRMatrix():  # also matches CSCMatrix, a CSRMatrix subclass
                # to_sparse (pydata/sparse) only reads COO; go through SciPy,
                # which reads CSR/CSC/COO, instead of densifying via COO.
                return to_scipy(array).toarray()
            case _:
                return np.asarray(to_sparse(array).todense())

    def to_binsparse(self, array):
        return from_numpy(np.asarray(array))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        match parse_einsum(prgm):
            # A two-array sum of products over floats, where every summed index
            # is in both arrays, is one batched matmul.
            case Einsum(
                Call("*", [Access(a, a_idxs), Access(b, b_idxs)]), "+", _, c_idxs
            ) if (
                np.result_type(kwargs[a], kwargs[b]).kind in "fc"
                and set(a_idxs) <= set(b_idxs) | set(c_idxs)
                and set(b_idxs) <= set(a_idxs) | set(c_idxs)
                and all(
                    kwargs[b].shape[b_idxs.index(i)] == n
                    for i, n in zip(a_idxs, kwargs[a].shape, strict=True)
                    if i in b_idxs
                )
            ):
                A, B = kwargs[a], kwargs[b]
                batch = [i for i in c_idxs if i in a_idxs and i in b_idxs]
                rows = [i for i in c_idxs if i in a_idxs and i not in b_idxs]
                cols = [i for i in c_idxs if i in b_idxs and i not in a_idxs]
                inner = [i for i in a_idxs if i not in c_idxs]
                sizes = dict(zip(a_idxs + b_idxs, A.shape + B.shape, strict=True))
                batch_shape = [sizes[i] for i in batch]
                m, k, n = (prod(sizes[i] for i in idxs) for idxs in (rows, inner, cols))
                A = np.transpose(A, [a_idxs.index(i) for i in batch + rows + inner])
                B = np.transpose(B, [b_idxs.index(i) for i in batch + inner + cols])
                C = np.matmul(
                    A.reshape(*batch_shape, m, k), B.reshape(*batch_shape, k, n)
                )
                labels = batch + rows + cols
                C = C.reshape([sizes[i] for i in labels])
                return np.transpose(C, [labels.index(i) for i in c_idxs])
            # Other sums of products go to np.einsum, the rest to the interpreter.
            case parsed:
                result = parsed.run_native(np, kwargs)
                return parsed.run(np, kwargs) if result is NotImplemented else result

    def pad_constant(self, x, pad_width, *, fill_value=0):
        pad_width = normalize_pad_width(np.ndim(x), pad_width)
        return np.pad(x, pad_width, mode="constant", constant_values=fill_value)

    def pad_reflect(self, x, pad_width):
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="reflect")

    def pad_edge(self, x, pad_width):
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="edge")

    def pad_wrap(self, x, pad_width):
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="wrap")

    def unfold(self, x, kernel_shape, *, axes=None, strides=None, dilations=None):
        array = np.asarray(x)
        kernel_t = tuple(int(size) for size in kernel_shape)
        axes_t, strides_t, dilations_t = normalize_unfold_args(
            array.ndim,
            kernel_t,
            axes,
            strides,
            dilations,
        )
        effective_kernel = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_t, dilations_t, strict=True)
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

    def conv(
        self,
        x,
        w,
        *,
        strides=None,
        dilations=None,
        padding=None,
        padding_mode="constant",
        fill_value=0,
        group=1,
        combine="*",
        reduce="+",
    ):
        if (
            nary_ops[combine] == "multiply"
            and reduction_ops[reduce] == "sum"
            and x.ndim == w.ndim > 2
            and x.shape[1] == w.shape[1] == 1
            and np.result_type(x, w).kind == "f"
        ):
            axes = tuple(range(2, x.ndim))
            _, strides_t, dilations_t = normalize_unfold_args(
                x.ndim, w.shape[2:], axes, strides, dilations
            )
            if all(step == 1 for step in strides_t + dilations_t):
                x = self._pad(x, padding, axes, padding_mode, fill_value)
                # FFTs only beat matmul when each output sums one input channel.
                return np.concatenate(
                    [
                        signal.correlate(x, f[None], mode="valid", method="fft")
                        for f in w
                    ],
                    axis=1,
                )
        return super().conv(
            x,
            w,
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
            group=group,
            combine=combine,
            reduce=reduce,
        )

    def pool(
        self,
        x,
        kernel_shape,
        *,
        reduce,
        strides=None,
        dilations=None,
        padding=None,
        padding_mode="constant",
        fill_value=0,
    ):
        filters = {
            "max": ndimage.maximum_filter,
            "min": ndimage.minimum_filter,
            "mean": ndimage.uniform_filter,
        }
        operation = reduction_ops[reduce]
        if (
            operation in filters
            and x.ndim == len(kernel_shape) + 2
            and x.dtype.kind == "f"
        ):
            axes = tuple(range(2, x.ndim))
            _, strides_t, dilations_t = normalize_unfold_args(
                x.ndim, kernel_shape, axes, strides, dilations
            )
            # The filters compute every window, so they only win at stride 1.
            if all(step == 1 for step in strides_t + dilations_t):
                x = self._pad(x, padding, axes, padding_mode, fill_value)
                y = filters[operation](x, size=(1, 1, *kernel_shape))
                # A size k filter centers each window on its (k // 2)th entry.
                return y[
                    (
                        ...,
                        *(
                            slice(k // 2, n - (k - 1 - k // 2))
                            for k, n in zip(kernel_shape, x.shape[2:], strict=True)
                        ),
                    )
                ]
        return super().pool(
            x,
            kernel_shape,
            reduce=reduce,
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )

    def replace(self, arr, old, new):
        return np.where(np.isnan(arr) if old != old else arr == old, new, arr)

    def __getattr__(self, name):
        return getattr(np, name)


xp = NumpyFramework()
