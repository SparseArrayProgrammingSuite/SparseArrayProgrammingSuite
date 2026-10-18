from abc import ABC, abstractmethod

from saps_framework.einsum import nary_ops, reduction_ops
from saps_framework.unfold import normalize_pad_width, normalize_unfold_args


class Framework(ABC):
    # Benchmark Format -> Eager Tensor
    @abstractmethod
    def from_binsparse(self, array):
        pass

    # Eager Tensor -> Benchmark Format
    @abstractmethod
    def to_binsparse(self, array):
        pass

    # Eager Tensor -> Lazy Tensor
    @abstractmethod
    def lazy(self, array):
        pass

    # Lazy Tensor -> Eager Tensor
    @abstractmethod
    def compute(self, array):
        pass

    def compile(self, func):
        return func

    @abstractmethod
    def einsum(self, prgm, **kwargs):
        pass

    def pad_constant(self, x, padding, *, axes=None, fill_value=0):
        """Pad ``axes`` of x (default: all axes) with ``fill_value``.

        ``padding`` is an int, one int or ``(before, after)`` pair per axis, or
        ONNX-style ``(begin_0, ..., begin_n, end_0, ..., end_n)``.
        """
        pad_width = normalize_pad_width(x.ndim, padding, axes)
        for axis, (before, after) in enumerate(pad_width):
            if before or after:
                shape = list(x.shape)
                shape[axis] = before
                head = self.full(tuple(shape), fill_value, dtype=x.dtype)
                shape[axis] = after
                tail = self.full(tuple(shape), fill_value, dtype=x.dtype)
                x = self.concat([head, x, tail], axis=axis)
        return x

    def pad_reflect(self, x, padding, *, axes=None):
        """Pad by mirroring about the edge, which isn't repeated: 2 1|0 1 2 3|2 1."""
        pad_width = normalize_pad_width(x.ndim, padding, axes)
        for axis, (before, after) in enumerate(pad_width):
            if before or after:
                size = x.shape[axis]
                inner = [slice(None)] * x.ndim
                inner[axis] = slice(1, size - 1)
                # The mirror image repeats 0 1 2 3 2 1 0 1 2 3 2 1 ...
                period = self.concat(
                    [x, self.flip(x[tuple(inner)], axis=axis)], axis=axis
                )
                x = self._periodic_window(period, axis, -before, size + after)
        return x

    def pad_edge(self, x, padding, *, axes=None):
        """Pad by repeating the edge value: 0 0|0 1 2 3|3 3."""
        pad_width = normalize_pad_width(x.ndim, padding, axes)
        for axis, (before, after) in enumerate(pad_width):
            if before or after:
                size = x.shape[axis]
                if size == 0:
                    raise ValueError("can't pad an empty axis with its edge values")
                first = [slice(None)] * x.ndim
                first[axis] = slice(0, 1)
                last = [slice(None)] * x.ndim
                last[axis] = slice(size - 1, size)
                x = self.concat(
                    [x[tuple(first)]] * before + [x] + [x[tuple(last)]] * after,
                    axis=axis,
                )
        return x

    def pad_wrap(self, x, padding, *, axes=None):
        """Pad periodically: 2 3|0 1 2 3|0 1."""
        pad_width = normalize_pad_width(x.ndim, padding, axes)
        for axis, (before, after) in enumerate(pad_width):
            if before or after:
                x = self._periodic_window(x, axis, -before, x.shape[axis] + after)
        return x

    def _periodic_window(self, x, axis, start, stop):
        """Return positions start:stop along ``axis`` of x repeated end to end."""
        period = x.shape[axis]
        if period == 0:
            raise ValueError("can't pad an empty axis with values from it")
        first = start // period
        last = -(-stop // period)
        index = [slice(None)] * x.ndim
        index[axis] = slice(start - first * period, stop - first * period)
        return self.concat([x] * (last - first), axis=axis)[tuple(index)]

    def _pad(self, x, padding, axes, padding_mode, fill_value):
        if padding_mode == "constant":
            return self.pad_constant(x, padding, axes=axes, fill_value=fill_value)
        if padding_mode == "reflect":
            return self.pad_reflect(x, padding, axes=axes)
        if padding_mode == "edge":
            return self.pad_edge(x, padding, axes=axes)
        if padding_mode == "wrap":
            return self.pad_wrap(x, padding, axes=axes)
        raise ValueError(
            "padding_mode must be 'constant', 'reflect', 'edge' or 'wrap', "
            f"got {padding_mode!r}"
        )

    @abstractmethod
    def unfold(self, x, kernel_shape, *, axes=None, strides=None, dilations=None):
        """Return the sliding windows of x.

        Each of ``axes`` (default: the last ``len(kernel_shape)``) is replaced by
        its number of windows and the kernel axes are appended, so that
        ``out[..., i, ..., j] = x[..., i * stride + j * dilation, ...]``.
        """

    def padded_unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        padding_mode="constant",
        fill_value=0,
    ):
        """Pad ``axes`` of x, then unfold them.

        ``padding_mode`` is "constant" (with ``fill_value``), "reflect", "edge"
        or "wrap". Backends override this to avoid building the padded input.
        """
        axes_t, strides_t, dilations_t = normalize_unfold_args(
            x.ndim, kernel_shape, axes, strides, dilations
        )
        x = self._pad(x, padding, axes_t, padding_mode, fill_value)
        return self.unfold(
            x, kernel_shape, axes=axes_t, strides=strides_t, dilations=dilations_t
        )

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
        """Convolve x of shape (N, C, *spatial) with w of shape (M, C / group, *kernel).

        Returns shape (N, M, *windows), where with one group

            Y[n, m, s] reduce= P[n, c, s, k] combine W[m, c, k]

        and P is ``padded_unfold`` of x's spatial axes. ``combine`` and
        ``reduce`` are einsum operators, e.g. combine="+", reduce="max".
        """
        batch, channels, *spatial_shape = x.shape
        out_channels, group_channels, *kernel_shape = w.shape
        rank = len(kernel_shape)
        if len(spatial_shape) != rank:
            raise ValueError("x and w must have the same number of spatial axes")
        if channels != group * group_channels or out_channels % group != 0:
            raise ValueError(
                f"conv with group={group} can't apply weights of shape {w.shape} "
                f"to input of shape {x.shape}"
            )
        combine_op = getattr(self, nary_ops[combine])
        reduce_op = getattr(self, reduction_ops[reduce])
        window_args = {
            "strides": strides,
            "dilations": dilations,
            "padding": padding,
            "padding_mode": padding_mode,
            "fill_value": fill_value,
        }

        if (
            group == 1
            and nary_ops[combine] == "multiply"
            and reduction_ops[reduce] == "sum"
        ):
            # Contract (C, *kernel) without materializing every product.
            patches = self.padded_unfold(
                x, kernel_shape, axes=tuple(range(2, 2 + rank)), **window_args
            )
            y = self.tensordot(
                patches,
                w,
                axes=(
                    (1, *range(2 + rank, 2 + 2 * rank)),
                    (1, *range(2, 2 + rank)),
                ),
            )
            return self.permute_dims(y, (0, rank + 1, *range(1, rank + 1)))

        x = self.reshape(x, (batch, group, group_channels, *spatial_shape))
        patches = self.padded_unfold(
            x, kernel_shape, axes=tuple(range(3, 3 + rank)), **window_args
        )
        # Broadcast (N, G, 1, C, *windows, *kernel) against
        # (1, G, M, C, *ones, *kernel), then reduce C and the kernel.
        patches = self.reshape(patches, (batch, group, 1, *patches.shape[2:]))
        w = self.reshape(
            w,
            (
                1,
                group,
                out_channels // group,
                group_channels,
                *((1,) * rank),
                *kernel_shape,
            ),
        )
        y = reduce_op(combine_op(patches, w), axis=(3, *range(4 + rank, 4 + 2 * rank)))
        return self.reshape(y, (batch, out_channels, *y.shape[3:]))

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
        """Reduce each window of x of shape (N, C, *spatial) to shape (N, C, *windows).

        ``reduce`` is an einsum reduction such as "max", "+" or "mean". Padding
        is reduced like any other value, so max pooling usually wants a
        ``fill_value`` below every input.
        """
        if x.ndim != len(kernel_shape) + 2:
            raise ValueError(
                f"pool over kernel_shape {tuple(kernel_shape)} needs an input of "
                f"rank {len(kernel_shape) + 2}, got rank {x.ndim}"
            )
        reduce_op = getattr(self, reduction_ops[reduce])
        patches = self.padded_unfold(
            x,
            kernel_shape,
            axes=tuple(range(2, x.ndim)),
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )
        return reduce_op(patches, axis=tuple(range(x.ndim, patches.ndim)))

    @abstractmethod
    def replace(self, arr, old, new):
        """Return an array with every occurrence of scalar old replaced by new.

        Includes implicit sparse values and treats NaN as matching NaN.
        The input is not modified.
        """

    @abstractmethod
    def __getattr__(self, name):
        pass
