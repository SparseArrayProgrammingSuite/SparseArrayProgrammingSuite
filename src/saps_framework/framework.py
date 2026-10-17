from abc import ABC, abstractmethod

from saps_framework.unfold import normalize_pad_args, normalize_unfold_args


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

    @abstractmethod
    def pad_constant(self, x, padding, *, axes=None, fill_value=0):
        """Pad ``axes`` of x (default: all axes) with ``fill_value``.

        ``padding`` is an int, one int or ``(before, after)`` pair per axis, or
        ONNX-style ``(begin_0, ..., begin_n, end_0, ..., end_n)``.
        """

    @abstractmethod
    def pad_reflect(self, x, padding, *, axes=None):
        """Pad by mirroring about the edge, which isn't repeated: 2 1|0 1 2 3|2 1."""

    @abstractmethod
    def pad_edge(self, x, padding, *, axes=None):
        """Pad by repeating the edge value: 0 0|0 1 2 3|3 3."""

    @abstractmethod
    def pad_wrap(self, x, padding, *, axes=None):
        """Pad periodically: 2 3|0 1 2 3|0 1."""

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
        _, padding_t = normalize_pad_args(x.ndim, padding, axes_t)
        if padding_mode not in ("constant", "reflect", "edge", "wrap"):
            raise ValueError(
                "padding_mode must be 'constant', 'reflect', 'edge' or 'wrap', "
                f"got {padding_mode!r}"
            )
        if any(pair != (0, 0) for pair in padding_t):
            if padding_mode == "constant":
                x = self.pad_constant(x, padding_t, axes=axes_t, fill_value=fill_value)
            elif padding_mode == "reflect":
                x = self.pad_reflect(x, padding_t, axes=axes_t)
            elif padding_mode == "edge":
                x = self.pad_edge(x, padding_t, axes=axes_t)
            else:
                x = self.pad_wrap(x, padding_t, axes=axes_t)
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
        if len(spatial_shape) != len(kernel_shape):
            raise ValueError("x and w must have the same number of spatial axes")
        if channels != group * group_channels or out_channels % group != 0:
            raise ValueError(
                f"conv with group={group} can't apply weights of shape {w.shape} "
                f"to input of shape {x.shape}"
            )
        x = self.reshape(x, (batch, group, group_channels, *spatial_shape))
        w = self.reshape(
            w, (group, out_channels // group, group_channels, *kernel_shape)
        )
        patches = self.padded_unfold(
            x,
            kernel_shape,
            axes=tuple(range(3, x.ndim)),
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )
        windows = ",".join(f"s{i}" for i in range(len(kernel_shape)))
        kernel = ",".join(f"k{i}" for i in range(len(kernel_shape)))
        patch = f"P[n,g,c,{windows},{kernel}]"
        weight = f"W[g,m,c,{kernel}]"
        if combine.isidentifier() and combine not in ("and", "or"):
            term = f"{combine}({patch}, {weight})"
        else:
            term = f"{patch} {combine} {weight}"
        y = self.einsum(f"Y[n,g,m,{windows}] {reduce}= {term}", P=patches, W=w)
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
        windows = ",".join(f"s{i}" for i in range(len(kernel_shape)))
        kernel = ",".join(f"k{i}" for i in range(len(kernel_shape)))
        return self.einsum(f"Y[n,c,{windows}] {reduce}= P[n,c,{windows},{kernel}]", P=patches)

    @abstractmethod
    def replace(self, arr, old, new):
        """Return an array with every occurrence of scalar old replaced by new.

        Includes implicit sparse values and treats NaN as matching NaN.
        The input is not modified.
        """

    @abstractmethod
    def __getattr__(self, name):
        pass
