from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import cast


def normalize_unfold_args(
    ndim: int,
    kernel_shape: Sequence[int],
    axes: Sequence[int] | int | None,
    strides: Sequence[int] | int | None,
    dilations: Sequence[int] | int | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    kernel = tuple(int(size) for size in kernel_shape)
    rank = len(kernel)
    if rank == 0:
        raise ValueError("kernel_shape must contain at least one dimension")
    if any(size <= 0 for size in kernel):
        raise ValueError("kernel_shape values must be positive")

    if axes is None:
        axes = tuple(range(ndim - rank, ndim))
    axes_t = _normalize_axes(ndim, axes)
    if len(axes_t) != rank:
        raise ValueError("axes must have the same length as kernel_shape")
    strides_t = _normalize_axis_values(strides, rank, default=1, name="strides")
    dilations_t = _normalize_axis_values(dilations, rank, default=1, name="dilations")

    if any(step <= 0 for step in strides_t):
        raise ValueError("strides values must be positive")
    if any(dilation <= 0 for dilation in dilations_t):
        raise ValueError("dilations values must be positive")

    return axes_t, strides_t, dilations_t


def normalize_pad_width(
    ndim: int,
    padding: Sequence[int | Sequence[int]] | int | None,
    axes: Sequence[int] | int | None,
) -> list[tuple[int, int]]:
    """Return a (before, after) pair for every axis, like numpy.pad's pad_width.

    ``padding`` applies to ``axes`` (default: all axes).
    """
    axes_t = _normalize_axes(ndim, range(ndim) if axes is None else axes)
    padding_t = _normalize_padding(padding, len(axes_t))
    if any(before < 0 or after < 0 for before, after in padding_t):
        raise ValueError("padding values must be non-negative")
    pad_width = [(0, 0)] * ndim
    for axis, pad_pair in zip(axes_t, padding_t, strict=True):
        pad_width[axis] = pad_pair
    return pad_width


def unfold_output_shape(
    input_shape: Sequence[int],
    kernel_shape: Sequence[int],
    axes: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
) -> tuple[int, ...]:
    output = [int(dim) for dim in input_shape]
    for axis, kernel, stride, dilation in zip(
        axes,
        kernel_shape,
        strides,
        dilations,
        strict=True,
    ):
        effective_kernel = (int(kernel) - 1) * int(dilation) + 1
        output[axis] = max(
            (int(input_shape[axis]) - effective_kernel) // int(stride) + 1, 0
        )
    return (*output, *(int(size) for size in kernel_shape))


def _normalize_axes(ndim: int, axes: Sequence[int] | int) -> tuple[int, ...]:
    axes_t = (axes,) if isinstance(axes, int) else tuple(int(axis) for axis in axes)

    normalized = []
    for axis in axes_t:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {axis} is out of bounds for array of rank {ndim}")
        normalized.append(axis)

    if len(set(normalized)) != len(normalized):
        raise ValueError("axes values must be unique")

    return tuple(normalized)


def _normalize_axis_values(
    values: Sequence[int] | int | None,
    rank: int,
    *,
    default: int,
    name: str,
) -> tuple[int, ...]:
    if values is None:
        return (default,) * rank
    if isinstance(values, int):
        return (int(values),) * rank
    values_t = tuple(int(value) for value in values)
    if len(values_t) != rank:
        raise ValueError(f"{name} must have the same length as kernel_shape")
    return values_t


def _normalize_padding(
    padding: Sequence[int | Sequence[int]] | int | None,
    rank: int,
) -> tuple[tuple[int, int], ...]:
    if padding is None:
        return ((0, 0),) * rank
    if isinstance(padding, Integral):
        pad = int(padding)
        return ((pad, pad),) * rank

    padding_t = tuple(cast(Sequence[int | Sequence[int]], padding))
    if len(padding_t) == rank:
        pairs = []
        for value in padding_t:
            if isinstance(value, Integral):
                pad = int(value)
                pairs.append((pad, pad))
            else:
                pair = tuple(int(item) for item in cast(Sequence[int], value))
                if len(pair) != 2:
                    raise ValueError("padding pairs must contain two values")
                pairs.append(pair)
        return tuple(pairs)

    flat_padding = []
    for value in padding_t:
        if not isinstance(value, Integral):
            break
        flat_padding.append(int(value))
    else:
        if len(flat_padding) == 2 * rank:
            return tuple((flat_padding[i], flat_padding[i + rank]) for i in range(rank))

    raise ValueError(
        "padding must be an int, one value per axis, one pair per axis, "
        "or ONNX-style begin/end values"
    )
