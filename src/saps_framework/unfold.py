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
    padding: Sequence[int | Sequence[int]] | int | None,
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]
]:
    kernel = tuple(int(size) for size in kernel_shape)
    rank = len(kernel)
    if rank == 0:
        raise ValueError("kernel_shape must contain at least one dimension")
    if any(size <= 0 for size in kernel):
        raise ValueError("kernel_shape values must be positive")

    axes_t = _normalize_axes(ndim, axes, rank)
    strides_t = _normalize_axis_values(strides, rank, default=1, name="strides")
    dilations_t = _normalize_axis_values(dilations, rank, default=1, name="dilations")
    padding_t = _normalize_padding(padding, rank)

    if any(step <= 0 for step in strides_t):
        raise ValueError("strides values must be positive")
    if any(dilation <= 0 for dilation in dilations_t):
        raise ValueError("dilations values must be positive")
    if any(before < 0 or after < 0 for before, after in padding_t):
        raise ValueError("padding values must be non-negative")

    return axes_t, strides_t, dilations_t, padding_t


def unfold_output_shape(
    input_shape: Sequence[int],
    kernel_shape: Sequence[int],
    axes: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    output = [int(dim) for dim in input_shape]
    for axis, kernel, stride, dilation, pad_pair in zip(
        axes,
        kernel_shape,
        strides,
        dilations,
        padding,
        strict=True,
    ):
        effective_kernel = (int(kernel) - 1) * int(dilation) + 1
        padded = int(input_shape[axis]) + int(pad_pair[0]) + int(pad_pair[1])
        output[axis] = max((padded - effective_kernel) // int(stride) + 1, 0)
    return (*output, *(int(size) for size in kernel_shape))


def _normalize_axes(
    ndim: int,
    axes: Sequence[int] | int | None,
    rank: int,
) -> tuple[int, ...]:
    if axes is None:
        axes_t = tuple(range(ndim - rank, ndim))
    elif isinstance(axes, int):
        axes_t = (axes,)
    else:
        axes_t = tuple(int(axis) for axis in axes)

    if len(axes_t) != rank:
        raise ValueError("axes must have the same length as kernel_shape")

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
