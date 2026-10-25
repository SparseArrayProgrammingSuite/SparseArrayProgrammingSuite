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
    pad_width: Sequence[Sequence[int]] | Sequence[int] | int,
) -> list[tuple[int, int]]:
    """Return a (before, after) pair for each of the ndim axes from pad_width."""
    return _normalize_pad_pairs(pad_width, ndim, "pad_width")


def normalize_unfold_padding(
    ndim: int,
    padding: Sequence[Sequence[int]] | Sequence[int] | int | None,
    axes: Sequence[int] | int,
) -> list[tuple[int, int]]:
    """Return the pad_width that pads each of ``axes`` by its pair in ``padding``."""
    axes_t = _normalize_axes(ndim, axes)
    pad_width = [(0, 0)] * ndim
    if padding is not None:
        pairs = _normalize_pad_pairs(padding, len(axes_t), "padding")
        for axis, pair in zip(axes_t, pairs, strict=True):
            pad_width[axis] = pair
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


def _normalize_pad_pairs(
    pairs: Sequence[Sequence[int]] | Sequence[int] | int,
    count: int,
    name: str,
) -> list[tuple[int, int]]:
    message = (
        f"{name} must be an int, a (before, after) pair, or one pair for each "
        f"of {count} axes, got {pairs!r}"
    )
    if isinstance(pairs, Integral):
        pairs_t = [(int(pairs), int(pairs))] * count
    elif (
        isinstance(pairs, Sequence)
        and len(pairs) == 2
        and all(isinstance(width, Integral) for width in pairs)
    ):
        before, after = cast(Sequence[int], pairs)
        pairs_t = [(int(before), int(after))] * count
    else:
        try:
            pairs_t = [
                (int(before), int(after))
                for before, after in cast(Sequence[Sequence[int]], pairs)
            ]
        except (TypeError, ValueError):
            raise ValueError(message) from None
    if len(pairs_t) != count:
        raise ValueError(message)
    if any(before < 0 or after < 0 for before, after in pairs_t):
        raise ValueError(f"{name} values must be non-negative")
    return pairs_t
