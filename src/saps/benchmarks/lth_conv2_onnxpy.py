"""Generated Conv-2 ONNXPY benchmark wrapper for SAPS.

Regenerate from the ONNXPY checkout with:
    poetry run python scripts/print_conv2lth_numpy.py --write-saps-benchmark
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import gdown
import onnx
from binsparse.conversions import from_numpy, to_numpy
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)

_MODEL_ENV = "LTH_CONV2_ONNX"
_MODEL_FILE_NAME = "conv2_pruned_dense.onnx"
_MODEL_DATA_FILE_NAME = "conv2_pruned_dense.onnx.data"

_MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1jxF9fFXidr9h_p6fEQGiRQXZkqkYfn14"
)
_MODEL_DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1FB_CE91DlFbrPWWT7bQ-ETGRXlzIqEcq"
)


model_inputs = (
    "input",
    "layers.0.conv1.weight",
    "layers.0.conv1.bias",
    "layers.0.conv2.weight",
    "layers.0.conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
    "val_4",
)
tensor_inputs = (
    "layers.0.conv1.weight",
    "layers.0.conv1.bias",
    "layers.0.conv2.weight",
    "layers.0.conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
    "val_4",
)


_ONNX_DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    16: "bfloat16",
}


def _onnxpy_dtype_name(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {"double": "float64", "float": "float32", "bool_": "bool"}.get(
            value, value
        )
    return _ONNX_DTYPE_NAMES[int(value)]


def _onnxpy_dtype(xp, value):
    name = _onnxpy_dtype_name(value)
    if name is None:
        return None
    if hasattr(xp, name):
        return getattr(xp, name)
    raise TypeError(f"Array namespace does not define dtype {name!r}")


def _onnxpy_asarray(value, dtype, xp):
    resolved = _onnxpy_dtype(xp, dtype)
    if resolved is None:
        return xp.asarray(value)
    return xp.asarray(value, dtype=resolved)


def _onnxpy_to_python(value):
    if isinstance(value, list | tuple):
        return list(value)
    if hasattr(value, "shape"):
        if len(value.shape) == 0:
            return int(value)
        if len(value.shape) == 1:
            return [int(value[i]) for i in range(value.shape[0])]
    return value


def _onnxpy_to_tuple(value):
    py = _onnxpy_to_python(value)
    if isinstance(py, list | tuple):
        return tuple(int(item) for item in py)
    return (int(py),)


def _onnxpy_indices(shape):
    shape_t = tuple(int(dim) for dim in shape)
    if not shape_t:
        yield ()
        return
    for i in range(shape_t[0]):
        for rest in _onnxpy_indices(shape_t[1:]):
            yield (i, *rest)


def _onnxpy_device(array):
    return getattr(array, "device", None)


def _onnxpy_asarray_like(value, like, xp):
    dtype = getattr(like, "dtype", None)
    device = _onnxpy_device(like)
    try:
        return xp.asarray(value, dtype=dtype, device=device)
    except TypeError:
        return xp.asarray(value, dtype=dtype)


def _onnxpy_zeros(shape, like, xp):
    dtype = getattr(like, "dtype", None)
    device = _onnxpy_device(like)
    try:
        return xp.zeros(tuple(int(dim) for dim in shape), dtype=dtype, device=device)
    except TypeError:
        return xp.zeros(tuple(int(dim) for dim in shape), dtype=dtype)


def _onnxpy_ones(shape, like, xp):
    dtype = getattr(like, "dtype", None)
    device = _onnxpy_device(like)
    try:
        return xp.ones(tuple(int(dim) for dim in shape), dtype=dtype, device=device)
    except TypeError:
        return xp.ones(tuple(int(dim) for dim in shape), dtype=dtype)


def _onnxpy_min_value(like, xp):
    try:
        if xp.isdtype(like.dtype, "integral"):
            return _onnxpy_asarray_like(xp.iinfo(like.dtype).min, like, xp)
    except (AttributeError, TypeError):
        pass
    return _onnxpy_asarray_like(float("-inf"), like, xp)


def _onnxpy_stack_nested(values, xp):
    if isinstance(values, list):
        return xp.stack(
            tuple(_onnxpy_stack_nested(value, xp) for value in values), axis=0
        )
    return values


def _onnxpy_unfold(
    x, kernel_shape, *, axes, strides, dilations, padding, fill_value=0, xp
):
    if hasattr(xp, "unfold"):
        return xp.unfold(
            x,
            kernel_shape,
            axes=axes,
            strides=strides,
            dilations=dilations,
            padding=padding,
            fill_value=fill_value,
        )

    rank = len(kernel_shape)
    output_spatial = _onnxpy_output_spatial(
        [x.shape[axis] for axis in axes],
        kernel_shape,
        strides,
        dilations,
        padding,
        0,
    )

    def element(base_index, out_index, kernel_index):
        input_index = list(base_index)
        for i, axis in enumerate(axes):
            source = int(out_index[i]) * int(strides[i])
            source += int(kernel_index[i]) * int(dilations[i])
            source -= int(padding[i][0])
            if source < 0 or source >= int(x.shape[axis]):
                return fill_value
            input_index[axis] = source
        return x[tuple(input_index)]

    base_shape = tuple(
        int(dim) for axis, dim in enumerate(x.shape) if axis not in set(axes)
    )
    original_axis_order = tuple(axis for axis in range(x.ndim) if axis not in set(axes))

    def window(base_index, out_index, kernel_index_prefix=()):
        if len(kernel_index_prefix) == rank:
            full_base = [0] * x.ndim
            for i, axis in enumerate(original_axis_order):
                full_base[axis] = base_index[i]
            return element(tuple(full_base), out_index, kernel_index_prefix)
        axis = len(kernel_index_prefix)
        return [
            window(base_index, out_index, (*kernel_index_prefix, i))
            for i in range(kernel_shape[axis])
        ]

    def output_at(base_index, out_index_prefix=()):
        if len(out_index_prefix) == rank:
            return window(base_index, out_index_prefix)
        axis = len(out_index_prefix)
        return [
            output_at(base_index, (*out_index_prefix, i))
            for i in range(output_spatial[axis])
        ]

    def base_output(base_index_prefix=()):
        if len(base_index_prefix) == len(base_shape):
            return output_at(base_index_prefix)
        axis = len(base_index_prefix)
        return [base_output((*base_index_prefix, i)) for i in range(base_shape[axis])]

    nested = base_output()
    result = _onnxpy_stack_nested(nested, xp)

    current_axes = list(original_axis_order) + [*axes] + [*range(x.ndim, x.ndim + rank)]
    desired_axes = [*range(x.ndim), *range(x.ndim, x.ndim + rank)]
    perm = tuple(current_axes.index(axis) for axis in desired_axes)
    return xp.permute_dims(result, perm)


def _onnxpy_resolve_pads(
    input_spatial, kernel_shape, strides, dilations, auto_pad, pads
):
    rank = len(kernel_shape)
    mode = (auto_pad or "NOTSET").upper()
    if mode == "NOTSET":
        if pads is None:
            return tuple((0, 0) for _ in range(rank))
        flat = list(pads)
        return tuple((int(flat[i]), int(flat[i + rank])) for i in range(rank))
    if mode == "VALID":
        return tuple((0, 0) for _ in range(rank))
    pairs = []
    for i in range(rank):
        in_size = int(input_spatial[i])
        s = int(strides[i])
        d = int(dilations[i])
        k = int(kernel_shape[i])
        out_size = -(-in_size // s)
        eff_k = (k - 1) * d + 1
        total = max((out_size - 1) * s + eff_k - in_size, 0)
        if mode == "SAME_UPPER":
            pairs.append((total // 2, total - total // 2))
        else:
            pairs.append((total - total // 2, total // 2))
    return tuple(pairs)


def _onnxpy_output_spatial(
    input_spatial, kernel_shape, strides, dilations, pad_pairs, ceil_mode
):
    rank = len(kernel_shape)
    output = []
    for i in range(rank):
        in_size = int(input_spatial[i])
        k = int(kernel_shape[i])
        s = int(strides[i])
        d = int(dilations[i])
        pad_before, pad_after = pad_pairs[i]
        effective_kernel = (k - 1) * d + 1
        numerator = in_size + pad_before + pad_after - effective_kernel
        if int(ceil_mode):
            out_size = numerator // s + 1
            if numerator % s:
                out_size += 1
        else:
            out_size = numerator // s + 1
        output.append(max(out_size, 0))
    return tuple(output)


def _onnxpy_pads_for_output(
    input_spatial, kernel_shape, strides, dilations, pad_pairs, output_spatial
):
    expanded = []
    for i, out_size in enumerate(output_spatial):
        in_size = int(input_spatial[i])
        k = int(kernel_shape[i])
        s = int(strides[i])
        d = int(dilations[i])
        before, after = pad_pairs[i]
        effective_kernel = (k - 1) * d + 1
        needed = max((int(out_size) - 1) * s + effective_kernel - in_size, 0)
        existing = before + after
        expanded.append((before, after + max(needed - existing, 0)))
    return tuple(expanded)


def _onnxpy_spatial_in_bounds(index, spatial_shape):
    return all(
        0 <= int(i) < int(limit) for i, limit in zip(index, spatial_shape, strict=True)
    )


def _onnxpy_conv(
    x,
    w,
    b=None,
    *,
    kernel_shape=None,
    strides=None,
    pads=None,
    dilations=None,
    group=1,
    auto_pad="NOTSET",
    xp,
):
    rank = x.ndim - 2
    kernel = tuple(int(k) for k in (kernel_shape or w.shape[2:]))
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = tuple(
        int(d) for d in (dilations if dilations is not None else [1] * rank)
    )
    group_i = int(group)
    pad_pairs = _onnxpy_resolve_pads(
        x.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    out_spatial = _onnxpy_output_spatial(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, 0
    )
    pad_pairs = _onnxpy_pads_for_output(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, out_spatial
    )
    c_per_group = x.shape[1] // group_i
    m_out = w.shape[0]
    m_per_group = m_out // group_i

    kernel_size = 1
    for size in kernel:
        kernel_size *= int(size)

    patches = _onnxpy_unfold(
        x,
        kernel,
        axes=tuple(range(2, x.ndim)),
        strides=strides_t,
        dilations=dilations_t,
        padding=pad_pairs,
        fill_value=_onnxpy_zeros((), x, xp),
        xp=xp,
    )
    spatial_axes = tuple(range(2, 2 + rank))
    kernel_axes = tuple(range(2 + rank, 2 + 2 * rank))
    groups = []
    for group_index in range(group_i):
        channel_start = group_index * c_per_group
        channel_end = channel_start + c_per_group
        output_start = group_index * m_per_group
        output_end = output_start + m_per_group
        group_patches = patches[:, channel_start:channel_end, ...]
        group_weights = w[output_start:output_end, ...]
        cols = xp.reshape(
            xp.permute_dims(
                group_patches,
                (0, *spatial_axes, 1, *kernel_axes),
            ),
            (-1, c_per_group * kernel_size),
        )
        rows = xp.reshape(group_weights, (m_per_group, c_per_group * kernel_size))
        group_out = cols @ xp.permute_dims(rows, (1, 0))
        group_out = xp.reshape(group_out, (x.shape[0], *out_spatial, m_per_group))
        groups.append(xp.permute_dims(group_out, (0, rank + 1, *range(1, rank + 1))))

    out = groups[0] if len(groups) == 1 else xp.concat(tuple(groups), axis=1)
    if b is not None:
        out = out + xp.reshape(b, (1, m_out, *((1,) * rank)))
    return out


def _onnxpy_max_pool(
    x,
    *,
    kernel_shape,
    strides=None,
    pads=None,
    dilations=None,
    ceil_mode=0,
    auto_pad="NOTSET",
    xp,
):
    rank = x.ndim - 2
    kernel = tuple(int(k) for k in kernel_shape)
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = tuple(
        int(d) for d in (dilations if dilations is not None else [1] * rank)
    )
    pad_pairs = _onnxpy_resolve_pads(
        x.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    out_spatial = _onnxpy_output_spatial(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, ceil_mode
    )
    pad_pairs = _onnxpy_pads_for_output(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, out_spatial
    )
    patches = _onnxpy_unfold(
        x,
        kernel,
        axes=tuple(range(2, x.ndim)),
        strides=strides_t,
        dilations=dilations_t,
        padding=pad_pairs,
        fill_value=_onnxpy_min_value(x, xp),
        xp=xp,
    )
    return xp.max(patches, axis=tuple(range(2 + rank, 2 + 2 * rank)))


def _onnxpy_avg_pool(
    x,
    *,
    kernel_shape,
    strides=None,
    pads=None,
    ceil_mode=0,
    auto_pad="NOTSET",
    count_include_pad=0,
    xp,
):
    rank = x.ndim - 2
    kernel = tuple(int(k) for k in kernel_shape)
    strides_t = tuple(int(s) for s in (strides if strides is not None else [1] * rank))
    dilations_t = (1,) * rank
    pad_pairs = _onnxpy_resolve_pads(
        x.shape[2:], kernel, strides_t, dilations_t, auto_pad, pads
    )
    out_spatial = _onnxpy_output_spatial(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, ceil_mode
    )
    pad_pairs = _onnxpy_pads_for_output(
        x.shape[2:], kernel, strides_t, dilations_t, pad_pairs, out_spatial
    )
    patches = _onnxpy_unfold(
        x,
        kernel,
        axes=tuple(range(2, x.ndim)),
        strides=strides_t,
        dilations=dilations_t,
        padding=pad_pairs,
        fill_value=_onnxpy_zeros((), x, xp),
        xp=xp,
    )
    kernel_axes = tuple(range(2 + rank, 2 + 2 * rank))
    if int(count_include_pad):
        return xp.mean(patches, axis=kernel_axes)

    mask = _onnxpy_ones(x.shape, x, xp)
    counts = xp.sum(
        _onnxpy_unfold(
            mask,
            kernel,
            axes=tuple(range(2, x.ndim)),
            strides=strides_t,
            dilations=dilations_t,
            padding=pad_pairs,
            fill_value=_onnxpy_zeros((), x, xp),
            xp=xp,
        ),
        axis=kernel_axes,
    )
    counts = xp.maximum(counts, _onnxpy_asarray_like(1, counts, xp))
    return xp.sum(patches, axis=kernel_axes) / counts


def _onnxpy_global_avg_pool(x, *, xp):
    return xp.mean(x, axis=tuple(range(2, x.ndim)), keepdims=True)


def _onnxpy_softmax(x, *, axis=-1, xp):
    a = int(axis)
    if a < 0:
        a += x.ndim
    shifted = x - xp.max(x, axis=a, keepdims=True)
    e = xp.exp(shifted)
    return e / xp.sum(e, axis=a, keepdims=True)


def _onnxpy_log_softmax(x, *, axis=-1, xp):
    a = int(axis)
    if a < 0:
        a += x.ndim
    shifted = x - xp.max(x, axis=a, keepdims=True)
    return shifted - xp.log(xp.sum(xp.exp(shifted), axis=a, keepdims=True))


def _onnxpy_flatten(x, *, axis=1, xp):
    a = int(axis)
    if a < 0:
        a += x.ndim
    if a == 0:
        return xp.reshape(x, (1, -1))
    leading = 1
    for d in x.shape[:a]:
        leading *= int(d)
    return xp.reshape(x, (leading, -1))


def _onnxpy_reshape(data, shape, *, allow_zero=0, xp):
    shape_py = _onnxpy_to_python(shape)
    if not isinstance(shape_py, list | tuple):
        shape_py = [int(shape_py)]
    target = []
    in_shape = tuple(int(d) for d in data.shape)
    for i, dim in enumerate(shape_py):
        d = int(dim)
        if d == 0 and not int(allow_zero):
            target.append(in_shape[i])
        else:
            target.append(d)
    return xp.reshape(data, tuple(target))


def _onnxpy_transpose(x, *, perm=None, xp):
    if perm is None:
        perm = tuple(range(x.ndim - 1, -1, -1))
    return xp.permute_dims(x, tuple(int(p) for p in perm))


def _onnxpy_concat(values, *, axis=0, xp):
    return xp.concat(tuple(values), axis=int(axis))


def _onnxpy_gemm(a, b, c=None, *, alpha=1.0, beta=1.0, trans_a=0, trans_b=0, xp):
    a2 = xp.permute_dims(a, (1, 0)) if int(trans_a) else a
    b2 = xp.permute_dims(b, (1, 0)) if int(trans_b) else b
    out = float(alpha) * (a2 @ b2)
    if c is not None:
        out = out + float(beta) * c
    return out


def _onnxpy_cast(x, *, to, xp):
    dtype = _onnxpy_dtype(xp, to)
    try:
        return xp.astype(x, dtype, copy=False)
    except TypeError:
        return xp.astype(x, dtype)


def model(
    input,
    layers_0_conv1_weight,
    layers_0_conv1_bias,
    layers_0_conv2_weight,
    layers_0_conv2_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    fc3_weight,
    fc3_bias,
    val_4,
    *,
    xp=np,
):
    conv2d = _onnxpy_conv(
        input,
        layers_0_conv1_weight,
        layers_0_conv1_bias,
        kernel_shape=None,
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        group=1,
        auto_pad="NOTSET",
        xp=xp,
    )
    relu = xp.maximum(conv2d, 0)
    conv2d_1 = _onnxpy_conv(
        relu,
        layers_0_conv2_weight,
        layers_0_conv2_bias,
        kernel_shape=None,
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        group=1,
        auto_pad="NOTSET",
        xp=xp,
    )
    relu_1 = xp.maximum(conv2d_1, 0)
    max_pool2d = _onnxpy_max_pool(
        relu_1,
        kernel_shape=(2, 2),
        strides=(2, 2),
        pads=(0, 0, 0, 0),
        dilations=(1, 1),
        ceil_mode=0,
        auto_pad="NOTSET",
        xp=xp,
    )
    view = _onnxpy_reshape(max_pool2d, val_4, allow_zero=1, xp=xp)
    linear = _onnxpy_gemm(
        view, fc1_weight, fc1_bias, alpha=1.0, beta=1.0, trans_a=0, trans_b=1, xp=xp
    )
    relu_2 = xp.maximum(linear, 0)
    linear_1 = _onnxpy_gemm(
        relu_2, fc2_weight, fc2_bias, alpha=1.0, beta=1.0, trans_a=0, trans_b=1, xp=xp
    )
    relu_3 = xp.maximum(linear_1, 0)
    return _onnxpy_gemm(
        relu_3, fc3_weight, fc3_bias, alpha=1.0, beta=1.0, trans_a=0, trans_b=1, xp=xp
    )


def _references() -> list[Ref]:
    return [
        Ref(
            title=(
                "The Lottery Ticket Hypothesis: Finding Sparse, "
                "Trainable Neural Networks"
            ),
            authors=[Author("Jonathan Frankle"), Author("Michael Carbin")],
            conference="ICLR",
            year=2019,
        ),
        Ref(
            title="Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask",
            authors=[
                Author("Hattie Zhou"),
                Author("Janice Lan"),
                Author("Rosanne Liu"),
                Author("Jason Yosinski"),
            ],
            conference="NeurIPS",
            year=2019,
        ),
    ]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "lth"


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if partial.exists():
        partial.unlink()

    result = gdown.download(url=url, output=str(partial), quiet=False)

    if result is None or not partial.is_file() or partial.stat().st_size == 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"Failed to download LTH model artifact from {url}")

    partial.replace(destination)


def _model_path() -> Path:
    configured = os.environ.get(_MODEL_ENV)

    if configured:
        model_path = Path(configured).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"{_MODEL_ENV} does not point to a file: {model_path}"
            )

        data_path = model_path.with_name(_MODEL_DATA_FILE_NAME)
        if not data_path.is_file():
            raise FileNotFoundError(f"ONNX external-data file not found: {data_path}")

        return model_path

    root = _default_data_dir()
    model_path = root / _MODEL_FILE_NAME
    data_path = root / _MODEL_DATA_FILE_NAME

    _download_if_missing(_MODEL_URL, model_path)
    _download_if_missing(_MODEL_DATA_URL, data_path)

    return model_path.resolve()


class LTHConv2Dataset(Dataset):
    @property
    def name(self) -> str:
        return "conv2_pruned"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 Pruned Model"

    @property
    def description(self) -> str:
        return "Pruned CIFAR-10 Conv-2 Lottery Ticket model exported to ONNX."

    @property
    def suites(self) -> list[str]:
        return ["lth"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class LTHConv2ONNXPYGenerator(Generator[LTHConv2Dataset]):
    @property
    def name(self) -> str:
        return "lth_conv2_onnxpy_inputs"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 ONNXPY Inputs"

    @property
    def description(self) -> str:
        return "Generates a deterministic runtime input for the Conv-2 model."

    @property
    def suites(self) -> list[str]:
        return ["lth"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ramya Polaki", "rpolaki3@gatech.edu"),
            Contributor("Michael Wang", "mwang764@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return _references()

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist with adapting existing benchmark "
            "integration code to the current SAPS and ONNXPY APIs. The model "
            "training, pruning workflow, benchmark objective, and model artifacts "
            "were created by the contributors."
        )

    @property
    def motivation(self) -> str:
        return (
            "Lottery-ticket pruning produces neural-network parameters containing "
            "many exact zeros. This benchmark establishes the complete "
            "ONNX-to-ONNXPY-to-SAPS inference path for correctness and timing."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LTHConv2Dataset]:
        return [LTHConv2Dataset()]

    def generate(self, _dataset: LTHConv2Dataset) -> DataInstance:
        model = onnx.load(str(_model_path()), load_external_data=True)

        initializer_names = {tensor.name for tensor in model.graph.initializer}
        real_inputs = [
            value_info
            for value_info in model.graph.input
            if value_info.name not in initializer_names
        ]

        if len(real_inputs) != 1:
            raise ValueError(f"Expected one runtime input, found {len(real_inputs)}.")

        input_info = real_inputs[0]

        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise ValueError(f"Input {input_info.name!r} must have a static shape.")
            shape.append(int(dim.dim_value))

        dtype = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(input_info.type.tensor_type.elem_type)
        )

        rng = np.random.default_rng(0)
        model_input = rng.standard_normal(tuple(shape)).astype(dtype)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        missing = [name for name in tensor_inputs if name not in initializers]
        if missing:
            raise ValueError(
                "Expected ONNX initializers not found: " + ", ".join(missing)
            )

        return DataInstance(
            inputs=[
                from_numpy(model_input),
                *(
                    from_numpy(np.asarray(numpy_helper.to_array(initializers[name])))
                    for name in tensor_inputs
                ),
            ],
            meta={
                "onnx_input_name": input_info.name,
                "onnx_initializer_names": tensor_inputs,
            },
        )


class LTHConv2ONNXPYBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "lth_conv2_onnxpy"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 via ONNXPY"

    @property
    def description(self) -> str:
        return "Runs the pruned Conv-2 ONNX graph through ONNXPY-generated Python."

    @property
    def suites(self) -> list[str]:
        return ["lth"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ramya Polaki", "rpolaki3@gatech.edu"),
            Contributor("Michael Wang", "mwang764@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return _references()

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist with adapting existing benchmark "
            "integration code to the current SAPS and ONNXPY APIs. The model "
            "training, pruning workflow, benchmark objective, and model artifacts "
            "were created by the contributors."
        )

    @property
    def motivation(self) -> str:
        return (
            "The ONNX model remains the source of truth and ONNXPY translates "
            "the complete graph instead of manually reimplementing Conv-2."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [LTHConv2ONNXPYGenerator()]

    def setup(self, param, *, use_cache: bool = True, xp=None):
        model_path = _model_path()

        super().setup(param, use_cache=use_cache, xp=xp)

        dense_input = to_numpy(self._input[0])
        input_name = self._meta["onnx_input_name"]

        model = onnx.load(str(model_path), load_external_data=True)
        outputs = ReferenceEvaluator(model).run(
            None,
            {input_name: dense_input},
        )
        if isinstance(outputs, dict):
            raise TypeError("Expected ReferenceEvaluator outputs as a list")
        expected = outputs[0]
        self._ref_outputs = [from_numpy(np.asarray(expected))]
        self._ref_meta = {
            "rtol": 1e-4,
            "atol": 1e-4,
        }

    def benchmark(
        self,
        xp,
        data: list[Any],
        meta: dict[str, Any],
    ):
        return [model(*data, xp=xp)]

    def check(self, param):
        actual = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=self._ref_meta["rtol"],
            atol=self._ref_meta["atol"],
        )

    def teardown(self, param):
        super().teardown(param)
