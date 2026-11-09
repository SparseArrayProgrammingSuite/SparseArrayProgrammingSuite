"""Per-op emitters that turn an ONNX ``NodeProto`` into a Python expression.

Each emitter receives an :class:`EmitContext` (defined as a Protocol so the
emit module can supply the concrete implementation) and the ONNX node, and
returns the right-hand side of the assignment that produces the node's
single output. Multi-step operations append statements with ``ctx.emit``.
The orchestrator in ``emit.py`` is responsible for emitting
``<lhs> = <rhs>`` and managing the surrounding state.

Multi-output ops are not part of the MVP and will raise here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from onnx import NodeProto


class EmitContext(Protocol):
    """Minimal surface that op emitters depend on."""

    def name(self, ssa_name: str) -> str:
        """Map an ONNX SSA name to its Python identifier."""

    def attr(self, node: NodeProto, name: str, default: Any = None) -> Any:
        """Return the Python value of ``node``'s attribute ``name``."""

    def attr_tensor(self, node: NodeProto, name: str) -> np.ndarray | None:
        """Return the TensorProto attribute ``name`` as a numpy array, or None."""

    def literal(self, value: Any) -> str:
        """Render a Python literal as source code (quotes, tuples, etc.)."""

    def register_constant(self, hint: str, array: np.ndarray) -> str:
        """Register ``array`` as a computation input and return its identifier."""

    def temp(self, hint: str) -> str:
        """Allocate a temporary identifier separate from graph SSA names."""

    def rank(self, ssa_name: str) -> int | None:
        """Return a tensor's known rank, without requiring static dimensions."""

    def emit(self, source: str) -> None:
        """Append dedented statements to the generated model function."""


Emitter = Callable[[EmitContext, NodeProto], str]


def _require_single_output(node: NodeProto) -> None:
    if len(node.output) != 1:
        raise NotImplementedError(
            f"{node.op_type} produces {len(node.output)} outputs; "
            "multi-output ops are not in the MVP",
        )


# Pointwise ops mapped to a Python operator. Each op has 2 inputs.
_BINARY_OPERATORS = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
}


def _emit_binary(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    op = _BINARY_OPERATORS[node.op_type]
    lhs = ctx.name(node.input[0])
    rhs = ctx.name(node.input[1])
    return f"({lhs} {op} {rhs})"


def _emit_relu(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return f"xp.maximum({ctx.name(node.input[0])}, 0)"


def _emit_sigmoid(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    return f"(1 / (1 + xp.exp(-{x})))"


def _emit_tanh(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return f"xp.tanh({ctx.name(node.input[0])})"


def _emit_identity(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    return ctx.name(node.input[0])


def _emit_matmul(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    a = ctx.name(node.input[0])
    b = ctx.name(node.input[1])
    return f"({a} @ {b})"


def _emit_gemm(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    a = ctx.name(node.input[0])
    b = ctx.name(node.input[1])
    c = ctx.name(node.input[2]) if len(node.input) > 2 and node.input[2] else "None"
    alpha = float(ctx.attr(node, "alpha", 1.0))
    beta = float(ctx.attr(node, "beta", 1.0))
    trans_a = int(ctx.attr(node, "transA", 0))
    trans_b = int(ctx.attr(node, "transB", 0))
    a_expr = f"xp.permute_dims({a}, (1, 0))" if trans_a else a
    b_expr = f"xp.permute_dims({b}, (1, 0))" if trans_b else b
    result = f"({alpha!r} * ({a_expr} @ {b_expr}))"
    if c != "None":
        result = f"({result} + {beta!r} * {c})"
    return result


def _emit_reshape(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    data = ctx.name(node.input[0])
    shape = ctx.name(node.input[1])
    allow_zero = int(ctx.attr(node, "allowzero", 0))
    target = (
        f"tuple(int({data}.shape[i]) if int(dim) == 0 and not {allow_zero} "
        f"else int(dim) for i, dim in enumerate({shape}))"
    )
    return f"xp.reshape({data}, {target})"


def _emit_flatten(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", 1))
    x = ctx.name(node.input[0])
    normalized_axis = f"({axis} + {x}.ndim if {axis} < 0 else {axis})"
    leading = f"math.prod({x}.shape[:{normalized_axis}])"
    return f"xp.reshape({x}, ({leading}, -1))"


def _emit_transpose(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    perm = ctx.attr(node, "perm", None)
    perm_expr = "None" if perm is None else ctx.literal(tuple(int(p) for p in perm))
    x = ctx.name(node.input[0])
    if perm is None:
        perm_expr = f"tuple(range({x}.ndim - 1, -1, -1))"
    return f"xp.permute_dims({x}, {perm_expr})"


def _emit_concat(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", 0))
    inputs = ", ".join(ctx.name(i) for i in node.input)
    return f"xp.concat(({inputs},), axis={axis})"


def _emit_softmax(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", -1))
    x = ctx.name(node.input[0])
    normalized_axis = f"({axis} + {x}.ndim if {axis} < 0 else {axis})"
    shifted = f"({x} - xp.max({x}, axis={normalized_axis}, keepdims=True))"
    return f"(xp.exp({shifted}) / xp.sum(xp.exp({shifted}), axis={normalized_axis}, keepdims=True))"


def _emit_log_softmax(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    axis = int(ctx.attr(node, "axis", -1))
    x = ctx.name(node.input[0])
    normalized_axis = f"({axis} + {x}.ndim if {axis} < 0 else {axis})"
    shifted = f"({x} - xp.max({x}, axis={normalized_axis}, keepdims=True))"
    return f"({shifted} - xp.log(xp.sum(xp.exp({shifted}), axis={normalized_axis}, keepdims=True)))"


def _emit_global_average_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    return f"xp.mean({x}, axis=tuple(range(2, {x}.ndim)), keepdims=True)"


def _emit_spatial_setup(ctx: EmitContext, node: NodeProto, x: str, kernel: str) -> str:
    """Expand padding arithmetic at compile time so the model has no Python loops."""

    kernel_shape = ctx.attr(node, "kernel_shape", None)
    tensor_rank = ctx.rank(node.input[0])
    if tensor_rank is None and node.op_type == "Conv":
        tensor_rank = ctx.rank(node.input[1])
    if kernel_shape is not None:
        rank = len(kernel_shape)
    elif tensor_rank is not None:
        rank = tensor_rank - 2
    else:
        # Attribute lengths can establish rank even when value_info omits it.
        strides = ctx.attr(node, "strides", None)
        dilations = ctx.attr(node, "dilations", None)
        pads = ctx.attr(node, "pads", None)
        if strides is not None:
            rank = len(strides)
        elif dilations is not None:
            rank = len(dilations)
        elif pads is not None:
            rank = len(pads) // 2
        else:
            raise NotImplementedError(f"{node.op_type} requires a known spatial rank")
    if rank < 1:
        raise ValueError(f"{node.op_type} requires at least one spatial dimension")

    p = ctx.temp(node.op_type.lower())
    strides = ctx.literal(tuple(int(s) for s in ctx.attr(node, "strides", (1,) * rank)))
    dilations = ctx.literal(tuple(int(d) for d in ctx.attr(node, "dilations", (1,) * rank)))
    pads = ctx.attr(node, "pads", None)
    auto_pad = str(ctx.attr(node, "auto_pad", "NOTSET") or "NOTSET").upper()
    ctx.emit(f"""
        {p}_rank = {rank}
        {p}_kernel = {kernel}
        {p}_strides = {strides}
        {p}_dilations = {dilations}
    """)
    if auto_pad in {"NOTSET", "VALID"}:
        pairs = ((0, 0),) * rank
        if auto_pad == "NOTSET" and pads is not None:
            pairs = tuple((int(pads[i]), int(pads[i + rank])) for i in range(rank))
        ctx.emit(f"{p}_pads = {ctx.literal(pairs)}")
    elif auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        pad_exprs = []
        for i in range(rank):
            total = f"{p}_total_{i}"
            ctx.emit(f"""
                {p}_size = int({x}.shape[{2 + i}])
                {p}_stride = {p}_strides[{i}]
                {p}_effective = ({p}_kernel[{i}] - 1) * {p}_dilations[{i}] + 1
                {p}_out_size = -(-{p}_size // {p}_stride)
                {total} = max(({p}_out_size - 1) * {p}_stride + {p}_effective - {p}_size, 0)
            """)
            before, after = f"{total} // 2", f"{total} - {total} // 2"
            if auto_pad == "SAME_LOWER":
                before, after = after, before
            pad_exprs.append(f"({before}, {after})")
        ctx.emit(f"{p}_pads = ({', '.join(pad_exprs)},)")
    else:
        raise ValueError(f"Unsupported auto_pad mode: {auto_pad!r}")

    if int(ctx.attr(node, "ceil_mode", 0)):
        pad_exprs = []
        for i in range(rank):
            pair = f"{p}_expanded_{i}"
            ctx.emit(f"""
                {p}_size = int({x}.shape[{2 + i}])
                {p}_stride = {p}_strides[{i}]
                {p}_effective = ({p}_kernel[{i}] - 1) * {p}_dilations[{i}] + 1
                {p}_before, {p}_after = {p}_pads[{i}]
                {p}_numerator = {p}_size + {p}_before + {p}_after - {p}_effective
                {p}_out_size = max(-(-{p}_numerator // {p}_stride) + 1, 0)
                {p}_needed = max(({p}_out_size - 1) * {p}_stride + {p}_effective - {p}_size, 0)
                {pair} = (
                    {p}_before, {p}_after + max({p}_needed - {p}_before - {p}_after, 0)
                )
            """)
            pad_exprs.append(pair)
        ctx.emit(f"{p}_pads = ({', '.join(pad_exprs)},)")
    return p


def _pool_expr(x: str, p: str, reduce: str, fill_value: str) -> str:
    return (
        f"xp.pool({x}, {p}_kernel, reduce={reduce!r}, strides={p}_strides, "
        f"dilations={p}_dilations, padding={p}_pads, fill_value={fill_value})"
    )


def _emit_array_like(ctx: EmitContext, value: str, like: str) -> str:
    result = ctx.temp("scalar")
    ctx.emit(f"""
        try:
            {result} = xp.asarray(
                {value}, dtype={like}.dtype, device=getattr({like}, "device", None)
            )
        except TypeError:
            {result} = xp.asarray({value}, dtype={like}.dtype)
    """)
    return result


def _emit_average_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    kernel = _list_attr_expr(ctx, node, "kernel_shape")
    p = _emit_spatial_setup(ctx, node, x, kernel)
    if int(ctx.attr(node, "count_include_pad", 0)):
        return _pool_expr(x, p, "mean", "0")

    # Average only the input entries in each window: sum them, then divide by
    # how many there are.
    ctx.emit(f"""
        try:
            {p}_mask = xp.ones({x}.shape, dtype={x}.dtype, device=getattr({x}, "device", None))
        except TypeError:
            {p}_mask = xp.ones({x}.shape, dtype={x}.dtype)
        {p}_counts = {_pool_expr(f"{p}_mask", p, "+", "0")}
    """)
    one = _emit_array_like(ctx, "1", f"{p}_counts")
    ctx.emit(f"{p}_counts = xp.maximum({p}_counts, {one})")
    return f"{_pool_expr(x, p, '+', '0')} / {p}_counts"


def _emit_max_pool(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    kernel = _list_attr_expr(ctx, node, "kernel_shape")
    p = _emit_spatial_setup(ctx, node, x, kernel)
    ctx.emit(f"""
        {p}_minimum = float("-inf")
        try:
            if xp.isdtype({x}.dtype, "integral"):
                {p}_minimum = xp.iinfo({x}.dtype).min
        except (AttributeError, TypeError):
            pass
    """)
    return _pool_expr(x, p, "max", f"{p}_minimum")


def _emit_conv(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    x = ctx.name(node.input[0])
    w = ctx.name(node.input[1])
    kernel = _list_attr_expr(ctx, node, "kernel_shape")
    if kernel == "None":
        kernel = f"{w}.shape[2:]"
    group = int(ctx.attr(node, "group", 1))
    if group < 1:
        raise ValueError("Conv group must be positive")
    p = _emit_spatial_setup(ctx, node, x, kernel)
    result = (
        f"xp.conv({x}, {w}, strides={p}_strides, dilations={p}_dilations, "
        f"padding={p}_pads, group={group})"
    )
    if len(node.input) > 2 and node.input[2]:
        b = ctx.name(node.input[2])
        result = f"({result} + xp.reshape({b}, (1, {w}.shape[0], *((1,) * {p}_rank))))"
    return result


def _emit_cast(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    to = int(ctx.attr(node, "to"))
    dtype_names = {
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
    try:
        dtype_name = dtype_names[to]
    except KeyError as exc:
        raise NotImplementedError(f"ONNX Cast dtype {to} is not supported") from exc
    return f"xp.astype({ctx.name(node.input[0])}, xp.{dtype_name}, copy=False)"


def _emit_constant(ctx: EmitContext, node: NodeProto) -> str:
    _require_single_output(node)
    tensor = ctx.attr_tensor(node, "value")
    if tensor is not None:
        return ctx.register_constant(node.output[0], tensor)
    int_val = ctx.attr(node, "value_int", None)
    if int_val is not None:
        return f"xp.asarray({int(int_val)!r}, dtype=xp.int64)"
    ints = ctx.attr(node, "value_ints", None)
    if ints is not None:
        return f"xp.asarray({ctx.literal([int(i) for i in ints])}, dtype=xp.int64)"
    float_val = ctx.attr(node, "value_float", None)
    if float_val is not None:
        return f"xp.asarray({float(float_val)!r}, dtype=xp.float32)"
    floats = ctx.attr(node, "value_floats", None)
    if floats is not None:
        return f"xp.asarray({ctx.literal([float(f) for f in floats])}, dtype=xp.float32)"
    raise NotImplementedError("Constant op variant not supported in MVP (need 'value' or value_*)")


def _list_attr_expr(ctx: EmitContext, node: NodeProto, name: str) -> str:
    """Render an int-list attribute as a tuple literal, or ``None`` if absent."""

    value = ctx.attr(node, name, None)
    if value is None:
        return "None"
    return ctx.literal(tuple(int(v) for v in value))


EMITTERS: dict[str, Emitter] = {
    "Add": _emit_binary,
    "Sub": _emit_binary,
    "Mul": _emit_binary,
    "Div": _emit_binary,
    "Relu": _emit_relu,
    "Sigmoid": _emit_sigmoid,
    "Tanh": _emit_tanh,
    "Identity": _emit_identity,
    "MatMul": _emit_matmul,
    "Gemm": _emit_gemm,
    "Reshape": _emit_reshape,
    "Flatten": _emit_flatten,
    "Transpose": _emit_transpose,
    "Concat": _emit_concat,
    "Softmax": _emit_softmax,
    "LogSoftmax": _emit_log_softmax,
    "GlobalAveragePool": _emit_global_average_pool,
    "AveragePool": _emit_average_pool,
    "MaxPool": _emit_max_pool,
    "Conv": _emit_conv,
    "Cast": _emit_cast,
    "Constant": _emit_constant,
}


def supported_ops() -> frozenset[str]:
    """Return the set of ONNX op types this MVP knows how to emit."""

    return frozenset(EMITTERS)


__all__ = ["EMITTERS", "EmitContext", "Emitter", "supported_ops"]
