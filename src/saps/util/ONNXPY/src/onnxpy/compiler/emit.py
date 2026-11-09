"""Direct ONNX → Python source generator.

Walks an :class:`onnx.ModelProto` in topological order and emits a
single inline function with an explicit array namespace argument.
Initializers and inline ``Constant`` op tensors are emitted as ordinary
computation inputs.
"""

from __future__ import annotations

import builtins
import keyword
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, numpy_helper

from .ops import EMITTERS

_RESERVED_NAMES = frozenset(
    {
        "np",
        "math",
        "xp",
    }
    | set(dir(builtins))
)


@dataclass
class _NameAllocator:
    """Sanitize ONNX SSA names into unique Python identifiers."""

    used: set[str] = field(default_factory=set)
    mapping: dict[str, str] = field(default_factory=dict)

    def allocate(self, name: str, *, prefix: str = "v") -> str:
        if name in self.mapping:
            return self.mapping[name]
        cleaned = re.sub(r"\W+", "_", name).strip("_").lower()
        if not cleaned:
            cleaned = prefix
        if cleaned[0].isdigit():
            cleaned = f"{prefix}_{cleaned}"
        if keyword.iskeyword(cleaned) or cleaned in _RESERVED_NAMES:
            cleaned = f"{cleaned}_"
        candidate = cleaned
        counter = 1
        while candidate in self.used:
            candidate = f"{cleaned}_{counter}"
            counter += 1
        self.used.add(candidate)
        self.mapping[name] = candidate
        return candidate


@dataclass
class _EmitContextImpl:
    """Concrete implementation of :class:`onnxpy.compiler.ops.EmitContext`."""

    names: _NameAllocator
    tensor_inputs: dict[str, np.ndarray]
    ranks: dict[str, int]
    body: list[str] = field(default_factory=list)
    _const_counter: int = 0
    _temp_counter: int = 0

    def temp(self, hint: str) -> str:
        # SSA names have leading underscores stripped, so this namespace is private.
        name = f"_onnxpy_{self._temp_counter}_{hint}"
        self._temp_counter += 1
        return name

    def rank(self, ssa_name: str) -> int | None:
        if ssa_name in self.tensor_inputs:
            return self.tensor_inputs[ssa_name].ndim
        return self.ranks.get(ssa_name)

    def emit(self, source: str) -> None:
        self.body.extend(
            f"    {line}" if line else "" for line in textwrap.dedent(source).strip().splitlines()
        )

    def name(self, ssa_name: str) -> str:
        if not ssa_name:
            raise ValueError("Empty SSA name encountered while emitting an op.")
        return self.names.allocate(ssa_name)

    def attr(self, node: NodeProto, name: str, default: Any = None) -> Any:
        for a in node.attribute:
            if a.name == name:
                return _attr_value(a)
        return default

    def attr_tensor(self, node: NodeProto, name: str) -> np.ndarray | None:
        for a in node.attribute:
            if a.name == name and a.type == AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t)
        return None

    def literal(self, value: Any) -> str:
        return _literal(value)

    def register_constant(self, hint: str, array: np.ndarray) -> str:
        key = hint or f"_const_{self._const_counter}"
        if key in self.tensor_inputs:
            self._const_counter += 1
            key = f"{hint}__const_{self._const_counter}"
        self.tensor_inputs[key] = array
        return self.names.allocate(key, prefix="c")


def _attr_value(a: AttributeProto) -> Any:
    t = a.type
    if t == AttributeProto.INT:
        return int(a.i)
    if t == AttributeProto.FLOAT:
        return float(a.f)
    if t == AttributeProto.STRING:
        return a.s.decode("utf-8")
    if t == AttributeProto.INTS:
        return [int(v) for v in a.ints]
    if t == AttributeProto.FLOATS:
        return [float(v) for v in a.floats]
    if t == AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in a.strings]
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    type_name = AttributeProto.AttributeType.Name(t)
    raise NotImplementedError(f"Attribute type {type_name} not supported")


def _literal(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        inner = ", ".join(_literal(v) for v in value)
        return f"[{inner}]"
    return repr(value)


def compile_to_source(
    model: ModelProto,
    *,
    function_name: str = "model",
    extra_parameters: Sequence[str] = (),
) -> tuple[str, dict[str, np.ndarray]]:
    """Return a single function definition and its tensor input values.

    The signature is ``function(xp, *extra_parameters, *graph_inputs, *tensors)``
    with one named positional parameter per input. ``extra_parameters`` are
    caller-owned arguments, unused by the generated computation. Their names
    are reserved so ONNX values cannot shadow them.

    Source contains only the inline function, with no imports or module globals.
    Supply ``math`` in its globals for models using Flatten. Tensor values are
    returned in parameter order, after graph inputs excluding initializers.
    """
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        raise ValueError(f"Invalid function name {function_name!r}")
    if len(set(extra_parameters)) != len(extra_parameters):
        raise ValueError("Extra parameter names must be unique")
    for name in extra_parameters:
        if (
            not name.isidentifier()
            or keyword.iskeyword(name)
            or name in _RESERVED_NAMES
            or name.startswith("_onnxpy_")
        ):
            raise ValueError(f"Invalid or reserved extra parameter name {name!r}")

    graph = model.graph
    _validate_op_coverage(graph)

    names = _NameAllocator(used=set(extra_parameters))
    tensor_inputs: dict[str, np.ndarray] = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer
    }
    initializer_names = set(tensor_inputs)

    real_inputs = [vi for vi in graph.input if vi.name not in initializer_names]
    arg_names = [names.allocate(vi.name, prefix="x") for vi in real_inputs]
    for tensor in graph.initializer:
        names.allocate(tensor.name, prefix="w")

    try:
        shape_graph = onnx.shape_inference.infer_shapes(model).graph
    except onnx.shape_inference.InferenceError:
        shape_graph = graph
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (*shape_graph.input, *shape_graph.value_info, *shape_graph.output)
        if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")
    }
    ctx = _EmitContextImpl(names=names, tensor_inputs=tensor_inputs, ranks=ranks)
    body = ctx.body

    for node in graph.node:
        if not node.output:
            raise ValueError(f"Node of type {node.op_type!r} has no outputs.")
        if len(node.output) > 1:
            raise NotImplementedError(
                f"Multi-output op {node.op_type!r} not supported in MVP",
            )
        emitter = EMITTERS[node.op_type]
        rhs = emitter(ctx, node)
        lhs = ctx.name(node.output[0])
        if lhs != rhs:
            body.append(f"    {lhs} = {rhs}")

    if not graph.output:
        raise ValueError("Graph has no outputs.")
    output_names = [ctx.name(o.name) for o in graph.output]
    if len(output_names) == 1:
        assignment_prefix = f"    {output_names[0]} = "
        if body and body[-1].startswith(assignment_prefix):
            body[-1] = f"    return {body[-1][len(assignment_prefix) :]}"
        else:
            body.append(f"    return {output_names[0]}")
    else:
        body.append(f"    return ({', '.join(output_names)},)")

    tensor_arg_names = [names.allocate(name, prefix="w") for name in tensor_inputs]
    all_arg_names = ["xp", *extra_parameters, *arg_names, *tensor_arg_names]
    signature = f"def {function_name}({', '.join(all_arg_names)}):"
    return "\n".join([signature, *body]).rstrip() + "\n", tensor_inputs


def _validate_op_coverage(graph: GraphProto) -> None:
    seen = {n.op_type for n in graph.node}
    unsupported = sorted(seen - set(EMITTERS))
    if unsupported:
        raise NotImplementedError(
            f"Unsupported ONNX op types in graph: {', '.join(unsupported)}",
        )


__all__ = ["compile_to_source"]
