"""Function-source API and caller-owned parameter tests."""

import ast
import inspect
import math

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnxpy import compile_to_source


def _model():
    weight = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)
    bias = np.array([0.5, 1.5], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["context", "weight"], ["state"]),
            helper.make_node("Flatten", ["state"], ["flat"], axis=1),
            helper.make_node("Constant", [], ["bias"], value=numpy_helper.from_array(bias)),
            helper.make_node("Add", ["flat", "bias"], ["output"]),
        ],
        "parameter_collisions",
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])],
        initializer=[numpy_helper.from_array(weight, name="weight")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_emits_only_one_inline_function():
    source, tensors = compile_to_source(_model())
    tree = ast.parse(source)
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef)
    assert not any(isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(tree))
    assert sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)) == 1
    scope = {"math": math}
    exec(source, scope)
    function = scope["model"]
    parameters = inspect.signature(function).parameters
    assert list(parameters) == ["xp", "context", "weight", "bias"]
    assert all(
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is inspect.Parameter.empty
        for p in parameters.values()
    )
    assert tuple(tensors) == ("weight", "bias")
    x = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(
        function(np, x, *tensors.values()), x @ tensors["weight"] + tensors["bias"]
    )


def test_extra_parameters_do_not_collide_with_graph_values():
    extras = ("context", "weight", "state", "bias")
    source, tensors = compile_to_source(
        _model(), function_name="inference", extra_parameters=extras
    )
    scope = {"math": math}
    exec(source, scope)
    function = scope["inference"]
    assert list(inspect.signature(function).parameters) == [
        "xp",
        *extras,
        "context_1",
        "weight_1",
        "bias_1",
    ]
    x = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float32)
    # Opaque caller-owned arguments cannot participate in array arithmetic.
    actual = function(np, *(object() for _ in extras), x, *tensors.values())
    np.testing.assert_allclose(actual, x @ tensors["weight"] + tensors["bias"])


@pytest.mark.parametrize(
    "extra_parameters",
    [
        ("context", "context"),
        ("not-valid",),
        ("class",),
        ("xp",),
        ("math",),
        ("np",),
        ("int",),
        ("_onnxpy_0_shape",),
    ],
)
def test_rejects_invalid_extra_parameters(extra_parameters):
    with pytest.raises(ValueError):
        compile_to_source(_model(), extra_parameters=extra_parameters)


@pytest.mark.parametrize("function_name", ["", "not-valid", "class"])
def test_rejects_invalid_function_names(function_name):
    with pytest.raises(ValueError, match="Invalid function name"):
        compile_to_source(_model(), function_name=function_name)


def test_graph_without_inputs_emits_namespace_only_signature():
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["value"], value_float=2.0)],
        "scalar_constant",
        [],
        [helper.make_tensor_value_info("value", TensorProto.FLOAT, [])],
    )
    source, tensors = compile_to_source(helper.make_model(graph))
    assert source.startswith("def model(xp):\n")
    assert not tensors
    scope = {}
    exec(source, scope)
    np.testing.assert_array_equal(scope["model"](np), 2.0)
