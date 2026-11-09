"""Per-op round-trip tests vs onnxruntime."""

from __future__ import annotations

import ast

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnxpy import compile_to_source


def _vi(name: str, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, None if shape is None else list(shape))


def _model(nodes, inputs, outputs, initializers=(), opset_version: int = 18) -> onnx.ModelProto:
    if isinstance(nodes, onnx.NodeProto):
        nodes = [nodes]
    graph = helper.make_graph(
        nodes=list(nodes),
        name="g",
        inputs=list(inputs),
        outputs=list(outputs),
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    return model


@pytest.mark.parametrize(
    "op,expected_factory",
    [
        ("Add", lambda a, b: a + b),
        ("Sub", lambda a, b: a - b),
        ("Mul", lambda a, b: a * b),
        ("Div", lambda a, b: a / b),
    ],
)
def test_binary_pointwise(op, expected_factory, assert_matches_ort):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    model = _model(
        helper.make_node(op, ["a", "b"], ["y"]),
        [_vi("a", a.shape), _vi("b", b.shape)],
        [_vi("y", a.shape)],
    )
    assert_matches_ort(model, {"a": a, "b": b})


def test_relu(assert_matches_ort):
    x = np.array([[-2.0, -0.5, 0.0, 3.0]], dtype=np.float32)
    model = _model(
        helper.make_node("Relu", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


def test_sigmoid(assert_matches_ort):
    x = np.linspace(-3, 3, 8, dtype=np.float32).reshape(1, 8)
    model = _model(
        helper.make_node("Sigmoid", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


def test_tanh(assert_matches_ort):
    x = np.linspace(-2, 2, 6, dtype=np.float32).reshape(2, 3)
    model = _model(
        helper.make_node("Tanh", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


def test_identity(assert_matches_ort):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = _model(
        helper.make_node("Identity", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


def test_matmul(assert_matches_ort):
    a = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    b = np.random.RandomState(1).randn(4, 5).astype(np.float32)
    model = _model(
        helper.make_node("MatMul", ["a", "b"], ["y"]),
        [_vi("a", a.shape), _vi("b", b.shape)],
        [_vi("y", (3, 5))],
    )
    assert_matches_ort(model, {"a": a, "b": b})


@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_gemm(trans_a, trans_b, assert_matches_ort):
    rng = np.random.RandomState(42)
    a = rng.randn(*((4, 3) if trans_a else (3, 4))).astype(np.float32)
    b = rng.randn(*((5, 4) if trans_b else (4, 5))).astype(np.float32)
    c = rng.randn(3, 5).astype(np.float32)
    node = helper.make_node(
        "Gemm",
        ["a", "b", "c"],
        ["y"],
        alpha=0.7,
        beta=1.3,
        transA=trans_a,
        transB=trans_b,
    )
    model = _model(
        node,
        [_vi("a", a.shape), _vi("b", b.shape), _vi("c", c.shape)],
        [_vi("y", (3, 5))],
    )
    assert_matches_ort(model, {"a": a, "b": b, "c": c})


def test_reshape(assert_matches_ort):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    shape = np.array([2, 12], dtype=np.int64)
    init = numpy_helper.from_array(shape, name="shape")
    model = _model(
        helper.make_node("Reshape", ["x", "shape"], ["y"]),
        [_vi("x", x.shape), _vi("shape", shape.shape, TensorProto.INT64)],
        [_vi("y", (2, 12))],
        initializers=[init],
    )
    assert_matches_ort(model, {"x": x})


def test_reshape_with_zero_keep(assert_matches_ort):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    shape = np.array([0, -1], dtype=np.int64)  # keep dim 0, infer rest
    init = numpy_helper.from_array(shape, name="shape")
    model = _model(
        helper.make_node("Reshape", ["x", "shape"], ["y"]),
        [_vi("x", x.shape), _vi("shape", shape.shape, TensorProto.INT64)],
        [_vi("y", (2, 12))],
        initializers=[init],
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_flatten(axis, assert_matches_ort):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = _model(
        helper.make_node("Flatten", ["x"], ["y"], axis=axis),
        [_vi("x", x.shape)],
        [_vi("y", None)],
    )
    assert_matches_ort(model, {"x": x})


def test_transpose_perm(assert_matches_ort):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = _model(
        helper.make_node("Transpose", ["x"], ["y"], perm=[2, 0, 1]),
        [_vi("x", x.shape)],
        [_vi("y", (4, 2, 3))],
    )
    assert_matches_ort(model, {"x": x})


def test_transpose_default(assert_matches_ort):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = _model(
        helper.make_node("Transpose", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", (3, 2))],
    )
    assert_matches_ort(model, {"x": x})


def test_concat(assert_matches_ort):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = (np.arange(8, dtype=np.float32) + 100).reshape(2, 4)
    model = _model(
        helper.make_node("Concat", ["a", "b"], ["y"], axis=1),
        [_vi("a", a.shape), _vi("b", b.shape)],
        [_vi("y", (2, 7))],
    )
    assert_matches_ort(model, {"a": a, "b": b})


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax(axis, assert_matches_ort):
    x = np.random.RandomState(7).randn(2, 5).astype(np.float32)
    model = _model(
        helper.make_node("Softmax", ["x"], ["y"], axis=axis),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_log_softmax(axis, assert_matches_ort):
    x = np.random.RandomState(8).randn(2, 5).astype(np.float32)
    model = _model(
        helper.make_node("LogSoftmax", ["x"], ["y"], axis=axis),
        [_vi("x", x.shape)],
        [_vi("y", x.shape)],
    )
    assert_matches_ort(model, {"x": x})


def test_global_average_pool(assert_matches_ort):
    x = np.random.RandomState(0).randn(2, 3, 5, 5).astype(np.float32)
    model = _model(
        helper.make_node("GlobalAveragePool", ["x"], ["y"]),
        [_vi("x", x.shape)],
        [_vi("y", (2, 3, 1, 1))],
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize("count_include_pad", [0, 1])
def test_average_pool(count_include_pad, assert_matches_ort):
    x = np.random.RandomState(1).randn(1, 1, 5, 5).astype(np.float32)
    model = _model(
        helper.make_node(
            "AveragePool",
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            count_include_pad=count_include_pad,
        ),
        [_vi("x", x.shape)],
        [_vi("y", None)],
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize("auto_pad", ["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"])
def test_max_pool(auto_pad, assert_matches_ort):
    x = np.random.RandomState(2).randn(1, 2, 6, 6).astype(np.float32)
    attrs = dict(kernel_shape=[3, 3], strides=[2, 2])
    if auto_pad == "NOTSET":
        attrs["pads"] = [1, 1, 1, 1]
    else:
        attrs["auto_pad"] = auto_pad
    model = _model(
        helper.make_node("MaxPool", ["x"], ["y"], **attrs),
        [_vi("x", x.shape)],
        [_vi("y", None)],
    )
    assert_matches_ort(model, {"x": x})


def test_max_pool_dilation(assert_matches_ort):
    x = np.random.RandomState(3).randn(1, 1, 7, 7).astype(np.float32)
    model = _model(
        helper.make_node(
            "MaxPool",
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[2, 2],
        ),
        [_vi("x", x.shape)],
        [_vi("y", None)],
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize(
    "pads,strides,dilations",
    [
        ([0, 0, 0, 0], [1, 1], [1, 1]),
        ([1, 1, 1, 1], [1, 1], [1, 1]),
        ([2, 2, 2, 2], [2, 2], [1, 1]),
        ([0, 0, 0, 0], [1, 1], [2, 2]),
    ],
)
def test_conv_basic(pads, strides, dilations, assert_matches_ort):
    rng = np.random.RandomState(123)
    x = rng.randn(1, 2, 7, 7).astype(np.float32)
    w = rng.randn(4, 2, 3, 3).astype(np.float32)
    b = rng.randn(4).astype(np.float32)
    init = [
        numpy_helper.from_array(w, name="w"),
        numpy_helper.from_array(b, name="b"),
    ]
    node = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[3, 3],
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=1,
    )
    model = _model(
        node,
        [_vi("x", x.shape), _vi("w", w.shape), _vi("b", b.shape)],
        [_vi("y", None)],
        initializers=init,
    )
    assert_matches_ort(model, {"x": x}, atol=1e-5)


def test_conv_grouped(assert_matches_ort):
    rng = np.random.RandomState(7)
    x = rng.randn(1, 4, 5, 5).astype(np.float32)
    w = rng.randn(4, 2, 3, 3).astype(np.float32)  # 4 out, 2 in/group, group=2
    init = [numpy_helper.from_array(w, name="w")]
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=2,
    )
    model = _model(
        node,
        [_vi("x", x.shape), _vi("w", w.shape)],
        [_vi("y", None)],
        initializers=init,
    )
    assert_matches_ort(model, {"x": x}, atol=1e-5)


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER"])
def test_conv_auto_pad(auto_pad, assert_matches_ort):
    rng = np.random.RandomState(11)
    x = rng.randn(1, 1, 6, 6).astype(np.float32)
    w = rng.randn(2, 1, 3, 3).astype(np.float32)
    init = [numpy_helper.from_array(w, name="w")]
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        auto_pad=auto_pad,
    )
    model = _model(
        node,
        [_vi("x", x.shape), _vi("w", w.shape)],
        [_vi("y", None)],
        initializers=init,
    )
    assert_matches_ort(model, {"x": x}, atol=1e-5)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("auto_pad", ["NOTSET", "SAME_UPPER", "SAME_LOWER"])
def test_inline_spatial_pipeline(rank, auto_pad, assert_matches_ort):
    rng = np.random.RandomState(19)
    x = rng.randn(1, 4, *((8,) * rank)).astype(np.float32)
    w = rng.randn(6, 2, *((3,) * rank)).astype(np.float32)
    # Exercise SSA names matching builtins, old helper locals and temporary names.
    weight_name = "_onnxpy_0_conv_rank"
    padding = {"pads": [1] * (2 * rank)} if auto_pad == "NOTSET" else {"auto_pad": auto_pad}
    model = _model(
        [
            helper.make_node(
                "Conv",
                ["range", weight_name],
                ["kernel"],
                group=2,
                strides=[2] * rank,
                **padding,
            ),
            helper.make_node(
                "MaxPool",
                ["kernel"],
                ["patches"],
                kernel_shape=[2] * rank,
                strides=[2] * rank,
            ),
            helper.make_node(
                "AveragePool",
                ["patches"],
                ["rank"],
                kernel_shape=[2] * rank,
            ),
        ],
        [_vi("range", (None, 4, *((None,) * rank)))],
        [_vi(name, None) for name in ("kernel", "patches", "rank", "range")],
        initializers=[numpy_helper.from_array(w, name=weight_name)],
    )
    source, _ = compile_to_source(model)
    definitions = [
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)
    ]
    assert [node.name for node in definitions] == ["model"]
    assert "unfold" not in source
    assert "xp.conv(" in source
    assert "xp.pool(" in source
    assert not any(
        isinstance(node, ast.For | ast.While | ast.comprehension)
        for node in ast.walk(ast.parse(source))
    )
    assert_matches_ort(model, {"range": x}, atol=1e-5)


@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
def test_pool_ceil_mode(op, assert_matches_ort):
    x = np.random.RandomState(20).randn(1, 2, 6, 6).astype(np.float32)
    model = _model(
        helper.make_node(
            op,
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            ceil_mode=1,
        ),
        [_vi("x", x.shape)],
        [_vi("y", None)],
    )
    source, _ = compile_to_source(model)
    assert not any(
        isinstance(node, ast.For | ast.While | ast.comprehension)
        for node in ast.walk(ast.parse(source))
    )
    assert_matches_ort(model, {"x": x})


@pytest.mark.parametrize(
    "dtype,onnx_dtype", [(np.int8, TensorProto.INT8), (np.uint8, TensorProto.UINT8)]
)
def test_max_pool_integer_padding(dtype, onnx_dtype, assert_matches_ort):
    x = np.arange(1, 10, dtype=dtype).reshape(1, 1, 3, 3)
    if dtype == np.int8:
        x = -x
    model = _model(
        helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], pads=[1, 1, 1, 1]),
        [_vi("x", x.shape, onnx_dtype)],
        [_vi("y", None, onnx_dtype)],
    )
    assert_matches_ort(model, {"x": x})


def test_constant(assert_matches_ort):
    value = np.array([[1.5, -2.0], [0.0, 3.25]], dtype=np.float32)
    tensor = numpy_helper.from_array(value, name="const_value")
    const_node = helper.make_node("Constant", [], ["c"], value=tensor)
    add_node = helper.make_node("Add", ["x", "c"], ["y"])
    model = _model(
        [const_node, add_node],
        [_vi("x", value.shape)],
        [_vi("y", value.shape)],
    )
    assert_matches_ort(model, {"x": np.zeros_like(value)})


def test_tensors_are_emitted_as_model_inputs():
    shape = np.array([2, 2], dtype=np.int64)
    bias = np.array([1.0, 2.0], dtype=np.float32)
    const_value = np.array([[3.0, 4.0]], dtype=np.float32)
    model = _model(
        [
            helper.make_node("Constant", [], ["c"], value=numpy_helper.from_array(const_value)),
            helper.make_node("Add", ["x", "bias"], ["xb"]),
            helper.make_node("Reshape", ["xb", "shape"], ["reshaped"]),
            helper.make_node("Add", ["reshaped", "c"], ["y"]),
        ],
        [
            _vi("x", (4,)),
            _vi("shape", shape.shape, TensorProto.INT64),
            _vi("bias", bias.shape),
        ],
        [_vi("y", None)],
        initializers=[
            numpy_helper.from_array(shape, name="shape"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )

    source, tensor_inputs = compile_to_source(model)

    assert source.startswith("def model(xp, x, shape, bias, c):\n")
    assert tuple(tensor_inputs) == ("shape", "bias", "c")


def test_cast_float_to_int(assert_matches_ort):
    x = np.array([[1.7, -0.5, 2.9]], dtype=np.float32)
    model = _model(
        helper.make_node("Cast", ["x"], ["y"], to=TensorProto.INT64),
        [_vi("x", x.shape)],
        [_vi("y", x.shape, TensorProto.INT64)],
    )
    assert_matches_ort(model, {"x": x})
