import ast
import inspect
from unittest.mock import patch

import pytest

import numpy as np

import onnx
import onnxpy
from binsparse.conversions import to_numpy
from onnx import TensorProto, helper, numpy_helper

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.lth_conv2 import LTHConv2Benchmark


@pytest.fixture
def conv_model(tmp_path, monkeypatch):
    """A small Conv/Relu/Pool/Flatten/MatMul graph with external weights."""
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    weights[weights < 0] = 0
    bias = rng.standard_normal(2).astype(np.float32)
    dense_weights = rng.standard_normal((8, 3)).astype(np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["input", "conv.weight", "conv.bias"],
                    ["conv"],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Relu", ["conv"], ["relu"]),
                helper.make_node(
                    "MaxPool",
                    ["relu"],
                    ["pool"],
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                ),
                helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
                helper.make_node(
                    "Constant",
                    [],
                    ["dense.weight"],
                    value=numpy_helper.from_array(dense_weights),
                ),
                helper.make_node("MatMul", ["flat", "dense.weight"], ["output"]),
            ],
            "tiny_conv",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4]),
                # Initializers listed as graph inputs must not become runtime inputs.
                helper.make_tensor_value_info("conv.bias", TensorProto.FLOAT, [2]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
            initializer=[
                numpy_helper.from_array(weights, name="conv.weight"),
                numpy_helper.from_array(bias, name="conv.bias"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model_path = tmp_path / "conv2_pruned_dense.onnx"
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=model_path.name + ".data",
        size_threshold=0,
    )
    monkeypatch.setenv("LTH_CONV2_ONNX", str(model_path))
    return model_path


def test_generated_conv_matches_reference_without_compiling_during_run(conv_model):
    benchmark = LTHConv2Benchmark()
    param = benchmark.params[0]
    with patch.object(
        onnxpy, "compile_to_source", wraps=onnxpy.compile_to_source
    ) as emit:
        benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
        assert emit.call_count == 1
        # Includes Constant node tensors as well as external initializers.
        assert benchmark._meta["onnx_initializer_names"] == (
            "conv.weight",
            "conv.bias",
            "dense.weight",
        )
        assert len(benchmark._input) == 4
        for _ in range(2):
            benchmark.run(param)
            benchmark.check(param)
        assert emit.call_count == 1
    benchmark.teardown(param)


def test_generated_function_is_inline_fixed_arity_and_inspectable(conv_model):
    benchmark = LTHConv2Benchmark()
    param = benchmark.params[0]
    problem = param.generator.generate(param.dataset)
    # Function construction depends on this problem, not transient generator state.
    function = type(param.generator)().generate_benchmark_function(
        param.dataset,
        problem,
        benchmark.benchmark,
    )
    signature = inspect.signature(function)
    assert list(signature.parameters) == [
        "xp",
        "meta",
        "input_",
        "conv_weight",
        "conv_bias",
        "dense_weight",
    ]
    assert all(
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is inspect.Parameter.empty
        for p in signature.parameters.values()
    )
    source = inspect.getsource(function)
    assert source == problem.meta["benchmark_source"]
    nodes = list(ast.walk(ast.parse(source)))
    assert sum(isinstance(node, ast.FunctionDef) for node in nodes) == 1
    assert not any(isinstance(node, ast.For | ast.While | ast.Lambda) for node in nodes)
    assert "xp.unfold(" in source
    assert "xp.tensordot(" in source
    result = function(NumpyFramework(), problem.meta, *map(to_numpy, problem.inputs))
    np.testing.assert_allclose(
        result, to_numpy(problem.ref_outputs[0]), rtol=1e-4, atol=1e-4
    )
    again = param.generator.generate(param.dataset)
    assert again.meta == problem.meta
    for actual, expected in zip(again.inputs, problem.inputs, strict=True):
        np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))


def test_onnx_input_named_meta_does_not_collide_with_saps_metadata(conv_model):
    model = onnx.load(conv_model, load_external_data=False)
    model.graph.input[0].name = "meta"
    model.graph.node[0].input[0] = "meta"
    onnx.save_model(model, conv_model)
    benchmark = LTHConv2Benchmark()
    param = benchmark.params[0]
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    benchmark.check(param)
    benchmark.teardown(param)


@pytest.mark.parametrize("invalid_input", ["dynamic_shape", "multiple_inputs"])
def test_generator_validates_runtime_inputs(conv_model, invalid_input):
    model = onnx.load(conv_model, load_external_data=False)
    if invalid_input == "dynamic_shape":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        message = "must have a static shape"
    else:
        model.graph.input.append(
            helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1])
        )
        message = "Expected one runtime input"
    onnx.save_model(model, conv_model)
    param = LTHConv2Benchmark().params[0]
    with pytest.raises(ValueError, match=message):
        param.generator.generate(param.dataset)
