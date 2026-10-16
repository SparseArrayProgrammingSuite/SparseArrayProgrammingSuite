import inspect

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.matrix_vector_multiplication import (
    MatrixVectorMultiplicationBenchmark,
)
from saps.benchmarks.model_counting import ModelCountingBenchmark
from saps.benchmarks.subgraph_matching import SubgraphMatchingBenchmark
from saps.benchmarks.weighted_model_counting import WeightedModelCountingBenchmark
from saps.codegen import (
    constant_function_source,
    define_function,
    einsum_function_source,
)


def _param(benchmark, dataset_name):
    return next(p for p in benchmark.params if p.dataset.name == dataset_name)


def _generated(benchmark, dataset_name):
    param = _param(benchmark, dataset_name)
    problem = param.generator.generate(param.dataset)
    function = param.generator.generate_benchmark_function(
        param.dataset, problem, benchmark.benchmark
    )
    return param, problem, function


def test_default_generator_passes_benchmark_method_through():
    benchmark = MatrixVectorMultiplicationBenchmark()
    param = _param(benchmark, "small")
    problem = param.generator.generate(param.dataset)
    function = param.generator.generate_benchmark_function(
        param.dataset, problem, benchmark.benchmark
    )
    assert function == benchmark.benchmark


@pytest.mark.parametrize(
    ("benchmark_cls", "dataset_name", "expected_params"),
    [
        (SubgraphMatchingBenchmark, "a_to_b", ["VA", "E0", "VB"]),
        (WeightedModelCountingBenchmark, "satisfiable", ["B", "W1", "W2"]),
        (ModelCountingBenchmark, "standard_sat", ["B"]),
    ],
)
def test_generator_builds_fixed_arity_function(
    benchmark_cls, dataset_name, expected_params
):
    benchmark = benchmark_cls()
    _, problem, function = _generated(benchmark, dataset_name)

    parameters = list(inspect.signature(function).parameters)
    assert parameters == ["xp", "meta", *expected_params]
    source = inspect.getsource(function)
    assert repr(problem.meta["expr"]) in source
    with pytest.raises(NotImplementedError):
        benchmark.benchmark(NumpyFramework(), problem.meta)


@pytest.mark.parametrize(
    ("benchmark_cls", "dataset_name"),
    [
        (SubgraphMatchingBenchmark, "a_to_b"),
        (WeightedModelCountingBenchmark, "satisfiable"),
        (WeightedModelCountingBenchmark, "no_clauses"),  # no clauses: constant total
        (ModelCountingBenchmark, "standard_sat"),
    ],
)
def test_generated_function_runs_through_setup(benchmark_cls, dataset_name):
    benchmark = benchmark_cls()
    param = _param(benchmark, dataset_name)
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    assert len(benchmark._output) == 1
    benchmark.teardown(param)  # the benchmark's own reference check


def test_define_function_source_is_inspectable():
    source = einsum_function_source("y[i] += A[i,j] * x[j]", ["A", "x"])
    function = define_function(source, "<saps-generated test.einsum>")
    assert inspect.getsource(function) == source
    A, x = np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0])
    xp = NumpyFramework()
    np.testing.assert_allclose(to_numpy(xp.to_binsparse(function(xp, {}, A, x))), A @ x)


def test_constant_function_source_uses_numpy_dtype():
    function = define_function(
        constant_function_source(0.25, "float64", ["B"]),
        "<saps-generated test.constant>",
        {"np": np},
    )
    result = function(np, {}, np.zeros(2))
    assert result.dtype == np.float64
    assert result == 0.25


def test_generated_parameter_names_are_validated():
    with pytest.raises(ValueError, match="Invalid parameter name"):
        einsum_function_source("s[] += A[i]", ["not-an-identifier"])
