import ast
import inspect

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import Benchmark, DataInstance, Param
from saps.benchmarks import ode
from saps.codegen import define_function


def test_ode_discovery_exposes_one_benchmark_per_method():
    benchmarks = [
        cls()
        for _, cls in inspect.getmembers(ode, inspect.isclass)
        if issubclass(cls, Benchmark) and not inspect.isabstract(cls)
    ]
    assert {benchmark.name for benchmark in benchmarks} == {
        "forward_euler",
        "backward_euler",
        "rk4",
    }
    expected_generators = {
        "ode_rc": 1,
        "ode_rlc": 1,
        "ode_lotka_volterra": 1,
        "ode_brusselator": 2,
        "ode_slicot": 10,
    }
    dataset_inventories = []
    for benchmark in benchmarks:
        inventory = {
            generator.name: generator.dataset_names
            for generator in benchmark.generators
        }
        assert {name: len(datasets) for name, datasets in inventory.items()} == (
            expected_generators
        )
        dataset_inventories.append(inventory)
        for metric in ("time", "peakmem"):
            assert [
                name for name in dir(benchmark) if name.startswith(f"{metric}_")
            ] == [f"{metric}_{benchmark.name}"]
    assert all(inventory == dataset_inventories[0] for inventory in dataset_inventories)


@pytest.mark.parametrize(
    ("benchmark_cls", "expected"),
    [
        (ode.ForwardEulerBenchmark, 0.9),
        (ode.BackwardEulerBenchmark, 1 / 1.1),
        (ode.RK4Benchmark, 0.9048375),
    ],
)
def test_ode_methods_integrate_exponential_decay(benchmark_cls, expected):
    data = [np.array([[-1.0]]), np.zeros((1, 1))]
    meta = {
        "problem_name": "ode_slicot",
        "span": (0.0, 0.2),
        "y0": [1.0],
        "step": 0.1,
        "input_value": 0.0,
    }

    benchmark = benchmark_cls()
    generator = ode.ODESLICOTGenerator()
    problem = DataInstance(inputs=[from_numpy(item) for item in data], meta=meta)
    function = generator.generate_benchmark_function(
        generator.datasets[0], problem, benchmark.benchmark
    )
    time, states = function(NumpyFramework(), meta, *data)

    np.testing.assert_allclose(time, [0.0, 0.1])
    np.testing.assert_allclose(states[:, 0], [1.0, expected], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "benchmark_cls",
    [ode.ForwardEulerBenchmark, ode.BackwardEulerBenchmark, ode.RK4Benchmark],
)
def test_ode_setup_preserves_non_slicot_timestep(benchmark_cls):
    generator = ode.ODERCGenerator()
    dataset = generator.datasets[0]
    benchmark = benchmark_cls()

    benchmark.setup(Param(generator, dataset), use_cache=False, xp=NumpyFramework())

    assert benchmark._meta["step"] == dataset.step
    assert benchmark._meta["problem_name"] == "ode_rc"


@pytest.mark.parametrize(
    "benchmark_cls",
    [ode.ForwardEulerBenchmark, ode.BackwardEulerBenchmark, ode.RK4Benchmark],
)
@pytest.mark.parametrize(
    ("generator_name", "input_names"),
    [
        ("ode_rc", []),
        ("ode_rlc", []),
        ("ode_lotka_volterra", []),
        ("ode_brusselator", ["C", "brusselator_cb"]),
        ("ode_slicot", ["A", "B"]),
    ],
)
def test_generated_ode_function_is_inline_and_matches_template(
    benchmark_cls, generator_name, input_names, monkeypatch
):
    benchmark = benchmark_cls()
    generator = next(g for g in benchmark.generators if g.name == generator_name)
    dataset = generator.datasets[0]
    if generator_name == "ode_slicot":
        problem = DataInstance(
            inputs=[
                from_numpy(np.array([[-1 + 0.5j, 0.1], [0, -2 - 0.3j]])),
                from_numpy(np.array([[1.0], [0.5]])),
            ],
            meta={
                "problem_name": generator_name,
                "span": (0.0, 0.03),
                "y0": [0.5, 1.0],
                "step": 0.01,
                "input_value": 2.0,
            },
        )
    else:
        problem = generator.generate(dataset)
        step = problem.meta["step"]
        # Include the RC/RLC input's switch from 0V to 5V.
        problem.meta["span"] = (-2 * step, 3 * step)
        if generator_name == "ode_brusselator":
            # Exercise the forcing branch, including within an RK4 step.
            problem.meta["span"] = (1.085, 1.125)
            problem.inputs[1] = from_numpy(np.ones(len(problem.meta["y0"])))

    function = generator.generate_benchmark_function(
        dataset, problem, benchmark.benchmark
    )
    parameters = inspect.signature(function).parameters
    assert list(parameters) == ["xp", "meta", *input_names]
    assert all(p.kind == p.POSITIONAL_OR_KEYWORD for p in parameters.values())
    source = inspect.getsource(function)
    assert (
        sum(isinstance(node, ast.FunctionDef) for node in ast.walk(ast.parse(source)))
        == 1
    )
    assert "_resolve_derivatives" not in source
    assert "_step_input(" not in source
    assert "dydt(" not in source
    inputs = [to_numpy(item) for item in problem.inputs]
    # Use the independent reference derivatives in the same solver template.
    arguments = ", ".join(("t", "state", "meta", *input_names))
    reference = define_function(
        benchmark.benchmark_source(
            tuple(input_names), f"dydt_vector = derivative({arguments})"
        ),
        f"<ode-reference {benchmark.name}.{generator_name}>",
        {"np": np, "derivative": ode._resolve_derivatives(generator_name)},
    )
    expected = reference(NumpyFramework(), problem.meta, *inputs)

    def fail(*args, **kwargs):
        pytest.fail("generated functions must not call derivative helpers at runtime")

    monkeypatch.setattr(ode, "_resolve_derivatives", fail)
    monkeypatch.setattr(ode, "_step_input", fail)
    actual = function(NumpyFramework(), problem.meta, *inputs)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
