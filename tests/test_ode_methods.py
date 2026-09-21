import inspect

import pytest

import numpy as np

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import Benchmark, Param
from saps.benchmarks import ode


def test_ode_discovery_exposes_one_benchmark_per_method():
    benchmarks = [
        cls()
        for _, cls in inspect.getmembers(ode, inspect.isclass)
        if issubclass(cls, Benchmark) and not inspect.isabstract(cls)
    ]
    assert {benchmark.name for benchmark in benchmarks} == {
        "forward_euler",
        "backward_euler",
        "runge_kutta",
    }
    expected_generators = {
        "rc": 1,
        "rlc": 1,
        "lotka_volterra": 1,
        "brusselator": 2,
        "slicot_ode": 10,
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
        (ode.ForwardEuler, 0.9),
        (ode.BackwardEuler, 1 / 1.1),
        (ode.RungeKutta, 0.9048375),
    ],
)
def test_ode_methods_integrate_exponential_decay(benchmark_cls, expected):
    data = [np.array([[-1.0]]), np.zeros((1, 1))]
    meta = {
        "problem_name": "slicot_ode",
        "span": (0.0, 0.2),
        "y0": [1.0],
        "step": 0.1,
        "input_value": 0.0,
    }

    time, states = benchmark_cls().benchmark(NumpyFramework(), data, meta)

    np.testing.assert_allclose(time, [0.0, 0.1])
    np.testing.assert_allclose(states[:, 0], [1.0, expected], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "benchmark_cls", [ode.ForwardEuler, ode.BackwardEuler, ode.RungeKutta]
)
def test_ode_setup_preserves_non_slicot_timestep(benchmark_cls):
    generator = ode.RCGenerator()
    dataset = generator.datasets[0]
    benchmark = benchmark_cls()

    benchmark.setup(Param(generator, dataset), use_cache=False, xp=NumpyFramework())

    assert benchmark._meta["step"] == dataset.step
    assert benchmark._meta["problem_name"] == "rc"
