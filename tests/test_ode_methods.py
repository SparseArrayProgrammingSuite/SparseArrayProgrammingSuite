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

    time, states = benchmark_cls().benchmark(NumpyFramework(), data, meta)

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
