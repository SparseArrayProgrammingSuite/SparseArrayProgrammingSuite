import numpy as np

import torch

import saps.benchmark as benchmark_module
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from saps import compile


def arithmetic_op_eager(x, y, z):
    return ((x + y) * z - y) / (x + 1.0)


@compile
def arithmetic_op(x, y, z):
    return arithmetic_op_eager(x, y, z)


class ToyFusionBenchmark:
    def benchmark(self, data, meta):
        x, y, z = data
        return [arithmetic_op(x, y, z)]


def test_benchmark_compile_pytorch():
    benchmark_module.xp = PytorchFramework()
    benchmark = ToyFusionBenchmark()
    x = torch.rand(1_000_000)
    y = torch.rand(1_000_000)
    z = torch.rand(1_000_000)

    output = benchmark.benchmark([x, y, z], {})[0]

    torch.testing.assert_close(output, arithmetic_op_eager(x, y, z))


def test_benchmark_compile_numpy():
    benchmark_module.xp = NumpyFramework()
    benchmark = ToyFusionBenchmark()
    x = np.random.default_rng(0).random(1_000_000)
    y = np.random.default_rng(1).random(1_000_000)
    z = np.random.default_rng(2).random(1_000_000)

    output = benchmark.benchmark([x, y, z], {})[0]

    np.testing.assert_allclose(output, arithmetic_op_eager(x, y, z))
