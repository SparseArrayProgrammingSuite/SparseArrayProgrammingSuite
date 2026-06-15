import time

import numpy as np

import torch

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework


def arithmetic_op_eager(x, y, z):
    out = (x + y) * z - y
    for _ in range(32):
        out = (out + x) * z - y
    return out / (x + 1.0)


def make_benchmark_cls():
    class ToyFusionBenchmark:
        def benchmark(self, data, meta):
            x, y, z = data
            return [arithmetic_op_eager(x, y, z)]

    return ToyFusionBenchmark


def test_numpy_passes_through():
    xp = NumpyFramework()
    cls = make_benchmark_cls()
    original = cls.benchmark

    cls.benchmark = xp.compile(cls.benchmark)

    assert cls.benchmark is original

    rng = np.random.default_rng(0)
    x, y, z = (rng.random(100_000) for _ in range(3))
    output = cls().benchmark([x, y, z], {})[0]

    np.testing.assert_allclose(output, arithmetic_op_eager(x, y, z))


def test_pytorch_compiles():
    xp = PytorchFramework()
    cls = make_benchmark_cls()
    original = cls.benchmark

    cls.benchmark = xp.compile(cls.benchmark)

    assert cls.benchmark is not original

    x, y, z = (torch.rand(100_000) for _ in range(3))
    output = cls().benchmark([x, y, z], {})[0]

    torch.testing.assert_close(output, arithmetic_op_eager(x, y, z))


def _median_time(fn, args):
    for _ in range(3):
        fn(*args)
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def test_pytorch_compiled_is_faster():
    # Compare PyTorch with and without compilation to confirm a speedup.
    xp = PytorchFramework()

    eager_cls = make_benchmark_cls()
    compiled_cls = make_benchmark_cls()
    compiled_cls.benchmark = xp.compile(compiled_cls.benchmark)

    eager = eager_cls()
    compiled = compiled_cls()

    x, y, z = (torch.rand(1_000_000) for _ in range(3))
    data = [x, y, z]

    torch.testing.assert_close(
        compiled.benchmark(data, {})[0], eager.benchmark(data, {})[0]
    )

    eager_time = _median_time(lambda: eager.benchmark(data, {}), ())
    compiled_time = _median_time(lambda: compiled.benchmark(data, {}), ())

    speedup = eager_time / compiled_time
    print(
        f"\npytorch eager: {eager_time * 1e3:.3f} ms, "
        f"compiled: {compiled_time * 1e3:.3f} ms, speedup: {speedup:.2f}x"
    )

    assert compiled_time <= eager_time
