import sys
import time

import pytest

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
        def benchmark(self, xp, data, meta):
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
    output = cls().benchmark(xp, [x, y, z], {})[0]

    np.testing.assert_allclose(output, arithmetic_op_eager(x, y, z))


def test_pytorch_compiles():
    xp = PytorchFramework()
    cls = make_benchmark_cls()
    original = cls.benchmark

    cls.benchmark = xp.compile(cls.benchmark)

    assert cls.benchmark is not original

    x, y, z = (torch.rand(100_000) for _ in range(3))
    output = cls().benchmark(xp, [x, y, z], {})[0]

    torch.testing.assert_close(output, arithmetic_op_eager(x, y, z))


def test_pytorch_einsum_compiles_tensor_operations():
    xp = PytorchFramework()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    Q, D = torch.randn(2, 4), torch.randn(3, 4)
    with torch._dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(xp.einsum, backend=backend)
        for operator in ("-", "+"):
            output = compiled(f"X[i, j, k] = Q[i, k] {operator} D[j, k]", Q=Q, D=D)
            expected = Q[:, None, :] + (-1 if operator == "-" else 1) * D[None, :, :]
            torch.testing.assert_close(output, expected)
        output = compiled("X[i, k] = Q[i, k] ** 2", Q=Q)
        torch.testing.assert_close(output, Q**2)

    targets = {node.target for graph in graphs for node in graph.graph.nodes}
    assert {torch.subtract, torch.add, torch.pow} <= targets


def test_pytorch_compiled_jl_matches_eager():
    from saps.benchmarks.approx_nn import JLApproxNearestNeighbor

    benchmark = JLApproxNearestNeighbor()
    param = next(
        param
        for param in benchmark.params
        if str(param) == "jl_projection_inputs.small"
    )
    xp = PytorchFramework()
    benchmark.setup(param, xp=xp, use_cache=False)
    data = [xp.from_binsparse(array) for array in benchmark._input]
    expected = benchmark.benchmark(xp, data, benchmark._meta)
    with torch._dynamo.config.patch(suppress_errors=False):
        actual = benchmark._compiled_benchmark(data, benchmark._meta)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference)


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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="torch.compile CPU timing is too noisy on Windows CI",
)
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
        compiled.benchmark(xp, data, {})[0], eager.benchmark(xp, data, {})[0]
    )

    eager_time = _median_time(lambda: eager.benchmark(xp, data, {}), ())
    compiled_time = _median_time(lambda: compiled.benchmark(xp, data, {}), ())

    speedup = eager_time / compiled_time
    print(
        f"\npytorch eager: {eager_time * 1e3:.3f} ms, "
        f"compiled: {compiled_time * 1e3:.3f} ms, speedup: {speedup:.2f}x"
    )

    assert compiled_time <= eager_time
