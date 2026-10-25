from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmarks.gcn import _SIGMAS
from saps.benchmarks.gcn_backward import (
    GCNBackwardBenchmark,
    _gcn_loss,
    _loss_and_sigma,
    _training_meta,
)


def _random_gcn(seed, n=120, features=12, hidden=8, outputs=2):
    rng = np.random.default_rng(seed)
    adjacency = sps.random_array(
        (n, n), density=0.05, rng=rng, format="csr", dtype=np.float32
    )
    return (
        adjacency,
        rng.standard_normal((n, features), dtype=np.float32),
        0.3 * rng.standard_normal((features, hidden), dtype=np.float32),
        np.zeros(hidden, dtype=np.float32),
        0.3 * rng.standard_normal((hidden, outputs), dtype=np.float32),
        np.zeros(outputs, dtype=np.float32),
        rng.standard_normal((n, outputs), dtype=np.float32),
    )


@pytest.mark.parametrize("seed", range(5))
def test_loss_sigma_bounds_float32_in_either_order(seed):
    adjacency, x, w1, b1, w2, b2, targets = _random_gcn(seed)
    exact = _gcn_loss(
        adjacency.astype(np.float64),
        *(a.astype(np.float64) for a in (x, w1, b1, w2, b2, targets)),
    )
    loss, sigma = _loss_and_sigma(adjacency, x, w1, b1, w2, b2, targets)
    assert loss == pytest.approx(exact, rel=1e-12)
    h = np.maximum(adjacency @ (x @ w1) + b1, 0)
    for predictions in (
        (adjacency @ np.maximum((adjacency @ x) @ w1 + b1, 0)) @ w2 + b2,
        adjacency @ (h @ w2) + b2,
    ):
        assert predictions.dtype == np.float32
        float32 = np.sum((predictions - targets) ** 2, dtype=np.float32) / len(targets)
        assert abs(float32 - exact) <= _SIGMAS * sigma


def _checked(arrays, meta, output):
    benchmark = GCNBackwardBenchmark()
    benchmark._input = [from_scipy(arrays[0]), *(from_numpy(a) for a in arrays[1:])]
    benchmark._meta = meta
    benchmark._ref_meta = {}
    benchmark._output = [from_numpy(np.zeros(1, np.float32)), *map(from_numpy, output)]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="random")))


def test_check_accepts_training_to_target_and_rejects_less():
    arrays = _random_gcn(0)
    dataset = SimpleNamespace(name="random", num_iterations=20, learning_rate=0.05)
    meta = _training_meta(*arrays, dataset)
    benchmark = GCNBackwardBenchmark()
    output = benchmark.benchmark(np, meta, *arrays)
    _checked(arrays, meta, [np.asarray(a, np.float32) for a in output[1:]])
    with pytest.raises(AssertionError, match="loss too high"):
        _checked(arrays, meta, arrays[2:6])


def test_training_meta_rejects_diverging_descent():
    arrays = _random_gcn(0)
    dataset = SimpleNamespace(name="random", num_iterations=20, learning_rate=50.0)
    with pytest.raises(ValueError, match="doesn't lower the loss"):
        _training_meta(*arrays, dataset)
