import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmarks.gcn import (
    _SIGMAS,
    GCNBenchmark,
    GCNTestGenerator,
    gcn_reference,
    gcn_reference_np,
)


def _random_gcn(seed, n=200, features=24, hidden=16, outputs=3):
    rng = np.random.default_rng(seed)
    adjacency = sps.random_array(
        (n, n), density=0.05, rng=rng, format="csr", dtype=np.float32
    )
    return (
        adjacency,
        rng.standard_normal((n, features), dtype=np.float32),
        rng.standard_normal((features, hidden), dtype=np.float32),
        rng.standard_normal(hidden, dtype=np.float32),
        rng.standard_normal((hidden, outputs), dtype=np.float32),
        rng.standard_normal(outputs, dtype=np.float32),
    )


@pytest.mark.parametrize("dataset", GCNTestGenerator().datasets, ids=str)
def test_reference_matches_hand_computed_outputs(dataset):
    if dataset.expected is None:
        pytest.skip("no hand-computed output")
    exact, _ = gcn_reference(
        dataset.adjacency,
        dataset.features,
        dataset.weights1,
        dataset.bias1,
        dataset.weights2,
        dataset.bias2,
    )
    np.testing.assert_allclose(exact, dataset.expected, rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_reference_is_exact_and_bounds_float32_rounding(seed):
    arrays = _random_gcn(seed)
    exact, sigma = gcn_reference(*arrays)

    as64 = [arrays[0].astype(np.float64), *(a.astype(np.float64) for a in arrays[1:])]
    np.testing.assert_allclose(exact, gcn_reference_np(*as64), rtol=1e-12)
    # Both orders of each layer's products round differently; both are bounded.
    float32 = gcn_reference_np(*arrays)
    adjacency, x, w1, b1, w2, b2 = arrays
    h = np.maximum(adjacency @ (x @ w1) + b1, 0)
    reassociated = adjacency @ (h @ w2) + b2
    for output in (float32, reassociated):
        assert output.dtype == np.float32
        assert np.all(np.abs(output - exact) <= _SIGMAS * sigma)


def test_check_accepts_rounding_and_rejects_larger_errors():
    benchmark = GCNBenchmark()
    param = next(p for p in benchmark.params if p.generator.name == "gcn_test")
    arrays = _random_gcn(0)
    exact, sigma = gcn_reference(*arrays)
    benchmark._ref_outputs = [from_numpy(exact), from_numpy(sigma)]
    benchmark._ref_meta = {"sigmas": _SIGMAS}

    benchmark._output = [from_numpy(gcn_reference_np(*arrays))]
    benchmark.check(param)

    off = exact.copy()
    off[7, 1] += 1.5 * _SIGMAS * sigma[7, 1]
    benchmark._output = [from_numpy(off)]
    with pytest.raises(AssertionError, match="standard deviations"):
        benchmark.check(param)
    assert to_numpy(benchmark._ref_outputs[1]).shape == exact.shape
