from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy

from saps.benchmarks.jl_kmeans import (
    JLKMeansBenchmark,
    _cost,
    _reference_cost,
)


def _blobs(seed, n=300, d=20, k=4):
    rng = np.random.default_rng(seed)
    centers = 5 * rng.standard_normal((k, d))
    points = centers[rng.integers(0, k, n)] + rng.standard_normal((n, d))
    projection = rng.choice([-1, 1], size=(d, 8)) / np.sqrt(8)
    return points.astype(np.float32), projection.astype(np.float32)


@pytest.mark.parametrize("seed", range(3))
def test_cost_matches_direct_sum_dense_and_sparse(seed):
    points, _ = _blobs(seed)
    labels = np.random.default_rng(seed).integers(0, 4, len(points))
    p = points.astype(np.float64)
    direct = sum(
        np.sum((p[labels == c] - p[labels == c].mean(axis=0)) ** 2) for c in range(4)
    )
    assert _cost(points, labels, 4) == pytest.approx(direct, rel=1e-10)
    assert _cost(sps.csr_array(points), labels, 4) == pytest.approx(direct, rel=1e-10)


def _checked(points, projection, labels, meta):
    benchmark = JLKMeansBenchmark()
    benchmark._input = [from_numpy(points), from_numpy(projection)]
    benchmark._meta = meta
    benchmark._ref_meta = {
        "reference_cost": _reference_cost(points, projection, 4, meta["max_iter"]),
        "rel_tol": 0.01,
    }
    benchmark._output = [from_numpy(labels)]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="blobs")))


def test_check_accepts_the_algorithm_and_rejects_worse_clusterings():
    points, projection = _blobs(0)
    meta = {"k": 4, "eps": 0.3, "c": 1, "max_iter": 50}
    labels = JLKMeansBenchmark().benchmark(np, meta, points, projection)
    _checked(points, projection, np.asarray(labels), meta)
    random = np.random.default_rng(0).integers(0, 4, len(points))
    with pytest.raises(AssertionError, match="cost too high"):
        _checked(points, projection, random, meta)
    with pytest.raises(AssertionError, match="Labels must be in"):
        _checked(points, projection, np.full(len(points), 4), meta)
