from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmarks.simhash_ann import SimHashANNBenchmark, _nearest_cosine_distances


def _points(seed, n=200, queries=40, d=16):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.standard_normal((queries, d))


@pytest.mark.parametrize("sparse", [False, True])
def test_nearest_cosine_distances_match_cdist(sparse):
    data, query = _points(0)
    expected = cdist(query, data, metric="cosine").min(axis=1)
    if sparse:
        data, query = sps.csr_array(data), sps.csr_array(query)
    np.testing.assert_allclose(
        _nearest_cosine_distances(data, query, chunk=7), expected
    )


def _checked(data, query, indices, target_probability=0.9, sparse=False):
    benchmark = SimHashANNBenchmark()
    convert = (lambda a: from_scipy(sps.coo_array(a))) if sparse else from_numpy
    benchmark._input = [convert(data), convert(query)]
    benchmark._meta = {"eps": 0.1, "target_probability": target_probability}
    benchmark._ref_outputs = [from_numpy(_nearest_cosine_distances(data, query))]
    benchmark._output = [from_numpy(indices), from_numpy(np.zeros(indices.shape))]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="points")))


@pytest.mark.parametrize("sparse", [False, True])
def test_check_counts_approximate_answers(sparse):
    data, query = _points(1)
    order = np.argsort(cdist(query, data, metric="cosine"), axis=1)[:, :3]
    _checked(data, query, order, sparse=sparse)
    # Answering the 50th nearest neighbor is not (1 + 0.1)-approximate.
    wrong = order.copy()
    wrong[:8, 0] = np.argsort(cdist(query[:8], data, metric="cosine"), axis=1)[:, 50]
    _checked(data, query, wrong, target_probability=0.8, sparse=sparse)
    with pytest.raises(AssertionError, match="approximate nearest neighbor"):
        _checked(data, query, wrong, target_probability=0.9, sparse=sparse)
