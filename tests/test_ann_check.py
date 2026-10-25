from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist

from binsparse.conversions import from_numpy, from_scipy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.simhash_ann import SimHashANNBenchmark, _simhash_reference


def _problem(seed, n=200, queries=40, d=16, candidate_target=20, sparse=False):
    rng = np.random.default_rng(seed)
    data, query = rng.standard_normal((n, d)), rng.standard_normal((queries, d))
    if sparse:
        data[rng.random(data.shape) < 0.6] = 0
        query[rng.random(query.shape) < 0.6] = 0
        data, query = sps.csr_array(data), sps.csr_array(query)
    projection = rng.standard_normal((d, 4 * 3))
    meta = {
        "k": 3,
        "n_projections": 3,
        "n_tables": 4,
        "candidate_target": candidate_target,
        "eps": 0.1,
        "target_probability": 0.9,
    }
    return data, query, projection, meta


def _dense(array):
    return array.toarray() if sps.issparse(array) else array


@pytest.mark.parametrize("sparse", [False, True])
def test_reference_with_every_point_a_candidate_is_exact_knn(sparse):
    data, query, projection, meta = _problem(0, candidate_target=200, sparse=sparse)
    indices, distances = _simhash_reference(data, query, projection, meta)
    exact = cdist(_dense(query), _dense(data), metric="cosine")
    expected = np.argsort(exact, axis=1, kind="stable")[:, :3]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(distances, np.take_along_axis(exact, expected, 1))


@pytest.mark.parametrize("sparse", [False, True])
def test_reference_matches_the_benchmark(sparse):
    data, query, projection, meta = _problem(1, sparse=sparse)
    xp = NumpyFramework()
    convert = (lambda a: from_scipy(sps.coo_array(a))) if sparse else from_numpy
    args = [xp.from_binsparse(convert(a)) for a in (data, query, projection)]
    got, _ = SimHashANNBenchmark().benchmark(xp, meta, *args)
    indices, _ = _simhash_reference(data, query, projection, meta)
    np.testing.assert_array_equal(to_numpy(xp.to_binsparse(got)), indices)


def _checked(data, query, projection, meta, indices, target_probability=0.9):
    benchmark = SimHashANNBenchmark()
    benchmark._input = [from_numpy(data), from_numpy(query)]
    benchmark._meta = {**meta, "target_probability": target_probability}
    _, distances = _simhash_reference(data, query, projection, meta)
    benchmark._ref_outputs = [from_numpy(distances[:, -1])]
    benchmark._output = [from_numpy(indices), from_numpy(np.zeros(indices.shape))]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="points")))


def test_check_grades_neighbors_within_the_reference_radius():
    data, query, projection, meta = _problem(2)
    indices, _ = _simhash_reference(data, query, projection, meta)
    _checked(data, query, projection, meta, indices)
    # The farthest point from a query is outside its radius; swapping one in
    # for every neighbor of 6 of 40 queries leaves 85% of neighbors within it.
    wrong = indices.copy()
    far = np.argmax(cdist(query, data, metric="cosine"), axis=1)
    wrong[:6] = far[:6, None]
    _checked(data, query, projection, meta, wrong, target_probability=0.85)
    with pytest.raises(AssertionError, match="within the reference radius"):
        _checked(data, query, projection, meta, wrong, target_probability=0.9)
