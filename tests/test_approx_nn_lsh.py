import pytest

import numpy as np
import scipy.sparse
from scipy.spatial.distance import cdist

from binsparse.conversions import from_numpy, from_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.approx_nn import SimHashApproxNearestNeighbor


@pytest.fixture(
    params=[
        (NumpyFramework, False),
        (SciPyFramework, False),
        (PytorchFramework, False),
        (PyDataSparseFramework, False),
        (PyDataSparseFramework, True),
    ],
    ids=["numpy", "scipy", "pytorch", "sparse-dense-input", "sparse"],
)
def run_lsh(request):
    framework, sparse_inputs = request.param
    xp = framework()

    def run(data, query, projection, **meta):
        n_projections = np.asarray(projection).shape[1] // meta["n_tables"]
        meta["n_projections"] = n_projections
        arrays = []
        for array in (data, query, projection):
            array = np.asarray(array, dtype=np.float64)
            tensor = (
                from_scipy(scipy.sparse.coo_array(array))
                if sparse_inputs
                else from_numpy(array)
            )
            arrays.append(xp.from_binsparse(tensor))
        outputs = SimHashApproxNearestNeighbor().benchmark(xp, arrays, meta)
        return [
            NumpyFramework().from_binsparse(xp.to_binsparse(output))
            for output in outputs
        ]

    return run


def test_lsh_exhaustive_candidates_match_cosine_knn(run_lsh):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(9, 4))
    query = rng.normal(size=(3, 4))
    projection = rng.normal(size=(4, 6))
    data[np.abs(data) < 0.5] = 0
    query[np.abs(query) < 0.5] = 0
    indices, distances = run_lsh(
        data, query, projection, k=3, n_tables=2, candidate_target=9
    )
    expected_distances = cdist(query, data, metric="cosine")
    expected_indices = np.argsort(expected_distances, axis=1)[:, :3]
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        distances,
        np.take_along_axis(expected_distances, expected_indices, axis=1),
        atol=1e-8,
    )


def test_lsh_queries_stop_independently_and_exclude_non_candidates(run_lsh):
    # d0 matches query 0 exactly on the full 2-sign prefix, so query 0 is
    # satisfied after the first (strictest) round. d2 only matches query 0's
    # first sign -- it needs the shorter, 1-sign prefix round -- but is
    # nonetheless *closer* in cosine distance than d0. Query 0 must not pick
    # it up once it has already stopped. Query 1 needs that same shorter
    # prefix to find d1 at all, so the search does keep going past round 0.
    data = [[1, 5], [-1, 5], [1, -0.05]]  # d0, d1, d2
    query = [[1, 0.1], [-1, -0.1]]
    projection = [[1, 0], [0, 1]]
    indices, distances = run_lsh(
        data,
        query,
        projection,
        k=1,
        n_tables=1,
        candidate_target=1,
    )
    expected_distances = cdist(query, data, metric="cosine")
    np.testing.assert_array_equal(indices, [[0], [1]])
    np.testing.assert_allclose(
        distances[:, 0],
        [expected_distances[0, 0], expected_distances[1, 1]],
        atol=1e-8,
    )
    # The trap point (d2) really is closer to query 0 than its accepted match.
    assert expected_distances[0, 2] < expected_distances[0, 0]


def test_lsh_unions_tables_and_returns_distinct_neighbors(run_lsh):
    # d0 agrees with the query's sign in both tables; d1 only agrees in
    # table 1 (feature 1's sign), and d2 only agrees in table 0 (feature 0's
    # sign). Each should be picked up via the OR across tables.
    data = [[0.1, 0.1], [-3.1, 0.1], [0.1, -2.1]]
    query = [[0.05, 0.05]]
    projection = [[1, 0], [0, 1]]
    indices, distances = run_lsh(
        data,
        query,
        projection,
        k=3,
        n_tables=2,
        candidate_target=3,
    )
    expected_distances = cdist(query, data, metric="cosine")
    expected_indices = np.argsort(expected_distances, axis=1)[:, :3]
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        distances,
        np.take_along_axis(expected_distances, expected_indices, axis=1),
        atol=1e-8,
    )


def test_lsh_shortens_the_required_prefix_until_opposite_signed_projections_collide(
    run_lsh,
):
    # Every hash bit depends only on feature 0's sign, so the query (positive
    # feature 0) disagrees with every point (all negative feature 0) on
    # every sign in every table -- no exact-match prefix length shorter
    # than the fully relaxed round (accept everyone) picks them up.
    data = [[-2, 0], [-1, 0], [-3, 1]]
    query = [[1, 0]]
    projection = [[1, 1, 1, 1], [0, 0, 0, 0]]
    indices, distances = run_lsh(
        data,
        query,
        projection,
        k=2,
        n_tables=2,
        candidate_target=1,
    )
    expected_distances = cdist(query, data, metric="cosine")
    expected_indices = np.argsort(expected_distances, axis=1)[:, :2]
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        distances,
        np.take_along_axis(expected_distances, expected_indices, axis=1),
        atol=1e-8,
    )
