import pytest

import numpy as np
import scipy.sparse

from binsparse.conversions import from_numpy, from_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.approx_nn import JLApproxNearestNeighbor


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
        arrays = []
        for array in (data, query, projection):
            array = np.asarray(array, dtype=np.float64)
            tensor = (
                from_scipy(scipy.sparse.coo_array(array))
                if sparse_inputs
                else from_numpy(array)
            )
            arrays.append(xp.from_binsparse(tensor))
        outputs = JLApproxNearestNeighbor().benchmark(xp, arrays, meta)
        return [
            NumpyFramework().from_binsparse(xp.to_binsparse(output))
            for output in outputs
        ]

    return run


def test_lsh_exhaustive_candidates_match_euclidean_knn(run_lsh):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(9, 4))
    query = rng.normal(size=(3, 4))
    projection = rng.normal(size=(4, 6))
    data[np.abs(data) < 0.5] = 0
    query[np.abs(query) < 0.5] = 0
    indices, distances = run_lsh(
        data, query, projection, k=3, hash_bits=3, n_tables=2, candidate_target=9
    )
    expected_distances = np.linalg.norm(query[:, None, :] - data[None, :, :], axis=2)
    expected_indices = np.argsort(expected_distances, axis=1)[:, :3]
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        distances, np.take_along_axis(expected_distances, expected_indices, axis=1)
    )


def test_lsh_queries_stop_independently_and_exclude_non_candidates(run_lsh):
    # Query 0 matches code 11 immediately. Query 1 has code 00 and must
    # shorten it to match 01; query 0 must not then acquire its closer 10 point.
    data = [[5, 5], [1, -1], [-2, 1]]
    query = [[0.1, 0.1], [-0.1, -0.1]]
    indices, distances = run_lsh(
        data, query, np.eye(2), k=1, hash_bits=2, n_tables=1, candidate_target=1
    )
    np.testing.assert_array_equal(indices, [[0], [2]])
    np.testing.assert_allclose(
        distances[:, 0], [np.hypot(4.9, 4.9), np.hypot(1.9, 1.1)]
    )


def test_lsh_unions_tables_and_returns_distinct_neighbors(run_lsh):
    # The first point collides in both tables; the next two each collide in one.
    indices, distances = run_lsh(
        [[3, 1], [3, -1], [-2, 1], [-2, -1]],
        [[1, 1]],
        np.eye(2),
        k=3,
        hash_bits=1,
        n_tables=2,
        candidate_target=3,
    )
    np.testing.assert_array_equal(indices, [[0, 1, 2]])
    np.testing.assert_allclose(distances, [[2, np.sqrt(8), 3]])


def test_lsh_32_bit_codes_reach_empty_prefix_and_at_least_k_candidates(run_lsh):
    # 0xffffffff must stay nonnegative so dropping all 32 bits reaches zero.
    indices, distances = run_lsh(
        [[-2, 0], [-1, 0], [-3, 1]],
        [[1, 0]],
        np.vstack([np.ones(64), np.zeros(64)]),
        k=2,
        hash_bits=32,
        n_tables=2,
        candidate_target=1,
    )
    np.testing.assert_array_equal(indices, [[1, 0]])
    np.testing.assert_allclose(distances, [[2, 3]])
