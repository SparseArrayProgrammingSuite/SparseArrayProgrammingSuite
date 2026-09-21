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
        pytest.param(
            (PytorchFramework, False),
            marks=pytest.mark.xfail(
                reason="PyTorch rejects unsigned tensor indices",
                raises=IndexError,
                strict=True,
            ),
        ),
        (PyDataSparseFramework, False),
        (PyDataSparseFramework, True),
    ],
    ids=["numpy", "scipy", "pytorch", "sparse-dense-input", "sparse"],
)
def run_lsh(request):
    framework, sparse_inputs = request.param
    xp = framework()

    def run(data, query, projection, *, eps=1.0, offsets=None, strides=None, **meta):
        arrays = []
        for array in (data, query, projection):
            array = np.asarray(array, dtype=np.float64)
            tensor = (
                from_scipy(scipy.sparse.coo_array(array))
                if sparse_inputs
                else from_numpy(array)
            )
            arrays.append(xp.from_binsparse(tensor))
        if offsets is None:
            offsets = np.full((meta["n_tables"], meta["hash_bits"]), 0.5)
        if strides is None:
            strides = np.arange(1, meta["hash_bits"] + 1, dtype=np.int64)
        arrays.extend(xp.from_binsparse(from_numpy(x)) for x in (offsets, strides))
        meta["eps"] = eps
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


@pytest.mark.parametrize(
    "eps,expected_indices,expected_distances",
    [(1.0, [[0], [2]], [0.98, 1.0]), (2.0, [[1], [2]], [0.02, 1.0])],
)
def test_lsh_queries_stop_independently_and_exclude_non_candidates(
    run_lsh, eps, expected_indices, expected_distances
):
    # Query 0 stops in bin 0. Query 1 needs width 2; query 0 must not then
    # acquire its closer point across the original negative bin boundary.
    data = [[0.49], [-0.51], [2.51]]
    query = [[-0.49], [1.51]]
    indices, distances = run_lsh(
        data,
        query,
        [[1, 0, 0, 0]],
        k=1,
        eps=eps,
        hash_bits=4,
        n_tables=1,
        candidate_target=1,
    )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances[:, 0], expected_distances)


def test_lsh_unions_tables_and_returns_distinct_neighbors(run_lsh):
    # The first point collides in both tables; the next two each collide in one.
    indices, distances = run_lsh(
        [[0.1, 0.1], [0.1, 2.1], [3.1, 0.1], [3.1, 2.1]],
        [[0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
        k=3,
        hash_bits=4,
        n_tables=2,
        candidate_target=3,
    )
    np.testing.assert_array_equal(indices, [[0, 1, 2]])
    np.testing.assert_allclose(
        distances, [[np.sqrt(0.02), np.sqrt(4.42), np.sqrt(9.62)]]
    )


@pytest.mark.parametrize(
    "run_lsh",
    [(PyDataSparseFramework, False), (PyDataSparseFramework, True)],
    indirect=True,
    ids=["dense-input", "sparse-input"],
)
def test_lsh_31_bit_codes_widen_bins_to_at_least_k_candidates(run_lsh):
    # Negative and positive projections must eventually share wide enough bins.
    indices, distances = run_lsh(
        [[-2, 0], [-1, 0], [-3, 1]],
        [[1, 0]],
        np.vstack([np.ones(62), np.zeros(62)]),
        k=2,
        hash_bits=31,
        n_tables=2,
        candidate_target=1,
    )
    np.testing.assert_array_equal(indices, [[1, 0]])
    np.testing.assert_allclose(distances, [[2, 3]])
