from unittest.mock import Mock

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse import COORMatrix, CustomTensor, ElementLevel, SparseLevel
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy, to_sparse

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmark import DataInstance
from saps.benchmarks import bellmanford, cp_als
from saps.storage import LocalStorageBackend


@pytest.mark.parametrize("keep_weights", [False, True])
@pytest.mark.parametrize("format", ["dense", "coo", "csr"])
def test_bellman_ford_distance_conversion_preserves_edges(format, keep_weights):
    adjacency = np.array([[9, 2, 0, 0], [0, 0, -3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    raw = (
        from_numpy(adjacency)
        if format == "dense"
        else from_scipy(getattr(sps, f"{format}_array")(adjacency))
    )

    distances = bellmanford._adjacency_to_distance(raw, keep_weights)

    expected = np.full((4, 4), np.inf)
    np.fill_diagonal(expected, 0)
    expected[0, 1] = 2 if keep_weights else 1
    expected[1, 2] = -3 if keep_weights else 1
    np.testing.assert_array_equal(to_sparse(distances).todense(), expected)
    assert distances.number_of_stored_values == 6
    assert distances.fill_value == np.inf
    # The cached source remains unchanged, including its self-loop.
    source = to_numpy(raw) if format == "dense" else to_scipy(raw).toarray()
    np.testing.assert_array_equal(source, adjacency)


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
def test_bellman_ford_runs_with_infinity_filled_sparse_input(framework):
    raw = from_scipy(
        sps.coo_array([[0, 2, 10, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
    distances = bellmanford._adjacency_to_distance(raw, keep_weights=True)
    xp = framework()

    result = bellmanford.BellmanFordBenchmark().benchmark(
        xp, [xp.from_binsparse(distances)], {"src": 0}
    )[0]

    if hasattr(result, "todense"):
        result = result.todense()
    np.testing.assert_array_equal(result, [0, 2, 5, np.inf])


def test_bellman_ford_large_graph_setup_and_cache_stay_sparse(monkeypatch, tmp_path):
    size = 1_000_000
    raw = from_scipy(
        sps.coo_array(([2.0, -3.0], ([0, 1], [1, size - 1])), shape=(size, size))
    )
    numpy_full = np.full

    def forbid_dense_matrix(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or len(shape) != 2
        return numpy_full(shape, *args, **kwargs)

    monkeypatch.setattr(bellmanford.np, "full", forbid_dense_matrix)
    monkeypatch.setattr(
        bellmanford, "to_numpy", Mock(side_effect=AssertionError("densification"))
    )
    monkeypatch.setattr(
        bellmanford,
        "fetch_suitesparse_matrix",
        lambda _: DataInstance(inputs=[raw], meta={}),
    )
    generator = bellmanford.BellmanFordGAPGenerator()
    problem = generator.generate(generator.datasets[0])
    distances = problem.inputs[0]

    assert distances.shape == (size, size)
    assert distances.number_of_stored_values == size + 2
    assert distances.fill_value == np.inf
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    path = tmp_path / "distances.bsp.h5"
    backend.serialize_data_to_file(problem, path)
    restored = backend.deserialize_data_from_file(path).inputs[0]
    assert restored.fill_value == np.inf
    assert restored.number_of_stored_values == size + 2
    np.testing.assert_array_equal(restored.values, distances.values)


def test_bellman_ford_coalesces_duplicates_and_ignores_zero_edges():
    raw = COORMatrix(
        (3, 3),
        4,
        values=np.array([2.0, 3.0, 0.0, 7.0]),
        indices_0=np.array([0, 0, 1, 2]),
        indices_1=np.array([1, 1, 2, 2]),
    )

    result = bellmanford._adjacency_to_distance(raw, keep_weights=True)

    np.testing.assert_array_equal(
        to_sparse(result).todense(),
        [[0, 5, np.inf], [np.inf, 0, np.inf], [np.inf, np.inf, 0]],
    )
    np.testing.assert_array_equal(raw.values, [2.0, 3.0, 0.0, 7.0])


@pytest.mark.parametrize("order", [3, 4, 5])
@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32, np.float64])
def test_cp_frostt_setup_reads_sparse_values_and_builds_float_factors(
    monkeypatch, order, dtype
):
    shape = tuple(1000 + mode for mode in range(order))
    tensor = CustomTensor(
        shape,
        1,
        level=SparseLevel(
            order,
            ElementLevel(np.array([1], dtype=dtype)),
            tuple(np.array([0]) for _ in shape),
        ),
    )
    monkeypatch.setattr(
        cp_als,
        "fetch_frostt_tensor",
        lambda _: DataInstance(inputs=[tensor], meta={"shape": shape}),
    )
    monkeypatch.setattr(
        cp_als, "to_numpy", Mock(side_effect=AssertionError("densification"))
    )
    dataset = cp_als.CPFrosttDataset("example", "Example", "example", order, rank=2)
    generator = cp_als.CPNFrosttGenerator()

    problem = generator.generate(dataset)
    repeated = generator.generate(dataset)

    assert problem.inputs[0] is tensor
    for mode, (factor, repeat) in enumerate(
        zip(problem.inputs[1:], repeated.inputs[1:], strict=True)
    ):
        array = to_numpy(factor)
        assert array.shape == (shape[mode], 2)
        assert array.dtype.kind == "f"
        assert np.all((array > 0) & (array < 1))
        if dtype == np.float32:
            assert array.dtype == np.float32
        np.testing.assert_array_equal(array, to_numpy(repeat))
