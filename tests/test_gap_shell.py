from unittest.mock import Mock

import pytest

import numpy as np
from scipy.sparse import coo_array

from binsparse.conversions import from_scipy, to_numpy, to_sparse

from saps.benchmark import DataInstance
from saps.benchmarks import gap
from saps.benchmarks.BFS import (
    BreadthFirstSearchDataset,
    BreadthFirstSearchGAPGenerator,
)
from saps.benchmarks.gap import (
    _MAX_DEGREES,
    GAPGraphBenchmark,
    GAPGraphGenerator,
    fetch_gap_graph,
)
from saps.benchmarks.suitesparse import SuiteSparseMatrixGenerator
from saps.metadata import _benchmark_instances


def test_gap_shell_inventory():
    generator = GAPGraphGenerator()
    datasets = generator.datasets
    assert [d.name for d in datasets] == [
        "GAP-road",
        "GAP-twitter",
        "GAP-web",
        "GAP-kron",
        "GAP-urand",
    ]
    assert {d.source_name for d in datasets} <= {
        d.source_name for d in SuiteSparseMatrixGenerator().datasets
    }
    assert set(_MAX_DEGREES) == {d.source_name for d in datasets}
    assert datasets[0].sources[:3] == [4795720, 21003853, 417968]
    assert not generator.cacheable
    assert GAPGraphBenchmark().name == "gap_graph_shell"
    assert any(isinstance(b, GAPGraphBenchmark) for b in _benchmark_instances())


def test_gap_metadata_reports_max_degree_and_sources():
    for dataset in GAPGraphGenerator().datasets:
        metadata = dataset.metadata
        assert metadata["source_name"] == f"GAP/{dataset.name}"
        assert metadata["max_degree"] == _MAX_DEGREES[dataset.source_name] > 0
        assert metadata["sources"] == dataset.sources
        assert len(set(dataset.sources)) == len(dataset.sources) == 64
    road = GAPGraphGenerator().datasets[0]
    assert road.max_degree == 9


@pytest.fixture
def fetch(monkeypatch):
    adjacency, rhs = object(), object()
    raw = DataInstance(inputs=[adjacency, rhs], meta={"shape": (3, 3), "nnz": 2})
    fetch = Mock(return_value=raw)
    monkeypatch.setattr(gap, "fetch_suitesparse_matrix", fetch)
    return fetch


def test_gap_shell_returns_matrix_max_degree_and_sources(fetch):
    road = GAPGraphGenerator().datasets[0]
    problem = fetch_gap_graph("GAP-road")
    fetch.assert_called_once_with("GAP/GAP-road")
    assert problem.inputs == fetch.return_value.inputs[:1]
    assert problem.meta == {"max_degree": 9, "sources": road.sources}
    # The shell hands out copies, so consumers cannot edit the shared sources.
    problem.meta["sources"].append(-1)
    assert -1 not in road.sources


def test_gap_source_graph_attaches_one_published_source(fetch):
    road = GAPGraphGenerator().datasets[0]
    generator = BreadthFirstSearchGAPGenerator()
    datasets = [d for d in generator.datasets if d.source_name == road.name]
    assert [d.src for d in datasets] == road.sources
    assert len({d.name for d in datasets}) == len(datasets)
    dataset = datasets[3]
    problem = generator.generate(dataset)
    assert problem.meta == {
        "max_degree": 9,
        "sources": road.sources,
        "src": road.sources[3],
    }
    assert dataset.name == f"GAP/GAP-road_{road.sources[3]}"
    assert dataset.src == road.sources[3]
    assert dataset.source_name == "GAP-road"


def test_gap_generator_rejects_unpublished_source(fetch):
    dataset = BreadthFirstSearchDataset("invalid", source_name="GAP-road", src=-1)
    with pytest.raises(ValueError, match="not a published source"):
        BreadthFirstSearchGAPGenerator().generate(dataset)


def test_gap_with_suites_does_not_mutate_shared_graphs():
    graph = GAPGraphGenerator().datasets[0]
    selected = graph.with_suites(["standard"])
    assert selected.suites == ["standard"]
    assert graph.suites == []
    dataset = BreadthFirstSearchDataset(
        "road", source_name=selected.name, src=selected.sources[0], suites=["trace"]
    )
    assert dataset.suites == ["trace"]
    assert graph.suites == []


def test_gap_consumers_preserve_published_source_cases():
    from saps.benchmarks.bellmanford import BellmanFordGAPGenerator
    from saps.benchmarks.BFS import BreadthFirstSearchGAPGenerator
    from saps.benchmarks.multi_source_shortest_paths import (
        MultiSourceShortestPathsGAPGenerator,
    )

    graphs = GAPGraphGenerator().datasets
    expected = [(f"GAP/{g.name}_{src}", src) for g in graphs for src in g.sources]
    for generator in (BreadthFirstSearchGAPGenerator(), BellmanFordGAPGenerator()):
        assert [(d.name, d.src) for d in generator.datasets] == expected
    by_name = {d.name: d for d in MultiSourceShortestPathsGAPGenerator().datasets}
    for graph in graphs:
        assert by_name[graph.source_name].sources is None


@pytest.fixture
def weighted_graph():
    return DataInstance(
        inputs=[from_scipy(coo_array([[0, 7, 0], [0, 0, -3], [0, 0, 0]]))],
        meta={"sources": [2, 0, 2], "max_degree": 1},
    )


@pytest.mark.parametrize("symmetrize", [False, True])
def test_floyd_warshall_gap_conversion(monkeypatch, weighted_graph, symmetrize):
    from saps.benchmarks import floyd_warshall as fw

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(fw, "fetch_gap_graph", load)
    dataset = fw.FloydWarshallDataset("GAP/GAP-road", symmetrize=symmetrize)
    problem = fw.FloydWarshallGAPGenerator().generate(dataset)
    expected = np.array([[0, 1, np.inf], [np.inf, 0, 1], [np.inf, np.inf, 0]])
    if symmetrize:
        expected = np.minimum(expected, expected.T)
    np.testing.assert_array_equal(to_numpy(problem.inputs[0]), expected)
    load.assert_called_once_with("GAP-road")


@pytest.mark.parametrize("symmetrize", [False, True])
def test_multi_source_gap_conversion_uses_shell_sources(
    monkeypatch, weighted_graph, symmetrize
):
    from saps.benchmarks import multi_source_shortest_paths as mssp

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(mssp, "fetch_gap_graph", load)
    dataset = mssp.MultiSourceShortestPathsDataset(
        "GAP/GAP-road", symmetrize=symmetrize
    )
    problem = mssp.MultiSourceShortestPathsGAPGenerator().generate(dataset)
    expected = np.array([[0, 1, np.inf], [np.inf, 0, 1], [np.inf, np.inf, 0]])
    if symmetrize:
        expected = np.minimum(expected, expected.T)
    np.testing.assert_array_equal(to_sparse(problem.inputs[0]).todense(), expected)
    np.testing.assert_array_equal(
        to_sparse(problem.inputs[1]).todense(),
        [[np.inf, np.inf, 0], [0, np.inf, np.inf], [np.inf, np.inf, 0]],
    )
    assert problem.meta == {"sources": [2, 0, 2]}
    assert weighted_graph.meta == {"sources": [2, 0, 2], "max_degree": 1}
    load.assert_called_once_with("GAP-road")


def test_msbfs_gap_searches_from_every_published_source(monkeypatch, weighted_graph):
    from saps.benchmarks import MSBFS

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(MSBFS, "fetch_gap_graph", load)
    generator = MSBFS.MultiSourceBreadthFirstSearchGAPGenerator()
    assert [(d.name, d.source_name) for d in generator.datasets] == [
        (f"GAP/{g.name}", g.name) for g in GAPGraphGenerator().datasets
    ]
    problem = generator.generate(generator.datasets[0])
    edges = to_sparse(problem.inputs[0]).todense()
    assert edges.dtype == bool
    np.testing.assert_array_equal(edges, [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(
        to_sparse(problem.inputs[1]).todense(),
        [[0, 0, 1], [1, 0, 0], [0, 0, 1]],
    )
    assert problem.meta == {"sources": [2, 0, 2]}
    assert weighted_graph.meta == {"sources": [2, 0, 2], "max_degree": 1}
    load.assert_called_once_with("GAP-road")


def test_bellman_ford_gap_preserves_weights_and_shared_metadata(
    monkeypatch, weighted_graph
):
    from saps.benchmarks import bellmanford as bf

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(bf, "fetch_gap_graph", load)
    dataset = bf.BellmanFordDataset("road-source", source_name="GAP-road", src=2)
    problem = bf.BellmanFordGAPGenerator().generate(dataset)
    np.testing.assert_array_equal(
        to_sparse(problem.inputs[0]).todense(),
        [[0, 7, np.inf], [np.inf, 0, -3], [np.inf, np.inf, 0]],
    )
    assert problem.meta == {**weighted_graph.meta, "src": 2}
    assert "src" not in weighted_graph.meta
    load.assert_called_once_with("GAP-road")
