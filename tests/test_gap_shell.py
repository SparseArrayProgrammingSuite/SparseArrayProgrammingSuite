from unittest.mock import Mock

import pytest

import numpy as np
from scipy.sparse import coo_array

from binsparse.conversions import from_scipy, to_sparse

from saps.benchmark import DataInstance
from saps.benchmarks import gap
from saps.benchmarks.bfs import (
    BFSDataset,
    BFSGAPGenerator,
)
from saps.benchmarks.gap import (
    _MAX_DEGREES,
    GAPGraphGenerator,
    GAPGraphShellBenchmark,
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
    assert GAPGraphShellBenchmark().name == "gap_graph_shell"
    assert any(isinstance(b, GAPGraphShellBenchmark) for b in _benchmark_instances())


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
    adjacency = from_scipy(coo_array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    rhs = object()
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
    generator = BFSGAPGenerator()
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
    assert dataset.name == f"GAP-road_src{road.sources[3]}"
    assert dataset.src == road.sources[3]
    assert dataset.source_name == "GAP-road"


def test_gap_generator_rejects_unpublished_source(fetch):
    dataset = BFSDataset("invalid", source_name="GAP-road", src=-1)
    with pytest.raises(ValueError, match="not a published source"):
        BFSGAPGenerator().generate(dataset)


def test_gap_with_suites_does_not_mutate_shared_graphs():
    graph = GAPGraphGenerator().datasets[0]
    selected = graph.with_suites(["standard"])
    assert selected.suites == ["standard"]
    assert graph.suites == []
    dataset = BFSDataset(
        "road", source_name=selected.name, src=selected.sources[0], suites=["trace"]
    )
    assert dataset.suites == ["trace"]
    assert graph.suites == []


def test_gap_consumers_preserve_published_source_cases():
    from saps.benchmarks.bellman_ford import BellmanFordGAPGenerator
    from saps.benchmarks.bfs import BFSGAPGenerator
    from saps.benchmarks.mssp import (
        MSSPGAPGenerator,
    )

    graphs = GAPGraphGenerator().datasets
    expected = [(f"{g.name}_src{src}", src) for g in graphs for src in g.sources]
    for generator in (BFSGAPGenerator(), BellmanFordGAPGenerator()):
        assert [(d.name, d.src) for d in generator.datasets] == expected
    by_name = {d.name: d for d in MSSPGAPGenerator().datasets}
    for graph in graphs:
        assert by_name[graph.name].sources is None


@pytest.fixture
def weighted_graph():
    return DataInstance(
        inputs=[from_scipy(coo_array([[0, 7, 0], [0, 0, -3], [0, 0, 0]]))],
        meta={"sources": [2, 0, 2], "max_degree": 1},
    )


@pytest.mark.parametrize("name", ["road", "twitter", "web", "kron", "urand"])
def test_floyd_warshall_gap_keeps_weights_and_direction(
    monkeypatch, weighted_graph, name
):
    from saps.benchmarks import floyd_warshall as fw

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(fw, "fetch_gap_graph", load)
    dataset = next(
        d for d in fw.FloydWarshallGAPGenerator().datasets if d.name == f"GAP-{name}"
    )
    problem = fw.FloydWarshallGAPGenerator().generate(dataset)
    expected = np.array([[0, 7, np.inf], [np.inf, 0, -3], [np.inf, np.inf, 0]])
    G = to_sparse(problem.inputs[0])
    assert G.fill_value == np.inf
    assert G.nnz == np.isfinite(expected).sum()
    np.testing.assert_array_equal(G.todense(), expected)
    load.assert_called_once_with(f"GAP-{name}")


@pytest.mark.parametrize("symmetrize", [False, True])
def test_multi_source_gap_conversion_uses_shell_sources(
    monkeypatch, weighted_graph, symmetrize
):
    from saps.benchmarks import mssp

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(mssp, "fetch_gap_graph", load)
    dataset = mssp.MSSPDataset("GAP-road", symmetrize=symmetrize)
    problem = mssp.MSSPGAPGenerator().generate(dataset)
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
    from saps.benchmarks import msbfs

    load = Mock(return_value=weighted_graph)
    monkeypatch.setattr(msbfs, "fetch_gap_graph", load)
    generator = msbfs.MSBFSGAPGenerator()
    assert [(d.name, d.source_name) for d in generator.datasets] == [
        (g.name, g.name) for g in GAPGraphGenerator().datasets
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
    from saps.benchmarks import bellman_ford as bf

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
