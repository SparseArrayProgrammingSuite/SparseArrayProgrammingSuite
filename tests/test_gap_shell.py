from unittest.mock import Mock

import pytest

from saps.benchmark import DataInstance
from saps.benchmarks import gap
from saps.benchmarks.gap import (
    _MAX_DEGREES,
    GAPGraphBenchmark,
    GAPGraphGenerator,
    GAPSourceDataset,
    fetch_gap_graph,
    fetch_gap_source_graph,
    gap_source_datasets,
)
from saps.benchmarks.suitesparse import (
    _GAP_KRON_SOURCES,
    _GAP_ROAD_SOURCES,
    _GAP_TWITTER_SOURCES,
    _GAP_URAND_SOURCES,
    _GAP_WEB_SOURCES,
    SuiteSparseMatrixGenerator,
)
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
    assert [d.sources for d in datasets] == [
        _GAP_ROAD_SOURCES,
        _GAP_TWITTER_SOURCES,
        _GAP_WEB_SOURCES,
        _GAP_KRON_SOURCES,
        _GAP_URAND_SOURCES,
    ]
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
    assert road.sources == _GAP_ROAD_SOURCES


def test_gap_source_graph_attaches_one_published_source(fetch):
    road = GAPGraphGenerator().datasets[0]
    datasets = gap_source_datasets(road)
    assert [d.src for d in datasets] == road.sources
    assert len({d.name for d in datasets}) == len(datasets)
    dataset = datasets[3]
    problem = fetch_gap_source_graph(dataset)
    assert problem.meta == {
        "max_degree": 9,
        "sources": road.sources,
        "src": road.sources[3],
    }
    assert dataset.name == f"GAP-road_{road.sources[3]}"
    assert dataset.metadata["src"] == road.sources[3]
    assert dataset.metadata["graph"] == "GAP-road"
    assert dataset.metadata["max_degree"] == 9


def test_gap_source_dataset_rejects_unpublished_source():
    road = GAPGraphGenerator().datasets[0]
    with pytest.raises(ValueError, match="not a published source"):
        GAPSourceDataset(road, -1)


def test_gap_with_suites_does_not_mutate_shared_graphs():
    graph = GAPGraphGenerator().datasets[0]
    selected = graph.with_suites(["standard"])
    assert selected.suites == ["standard"]
    assert graph.suites == []
    assert GAPSourceDataset(selected, selected.sources[0], suites=["trace"]).suites == [
        "standard",
        "trace",
    ]
