from unittest.mock import Mock, call

import pytest

import numpy as np
from scipy.sparse import coo_array

import networkx as nx
from binsparse.conversions import from_scipy, to_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmark import DataInstance
from saps.benchmarks import betweenness_centrality as bc
from saps.benchmarks.gap import GAPGraphGenerator
from saps.benchmarks.snap import SNAPGraphGenerator

# These generators' references compute in-process here.
pytestmark = pytest.mark.usefixtures("uncached_references")


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
@pytest.mark.parametrize(
    "dataset", bc.BetweennessCentralityTestGenerator().datasets, ids=lambda d: d.name
)
def test_betweenness_test_problems(framework, dataset):
    problem = bc.BetweennessCentralityTestGenerator().generate(dataset)
    xp = framework()
    output = bc.BetweennessCentralityBenchmark().benchmark(
        xp, problem.meta, *(xp.from_binsparse(x) for x in problem.inputs)
    )
    np.testing.assert_allclose(
        to_numpy(xp.to_binsparse(output)), to_numpy(problem.ref_outputs[0])
    )


def test_gap_trials_partition_published_sources():
    datasets = bc.BetweennessCentralityGAPGenerator().datasets
    assert len(datasets) == len({d.name for d in datasets}) == 80
    for graph in GAPGraphGenerator().datasets:
        trials = [d for d in datasets if d.source_name == graph.name]
        assert [d.trial for d in trials] == list(range(16))
        assert [source for d in trials for source in d.sources] == graph.sources
        for dataset in trials:
            assert len(dataset.sources) == 4
            assert dataset.name == f"{graph.name}_trial{dataset.trial}"
            assert dataset.suites == ["standard"]
            assert dataset.metadata["sources"] == dataset.sources
            assert dataset.metadata["trial"] == dataset.trial


@pytest.mark.parametrize("trial", [-1, 16])
@pytest.mark.parametrize(
    ("dataset_class", "source_name"),
    [
        (bc.BetweennessCentralityGAPDataset, "GAP-road"),
        (bc.BetweennessCentralitySNAPDataset, "roadNet-CA"),
    ],
)
def test_trial_rejects_invalid_index(dataset_class, source_name, trial):
    with pytest.raises(ValueError, match="between 0 and 15"):
        dataset_class(source_name, trial)


def test_snap_trials_partition_source_seeds():
    datasets = bc.BetweennessCentralitySNAPGenerator().datasets
    graphs = SNAPGraphGenerator().datasets
    assert len(datasets) == len({d.name for d in datasets}) == 16 * len(graphs)
    for graph in graphs:
        trials = [d for d in datasets if d.source_name == graph.name]
        assert [d.trial for d in trials] == list(range(16))
        assert [seed for d in trials for seed in d.source_seeds] == list(range(64))
        for dataset in trials:
            assert dataset.name == f"{graph.name}_trial{dataset.trial}"
            assert "standard" in dataset.suites
            assert dataset.metadata["source_seeds"] == dataset.source_seeds
            assert dataset.metadata["trial"] == dataset.trial
    assert {d.source_name for d in datasets if "trace" in d.suites} == {
        "email-Eu-core",
        "email-Eu-core-temporal",
    }


def test_snap_generator_preserves_source_order_and_repeats(monkeypatch):
    # SNAP samples with replacement; repeated vertices must contribute again.
    sources = [0, 0, 2, 1, 2, 1, 1, 0] * 8
    raw = DataInstance(
        inputs=[from_scipy(coo_array([[0, -7, 0], [0, 0, 3], [0, 0, 0]]))],
        meta={"sources": sources, "max_degree": 1},
    )
    fetch = Mock(return_value=raw)
    monkeypatch.setattr(bc, "fetch_snap_graph", fetch)
    generator = bc.BetweennessCentralitySNAPGenerator()
    for trial in range(16):
        dataset = bc.BetweennessCentralitySNAPDataset("roadNet-CA", trial)
        problem = generator.generate(dataset)
        assert problem.meta["bc_sources"] == sources[4 * trial : 4 * (trial + 1)]
        assert problem.meta["trial"] == trial
        assert problem.meta["include_source"] is True
        assert problem.meta["normalize"] is True
        np.testing.assert_array_equal(
            to_scipy(problem.inputs[0]).toarray(), [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        )
        problem.meta["bc_sources"].append(-1)
    assert all(call.args == ("roadNet-CA",) for call in fetch.call_args_list)
    assert raw.meta == {"sources": sources, "max_degree": 1}
    assert len(sources) == 64


def test_snap_generator_rejects_incomplete_trial(monkeypatch):
    monkeypatch.setattr(
        bc, "fetch_snap_graph", lambda _: DataInstance(inputs=[], meta={"sources": [0]})
    )
    dataset = bc.BetweennessCentralitySNAPDataset("roadNet-CA", 0)
    with pytest.raises(ValueError, match="four source vertices"):
        bc.BetweennessCentralitySNAPGenerator().generate(dataset)


def test_gap_generator_selects_trial_and_discards_weights(monkeypatch):
    raw = DataInstance(
        inputs=[from_scipy(coo_array([[0, 7, 0], [0, 0, 3], [0, 0, 0]]))],
        meta={"sources": list(range(64)), "max_degree": 1},
    )
    fetch = Mock(return_value=raw)
    monkeypatch.setattr(bc, "fetch_gap_graph", fetch)
    dataset = bc.BetweennessCentralityGAPDataset("GAP-road", 3)
    # Published IDs are tested separately; numerical checks need local vertices.
    dataset.sources = [0, 1, 2, 0]
    problem = bc.BetweennessCentralityGAPGenerator().generate(dataset)
    # An in-process reference fetches the graph again.
    assert fetch.call_args_list == [call("GAP-road")] * fetch.call_count
    assert problem.meta["bc_sources"] == dataset.sources
    assert problem.meta["trial"] == 3
    assert problem.meta["include_source"] is True
    assert problem.meta["normalize"] is True
    np.testing.assert_array_equal(
        to_scipy(problem.inputs[0]).toarray(), [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )
    problem.meta["bc_sources"].append(-1)
    assert len(dataset.sources) == len(dataset.meta["bc_sources"]) == 4
    assert raw.meta == {"sources": list(range(64)), "max_degree": 1}


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
@pytest.mark.parametrize("sources", [[0, 2, 4, 5], [0, 0, 2, 5]])
@pytest.mark.parametrize("self_loops", [False, True])
def test_sampled_scores_match_brandes(framework, sources, self_loops):
    # Two shortest paths to 3, a directed cycle, and a disconnected component.
    adjacency = coo_array(
        (np.ones(6), ([0, 0, 1, 2, 3, 4], [1, 2, 3, 3, 0, 5])), shape=(6, 6)
    )
    if self_loops:
        adjacency.setdiag(1)
    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.DiGraph)
    expected = np.zeros(6)
    for source in sources:
        contribution = nx.betweenness_centrality_subset(
            graph, sources=[source], targets=list(graph), normalized=False
        )
        expected += np.array([contribution[v] for v in range(6)])
        expected[source] += len(nx.descendants(graph, source))
    expected /= expected.max()
    xp = framework()
    output = bc.BetweennessCentralityBenchmark().benchmark(
        xp,
        {"bc_sources": sources, "include_source": True, "normalize": True},
        xp.from_binsparse(from_scipy(adjacency)),
    )
    np.testing.assert_allclose(to_numpy(xp.to_binsparse(output)), expected)


@pytest.mark.parametrize("self_loops", [False, True])
def test_sampled_zero_scores_remain_finite(self_loops):
    output = bc.BetweennessCentralityBenchmark().benchmark(
        NumpyFramework(),
        {"bc_sources": [0, 1, 2, 3], "include_source": True, "normalize": True},
        np.eye(4) if self_loops else np.zeros((4, 4)),
    )
    np.testing.assert_array_equal(output, np.zeros(4))


def test_shared_graph_sources_do_not_change_all_source_cases():
    adjacency = np.diag(np.ones(3), k=1)
    output = bc.BetweennessCentralityBenchmark().benchmark(
        NumpyFramework(), {"sources": [0]}, adjacency
    )
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    expected = nx.betweenness_centrality(graph, normalized=False)
    np.testing.assert_allclose(output, list(expected.values()))
