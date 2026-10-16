import numpy as np
from scipy import sparse

from binsparse.conversions import from_scipy, to_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance
from saps.benchmarks import four_clique_counting as four


def test_all_four_clique_sources_are_loadable(monkeypatch):
    generator = four.FourCliqueCountingSNAPGenerator()
    adjacency = from_scipy(sparse.coo_matrix(np.triu(np.ones((4, 4)))))
    calls = []
    metadata = {"source": "stub"}

    def load_snap(name):
        calls.append(("snap", name))
        return DataInstance(inputs=[adjacency], meta=metadata)

    monkeypatch.setattr(four, "fetch_snap_graph", load_snap)
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        assert len(problem.inputs) == 1
        actual = to_scipy(problem.inputs[0])
        assert actual.dtype == np.int64
        reference = to_numpy(problem.ref_outputs[0])
        assert reference.dtype == np.int64
        assert reference.item() == dataset.expected
        np.testing.assert_array_equal(actual.toarray(), np.triu(np.ones((4, 4)), k=1))
        assert problem.meta == metadata
        assert calls[-1] == ("snap", dataset.name)
    assert len(calls) == len(generator.datasets)


def test_all_snap_four_clique_references_count_whole_cliques():
    datasets = four.FourCliqueCountingSNAPGenerator().datasets
    for dataset in datasets:
        assert isinstance(dataset.expected, int) and dataset.expected >= 0
    # GraphMiner publishes this count for the same simple undirected graph.
    assert next(d.expected for d in datasets if d.name == "cit-Patents") == 3_501_071


def test_four_clique_examples_count_each_clique_once():
    generator = four.FourCliqueCountingTestGenerator()
    benchmark = four.FourCliqueCountingBenchmark()
    xp = NumpyFramework()
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        A = to_numpy(problem.inputs[0])
        assert A.dtype == np.int64
        assert to_numpy(problem.ref_outputs[0]).dtype == np.int64
        np.testing.assert_array_equal(A, np.triu(A, k=1))
        assert benchmark.benchmark(xp, {}, A) == to_numpy(problem.ref_outputs[0])


def test_four_clique_parameters_include_full_snap_catalog_without_gap():
    from saps.benchmarks.snap import SNAPGraphGenerator

    parameters = four.FourCliqueCountingBenchmark().params
    snap = [
        p
        for p in parameters
        if isinstance(p.generator, four.FourCliqueCountingSNAPGenerator)
    ]
    assert {p.dataset.name for p in snap} == {
        d.name for d in SNAPGraphGenerator().datasets
    }
    assert not any(p.dataset.name.startswith("GAP-") for p in parameters)
    assert all(not p.generator.cacheable for p in snap)
