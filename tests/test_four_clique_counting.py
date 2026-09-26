import pytest

import numpy as np
from scipy import sparse

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmark import DataInstance
from saps.benchmarks import four_clique_counting as four


@pytest.mark.parametrize(
    "generator",
    [four.FourCliqueCountSNAPGenerator(), four.FourCliqueCountGAPGenerator()],
)
def test_all_four_clique_sources_are_loadable(monkeypatch, generator):
    adjacency = from_scipy(sparse.coo_matrix(np.ones((4, 4)) - np.eye(4)))
    extra_input = from_numpy(np.ones(4))
    calls = []
    metadata = {"source": "stub"}

    def load_snap(name):
        calls.append(("snap", name))
        return DataInstance(inputs=[adjacency], meta=metadata)

    def load_gap(name):
        calls.append(("gap", name))
        return DataInstance(inputs=[adjacency, extra_input], meta=metadata)

    monkeypatch.setattr(four, "fetch_snap_graph", load_snap)
    monkeypatch.setattr(four, "fetch_suitesparse_matrix", load_gap)
    source = (
        "gap" if isinstance(generator, four.FourCliqueCountGAPGenerator) else "snap"
    )
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        assert problem.inputs == [adjacency]
        assert problem.meta == metadata
        assert calls[-1] == (source, dataset.name)
    assert len(calls) == len(generator.datasets)


def test_four_clique_parameters_include_full_snap_catalog_and_standard_gap():
    from saps.benchmarks.snap import SNAPGraphGenerator

    parameters = four.FourCliqueCountBenchmark().params
    snap = [
        p
        for p in parameters
        if isinstance(p.generator, four.FourCliqueCountSNAPGenerator)
    ]
    assert {p.dataset.name for p in snap} == {
        d.name for d in SNAPGraphGenerator().datasets
    }
    gap = [
        p
        for p in parameters
        if isinstance(p.generator, four.FourCliqueCountGAPGenerator)
    ]
    assert len(gap) == 5
    assert all("standard" in p.dataset.suites for p in gap)
    assert all(not p.generator.cacheable for p in snap + gap)
