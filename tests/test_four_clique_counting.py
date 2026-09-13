import pytest

import numpy as np
from scipy import sparse

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmark import DataInstance
from saps.benchmarks import four_clique_counting as four


@pytest.mark.parametrize(
    "generator", [four.FourCliqueCountGenerator(), four.FourCliqueCountGAPGenerator()]
)
def test_all_four_clique_sources_are_loadable(monkeypatch, generator):
    adjacency = from_scipy(sparse.coo_matrix(np.ones((4, 4)) - np.eye(4)))
    extra_input = from_numpy(np.ones(4))
    calls = []
    metadata = {"source": "stub"}

    def load_snap(name):
        calls.append(("snap", name))
        return [adjacency], metadata

    def load_gap(name):
        calls.append(("gap", name))
        return DataInstance(inputs=[adjacency, extra_input], meta=metadata)

    monkeypatch.setattr(four, "download_snap_dataset", load_snap)
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


def test_standard_four_clique_parameters_include_snap_and_gap():
    parameters = [
        p
        for p in four.FourCliqueCountBenchmark().params
        if "standard" in p.dataset.suites
    ]
    assert len(parameters) == 7
    snap = [
        p for p in parameters if isinstance(p.generator, four.FourCliqueCountGenerator)
    ]
    assert {p.dataset.name for p in snap} == {
        "snap-email-Eu-core-temporal-Dept3",
        "snap-email-Eu-core-temporal-Dept4",
    }
    gap = [
        p
        for p in parameters
        if isinstance(p.generator, four.FourCliqueCountGAPGenerator)
    ]
    assert len(gap) == 5
    assert all(not p.generator.cacheable for p in gap)
