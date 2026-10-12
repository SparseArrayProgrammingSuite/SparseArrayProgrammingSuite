import pytest

import numpy as np
from scipy import sparse

from binsparse.conversions import from_scipy, to_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance
from saps.benchmarks import triangle_counting as triangle


@pytest.mark.parametrize(
    "generator",
    [triangle.TriangleCountingSNAPGenerator(), triangle.TriangleCountingGAPGenerator()],
)
def test_triangle_sources_are_upper_triangles_of_simple_undirected_graphs(
    monkeypatch, generator
):
    # A transitive directed triangle and a self-loop: trace(A^3) / 6 is 1/6
    # before normalization, and one ordinary triangle afterwards.
    adjacency = from_scipy(sparse.coo_array([[0, 2, -1], [0, 0, 3], [0, 0, 4]]))
    metadata = {"source": "stub"}

    def load(_):
        return DataInstance(inputs=[adjacency], meta=metadata)

    monkeypatch.setattr(triangle, "fetch_snap_graph", load)
    monkeypatch.setattr(triangle, "fetch_gap_graph", load)
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        A = to_scipy(problem.inputs[0]).toarray()
        np.testing.assert_array_equal(A, np.triu(np.ones((3, 3)), k=1))
        assert np.einsum("ij,jk,ik->", A, A, A) == 1
        assert A.dtype == np.int64
        assert dataset.simple_undirected
        assert "self-loops removed" in dataset.description
        assert problem.meta == metadata
        assert problem.ref_outputs is not None
        reference = to_numpy(problem.ref_outputs[0])
        assert reference.dtype == np.int64
        assert reference.item() == dataset.expected
        assert isinstance(dataset.expected, int) and dataset.expected >= 0


@pytest.mark.parametrize(
    "name,count",
    [
        ("wiki-Vote", 608_389),
        ("cit-Patents", 7_515_023),
        ("ca-GrQc", 48_260),
        ("as-735", 6_584),
    ],
)
def test_references_match_published_snap_triangle_counts(name, count):
    # https://snap.stanford.edu/data/{wiki-Vote,cit-Patents,ca-GrQc,as-733}.html
    dataset = next(
        d for d in triangle.TriangleCountingSNAPGenerator().datasets if d.name == name
    )
    assert dataset.expected == count


def test_triangle_examples_follow_the_same_convention():
    generator = triangle.TriangleCountingTestGenerator()
    benchmark = triangle.TriangleCountingBenchmark()
    xp = NumpyFramework()
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        A = to_numpy(problem.inputs[0])
        assert A.dtype == np.int64
        assert to_numpy(problem.ref_outputs[0]).dtype == np.int64
        np.testing.assert_array_equal(A, np.triu(A, k=1))
        assert not np.any(np.diag(A))
        assert np.einsum("ij,jk,ik->", A, A, A) == to_numpy(problem.ref_outputs[0])
        assert benchmark.benchmark(xp, {}, A) == to_numpy(problem.ref_outputs[0])
