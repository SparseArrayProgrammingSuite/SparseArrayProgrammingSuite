import pytest

import numpy as np
from scipy.sparse import coo_array

from binsparse.conversions import from_numpy, from_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import DataInstance
from saps.benchmarks import floyd_warshall as fw
from saps.benchmarks import transitive_closure as tc
from saps.util.adjacency import squaring_count


@pytest.mark.parametrize(
    "n,degree,density,steps",
    [
        (1000, 1, 0.01, 1),  # Four entries per row; the next step allows 16.
        (900, 2, 0.01, 1),  # Exactly 1%, including the diagonal.
        (899, 2, 0.01, 0),
        (1000, 500, 0.01, 0),
        (1, 0, 1.0, 0),
        (0, 0, 0.01, 0),
        (9, 1, 1.0, 3),
        (10, 1, 1.0, 4),
        (10**12, 10**9, 0.01, 0),
    ],
)
def test_squaring_budget(n, degree, density, steps):
    assert squaring_count(n, degree, density) == steps


@pytest.mark.parametrize("seed", range(5))
def test_budget_limits_actual_boolean_density(seed):
    rng = np.random.default_rng(seed)
    edges = rng.random((30, 30)) < 0.02
    np.fill_diagonal(edges, False)
    degree = int(edges.sum(axis=1).max())
    graph = edges | np.eye(len(edges), dtype=bool)
    for _ in range(squaring_count(len(edges), degree, 0.3)):
        graph = graph @ graph
    assert graph.mean() <= 0.3


@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_both_kernels_stop_at_same_hop_limit(framework):
    xp = framework()
    edges = np.eye(8, k=1, dtype=bool)
    distances = np.where(edges, -1.0, np.inf)  # Negative edges, no cycles.
    np.fill_diagonal(distances, 0.0)
    plan = {"max_squarings": squaring_count(8, 1, 0.5)}
    assert plan["max_squarings"] == 1
    closure = tc.TransitiveClosureBenchmark().benchmark(
        xp, plan, xp.from_binsparse(from_numpy(edges))
    )
    shortest = fw.FloydWarshallBenchmark().benchmark(
        xp, plan, xp.from_binsparse(from_numpy(distances))
    )

    def convert(a):
        return NumpyFramework().from_binsparse(xp.to_binsparse(a))

    closure, shortest = convert(closure), convert(shortest)
    hops = np.arange(8)[None, :] - np.arange(8)[:, None]
    expected = (hops >= 0) & (hops <= 2)
    np.testing.assert_array_equal(closure, expected)
    np.testing.assert_array_equal(shortest, np.where(expected, -hops, np.inf))
    assert np.isfinite(shortest).mean() <= 0.5


def test_zero_budget_does_not_contract():
    class NoContractions(NumpyFramework):
        def einsum(self, *args, **kwargs):
            pytest.fail("No squarings fit the density budget")

    xp = NoContractions()
    edges = np.eye(8, k=1, dtype=bool)
    distances = np.where(edges, 1.0, np.inf)
    np.fill_diagonal(distances, 0.0)
    plan = {"max_squarings": squaring_count(8, 1)}
    assert plan["max_squarings"] == 0
    np.testing.assert_array_equal(
        tc.TransitiveClosureBenchmark().benchmark(xp, plan, edges),
        edges | np.eye(8, dtype=bool),
    )
    np.testing.assert_array_equal(
        fw.FloydWarshallBenchmark().benchmark(xp, plan, distances), distances
    )


@pytest.mark.parametrize("source", ["gap", "snap"])
def test_closure_generators_attach_plan(monkeypatch, source):
    raw = DataInstance(
        inputs=[from_scipy(coo_array((1000, 1000)))],
        meta={"max_degree": 1, "sources": [0]},
    )
    monkeypatch.setattr(tc, f"fetch_{source}_graph", lambda _: raw)
    generator = getattr(tc, f"TransitiveClosure{source.upper()}Generator")()
    problem = generator.generate(tc.TransitiveClosureDataset("graph"))
    assert problem.meta == {**raw.meta, "max_squarings": 1}


@pytest.mark.parametrize("source", ["gap", "snap"])
def test_distance_generators_attach_plan(monkeypatch, source):
    raw = DataInstance(
        inputs=[from_scipy(coo_array((1000, 1000)))],
        meta={"max_degree": 1, "sources": [0]},
    )
    fetch, generator_class = {
        "gap": ("fetch_gap_graph", fw.FloydWarshallGAPGenerator),
        "snap": ("fetch_snap_graph", fw.FloydWarshallSNAPGenerator),
    }[source]
    monkeypatch.setattr(fw, fetch, lambda _: raw)
    generator = generator_class()
    problem = generator.generate(fw.FloydWarshallDataset("graph", max_density=0.02))
    assert problem.meta == {**raw.meta, "max_squarings": 2}


def test_floyd_warshall_snap_matches_closure_inventory():
    shortest = fw.FloydWarshallSNAPGenerator()
    closure = tc.TransitiveClosureSNAPGenerator()
    assert shortest.suites == closure.suites == ["standard"]
    assert [d.name for d in shortest.datasets] == [d.name for d in closure.datasets]
    assert all("standard" in d.suites for d in shortest.datasets)
    assert any(
        isinstance(g, fw.FloydWarshallSNAPGenerator)
        for g in fw.FloydWarshallBenchmark().generators
    )
