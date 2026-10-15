import importlib
from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

import igraph as ig
import networkx as nx
from binsparse.conversions import from_numpy, from_scipy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import DataInstance
from saps.storage import LocalStorageBackend
from saps.util.adjacency import distance_matrix

GRAPH_CLASSES = [
    ("bfs", "BFS"),
    ("msbfs", "MSBFS"),
    ("bellman_ford", "BellmanFord"),
    ("mssp", "MSSP"),
    ("connected_components", "ConnectedComponents"),
    ("fastsv", "FastSV"),
    ("floyd_warshall", "FloydWarshall"),
    ("transitive_closure", "TransitiveClosure"),
    ("transitive_reduction", "TransitiveReduction"),
    ("pagerank", "PageRank"),
    ("betweenness_centrality", "BetweennessCentrality"),
    ("mcl", "MCL"),
]


def test_bfs_levels_are_deterministic_when_parent_trees_differ(monkeypatch):
    original = sp.csr_array(
        (np.ones(4, dtype=bool), [1, 2, 3, 3], [0, 2, 3, 4, 4, 4]), shape=(5, 5)
    )
    parents = []
    # Relabel tied neighbors to make SciPy choose each valid parent in turn.
    for order in (np.arange(5), np.array([0, 2, 1, 3, 4])):
        graph = original[order, :][:, order]
        _, predecessor = csgraph.breadth_first_order(graph, 0)
        parents.append(order[predecessor[3]])
        problem = reference_problem(monkeypatch, "bfs", from_scipy(graph), {})
        np.testing.assert_array_equal(to_numpy(problem.ref_outputs[0]), [1, 2, 2, 3, 0])
    assert parents == [1, 2]


def reference_problem(monkeypatch, name, graph, meta):
    """Run the real generator reference against a specified prepared input."""
    module = importlib.import_module(f"saps.benchmarks.{name}")
    prefix = dict(GRAPH_CLASSES)[name]
    generator = getattr(module, f"{prefix}SNAPGenerator")()
    dataset = generator.datasets[0]
    if hasattr(dataset, "meta"):
        dataset.meta.update(meta)
    sources = meta.get("sources", meta.get("bc_sources", [0] * 64))
    raw = DataInstance([graph], {"max_degree": 2, "sources": sources, **meta})
    monkeypatch.setattr(module, "fetch_snap_graph", lambda _: raw)
    for converter in ("zero_one_adjacency", "distance_matrix", "_unweighted_distances"):
        if hasattr(module, converter):
            monkeypatch.setattr(module, converter, lambda *args, **kwargs: graph)
    if "max_squarings" in meta:
        monkeypatch.setattr(
            module, "squaring_count", lambda *args, **kwargs: meta["max_squarings"]
        )
    return generator.generate(dataset)


def check_reference(name, problem, output):
    module = importlib.import_module(f"saps.benchmarks.{name}")
    benchmark = getattr(module, f"{dict(GRAPH_CLASSES)[name]}Benchmark")()
    benchmark._ref_outputs = problem.ref_outputs
    benchmark._ref_meta = problem.ref_meta
    benchmark._output = [output]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="reference_case")))


@pytest.mark.parametrize("name,prefix", GRAPH_CLASSES)
@pytest.mark.parametrize("collection", ["SNAP", "GAP"])
@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_collection_references_are_cached_and_reject_wrong_results(
    monkeypatch, tmp_path, name, prefix, collection, framework
):
    module = importlib.import_module(f"saps.benchmarks.{name}")
    # A diamond, a cycle, an isolated self-loop, and an isolated vertex.
    adjacency = sp.coo_array(
        ([2, 3, 4, 5, 1, 9], ([0, 0, 1, 2, 3, 4], [1, 2, 3, 3, 0, 4])), shape=(6, 6)
    )
    raw = DataInstance(
        inputs=[from_scipy(adjacency)],
        meta={"sources": [0, 0, 2, 5] * 16, "max_degree": 2},
    )
    monkeypatch.setattr(module, f"fetch_{collection.lower()}_graph", lambda _: raw)
    generator = getattr(module, f"{prefix}{collection}Generator")()
    dataset = generator.datasets[0]
    if collection == "GAP" and name in {"bfs", "bellman_ford"}:
        dataset.src = 0
    if collection == "GAP" and name == "betweenness_centrality":
        dataset.sources = [0, 0, 2, 5]
    problem = generator.generate(dataset)
    assert problem.ref_outputs is not None
    backend = (
        "scipy"
        if name
        in {"floyd_warshall", "transitive_closure", "transitive_reduction", "mcl"}
        else "igraph"
    )
    assert problem.ref_meta["reference_backend"] == backend
    assert problem.ref_meta["reference_algorithm"] == name

    storage = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    file = tmp_path / "case.h5"
    storage.serialize_data_to_file(problem, file)
    saved = storage.deserialize_data_from_file(file)
    assert saved.ref_meta == problem.ref_meta

    # Checking saved results must not run SciPy's graph algorithms again.
    def no_reference(*args, **kwargs):
        pytest.fail("Reference recomputed during execution/checking")

    monkeypatch.setattr(csgraph, "shortest_path", no_reference)
    monkeypatch.setattr(csgraph, "connected_components", no_reference)
    monkeypatch.setattr(csgraph, "dijkstra", no_reference)
    for method in ["distances", "betweenness", "connected_components", "pagerank"]:
        monkeypatch.setattr(ig.Graph, method, no_reference)
    benchmark = getattr(module, f"{prefix}Benchmark")()
    xp = framework()
    output = benchmark.benchmark(
        xp, saved.meta, *(xp.from_binsparse(a) for a in saved.inputs)
    )
    benchmark._output = [xp.to_binsparse(output)]
    benchmark._ref_outputs = saved.ref_outputs
    benchmark._ref_meta = saved.ref_meta
    param = SimpleNamespace(dataset=dataset)
    benchmark.check(param)
    wrong = np.array(NumpyFramework().from_binsparse(benchmark._output[0]), copy=True)
    if wrong.dtype == bool:
        wrong.flat[0] = not wrong.flat[0]
    else:
        wrong.flat[0] = -100
    benchmark._output = [from_numpy(wrong)]
    with pytest.raises(AssertionError):
        benchmark.check(param)


@pytest.mark.parametrize("include_source", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_igraph_bc_matches_networkx_with_repeated_sources(
    monkeypatch, include_source, normalize
):
    rng = np.random.default_rng(3)
    graph = sp.csr_array(rng.random((25, 25)) < 0.08)
    graph.setdiag(True)
    nx_graph = nx.from_scipy_sparse_array(graph, create_using=nx.DiGraph)
    sources = [0, 0, 3, 24]
    expected = np.zeros(25)
    for source in sources:
        contribution = nx.betweenness_centrality_subset(
            nx_graph, [source], list(nx_graph), normalized=False
        )
        expected += np.array(list(contribution.values()))
        if include_source:
            expected[source] += len(nx.descendants(nx_graph, source))
    if normalize:
        expected /= expected.max()
    np.testing.assert_allclose(
        to_numpy(
            reference_problem(
                monkeypatch,
                "betweenness_centrality",
                from_scipy(graph.tocoo()),
                {
                    "bc_sources": sources,
                    "include_source": include_source,
                    "normalize": normalize,
                },
            ).ref_outputs[0]
        ),
        expected,
        atol=1e-12,
    )


def test_igraph_pagerank_preserves_column_orientation_and_dangling_nodes(monkeypatch):
    graph = sp.csr_array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    expected = nx.pagerank(
        nx.from_scipy_sparse_array(graph.T, create_using=nx.DiGraph), tol=1e-14
    )
    np.testing.assert_allclose(
        to_numpy(
            reference_problem(
                monkeypatch,
                "pagerank",
                from_scipy(graph.tocoo()),
                {"tol": 1e-13, "max_iter": 1000},
            ).ref_outputs[0]
        ),
        list(expected.values()),
        atol=1e-12,
    )


@pytest.mark.parametrize("algorithm", ["floyd_warshall", "transitive_closure"])
def test_bounded_reference_covers_every_pair_without_densifying(monkeypatch, algorithm):
    n = 20_000
    adjacency = from_scipy(
        sp.diags_array([np.ones(n - 1)], offsets=[1], shape=(n, n)).tocoo()
    )
    graph = distance_matrix(adjacency) if algorithm == "floyd_warshall" else adjacency
    original_full = np.full

    def forbid_all_pairs(shape, *args, **kwargs):
        assert shape != (n, n)
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(np, "full", forbid_all_pairs)
    problem = reference_problem(monkeypatch, algorithm, graph, {"max_squarings": 2})
    assert problem.ref_outputs[0].shape == (n, n)
    assert problem.ref_outputs[0].number_of_stored_values == 5 * n - 10
    assert "reference_rows" not in problem.ref_meta
    if algorithm == "transitive_closure":
        assert problem.ref_outputs[0].values.dtype == bool
    expected = sp.diags_array(
        [np.ones(n - k) for k in range(5)], offsets=range(5), shape=(n, n)
    ).tocoo()
    if algorithm == "floyd_warshall":
        import sparse
        from binsparse.conversions import from_sparse

        expected.data = (expected.col - expected.row).astype(float)
        output = from_sparse(
            sparse.COO(
                np.stack([expected.row, expected.col]),
                expected.data,
                shape=expected.shape,
                fill_value=np.inf,
            )
        )
    else:
        output = from_scipy(expected.astype(bool))
    check_reference(algorithm, problem, output)
    # Corrupt an entry far from the rows covered by the previous sampled check.
    corrupted = output.values.copy()
    position = np.flatnonzero(output.indices_0 == 12345)[0]
    corrupted[position] = False if algorithm == "transitive_closure" else 1000
    output.values[:] = corrupted
    with pytest.raises(AssertionError):
        check_reference(algorithm, problem, output)


def test_distance_reference_preserves_zero_length_edges_and_negative_weights(
    monkeypatch,
):
    import sparse
    from binsparse.conversions import from_sparse

    distances = np.array(
        [
            [0, 0, 5, np.inf],
            [np.inf, 0, -2, np.inf],
            [np.inf, np.inf, 0, np.inf],
            [np.inf, np.inf, np.inf, 0],
        ]
    )
    graph = from_sparse(sparse.COO.from_numpy(distances, fill_value=np.inf))
    problem = reference_problem(monkeypatch, "mssp", graph, {"sources": [0, 3]})
    np.testing.assert_array_equal(
        to_numpy(problem.ref_outputs[0]),
        [[0, 0, -2, np.inf], [np.inf, np.inf, np.inf, 0]],
    )


def test_component_check_rejects_wrong_partition_with_correct_count(monkeypatch):
    graph = from_scipy(sp.coo_array(([1, 1], ([0, 2], [1, 3])), shape=(4, 4)))
    problem = reference_problem(monkeypatch, "connected_components", graph, {})
    np.testing.assert_array_equal(to_numpy(problem.ref_outputs[0]), [0, 0, 2, 2])
    with pytest.raises(AssertionError):
        check_reference(
            "connected_components", problem, from_numpy(np.array([0, 2, 0, 2]))
        )


@pytest.mark.parametrize("max_squarings", [0, 1, 2, 3])
def test_weighted_hop_limit_is_not_unrestricted_shortest_paths(
    monkeypatch, max_squarings
):
    from saps.benchmarks.floyd_warshall import FloydWarshallBenchmark

    adjacency = from_scipy(
        sp.diags_array([-np.ones(7)], offsets=[1], shape=(8, 8)).tocoo()
    )
    graph = distance_matrix(adjacency, keep_weights=True)
    problem = reference_problem(
        monkeypatch, "floyd_warshall", graph, {"max_squarings": max_squarings}
    )
    xp = NumpyFramework()
    output = FloydWarshallBenchmark().benchmark(
        xp, problem.meta, xp.from_binsparse(graph)
    )
    check_reference("floyd_warshall", problem, from_numpy(output))


@pytest.mark.parametrize("x", [-3, 0, 1, 5])
def test_weighted_edge_reduction_rule(monkeypatch, x):
    from saps.benchmarks.transitive_reduction import TransitiveReductionBenchmark

    graph = np.array(
        [
            [np.inf, -2, 4, 7],
            [np.inf, np.inf, -3, 2],
            [np.inf, np.inf, np.inf, 1],
            [np.inf, np.inf, np.inf, np.inf],
        ]
    )
    problem = reference_problem(
        monkeypatch, "transitive_reduction", from_numpy(graph), {"x": x, "max_iters": 3}
    )
    output = TransitiveReductionBenchmark().benchmark(
        NumpyFramework(), problem.meta, graph
    )
    check_reference("transitive_reduction", problem, from_numpy(output))


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_full_weighted_reference_with_zero_edges_and_negative_cycles(
    monkeypatch, steps
):
    from saps.benchmarks.floyd_warshall import FloydWarshallBenchmark

    rng = np.random.default_rng(23)
    graph = np.where(rng.random((19, 19)) < 0.2, rng.integers(-3, 7, (19, 19)), np.inf)
    np.fill_diagonal(graph, 0)
    problem = reference_problem(
        monkeypatch, "floyd_warshall", from_numpy(graph), {"max_squarings": steps}
    )
    output = FloydWarshallBenchmark().benchmark(NumpyFramework(), problem.meta, graph)
    check_reference("floyd_warshall", problem, from_numpy(output))


@pytest.mark.parametrize("algorithm", ["floyd_warshall", "transitive_closure"])
def test_boolean_frontiers_do_not_overflow_on_many_paths(monkeypatch, algorithm):
    # There are 4096 length-four paths from 0 to 49: reachability must not
    # overflow a narrow integer path counter and disappear.
    graph = np.zeros((51, 51), dtype=bool)
    graph[0, 1:17] = True
    graph[1:17, 17:33] = True
    graph[17:33, 33:49] = True
    graph[33:49, 49] = True
    adjacency = from_scipy(sp.coo_array(graph))
    prepared = (
        distance_matrix(adjacency) if algorithm == "floyd_warshall" else adjacency
    )
    problem = reference_problem(monkeypatch, algorithm, prepared, {"max_squarings": 2})
    distances = csgraph.shortest_path(sp.csr_array(graph), unweighted=True, method="D")
    expected = distances if algorithm == "floyd_warshall" else np.isfinite(distances)
    check_reference(algorithm, problem, from_numpy(expected))


@pytest.mark.parametrize(
    "iterations,tolerance", [(0, 1e-8), (1, 1e-8), (4, 1e-8), (100, 1e-8), (100, 1e-3)]
)
def test_igraph_pagerank_check_allows_finite_iteration_error(
    monkeypatch, iterations, tolerance
):
    from saps.benchmarks.pagerank import PageRankBenchmark

    # Unequal weights, a dangling column, and asymmetric edge orientation.
    graph = np.array([[0, 0, 2, 0], [3, 0, 0, 0], [1, 2, 0, 0], [0, 1, 0, 0]])
    problem = reference_problem(
        monkeypatch,
        "pagerank",
        from_numpy(graph),
        {"alpha": 0.9, "max_iter": iterations, "tol": tolerance},
    )
    output = PageRankBenchmark().benchmark(NumpyFramework(), problem.meta, graph)
    check_reference("pagerank", problem, from_numpy(output))


def test_igraph_pagerank_check_rejects_wrong_distribution(monkeypatch):
    graph = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    problem = reference_problem(monkeypatch, "pagerank", from_numpy(graph), {})
    wrong = to_numpy(problem.ref_outputs[0]).copy()
    wrong[0] += 1e-4
    wrong[1] -= 1e-4
    with pytest.raises(AssertionError):
        check_reference("pagerank", problem, from_numpy(wrong))
