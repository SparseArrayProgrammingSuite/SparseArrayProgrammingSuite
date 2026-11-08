import importlib

import pytest

import numpy as np

from binsparse import COORMatrix

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance
from saps.util.adjacency import random_edge_lengths, zero_one_adjacency

# These generators' references compute in-process here.
pytestmark = pytest.mark.usefixtures("uncached_references")


def _signed_graph():
    """Edges 0->1 and 1->2, a weighted self-loop at 2, and an explicit zero 2->0.

    1->2 is stored twice with opposite signs, so summing duplicates cancels it.
    (Built directly, since ``from_scipy`` would sum the duplicates first.)
    """
    return COORMatrix(
        (3, 3),
        5,
        indices_0=np.array([0, 1, 1, 2, 2]),
        indices_1=np.array([1, 2, 2, 0, 2]),
        values=np.array([2.5, -1.0, 1.0, 0.0, 7.0]),
    )


_PATTERN = [[0, 1, 0], [0, 0, 1], [0, 0, 1]]
_UPPER_PATTERN = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
inf = np.inf

# (module, generator, expected first input, expected dtype)
_SNAP_CONSUMERS = [
    ("bfs", "BFSSNAPGenerator", _PATTERN, bool),
    ("msbfs", "MSBFSSNAPGenerator", _PATTERN, bool),
    (
        "betweenness_centrality",
        "BetweennessCentralitySNAPGenerator",
        _PATTERN,
        bool,
    ),
    ("connected_components", "ConnectedComponentsSNAPGenerator", _PATTERN, bool),
    ("fastsv", "FastSVSNAPGenerator", [[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool),
    (
        "four_clique_counting",
        "FourCliqueCountingSNAPGenerator",
        _UPPER_PATTERN,
        np.int64,
    ),
    ("mcl", "MCLSNAPGenerator", _PATTERN, bool),
    ("pagerank", "PageRankSNAPGenerator", _PATTERN, bool),
    ("transitive_closure", "TransitiveClosureSNAPGenerator", _PATTERN, bool),
    (
        "triangle_counting",
        "TriangleCountingSNAPGenerator",
        _UPPER_PATTERN,
        np.int64,
    ),
]


def test_zero_one_adjacency_keeps_only_nonzero_edges():
    for dtype in (bool, np.int64, np.float32):
        actual = NumpyFramework().from_binsparse(
            zero_one_adjacency(_signed_graph(), dtype)
        )
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, _PATTERN)


def test_simple_undirected_adjacency_drops_loops_and_unions_edges():
    graph = _signed_graph()
    actual = NumpyFramework().from_binsparse(
        zero_one_adjacency(graph, np.int64, simple_undirected=True)
    )
    np.testing.assert_array_equal(actual, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # Opposite signs at the same coordinate cannot cancel; the input is intact.
    np.testing.assert_array_equal(graph.values, [2.5, -1, 1, 0, 7])


def test_upper_triangle_unions_both_edge_directions_first():
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy, to_scipy

    graph = from_scipy(coo_array([[9, -1, 0], [0, 0, 0], [0, 2, 0]]))
    actual = zero_one_adjacency(
        graph, np.int64, simple_undirected=True, upper_triangle=True
    )
    np.testing.assert_array_equal(to_scipy(actual).toarray(), _UPPER_PATTERN)


@pytest.mark.parametrize(
    "module_name,class_prefix",
    [
        ("triangle_counting", "TriangleCounting"),
        ("four_clique_counting", "FourCliqueCounting"),
    ],
)
def test_directed_clique_inputs_match_published_convention(
    monkeypatch, module_name, class_prefix
):
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    generator = getattr(module, f"{class_prefix}SNAPGenerator")()
    dataset = next(d for d in generator.datasets if d.name == "twitter7")
    monkeypatch.setattr(
        module,
        "fetch_snap_graph",
        lambda _: DataInstance(inputs=[_signed_graph()], meta={}),
    )
    actual = NumpyFramework().from_binsparse(generator.generate(dataset).inputs[0])
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, _UPPER_PATTERN)
    assert "self-loops removed" in dataset.description


def _check_first_input(
    monkeypatch, module_name, generator_name, fetch, expected, dtype
):
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    generator = getattr(module, generator_name)()
    dataset = generator.datasets[0]
    src = getattr(dataset, "src", None)
    raw = DataInstance(
        inputs=[_signed_graph()],
        meta={"max_degree": 1, "sources": [0, 1, 2] * 4 if src is None else [src]},
    )
    monkeypatch.setattr(module, fetch, lambda _: raw)
    actual = NumpyFramework().from_binsparse(generator.generate(dataset).inputs[0])
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("module", "generator", "expected", "dtype"), _SNAP_CONSUMERS)
def test_snap_connectivity_inputs_are_zero_one(
    monkeypatch, module, generator, expected, dtype
):
    _check_first_input(
        monkeypatch, module, generator, "fetch_snap_graph", expected, dtype
    )


@pytest.mark.parametrize(
    ("module_name", "generator_name"),
    [
        ("bellman_ford", "BellmanFordSNAPGenerator"),
        ("floyd_warshall", "FloydWarshallSNAPGenerator"),
        ("mssp", "MSSPSNAPGenerator"),
    ],
)
def test_snap_distance_inputs_use_seeded_random_lengths(
    monkeypatch, module_name, generator_name
):
    # 0->1 and 1->2 are directed edges whatever their stored values; the
    # self-loop and the explicit zero are not edges.
    a, b = random_edge_lengths(np.array([0, 1]), np.array([1, 2]), 0)
    _check_first_input(
        monkeypatch,
        module_name,
        generator_name,
        "fetch_snap_graph",
        [[0, a, inf], [inf, 0, b], [inf, inf, 0]],
        np.float64,
    )


def test_transitive_reduction_snap_uses_seeded_lengths_without_diagonal(monkeypatch):
    # Pruning follows edges only, so no vertex is given a zero-length self-loop.
    a, b = random_edge_lengths(np.array([0, 1]), np.array([1, 2]), 0)
    _check_first_input(
        monkeypatch,
        "transitive_reduction",
        "TransitiveReductionSNAPGenerator",
        "fetch_snap_graph",
        [[inf, a, inf], [inf, inf, b], [inf, inf, inf]],
        np.float64,
    )


def test_random_edge_lengths_are_fixed_symmetric_and_seeded():
    rng = np.random.default_rng(0)
    row, col = rng.integers(0, 2**31, (2, 10_000))
    lengths = random_edge_lengths(row, col, 0)
    np.testing.assert_array_equal(lengths, random_edge_lengths(col, row, 0))
    assert lengths.min() == 1 and lengths.max() == 255
    assert np.all(lengths == np.round(lengths))
    assert np.mean(lengths != random_edge_lengths(row, col, 1)) > 0.99
    # Pinned: cached references depend on these exact lengths.
    np.testing.assert_array_equal(
        random_edge_lengths(np.arange(5), np.arange(1, 6), 0), [35, 41, 125, 131, 107]
    )
