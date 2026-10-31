import importlib

import pytest

import numpy as np

from binsparse import COORMatrix

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance
from saps.benchmarks.mcl import _edge_offsets
from saps.util.adjacency import zero_one_adjacency

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
# MCL weights each edge slightly differently to break symmetric ties.
_MCL_INPUT = np.zeros((3, 3))
_MCL_INPUT[np.nonzero(_PATTERN)] = 1 + _edge_offsets(*np.nonzero(_PATTERN), 3)
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
    ("mcl", "MCLSNAPGenerator", _MCL_INPUT, np.float64),
    ("pagerank", "PageRankSNAPGenerator", _PATTERN, bool),
    ("transitive_closure", "TransitiveClosureSNAPGenerator", _PATTERN, bool),
    (
        "triangle_counting",
        "TriangleCountingSNAPGenerator",
        _UPPER_PATTERN,
        np.int64,
    ),
    (
        "transitive_reduction",
        "TransitiveReductionSNAPGenerator",
        [[inf, 1, inf], [inf, inf, 1], [inf, inf, inf]],
        np.float64,
    ),
]
_GAP_CONSUMERS = [
    (module, generator.replace("SNAPGenerator", "GAPGenerator"), expected, dtype)
    for module, generator, expected, dtype in _SNAP_CONSUMERS
    if module != "four_clique_counting"
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
    "module_name,class_prefix,name",
    [
        ("triangle_counting", "TriangleCounting", "twitter7"),
        ("triangle_counting", "TriangleCounting", "GAP-twitter"),
        ("triangle_counting", "TriangleCounting", "GAP-web"),
        ("four_clique_counting", "FourCliqueCounting", "twitter7"),
    ],
)
def test_directed_clique_inputs_match_published_convention(
    monkeypatch, module_name, class_prefix, name
):
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    group = "GAP" if name.startswith("GAP-") else "SNAP"
    generator = getattr(module, f"{class_prefix}{group}Generator")()
    dataset = next(d for d in generator.datasets if d.name == name)
    monkeypatch.setattr(
        module,
        f"fetch_{group.lower()}_graph",
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
    if fetch == "fetch_gap_graph" and module_name == "bfs":
        dataset.src = 0
    if fetch == "fetch_gap_graph" and module_name == "betweenness_centrality":
        dataset.sources = [0, 1, 2, 0]
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


@pytest.mark.parametrize(("module", "generator", "expected", "dtype"), _GAP_CONSUMERS)
def test_gap_connectivity_inputs_are_zero_one(
    monkeypatch, module, generator, expected, dtype
):
    _check_first_input(
        monkeypatch, module, generator, "fetch_gap_graph", expected, dtype
    )


def test_floyd_warshall_snap_uses_directed_unit_edges(monkeypatch):
    _check_first_input(
        monkeypatch,
        "floyd_warshall",
        "FloydWarshallSNAPGenerator",
        "fetch_snap_graph",
        [[0, 1, inf], [inf, 0, 1], [inf, inf, 0]],
        np.float64,
    )
