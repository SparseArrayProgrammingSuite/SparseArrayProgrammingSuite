import importlib

import pytest

import numpy as np

from binsparse import COORMatrix

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance
from saps.benchmarks.adjacency import zero_one_adjacency


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
inf = np.inf

# (module, generator, expected first input, expected dtype)
_SNAP_CONSUMERS = [
    ("BFS", "BFSSNAPGenerator", _PATTERN, bool),
    ("MSBFS", "MSBFSSNAPGenerator", _PATTERN, bool),
    ("centrality", "BetweennessCentralitySNAPGenerator", _PATTERN, np.float64),
    ("connected_components", "ConnectedComponentsSNAPGenerator", _PATTERN, bool),
    ("fastsv", "FastSVSNAPGenerator", _PATTERN, bool),
    ("four_clique_counting", "FourCliqueCountingSNAPGenerator", _PATTERN, np.int64),
    ("mcl_benchmark", "MCLSNAPGenerator", _PATTERN, np.float32),
    ("pagerank", "PageRankSNAPGenerator", _PATTERN, np.float64),
    ("transitive_closure", "TransitiveClosureSNAPGenerator", _PATTERN, bool),
    ("triangle_counting", "TriangleCountingSNAPGenerator", _PATTERN, np.int64),
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
]


def test_zero_one_adjacency_keeps_only_nonzero_edges():
    for dtype in (bool, np.int64, np.float32):
        actual = NumpyFramework().from_binsparse(
            zero_one_adjacency(_signed_graph(), dtype)
        )
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, _PATTERN)


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
