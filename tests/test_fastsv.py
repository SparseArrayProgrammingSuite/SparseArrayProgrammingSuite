import numpy as np

import saps.benchmarks.connected_components as cc
import saps.benchmarks.fastsv as fastsv
from frameworks.saps_numpy import NumpyFramework
from saps_framework import BinsparseFormat


def _run_fastsv_case(A, expected):
    "This method will run all the different tests. It will assist with setup"
    xp = NumpyFramework()
    A_bin = BinsparseFormat.from_numpy(A)
    fastsv.xp = xp
    (bench_result,) = fastsv.FastSVBenchmark().benchmark((A_bin,), {})
    result = bench_result.ravel()
    assert np.array_equal(result, expected), (
        f"fastsv output mismatch.\nGot {result}, expected {expected}"
    )

    cc.xp = xp
    (bench_result,) = cc.SimplyConnectedComponentsBenchmark().benchmark((A_bin,), {})
    result = bench_result.ravel()
    assert np.array_equal(result, expected), (
        f"fastsv output mismatch.\nGot {result}, expected {expected}"
    )


def test_fastsv_no_edges():
    """Graph with no edges: every vertex is its own component."""
    A = np.zeros((5, 5), dtype=bool)
    expected = np.arange(5)  # each node isolated
    _run_fastsv_case(A, expected)


def test_fastsv_single_component():
    """Fully connected undirected graph: one component."""
    A = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    )
    expected = np.array([0, 0, 0, 0])
    _run_fastsv_case(A, expected)


def test_fastsv_two_components():
    """Two disconnected components of equal size."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    )
    expected = np.array([0, 0, 2, 2])
    _run_fastsv_case(A, expected)


def test_fastsv_chain():
    """A simple chain: 0-1-2-3-4 → one connected component."""
    A = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=bool,
    )
    expected = np.array([0, 0, 0, 0, 0])
    _run_fastsv_case(A, expected)


def test_fastsv_star():
    """Star graph: center is 0, connected to all others."""
    A = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    expected = np.array([0, 0, 0, 0, 0])
    _run_fastsv_case(A, expected)


def test_fastsv_isolated_and_connected():
    """One connected triple + two isolated nodes."""
    A = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    expected = np.array([0, 0, 0, 3, 4])
    _run_fastsv_case(A, expected)
