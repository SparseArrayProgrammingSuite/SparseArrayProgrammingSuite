import numpy as np

from sparseappbench.benchmarks.fastsv import benchmark_fastsv
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def _run_fastsv_case(A, expected):
    "This method will run all the different tests. It will assist with setup"
    xp = NumpyFramework()
    A_bin = BinsparseFormat.from_numpy(A)
    bench_result = benchmark_fastsv(xp, A_bin)
    result = xp.from_benchmark(bench_result).ravel()
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
