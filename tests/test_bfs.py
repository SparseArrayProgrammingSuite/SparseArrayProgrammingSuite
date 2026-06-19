import numpy as np

import saps.benchmarks.BFS as bfs
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.snap import load_toy_dataset
from saps_framework import BinsparseFormat


def _run_bfs_case(A, source: int, expected: np.ndarray):
    "This method will run all the different tests. It will asisst with setup"
    xp = NumpyFramework()
    bfs.xp = xp
    A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
    (bench_result,) = bfs.BreadthFirstSearchBenchmark().benchmark(
        [A_bin], {"src": source}
    )
    result = bench_result.ravel()
    assert np.array_equal(result, expected), (
        f"BFS output mismatch.\nGot {result}, expected {expected}"
    )


def test_bfs_basic():
    """Standard DAG benchmark graph."""
    A = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    _run_bfs_case(A, 0, np.array([1, 2, 2, 3, 3, 4], dtype=int))


def test_bfs_single_node():
    """Trivial graph with one vertex."""
    A = np.array([[0]], dtype=bool)
    _run_bfs_case(A, 0, np.array([1], dtype=int))


def test_bfs_disconnected():
    """Graph with unreachable vertices."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    _run_bfs_case(A, 0, np.array([1, 2, 0, 0], dtype=int))


def test_bfs_undirected():
    """Undirected symmetric adjacency."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    )
    _run_bfs_case(A, 0, np.array([1, 2, 3, 4], dtype=int))


def test_bfs_cycle():
    """Cycle graph: 0→1→2→3→0."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    )
    _run_bfs_case(A, 0, np.array([1, 2, 3, 4], dtype=int))


def test_bfs_snap_toy():
    data, meta = load_toy_dataset()
    _run_bfs_case(data[0], meta["src"], np.array([1, 2, 3], dtype=int))
