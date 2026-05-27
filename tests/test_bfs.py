import gzip

import numpy as np

import saps.benchmarks.BFS as bfs
from frameworks.saps_numpy import NumpyFramework


def _run_bfs_case(A, source, expected):
    "This method will run all the different tests. It will asisst with setup"
    xp = NumpyFramework()
    bfs.xp = xp
    (bench_result,) = bfs.BreadthFirstSearchBenchmark().benchmark((A,), {"src": source})
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


def test_bfs_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as file:
        file.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = bfs.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(bfs, "download_snap_dataset", download_from_tmp)

    dataset = bfs.BreadthFirstSearchDataset("snap-toy")
    data, meta = bfs.BreadthFirstSearchGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert np.array_equal(data[0].data["indices_0"], np.array([0, 1]))
    assert np.array_equal(data[0].data["indices_1"], np.array([1, 2]))
    assert np.array_equal(data[0].data["values"], np.array([True, True]))
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0
