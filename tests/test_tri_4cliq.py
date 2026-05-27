import gzip

import pytest

import numpy as np

import saps.benchmarks.tri_4cliq as tri_4cliq
from frameworks.saps_numpy import NumpyFramework


def run_count_benchmark(benchmark, xp, A):
    prev_xp = getattr(tri_4cliq, "xp", None)
    tri_4cliq.xp = xp
    try:
        (result,) = benchmark.benchmark([A], {})
    finally:
        tri_4cliq.xp = prev_xp
    return result


@pytest.mark.parametrize(
    "A, expected",
    [
        # Single triangle
        (
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ],
                dtype=int,
            ),
            1,
        ),
        # Path - no triangles
        (
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=int,
            ),
            0,
        ),
        # 4 clique - contains 4c3 = 4 triangles
        (
            np.array(
                [
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                ],
                dtype=int,
            ),
            4,
        ),
    ],
)
def test_triangle_count(A, expected):
    xp = NumpyFramework()
    result = run_count_benchmark(tri_4cliq.TriangleCountBenchmark(), xp, A).item()
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "A, expected",
    [
        # Complete graph K3 - no 4-cliques
        (
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ],
                dtype=int,
            ),
            0,
        ),
        # Single 4-clique (K4)
        (
            np.array(
                [
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                ],
                dtype=int,
            ),
            1,
        ),
        # Two overlapping 4-cliques sharing an edge, only 2 4-cliques should return 2.
        # Nodes {0,1,2,3} and {1,2,3,4}
        (
            np.array(
                [
                    [0, 1, 1, 1, 0],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ],
                dtype=int,
            ),
            2,
        ),
    ],
)
def test_4clique_count(A, expected):
    xp = NumpyFramework()
    result = run_count_benchmark(
        tri_4cliq.FourCliqueCountBenchmark(), xp, A
    ).item()
    assert np.allclose(result, expected)


def test_triangle_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = tri_4cliq.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(tri_4cliq, "download_snap_dataset", download_from_tmp)

    dataset = tri_4cliq.GraphCountingDataset("snap-toy")
    data, meta = tri_4cliq.TriangleCountGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0


def test_4clique_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = tri_4cliq.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(tri_4cliq, "download_snap_dataset", download_from_tmp)

    dataset = tri_4cliq.GraphCountingDataset("snap-toy")
    data, meta = tri_4cliq.FourCliqueCountGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0
