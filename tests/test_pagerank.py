import gzip

import pytest

import numpy as np

import networkx as nx

import saps.benchmarks.pagerank as pr
from frameworks.saps_numpy import NumpyFramework


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.array([[0, 1], [1, 0]], dtype=float), np.array([0.5, 0.5], dtype=float)),
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), None),
        (np.array([[0, 0], [1, 0]], dtype=float), None),
    ],
)
def test_basic_pagerank_cases(A, expected):
    xp = NumpyFramework()

    pr.xp = xp
    (result,) = pr.PageRankBenchmark().benchmark((A,), {})

    result = result.ravel()

    if expected is not None:
        assert np.allclose(result, expected, atol=1e-2)
    else:
        assert np.isclose(np.sum(result), 1.0, atol=1e-6)
        assert np.all(result >= 0)

        if A.shape == (3, 3) and np.all(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) == A
        ):
            eps = 1e-6
            assert (result[0] < result[1] - eps) and (result[1] < result[2] - eps)

        if A.shape == (2, 2) and np.all(np.array([[0, 0], [1, 0]]) == A):
            eps = 1e-6
            assert result[0] < result[1] - eps


def test_pagerank_against_networkx():
    xp = NumpyFramework()
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    A = nx.to_numpy_array(G, dtype=float)

    pr.xp = xp
    (result,) = pr.PageRankBenchmark().benchmark((A,), {})
    result = result.ravel()

    expected_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    expected = np.array([expected_dict[i] for i in range(len(G))])

    assert np.allclose(result, expected, atol=1e-2)


def test_pagerank_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as file:
        file.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = pr.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(pr, "download_snap_dataset", download_from_tmp)

    dataset = pr.PageRankDataset("snap-toy")
    data, meta = pr.PageRankGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert np.array_equal(data[0].data["indices_0"], np.array([0, 1]))
    assert np.array_equal(data[0].data["indices_1"], np.array([1, 2]))
    assert np.array_equal(data[0].data["values"], np.array([True, True]))
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0
