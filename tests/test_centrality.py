import gzip

import pytest

import numpy as np

import networkx as nx

import saps.benchmarks.centrality as centrality
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.snap import load_toy_dataset
from saps_framework import BinsparseFormat


def run_bc(A):
    xp = NumpyFramework()
    centrality.xp = xp
    A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
    (result,) = centrality.BetweennessCentralityBenchmark().benchmark([A_bin], {})
    return result.ravel()


# Modified the intended results because I am calculating
# unnormalized betweenness centrality.
def test_joels_case():
    A = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    result = run_bc(A)
    expected = np.array([0.0, 1.0, 1.0, 3.0, 0.0])

    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.zeros((3, 3)), np.array([0.0, 0.0, 0.0])),
        (
            np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
            np.array([0.0, 1.0, 0.0]),
        ),
        (
            np.array(
                [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                dtype=float,
            ),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_basic_bc(A, expected):
    result = run_bc(A)
    assert np.allclose(result, expected, atol=1e-6)


def reference_bc_alg_6_4(A):
    # Test for algorithm 6.4 from the Gilbert and Kempner book
    n = A.shape[0]
    BC = np.zeros(n)
    for s in range(n):
        stack = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1
        d = -np.ones(n)
        d[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            stack.append(v)
            for w in np.where(A[v, :] > 0)[0]:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                BC[w] += delta[w]
    return BC


def test_matrix_vertex_algorithm_comparison():
    # Test for comparing results from matrix and vertex-based algorithms
    rng = np.random.default_rng(42)
    n = 10
    A = (rng.random((n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0)

    result = run_bc(A)
    expected = reference_bc_alg_6_4(A)

    assert np.allclose(result, expected, atol=1e-6)


def test_undirected_graph():
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = 1
        A[i + 1, i] = 1

    result = run_bc(A)
    G = nx.DiGraph()
    for i in range(4):
        G.add_edge(i, i + 1)
        G.add_edge(i + 1, i)
    bc_nx = nx.betweenness_centrality(G, normalized=False)
    expected = np.array([bc_nx[i] for i in range(5)])

    assert np.allclose(result, expected, atol=1e-6)


def test_networkx():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 2),
        ]
    )

    A = nx.to_numpy_array(G, dtype=float)
    result = run_bc(A)

    bc = nx.betweenness_centrality(G, normalized=False)
    expected = np.array([bc[i] for i in range(len(G))])

    assert np.allclose(result, expected, atol=1e-6)


def test_centrality_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = centrality.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(centrality, "download_snap_dataset", download_from_tmp)

    dataset = centrality.BetweennessCentralityDataset("snap-toy")
    data, meta = centrality.BetweennessCentralityGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0


def test_centrality_snap_toy():
    data, _ = load_toy_dataset()
    result = run_bc(data[0])
    assert np.allclose(result, [0.0, 1.0, 0.0], atol=1e-6)
