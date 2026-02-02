import pytest

import numpy as np

import networkx as nx

from sparseappbench.benchmarks.centrality import betweenness_centrality
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def run_bc(xp, A):
    A_bin = BinsparseFormat.from_numpy(A)
    result_bin = betweenness_centrality(xp, A_bin)
    return xp.from_benchmark(result_bin).ravel()


# Modified the intended results because I am calculating
# unnormalized betweenness centrality.
def test_joels_case():
    xp = NumpyFramework()

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

    result = run_bc(xp, A)
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
    xp = NumpyFramework()
    result = run_bc(xp, A)
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
    # Test for comparing results from matrix and vertext-based algorithms
    xp = NumpyFramework()
    rng = np.random.default_rng(42)
    n = 10
    A = (rng.random((n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0)

    result = run_bc(xp, A)
    expected = reference_bc_alg_6_4(A)

    assert np.allclose(result, expected, atol=1e-6)


def test_undirected_graph():
    xp = NumpyFramework()
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = 1
        A[i + 1, i] = 1

    result = run_bc(xp, A)
    G = nx.DiGraph()
    for i in range(4):
        G.add_edge(i, i + 1)
        G.add_edge(i + 1, i)
    bc_nx = nx.betweenness_centrality(G, normalized=False)
    expected = np.array([bc_nx[i] for i in range(5)])

    assert np.allclose(result, expected, atol=1e-6)


def test_networkx():
    xp = NumpyFramework()
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
    result = run_bc(xp, A)

    bc = nx.betweenness_centrality(G, normalized=False)
    expected = np.array([bc[i] for i in range(len(G))])

    assert np.allclose(result, expected, atol=1e-6)
