import pytest

import numpy as np

from sparseappbench.benchmarks.bellmanford import bellman_ford
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework

xp = NumpyFramework()


def bellman_ford_reference(A, src):
    n = A.shape[0]
    D = np.full((n,), np.inf)
    D[src] = 0
    for _ in range(n):
        for v in range(n):
            for u in range(n):
                if A[u, v] + D[u] < D[v]:
                    D[v] = A[u, v] + D[u]
    return D


# --- TRIBES SOCIAL NETWORK ---
TRIBES_N = 16
TRIBES_EDGES = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (2, 3),
    (3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (4, 1),
    (0, 5),
    (5, 0),
    (1, 5),
    (5, 1),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (4, 6),
    (6, 4),
    (5, 6),
    (6, 5),
    (2, 7),
    (7, 2),
    (3, 7),
    (7, 3),
    (5, 7),
    (7, 5),
    (6, 7),
    (7, 6),
    (1, 8),
    (8, 1),
    (4, 8),
    (8, 4),
    (7, 8),
    (8, 7),
    (3, 9),
    (9, 3),
    (8, 9),
    (9, 8),
    (9, 10),
    (10, 9),
    (10, 11),
    (11, 10),
    (8, 11),
    (11, 8),
    (11, 12),
    (12, 11),
    (7, 12),
    (12, 7),
    (12, 13),
    (13, 12),
    (13, 14),
    (14, 13),
    (9, 14),
    (14, 9),
    (14, 15),
    (15, 14),
    (12, 15),
    (15, 12),
]


def build_tribes_matrix():
    A = np.full((TRIBES_N, TRIBES_N), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in TRIBES_EDGES:
        A[u, v] = 1.0
    return A


# --- CHESAPEAKE ROAD NETWORK ---
CHESAPEAKE_N = 39
CHESAPEAKE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (1, 14),
    (1, 15),
    (1, 16),
    (2, 9),
    (2, 10),
    (2, 17),
    (2, 18),
    (3, 11),
    (3, 12),
    (3, 19),
    (3, 20),
    (3, 21),
    (4, 13),
    (4, 22),
    (4, 23),
    (4, 24),
    (5, 14),
    (5, 22),
    (5, 25),
    (5, 26),
    (6, 15),
    (6, 23),
    (6, 27),
    (6, 28),
    (7, 16),
    (7, 24),
    (7, 29),
    (7, 30),
    (8, 17),
    (8, 18),
    (8, 19),
    (8, 20),
    (8, 21),
    (9, 22),
    (9, 31),
    (9, 32),
    (10, 23),
    (10, 31),
    (10, 33),
    (11, 24),
    (11, 32),
    (11, 34),
    (12, 25),
    (12, 26),
    (12, 35),
    (13, 27),
    (13, 36),
    (14, 28),
    (14, 37),
    (15, 29),
    (15, 38),
    (16, 30),
    (17, 31),
    (18, 32),
    (19, 33),
    (20, 34),
    (21, 35),
    (22, 36),
    (23, 37),
    (24, 38),
    (25, 27),
    (25, 29),
    (26, 28),
    (26, 30),
    (27, 31),
    (27, 33),
    (28, 32),
    (28, 34),
    (29, 35),
    (30, 36),
    (31, 37),
    (32, 38),
    (33, 35),
    (34, 36),
    (35, 37),
    (36, 38),
    (37, 38),
]


def build_chesapeake_matrix():
    A = np.full((CHESAPEAKE_N, CHESAPEAKE_N), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in CHESAPEAKE_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


@pytest.mark.parametrize(
    "matrix_builder,src",
    [
        (build_tribes_matrix, 0),
        (build_chesapeake_matrix, 0),
        (build_chesapeake_matrix, 10),
        (build_chesapeake_matrix, 38),
    ],
)
def test_bellman_ford_networks(matrix_builder, src):
    A = matrix_builder()
    bench = BinsparseFormat.from_numpy(A)
    result_bin = bellman_ford(xp, bench, src)
    result = xp.from_benchmark(result_bin).ravel()
    ref = bellman_ford_reference(A, src)
    assert np.allclose(result, ref, equal_nan=True)
