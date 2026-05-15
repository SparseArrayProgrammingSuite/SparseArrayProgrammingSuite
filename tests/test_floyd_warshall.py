import pytest

import numpy as np

import saps.benchmarks.floyd_warshall as floyd_warshall
from frameworks.saps_numpy import NumpyFramework


def _run_fw_case(xp, A, expected):
    """Run Floyd–Warshall and compare to an expected APSP distance matrix."""
    floyd_warshall.xp = xp
    out = floyd_warshall.FloydWarshallBenchmark().benchmark([A], {})[0]

    both_inf = np.isinf(out) & np.isinf(expected)
    both_finite = np.isfinite(out) & np.isfinite(expected)
    assert np.all(both_inf | (both_finite & (out == expected))), (
        f"Floyd–Warshall output mismatch.\nGot:\n{out}\nExpected:\n{expected}"
    )


def test_fw_tiny_cases():
    xp = NumpyFramework()

    A = np.array([[0.0]])
    expected = np.array([[0.0]])
    _run_fw_case(xp, A, expected)

    A = np.array([[0.0, 1.0], [np.inf, 0.0]])
    expected = np.array([[0.0, 1.0], [np.inf, 0.0]])
    _run_fw_case(xp, A, expected)

    A = np.array(
        [
            [0.0, 1.0, np.inf],
            [np.inf, 0.0, 1.0],
            [np.inf, np.inf, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.inf, 0.0, 1.0],
            [np.inf, np.inf, 0.0],
        ]
    )
    _run_fw_case(xp, A, expected)

    A = np.array(
        [
            [0.0, 1.0, 5.0],
            [np.inf, 0.0, 1.0],
            [np.inf, np.inf, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.inf, 0.0, 1.0],
            [np.inf, np.inf, 0.0],
        ]
    )
    _run_fw_case(NumpyFramework(), A, expected)

    A = np.array(
        [
            [0.0, 1.0, np.inf, np.inf],
            [1.0, 0.0, np.inf, np.inf],
            [np.inf, np.inf, 0.0, 1.0],
            [np.inf, np.inf, 1.0, 0.0],
        ]
    )
    expected = A.copy()
    _run_fw_case(NumpyFramework(), A, expected)


@pytest.mark.parametrize("dataset", floyd_warshall.FloydWarshallGenerator().datasets)
def test_fw_generated_datasets(dataset):
    xp = NumpyFramework()
    n = 39

    edges = [
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

    A = np.full((n, n), np.inf)
    np.fill_diagonal(A, 0.0)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    out = floyd_warshall.FloydWarshallBenchmark().benchmark((A,), {})[0]

    assert out.shape[0] == out.shape[1]
    assert np.all(np.diag(out) == 0.0)
    assert np.all(out >= 0.0)

    if dataset.symmetrize:
        assert np.all(out == out.T)

    rng = np.random.default_rng(0)
    for _ in range(50):
        i, j, k = rng.integers(0, out.shape[0], size=3)
        assert out[i, j] <= out[i, k] + out[k, j]
