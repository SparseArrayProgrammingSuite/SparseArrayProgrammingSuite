import pytest

import numpy as np

from sparseappbench.benchmarks.tri_4cliq import (
    benchmark_4clique_count,
    benchmark_triangle_count,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


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
    A_bin = BinsparseFormat.from_numpy(A)

    bench_result = benchmark_triangle_count(xp, A_bin)
    result = xp.from_benchmark(bench_result).item()

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
    A_bin = BinsparseFormat.from_numpy(A)

    bench_result = benchmark_4clique_count(xp, A_bin)
    result = xp.from_benchmark(bench_result).item()

    assert np.allclose(result, expected)
