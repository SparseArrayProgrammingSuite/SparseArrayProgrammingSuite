import pytest

import numpy as np

import saps.benchmarks.tri_4cliq as tri_4cliq
from frameworks.saps_numpy import NumpyFramework
from saps_framework import BinsparseFormat


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
    A_bin = BinsparseFormat.from_numpy(A)
    A_input = xp.from_binsparse(A_bin)

    result = run_count_benchmark(tri_4cliq.TriangleCountBenchmark(), xp, A_input).item()

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
    A_input = xp.from_binsparse(A_bin)

    result = run_count_benchmark(
        tri_4cliq.FourCliqueCountBenchmark(), xp, A_input
    ).item()

    assert np.allclose(result, expected)
