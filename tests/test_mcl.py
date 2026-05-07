import pytest

import numpy as np
import scipy.sparse as sp

import saps.benchmarks.mcl_benchmark as mcl
from saps_framework import BinsparseFormat
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework


def get_cluster_count(matrix):
    if hasattr(matrix, "todense") and not sp.isspmatrix(matrix):
        matrix = matrix.todense()
    if not sp.isspmatrix(matrix):
        matrix = sp.csc_matrix(matrix)
    elif not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    attractors = matrix.diagonal().nonzero()[0]

    clusters = set()
    for attractor in attractors:
        cluster_indices = matrix.getrow(attractor).nonzero()[1].tolist()
        clusters.add(tuple(sorted(cluster_indices)))

    return len(clusters)


def create_planted_clique(N, k):
    A = np.zeros((N, N), dtype=np.float32)
    A[:k, :k] = 1.0
    np.fill_diagonal(A, 0)
    return A


def run_mcl_benchmark(xp, A):
    benchmark = mcl.MCLBenchmark()
    prev_xp = getattr(mcl, "xp", None)
    mcl.xp = xp
    try:
        (result_matrix,) = benchmark.benchmark(
            [A], {"expansion": 2, "inflation": 2, "loop_value": 1}
        )
    finally:
        mcl.xp = prev_xp
    return result_matrix


@pytest.mark.parametrize(
    "xp, A, expected_count",
    [
        (
            NumpyFramework(),
            np.array(
                [
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            2,
        ),
        (
            PyDataSparseFramework(),
            np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                ],
                dtype=np.float32,
            ),
            3,
        ),
        (
            NumpyFramework(),
            create_planted_clique(N=10, k=4),
            7,
        ),
    ],
)
def test_mcl_solver(xp, A, expected_count):
    A_sparse = sp.coo_matrix(A)
    A_bin = BinsparseFormat.from_coo(
        (A_sparse.row, A_sparse.col), A_sparse.data, A_sparse.shape
    )
    A_input = xp.from_binsparse(A_bin)

    result_matrix = run_mcl_benchmark(xp, A_input)

    actual_count = get_cluster_count(result_matrix)

    assert actual_count == expected_count, (
        f"MCL failed for graph {A.shape}. "
        f"Found {actual_count} clusters, "
        f"but expected {expected_count}."
    )
