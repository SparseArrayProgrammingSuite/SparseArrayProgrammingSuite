import numpy as np

import saps.benchmarks.transitive_closure as tc
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.snap import load_toy_dataset
from saps_framework import BinsparseFormat


def _run_tc(A):
    xp = NumpyFramework()
    tc.xp = xp
    A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
    (res,) = tc.TransitiveClosureBenchmark().benchmark([A_bin], {})
    return res


def test_transitive_closure():
    # 6-node DAG.
    input_matrix = np.array(
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

    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=bool,
    )

    res = _run_tc(input_matrix)
    assert np.array_equal(res, expected)


def test_stc():
    # 8 node graph with 4 Stc.
    input_matrix = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )

    expected = 4

    res = _run_tc(input_matrix)

    # count stc.
    visited_set = set()
    count = 0
    for i in range(res.shape[0]):
        comp = tuple(res[i, :])
        if comp not in visited_set:
            count += 1
            visited_set.add(comp)

    assert count == expected


def test_stc_cycle():
    # one stc. one cycle
    input_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    res = _run_tc(input_matrix)
    # clique matrix
    expected = np.ones((3, 3), dtype=bool)
    assert np.array_equal(res, expected)


def test_stc_one_node():
    # one node
    input_matrix = np.array([[0]], dtype=bool)

    res = _run_tc(input_matrix)

    # simple 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_closure_one_node():
    # one node
    input_matrix = np.array([[0]], dtype=bool)

    res = _run_tc(input_matrix)

    # should be self loop, 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_snap_toy():
    data, _ = load_toy_dataset()
    res = _run_tc(data[0])
    expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
    assert np.array_equal(res, expected)
