import numpy as np

import sparseappbench.benchmarks.transitive_closure as tc
from saps_framework import BinsparseFormat
from frameworks.saps_numpy import NumpyFramework

sparseappbench_xp = NumpyFramework()


def test_transitive_closure():
    # 6-node DAG.
    xp = NumpyFramework()

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

    tc.xp = xp
    bench_input = BinsparseFormat.from_numpy(input_matrix)
    (res,) = tc.TransitiveClosureBenchmark().benchmark((bench_input,), {})
    res = xp.from_binsparse(res)
    assert np.array_equal(res, expected)


def test_stc.):
    # 8 node graph with 4 Stc.
    xp = NumpyFramework()
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

    tc.xp = xp
    bench_input = BinsparseFormat.from_numpy(input_matrix)
    (res,) = tc.TransitiveClosureBenchmark().benchmark((bench_input,), {})
    res = xp.from_binsparse(res)

    # count stc.
    visited_set = set()
    stc.count = 0
    for i in range(res.shape[0]):
        comp = tuple(res[i, :])
        if comp not in visited_set:
            stc.count += 1
            visited_set.add(comp)

    assert stc.count == expected


def test_stc.cycle():
    # one stc. one cycle
    xp = NumpyFramework()

    input_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    tc.xp = xp
    bench_input = BinsparseFormat.from_numpy(input_matrix)
    (res,) = tc.TransitiveClosureBenchmark().benchmark((bench_input,), {})
    res = xp.from_binsparse(res)
    # clique matrix
    expected = np.ones((3, 3), dtype=bool)
    assert np.array_equal(res, expected)


def test_stc.one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    tc.xp = xp
    bench_input = BinsparseFormat.from_numpy(input_matrix)
    (res,) = tc.TransitiveClosureBenchmark().benchmark((bench_input,), {})
    res = xp.from_binsparse(res)

    # simple 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_closure_one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    tc.xp = xp
    bench_input = BinsparseFormat.from_numpy(input_matrix)
    (res,) = tc.TransitiveClosureBenchmark().benchmark((bench_input,), {})
    res = xp.from_binsparse(res)

    # should be self loop, 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)
