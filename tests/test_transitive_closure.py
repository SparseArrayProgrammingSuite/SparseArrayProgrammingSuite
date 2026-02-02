import numpy as np

from sparseappbench.benchmarks.transitive_closure import (
    benchmark_simple_connected_components,
    benchmark_transitive_closure,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


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

    bench_input = BinsparseFormat.from_numpy(input_matrix)
    res = benchmark_transitive_closure(xp, bench_input)
    res = xp.from_benchmark(res)
    assert np.array_equal(res, expected)


def test_scc():
    # 8 node graph with 4 SCCs
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

    bench_input = BinsparseFormat.from_numpy(input_matrix)
    res = benchmark_simple_connected_components(xp, bench_input)
    res = xp.from_benchmark(res)

    # count sccs
    visited_set = set()
    scc_count = 0
    for i in range(res.shape[0]):
        comp = tuple(res[i, :])
        if comp not in visited_set:
            scc_count += 1
            visited_set.add(comp)

    assert scc_count == expected


def test_scc_cycle():
    # one scc, one cycle
    xp = NumpyFramework()

    input_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    bench_input = BinsparseFormat.from_numpy(input_matrix)
    res = benchmark_simple_connected_components(xp, bench_input)
    res = xp.from_benchmark(res)
    # clique matrix
    expected = np.ones((3, 3), dtype=bool)
    assert np.array_equal(res, expected)


def test_scc_one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    bench_input = BinsparseFormat.from_numpy(input_matrix)
    res = benchmark_simple_connected_components(xp, bench_input)
    res = xp.from_benchmark(res)

    # simple 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_closure_one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    bench_input = BinsparseFormat.from_numpy(input_matrix)
    res = benchmark_transitive_closure(xp, bench_input)
    res = xp.from_benchmark(res)

    # should be self loop, 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)
