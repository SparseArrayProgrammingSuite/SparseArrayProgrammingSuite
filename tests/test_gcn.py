import pytest

import numpy as np

from sparseappbench.benchmarks.gcn import (
    benchmark_gcn,
    dg_gcn_social_1,
    gcn_reference_np,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "xp,adjacency,features,weights1,bias1,weights2,bias2",
    [
        (
            NumpyFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[1, 0], [0, 1], [1, 1]]),
            np.array([[1, 0], [0, 1]]),
            np.array([0, 0]),
            np.array([[1], [1]]),
            np.array([0]),
        ),
        (
            CheckerFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[1, 0], [0, 1], [1, 1]]),
            np.array([[1, 0], [0, 1]]),
            np.array([0, 0]),
            np.array([[1], [1]]),
            np.array([0]),
        ),
    ],
)
def test_benchmark_gcn(xp, adjacency, features, weights1, bias1, weights2, bias2):
    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)

    output_b = benchmark_gcn(
        xp, adjacency_b, features_b, weights1_b, bias1_b, weights2_b, bias2_b
    )
    output_coo = BinsparseFormat.to_coo(output_b)

    expected = gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2)
    assert output_coo == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(expected))


def test_dg_gcn_social_1():
    """Test social network graph generator."""
    A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b = dg_gcn_social_1()
    xp = NumpyFramework()
    output_b = benchmark_gcn(
        xp, A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b
    )
    assert output_b is not None


def test_gcn_simple_2node():
    """Test GCN on a simple 2-node graph with hand-computed expected output.

    Graph: 0 -- 1 (single edge)
    Adjacency: [[0, 1], [1, 0]]

    Manual computation:
    Layer 1: h1 = A @ X @ W1 + b1
      A @ X = [[0, 1], [1, 0]] @ [[1], [2]] = [[2], [1]]
      h1 = [[2], [1]] @ [[2]] = [[4], [2]]
      h1 = ReLU([[4], [2]]) = [[4], [2]]

    Layer 2: output = A @ h1 @ W2 + b2
      A @ h1 = [[0, 1], [1, 0]] @ [[4], [2]] = [[2], [4]]
      output = [[2], [4]] @ [[3]] = [[6], [12]]
    """
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.float64)
    features = np.array([[1.0], [2.0]])
    weights1 = np.array([[2.0]])
    bias1 = np.array([0.0])
    weights2 = np.array([[3.0]])
    bias2 = np.array([0.0])

    expected = np.array([[6.0], [12.0]])

    output = gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2)
    np.testing.assert_allclose(output, expected, rtol=1e-10)

    # Also test with benchmark_gcn
    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)

    xp = NumpyFramework()
    output_b = benchmark_gcn(
        xp, adjacency_b, features_b, weights1_b, bias1_b, weights2_b, bias2_b
    )
    output_np = xp.from_benchmark(output_b)
    np.testing.assert_allclose(output_np, expected, rtol=1e-10)


def test_gcn_simple_3node_line():
    """Test GCN on a 3-node line graph with hand-computed expected output.

    Source: Computation methodology based on "Graph Convolutional Network (GCN) by Hand"
    byhand.ai.
    https://www.byhand.ai/p/17-can-you-calculate-a-graph-convolutional

    Test case manually computed following the GCN formula from GCN.py (lines 37-40).

    Graph: 0 -- 1 -- 2 (line graph)
    Adjacency: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    Manual computation:
    Layer 1: h1 = A @ X @ W1 + b1
      A @ X = [[0, 1, 0], [1, 0, 1], [0, 1, 0]] @ [[1], [0], [1]] = [[0], [2], [0]]
      h1 = [[0], [2], [0]] @ [[1]] = [[0], [2], [0]]
      h1 = ReLU([[0], [2], [0]]) = [[0], [2], [0]]

    Layer 2: output = A @ h1 @ W2 + b2
      A @ h1 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]] @ [[0], [2], [0]] = [[2], [0], [2]]
      output = [[2], [0], [2]] @ [[1]] = [[2], [0], [2]]
    """
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    features = np.array([[1.0], [0.0], [1.0]])
    weights1 = np.array([[1.0]])
    bias1 = np.array([0.0])
    weights2 = np.array([[1.0]])
    bias2 = np.array([0.0])

    expected = np.array([[2.0], [0.0], [2.0]])

    output = gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2)
    np.testing.assert_allclose(output, expected, rtol=1e-10)

    # Also test with benchmark_gcn
    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)

    xp = NumpyFramework()
    output_b = benchmark_gcn(
        xp, adjacency_b, features_b, weights1_b, bias1_b, weights2_b, bias2_b
    )
    output_np = xp.from_benchmark(output_b)
    np.testing.assert_allclose(output_np, expected, rtol=1e-10)


def test_gcn_with_relu_activation():
    """Test GCN with ReLU activation (negative values zeroed out).

    Source: Computation methodology based on "Graph Convolutional Network (GCN) by Hand"
    byhand.ai.
    https://www.byhand.ai/p/17-can-you-calculate-a-graph-convolutional

    Test case manually computed following the GCN formula from GCN.py (lines 37-40).
    This test verifies that ReLU activation works correctly by using
    weights that produce negative intermediate values.

    Graph: 0 -- 1
    """
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.float64)
    features = np.array([[1.0], [-1.0]])
    weights1 = np.array([[1.0]])
    bias1 = np.array([0.0])
    weights2 = np.array([[2.0]])
    bias2 = np.array([0.0])

    # Manual computation:
    # Layer 1: h1 = A @ X @ W1
    #   A @ X = [[0, 1], [1, 0]] @ [[1], [-1]] = [[-1], [1]]
    #   h1 = [[-1], [1]] @ [[1]] = [[-1], [1]]
    #   h1 = ReLU([[-1], [1]]) = [[0], [1]]  <- ReLU zeros out negative value
    # Layer 2: output = A @ h1 @ W2
    #   A @ h1 = [[0, 1], [1, 0]] @ [[0], [1]] = [[1], [0]]
    #   output = [[1], [0]] @ [[2]] = [[2], [0]]

    expected = np.array([[2.0], [0.0]])

    output = gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2)
    np.testing.assert_allclose(output, expected, rtol=1e-10)

    # Also test with benchmark_gcn
    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)

    xp = NumpyFramework()
    output_b = benchmark_gcn(
        xp, adjacency_b, features_b, weights1_b, bias1_b, weights2_b, bias2_b
    )
    output_np = xp.from_benchmark(output_b)
    np.testing.assert_allclose(output_np, expected, rtol=1e-10)
