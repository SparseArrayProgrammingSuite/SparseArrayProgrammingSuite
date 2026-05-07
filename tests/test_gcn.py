import pytest

import numpy as np

import saps.benchmarks.gcn as gcn
from frameworks.saps_numpy import NumpyFramework

def gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2):
    """Reference NumPy implementation of the 2-layer GCN used for tests.

    Inputs are dense NumPy arrays; adjacency is treated as a dense matrix for
    simplicity in tests (small graphs).
    """
    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = np.maximum(h1, 0)

    h2 = adjacency @ h1
    return h2 @ weights2 + bias2


def run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2):
    xp = NumpyFramework()
    benchmark = gcn.GCNBenchmark()
    prev_xp = getattr(gcn, "xp", None)
    gcn.xp = xp
    try:
        output_b = benchmark.benchmark(
            [adjacency, features, weights1, bias1, weights2, bias2],
            {},
        )
    finally:
        gcn.xp = prev_xp
    return xp.from_binsparse(output_b)

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
    ],
)
def test_benchmark_gcn(xp, adjacency, features, weights1, bias1, weights2, bias2):
    expected = gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2)
    output = run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2)
    np.testing.assert_allclose(output, expected, rtol=1e-10)


def test_gcn_benchmark_smoke():
    """Smoke test for the class-based benchmark interface."""
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias1 = np.array([0.0, 0.0])
    weights2 = np.array([[1.0], [1.0]])
    bias2 = np.array([0.0])

    output = run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2)
    assert output.shape == (3, 1)


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
    output_np = run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2)
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
    output_np = run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2)
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

    output_np = run_gcn_benchmark(adjacency, features, weights1, bias1, weights2, bias2)
    np.testing.assert_allclose(output_np, expected, rtol=1e-10)
