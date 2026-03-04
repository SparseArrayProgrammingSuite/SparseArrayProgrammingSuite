import numpy as np

from sparseappbench.benchmarks.gcn_backward import benchmark_gcn_backward
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_gcn_backward_2node():
    """Test backward pass on simple 2-node graph."""
    # Graph: 0 -- 1
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.float64)
    adjacency_T = adjacency.T
    features = np.array([[1.0], [2.0]])
    weights1 = np.array([[1.0]])
    bias1 = np.array([0.0])
    weights2 = np.array([[1.0]])
    bias2 = np.array([0.0])
    targets = np.array([[2.0], [1.0]])

    xp = NumpyFramework()

    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    adjacency_T_b = BinsparseFormat.from_numpy(adjacency_T)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)
    targets_b = BinsparseFormat.from_numpy(targets)

    loss, w1, b1, w2, b2 = benchmark_gcn_backward(
        xp,
        adjacency_b,
        adjacency_T_b,
        features_b,
        weights1_b,
        bias1_b,
        weights2_b,
        bias2_b,
        targets_b,
        num_iterations=10,
        learning_rate=0.01,
    )

    # Should return valid outputs
    assert loss is not None
    assert w1 is not None
    assert b1 is not None
    assert w2 is not None
    assert b2 is not None


def test_gcn_backward_multidim():
    """Test backward pass with multi-dimensional features and hidden layers."""
    # 4-node graph with 2D features, 3D hidden, 2D output
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    adjacency_T = adjacency.T

    features = np.array(
        [
            [1.0, 0.5],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    weights1 = np.array([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]])  # (2, 3)
    bias1 = np.zeros(3)
    weights2 = np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])  # (3, 2)
    bias2 = np.zeros(2)
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )

    xp = NumpyFramework()

    adjacency_b = BinsparseFormat.from_numpy(adjacency)
    adjacency_T_b = BinsparseFormat.from_numpy(adjacency_T)
    features_b = BinsparseFormat.from_numpy(features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)
    targets_b = BinsparseFormat.from_numpy(targets)

    # Get initial loss (1 iteration)
    loss_1, _, _, _, _ = benchmark_gcn_backward(
        xp,
        adjacency_b,
        adjacency_T_b,
        features_b,
        weights1_b,
        bias1_b,
        weights2_b,
        bias2_b,
        targets_b,
        num_iterations=1,
        learning_rate=0.01,
    )

    # Get loss after training
    loss_100, w1, b1, w2, b2 = benchmark_gcn_backward(
        xp,
        adjacency_b,
        adjacency_T_b,
        features_b,
        weights1_b,
        bias1_b,
        weights2_b,
        bias2_b,
        targets_b,
        num_iterations=100,
        learning_rate=0.01,
    )

    assert loss_100 < loss_1, f"Loss should decrease: {loss_100} < {loss_1}"

    # Check output shapes
    w1_np = xp.from_benchmark(w1)
    b1_np = xp.from_benchmark(b1)
    w2_np = xp.from_benchmark(w2)
    b2_np = xp.from_benchmark(b2)

    assert w1_np.shape == (2, 3)
    assert b1_np.shape == (3,)
    assert w2_np.shape == (3, 2)
    assert b2_np.shape == (2,)


def test_gcn_backward_degree_prediction():
    """Test that GCN learns to predict node degrees from graph structure.

    Training graph: Star with tail + singleton (7 nodes)
        Node 0 is hub connected to nodes 1, 2, 3, 4
        Node 4 is bridge also connected to node 5
        Node 6 is a singleton (no connections)
        Degrees: [4, 1, 1, 1, 2, 1, 0]

    Test graph: Different structure (6 nodes)
        Node 0 connected to 1, 2, 3 (degree 3)
        Node 1 connected to 0, 2 (degree 2)
        Node 2 connected to 0, 1, 4 (degree 3)
        Node 3 connected to 0 (degree 1)
        Node 4 connected to 2 (degree 1)
        Node 5 is a singleton (degree 0)
        Degrees: [3, 2, 3, 1, 1, 0]

    Uses constant features (all 1s) to force learning from structure alone.
    After training on one graph, tests on a different graph to verify the
    network learned to predict degrees, not just memorize the training data.
    """
    # Training graph: Star with tail + singleton (7 nodes)
    train_adj = np.array(
        [
            [0, 1, 1, 1, 1, 0, 0],  # node 0: degree 4
            [1, 0, 0, 0, 0, 0, 0],  # node 1: degree 1
            [1, 0, 0, 0, 0, 0, 0],  # node 2: degree 1
            [1, 0, 0, 0, 0, 0, 0],  # node 3: degree 1
            [1, 0, 0, 0, 0, 1, 0],  # node 4: degree 2
            [0, 0, 0, 0, 1, 0, 0],  # node 5: degree 1
            [0, 0, 0, 0, 0, 0, 0],  # node 6: degree 0 (singleton)
        ],
        dtype=np.float64,
    )
    train_adj_T = train_adj.T
    train_features = np.ones((7, 1))
    train_degrees = train_adj.sum(axis=1, keepdims=True)
    train_targets = train_degrees / train_degrees.max()

    # Test graph: Different structure (6 nodes)
    test_adj = np.array(
        [
            [0, 1, 1, 1, 0, 0],  # node 0: degree 3
            [1, 0, 1, 0, 0, 0],  # node 1: degree 2
            [1, 1, 0, 0, 1, 0],  # node 2: degree 3
            [1, 0, 0, 0, 0, 0],  # node 3: degree 1
            [0, 0, 1, 0, 0, 0],  # node 4: degree 1
            [0, 0, 0, 0, 0, 0],  # node 5: degree 0 (singleton)
        ],
        dtype=np.float64,
    )
    test_features = np.ones((6, 1))

    # Initialize weights (input_dim=1, hidden_dim=4, output_dim=1)
    rng = np.random.default_rng(42)
    weights1 = rng.standard_normal((1, 4)) * 0.5
    bias1 = np.zeros(4)
    weights2 = rng.standard_normal((4, 1)) * 0.5
    bias2 = np.zeros(1)

    xp = NumpyFramework()

    # Train on training graph
    train_adj_b = BinsparseFormat.from_numpy(train_adj)
    train_adj_T_b = BinsparseFormat.from_numpy(train_adj_T)
    train_features_b = BinsparseFormat.from_numpy(train_features)
    weights1_b = BinsparseFormat.from_numpy(weights1)
    bias1_b = BinsparseFormat.from_numpy(bias1)
    weights2_b = BinsparseFormat.from_numpy(weights2)
    bias2_b = BinsparseFormat.from_numpy(bias2)
    train_targets_b = BinsparseFormat.from_numpy(train_targets)

    _, w1_b, b1_b, w2_b, b2_b = benchmark_gcn_backward(
        xp,
        train_adj_b,
        train_adj_T_b,
        train_features_b,
        weights1_b,
        bias1_b,
        weights2_b,
        bias2_b,
        train_targets_b,
        num_iterations=500,
        learning_rate=0.01,
    )

    # Get trained weights
    w1_trained = xp.from_benchmark(w1_b)
    b1_trained = xp.from_benchmark(b1_b)
    w2_trained = xp.from_benchmark(w2_b)
    b2_trained = xp.from_benchmark(b2_b)

    # Run forward pass on TEST graph with trained weights
    Z1 = test_adj @ test_features
    H1_pre = Z1 @ w1_trained + b1_trained
    H1 = np.maximum(H1_pre, 0)
    Z2 = test_adj @ H1
    predictions = Z2 @ w2_trained + b2_trained

    # Test graph degrees: [3, 2, 3, 1, 1, 0]
    # Nodes 0 and 2 have degree 3 (highest)
    # Node 1 has degree 2 (middle)
    # Nodes 3 and 4 have degree 1 (low)
    # Node 5 has degree 0 (singleton)
    high_degree_preds = [predictions[0, 0], predictions[2, 0]]
    mid_degree_pred = predictions[1, 0]
    low_degree_preds = [predictions[3, 0], predictions[4, 0]]
    singleton_pred = predictions[5, 0]

    min_high = min(high_degree_preds)
    max_low = max(low_degree_preds)

    # High degree nodes should have higher predictions than low degree nodes
    assert min_high > max_low, (
        f"High degree predictions ({high_degree_preds}) should be > "
        f"low degree predictions ({low_degree_preds})"
    )
    # Mid degree node should be between high and low
    assert min_high > mid_degree_pred > max_low, (
        f"Mid degree prediction ({mid_degree_pred:.3f}) should be between "
        f"high ({min_high:.3f}) and low ({max_low:.3f})"
    )
    # Singleton should have lowest prediction (near zero)
    assert max_low > singleton_pred, (
        f"Low degree predictions ({low_degree_preds}) should be > "
        f"singleton prediction ({singleton_pred:.3f})"
    )
    assert abs(singleton_pred) < 0.1, (
        f"Singleton prediction ({singleton_pred:.3f}) should be near zero"
    )
