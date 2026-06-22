import os
from typing import Any

import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


class GCNTrainingDataset(Dataset):
    def __init__(
        self,
        name: str,
        description: str | None = None,
        source_name: str | None = None,
        *,
        feature_dim: int = 1,
        hidden_dim: int = 4,
        out_dim: int = 1,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
    ):
        self._suites: list[str] = []
        self._name = name
        self._description = description
        self.source_name = source_name or name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return f"GCN {self._name}"

    @property
    def description(self) -> str:
        if self._description is not None:
            return self._description
        return f"SuiteSparse matrix {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_name"] = self.source_name
        data["feature_dim"] = self.feature_dim
        data["hidden_dim"] = self.hidden_dim
        data["out_dim"] = self.out_dim
        data["num_iterations"] = self.num_iterations
        data["learning_rate"] = self.learning_rate
        return data


# BEGIN COPIED TEST FILE: tests/test_gcn_backward.py
# import numpy as np
#
# import saps.benchmarks.gcn_backward as gcn_backward
# from frameworks.saps_numpy import NumpyFramework
#
#
# def run_gcn_backward_benchmark(
#     xp,
#     adjacency,
#     adjacency_T,
#     features,
#     weights1,
#     bias1,
#     weights2,
#     bias2,
#     targets,
#     num_iterations=10,
#     learning_rate=0.01,
# ):
#     benchmark = gcn_backward.GCNBackwardBenchmark()
#     prev_xp = getattr(gcn_backward, "xp", None)
#     gcn_backward.xp = xp
#     try:
#         return benchmark.benchmark(
#             [
#                 adjacency,
#                 adjacency_T,
#                 features,
#                 weights1,
#                 bias1,
#                 weights2,
#                 bias2,
#                 targets,
#             ],
#             {
#                 "num_iterations": num_iterations,
#                 "learning_rate": learning_rate,
#             },
#         )
#     finally:
#         gcn_backward.xp = prev_xp
#
#
# def test_gcn_backward_2node():
#     """Test backward pass on simple 2-node graph."""
#     # Graph: 0 -- 1
#     adjacency = np.array([[0, 1], [1, 0]], dtype=np.float64)
#     adjacency_T = adjacency.T
#     features = np.array([[1.0], [2.0]])
#     weights1 = np.array([[1.0]])
#     bias1 = np.array([0.0])
#     weights2 = np.array([[1.0]])
#     bias2 = np.array([0.0])
#     targets = np.array([[2.0], [1.0]])
#
#     xp = NumpyFramework()
#
#     loss, w1, b1, w2, b2 = run_gcn_backward_benchmark(
#         xp,
#         adjacency,
#         adjacency_T,
#         features,
#         weights1,
#         bias1,
#         weights2,
#         bias2,
#         targets,
#         num_iterations=10,
#         learning_rate=0.01,
#     )
#
#     # Should return valid outputs
#     assert loss is not None
#     assert w1 is not None
#     assert b1 is not None
#     assert w2 is not None
#     assert b2 is not None
#
#
# def test_gcn_backward_multidim():
#     """Test backward pass with multi-dimensional features and hidden layers."""
#     # 4-node graph with 2D features, 3D hidden, 2D output
#     adjacency = np.array(
#         [
#             [0, 1, 1, 0],
#             [1, 0, 1, 1],
#             [1, 1, 0, 1],
#             [0, 1, 1, 0],
#         ],
#         dtype=np.float64,
#     )
#     adjacency_T = adjacency.T
#
#     features = np.array(
#         [
#             [1.0, 0.5],
#             [0.0, 1.0],
#             [1.0, 1.0],
#             [0.5, 0.5],
#         ]
#     )
#     weights1 = np.array([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]])  # (2, 3)
#     bias1 = np.zeros(3)
#     weights2 = np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])  # (3, 2)
#     bias2 = np.zeros(2)
#     targets = np.array(
#         [
#             [1.0, 0.0],
#             [0.0, 1.0],
#             [1.0, 1.0],
#             [0.5, 0.5],
#         ]
#     )
#
#     xp = NumpyFramework()
#
#     # Get initial loss (1 iteration)
#     loss_1, _, _, _, _ = run_gcn_backward_benchmark(
#         xp,
#         adjacency,
#         adjacency_T,
#         features,
#         weights1,
#         bias1,
#         weights2,
#         bias2,
#         targets,
#         num_iterations=1,
#         learning_rate=0.01,
#     )
#
#     # Get loss after training
#     loss_100, w1, b1, w2, b2 = run_gcn_backward_benchmark(
#         xp,
#         adjacency,
#         adjacency_T,
#         features,
#         weights1,
#         bias1,
#         weights2,
#         bias2,
#         targets,
#         num_iterations=100,
#         learning_rate=0.01,
#     )
#
#     assert loss_100 < loss_1, f"Loss should decrease: {loss_100} < {loss_1}"
#
#     # Check output shapes
#     w1_np = w1
#     b1_np = b1
#     w2_np = w2
#     b2_np = b2
#
#     assert w1_np.shape == (2, 3)
#     assert b1_np.shape == (3,)
#     assert w2_np.shape == (3, 2)
#     assert b2_np.shape == (2,)
#
#
# def test_gcn_backward_degree_prediction():
#     """Test that GCN learns to predict node degrees from graph structure.
#
#     Training graph: Star with tail + singleton (7 nodes)
#         Node 0 is hub connected to nodes 1, 2, 3, 4
#         Node 4 is bridge also connected to node 5
#         Node 6 is a singleton (no connections)
#         Degrees: [4, 1, 1, 1, 2, 1, 0]
#
#     Test graph: Different structure (6 nodes)
#         Node 0 connected to 1, 2, 3 (degree 3)
#         Node 1 connected to 0, 2 (degree 2)
#         Node 2 connected to 0, 1, 4 (degree 3)
#         Node 3 connected to 0 (degree 1)
#         Node 4 connected to 2 (degree 1)
#         Node 5 is a singleton (degree 0)
#         Degrees: [3, 2, 3, 1, 1, 0]
#
#     Uses constant features (all 1s) to force learning from structure alone.
#     After training on one graph, tests on a different graph to verify the
#     network learned to predict degrees, not just memorize the training data.
#     """
#     # Training graph: Star with tail + singleton (7 nodes)
#     train_adj = np.array(
#         [
#             [0, 1, 1, 1, 1, 0, 0],  # node 0: degree 4
#             [1, 0, 0, 0, 0, 0, 0],  # node 1: degree 1
#             [1, 0, 0, 0, 0, 0, 0],  # node 2: degree 1
#             [1, 0, 0, 0, 0, 0, 0],  # node 3: degree 1
#             [1, 0, 0, 0, 0, 1, 0],  # node 4: degree 2
#             [0, 0, 0, 0, 1, 0, 0],  # node 5: degree 1
#             [0, 0, 0, 0, 0, 0, 0],  # node 6: degree 0 (singleton)
#         ],
#         dtype=np.float64,
#     )
#     train_adj_T = train_adj.T
#     train_features = np.ones((7, 1))
#     train_degrees = train_adj.sum(axis=1, keepdims=True)
#     train_targets = train_degrees / train_degrees.max()
#
#     # Test graph: Different structure (6 nodes)
#     test_adj = np.array(
#         [
#             [0, 1, 1, 1, 0, 0],  # node 0: degree 3
#             [1, 0, 1, 0, 0, 0],  # node 1: degree 2
#             [1, 1, 0, 0, 1, 0],  # node 2: degree 3
#             [1, 0, 0, 0, 0, 0],  # node 3: degree 1
#             [0, 0, 1, 0, 0, 0],  # node 4: degree 1
#             [0, 0, 0, 0, 0, 0],  # node 5: degree 0 (singleton)
#         ],
#         dtype=np.float64,
#     )
#     test_features = np.ones((6, 1))
#
#     # Initialize weights (input_dim=1, hidden_dim=4, output_dim=1)
#     rng = np.random.default_rng(42)
#     weights1 = rng.standard_normal((1, 4)) * 0.5
#     bias1 = np.zeros(4)
#     weights2 = rng.standard_normal((4, 1)) * 0.5
#     bias2 = np.zeros(1)
#
#     xp = NumpyFramework()
#
#     _, w1_b, b1_b, w2_b, b2_b = run_gcn_backward_benchmark(
#         xp,
#         train_adj,
#         train_adj_T,
#         train_features,
#         weights1,
#         bias1,
#         weights2,
#         bias2,
#         train_targets,
#         num_iterations=500,
#         learning_rate=0.01,
#     )
#
#     # Get trained weights
#     w1_trained = w1_b
#     b1_trained = b1_b
#     w2_trained = w2_b
#     b2_trained = b2_b
#
#     # Run forward pass on TEST graph with trained weights
#     Z1 = test_adj @ test_features
#     H1_pre = Z1 @ w1_trained + b1_trained
#     H1 = np.maximum(H1_pre, 0)
#     Z2 = test_adj @ H1
#     predictions = Z2 @ w2_trained + b2_trained
#
#     # Test graph degrees: [3, 2, 3, 1, 1, 0]
#     # Nodes 0 and 2 have degree 3 (highest)
#     # Node 1 has degree 2 (middle)
#     # Nodes 3 and 4 have degree 1 (low)
#     # Node 5 has degree 0 (singleton)
#     high_degree_preds = [predictions[0, 0], predictions[2, 0]]
#     mid_degree_pred = predictions[1, 0]
#     low_degree_preds = [predictions[3, 0], predictions[4, 0]]
#     singleton_pred = predictions[5, 0]
#
#     min_high = min(high_degree_preds)
#     max_low = max(low_degree_preds)
#
#     # High degree nodes should have higher predictions than low degree nodes
#     assert min_high > max_low, (
#         f"High degree predictions ({high_degree_preds}) should be > "
#         f"low degree predictions ({low_degree_preds})"
#     )
#     # Mid degree node should be between high and low
#     assert min_high > mid_degree_pred > max_low, (
#         f"Mid degree prediction ({mid_degree_pred:.3f}) should be between "
#         f"high ({min_high:.3f}) and low ({max_low:.3f})"
#     )
#     # Singleton should have lowest prediction (near zero)
#     assert max_low > singleton_pred, (
#         f"Low degree predictions ({low_degree_preds}) should be > "
#         f"singleton prediction ({singleton_pred:.3f})"
#     )
#     assert abs(singleton_pred) < 0.1, (
#         f"Singleton prediction ({singleton_pred:.3f}) should be near zero"
#     )
# END COPIED TEST FILE: tests/test_gcn_backward.py

class GCNTrainingGenerator(Generator[GCNTrainingDataset]):
    @property
    def name(self) -> str:
        return "gcn_weights"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Weights"

    @property
    def description(self) -> str:
        return "Generates random weights for a 2-layer Graph Convolutional Network."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Scorch: A Library for Sparse Deep Learning",
                authors=[
                    Author("Bobby Yan"),
                    Author("Alexander J. Root"),
                    Author("Trevor Gale"),
                    Author("David Broman"),
                    Author("Fredrik Kjolstad"),
                ],
                journal="Arxiv",
                volume="arXiv:2405.16883",
                year=2024,
                url="https://anonymous.4open.science/r/scorch/README.md",
            ),
            Ref(
                title="Open Graph Benchmark: Datasets for Machine Learning on Graphs",
                authors=[
                    Author("Weihua Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal="Arxiv",
                volume="arXiv:2005.00687",
                year=2020,
                url="https://arxiv.org/abs/2005.00687",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting systems of interacting objects,"
            " such as social networks (Easley et al., 2010), knowledge graphs (Nickel"
            " et al., 2015), molecular graphs (Wu et al., 2018), and biological"
            " networks (Barabasi & Oltvai, 2004), as well as for modeling 3D objects"
            " (Simonovsky & Komodakis, 2017), manifolds (Bronstein et al., 2017), and"
            " source code (Allamanis et al., 2017). Machine learning (ML), especially"
            " deep learning, on graphs is an emerging field (Hamilton et al., 2017b;"
            " Bronstein et al., 2017). W. Hu et al. To represent a graph, an adjaceny"
            " matrix is used, which is inherently sparse."
        )

    @property
    def datasets(self) -> list[GCNTrainingDataset]:
        return [
            GCNTrainingDataset(
                "dg_gcn_social_1",
                "Small social network graph.",
                "karate",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_social_2",
                "Medium social network graph.",
                "dolphins",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_social_3",
                "Larger social network graph.",
                "ca-GrQc",
                feature_dim=32,
                hidden_dim=16,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_road_1",
                "Small road network graph.",
                "chesapeake",
                feature_dim=8,
                hidden_dim=4,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_road_2",
                "Medium road network graph.",
                "road_central",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_molecular_1",
                "Small molecular graph. - Email network.",
                "email",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_molecular_2",
                "Medium molecular graph - PDDB protein structure.",
                "Chebyshev3",
                feature_dim=24,
                hidden_dim=12,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_citation_1",
                "Large citation network graph (AIDS-like size).",
                "ca-HepPh",
                feature_dim=64,
                hidden_dim=32,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_large_2",
                "Very large road network.",
                "road_usa",
                feature_dim=64,
                hidden_dim=32,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_bcsstk01",
                "Original small structural engineering matrix"
                " (for backward compatibility).",
                "bcsstk01",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
        ]

    def generate(self, dataset: GCNTrainingDataset):
        from scipy.io import mmread

        import ssgetpy

        feature_dim = dataset.feature_dim
        hidden_dim = dataset.hidden_dim
        out_dim = dataset.out_dim
        source = dataset.source_name

        matrices = ssgetpy.search(name=source)
        if not matrices:
            raise ValueError(f"No matrix found with name '{source}'")
        matrix = matrices[0]
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        rng = np.random.default_rng(0)
        A = A.tocoo()

        # Create feature/weight arrays using the RNG (deterministic)
        n = A.shape[0]
        features = rng.standard_normal((n, feature_dim))
        weights1 = rng.standard_normal((feature_dim, hidden_dim))
        bias1 = np.zeros((hidden_dim,))
        weights2 = rng.standard_normal((hidden_dim, out_dim))
        bias2 = np.zeros((out_dim,))
        targets = rng.standard_normal((n, out_dim))
        A_T = A.T.tocoo()

        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        A_T_bin = BinsparseFormat.from_coo((A_T.row, A_T.col), A_T.data, A_T.shape)
        features_b = BinsparseFormat.from_numpy(features)
        weights1_b = BinsparseFormat.from_numpy(weights1)
        bias1_b = BinsparseFormat.from_numpy(bias1)
        weights2_b = BinsparseFormat.from_numpy(weights2)
        bias2_b = BinsparseFormat.from_numpy(bias2)
        targets_b = BinsparseFormat.from_numpy(targets)
        return DataInstance(
            inputs=[
                A_bin,
                A_T_bin,
                features_b,
                weights1_b,
                bias1_b,
                weights2_b,
                bias2_b,
                targets_b,
            ],
            meta={
                "num_iterations": dataset.num_iterations,
                "learning_rate": dataset.learning_rate,
            },
        )


class GCNBackwardBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gcn_backward"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Inference"

    @property
    def description(self) -> str:
        return """
Benchmarks 2-layer GCN training loop (forward, backward, weight updates).

Each iteration:
    Forward pass:
        Z1 = A @ X
        H1_pre = Z1 @ W1 + b1
        H1 = ReLU(H1_pre)
        Z2 = A @ H1
        Y = Z2 @ W2 + b2

    Backward pass (MSE loss):
        dY = (2/N) * (Y - T)
        dW2 = Z2.T @ dY
        db2 = sum(dY, axis=0)
        dZ2 = dY @ W2.T
        dH1 = A.T @ dZ2
        dH1_pre = dH1 * (H1_pre > 0)
        dW1 = Z1.T @ dH1_pre
        db1 = sum(dH1_pre, axis=0)

    Weight updates:
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        """

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Scorch: A Library for Sparse Deep Learning",
                authors=[
                    Author("Bobby Yan"),
                    Author("Alexander J. Root"),
                    Author("Trevor Gale"),
                    Author("David Broman"),
                    Author("Fredrik Kjolstad"),
                ],
                journal="Arxiv",
                volume="arXiv:2405.16883",
                year=2024,
                url="https://anonymous.4open.science/r/scorch/README.md",
            ),
            Ref(
                title="Open Graph Benchmark: Datasets for Machine Learning on Graphs",
                authors=[
                    Author("Weihua Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal="Arxiv",
                volume="arXiv:2005.00687",
                year=2020,
                url="https://arxiv.org/abs/2005.00687",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting systems of interacting objects,"
            " such as social networks (Easley et al., 2010), knowledge graphs (Nickel"
            " et al., 2015), molecular graphs (Wu et al., 2018), and biological"
            " networks (Barabasi & Oltvai, 2004), as well as for modeling 3D objects"
            " (Simonovsky & Komodakis, 2017), manifolds (Bronstein et al., 2017), and"
            " source code (Allamanis et al., 2017). Machine learning (ML), especially"
            " deep learning, on graphs is an emerging field (Hamilton et al., 2017b;"
            " Bronstein et al., 2017). W. Hu et al. To represent a graph, an adjaceny"
            " matrix is used, which is inherently sparse."
        )

    @property
    def generators(self):
        return [GCNTrainingGenerator()]

    """
    Args:
    ----
    xp : array_api
        Array API module (e.g. numpy, cupy, torch)
    adjacency_bench : BinsparseFormat
        Sparse adjacency matrix A (N x N)
    adjacency_T_bench : BinsparseFormat
        Sparse transposed adjacency matrix A.T (N x N)
    features_bench : BinsparseFormat
        Node feature matrix X (N x F)
    weights1_bench : BinsparseFormat
        Initial weights for first GCN layer W1 (F x H)
    bias1_bench : BinsparseFormat
        Initial bias for first GCN layer b1 (H,)
    weights2_bench : BinsparseFormat
        Initial weights for second GCN layer W2 (H x O)
    bias2_bench : BinsparseFormat
        Initial bias for second GCN layer b2 (O,)
    targets_bench : BinsparseFormat
        Target values T (N x O) for MSE loss
    num_iterations : int
        Number of training iterations (default 10)
    learning_rate : float
        Learning rate for gradient descent (default 0.01)

    Returns:
    -------
    tuple
        (final_loss, final_W1, final_b1, final_W2, final_b2)
    """

    def benchmark(self, data: list, meta: dict):
        adjacency, adjacency_T, features, weights1, bias1, weights2, bias2, targets = (
            data
        )
        num_iterations = meta["num_iterations"]
        learning_rate = meta["learning_rate"]

        for _ in range(num_iterations):
            # Forward pass
            Z1 = adjacency @ features  # (N × F)
            H1_pre = Z1 @ weights1 + bias1  # (N × H)
            H1 = xp.maximum(H1_pre, 0)  # (N × H) ReLU
            Z2 = adjacency @ H1  # (N × H)
            Y = Z2 @ weights2 + bias2  # (N × O)

            # MSE loss
            N = Y.shape[0]
            diff = Y - targets
            loss = xp.sum(diff * diff) / N

            # Backward pass
            dY = (2 / N) * diff

            # Layer 2 gradients
            dW2 = Z2.T @ dY  # (H × O)
            db2 = xp.sum(dY, axis=0)  # (O,)
            dZ2 = dY @ weights2.T  # (N × H)

            # Backprop through adjacency
            dH1 = adjacency_T @ dZ2  # (N × H)

            # Backprop through ReLU
            dH1_pre = dH1 * (H1_pre > 0)  # (N × H)

            # Layer 1 gradients
            dW1 = Z1.T @ dH1_pre  # (F × H)
            db1 = xp.sum(dH1_pre, axis=0)  # (H,)

            weights1 = weights1 - learning_rate * dW1
            bias1 = bias1 - learning_rate * db1
            weights2 = weights2 - learning_rate * dW2
            bias2 = bias2 - learning_rate * db2

        # Compute final outputs
        loss_out = loss
        weights1_out = weights1
        bias1_out = bias1
        weights2_out = weights2
        bias2_out = bias2

        return (
            loss_out,
            weights1_out,
            bias1_out,
            weights2_out,
            bias2_out,
        )
