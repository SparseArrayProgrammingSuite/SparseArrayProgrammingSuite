import os
from typing import Any

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = sparseappbench.xp

import os

import numpy as np
from scipy.io import mmread

import ssgetpy

import sparseappbench
from saps_framework.binsparse_format import BinsparseFormat

xp = sparseappbench.xp







class GCNTrainingDataset(Dataset):
    def __init__(
        self, source_name: str, has_b_file: bool = False, nnz: int | None = None
    ):
        self.source_name = source_name
        self.has_b_file = has_b_file
        self.nnz = nnz

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"GCN {self.source_name}"

    @property
    def description(self) -> str:
        return f"SuiteSparse matrix {self.source_name}."

    @property
    def tags(self) -> list[str]:
        return ["suitesparse", "sparse"]

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        data["has_b_file"] = self.has_b_file
        return data


class GCNTrainingGenerator(Generator[GCNTrainingDataset]):
    @property
    def name(self) -> str:
        return "gcn_weights"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Weights"

    @property
    def description(self) -> str:
        return (
            "Generates random weights for a 2-layer Graph Convolutional Network."
        )

    @property
    def tags(self) -> list[str]:
        return ["machine learning", "sparse"]

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
                url="https://anonymous.4open.science/r/scorch/README.md"
            ),
            Ref(
                title = "Open Graph Benchmark: Datasets for Machine Learning on Graph",
                authors = [
                    Author("Wenbing Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal = "Arxiv",
                volume = "arXiv:2005.00687",
                year = 2021,
                url = "https://arxiv.org/abs/2005.00687"
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function "
            "itself. Generative AI might have been used to construct tests. This statement "
            "was written by hand. "
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting  systems of interacting objects, "
            "such as social networks (Easley et al., 2010), knowledge graphs (Nickel et al., 2015), "
            "molecular graphs (Wu et al., 2018), and biological networks (Barabasi & Oltvai, 2004), "
            "as well as for modeling 3D objects (Simonovsky & Komodakis, 2017), "
            "manifolds (Bronstein et al., 2017), and source code (Allamanis et al., "
            "2017). Machine learning (ML), especially deep learning, "
            "on graphs is an emerging field (Hamilton et al., 2017b; Bronstein et al., 2017). "
            "W. Hu et al. To represent a graph, an adjaceny matrix is used, which is inherently sparse. "
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
                "Original small structural engineering matrix (for backward compatibility).",
                "bcsstk01",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
        ]

    def generate(self, dataset: GCNTrainingDataset):
        feature_dim = dataset.feature_dim
        hidden_dim = dataset.hidden_dim
        out_dim = dataset.out_dim

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

        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        features_b = BinsparseFormat.from_numpy(features)
        weights1_b = BinsparseFormat.from_numpy(weights1)
        bias1_b = BinsparseFormat.from_numpy(bias1)
        weights2_b = BinsparseFormat.from_numpy(weights2)
        bias2_b = BinsparseFormat.from_numpy(bias2)
        return (A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b)

class GCNBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gcn"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Inference"

    @property
    def description(self) -> str:
        return ("""
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
        """)

    @property
    def tags(self) -> list[str]:
        return ["machine learning", "sparse"]

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
                url="https://anonymous.4open.science/r/scorch/README.md"
            ),
            Ref(
                title = "Open Graph Benchmark: Datasets for Machine Learning on Graph",
                authors = [
                    Author("Wenbing Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal = "Arxiv",
                volume = "arXiv:2005.00687",
                year = 2021,
                url = "https://arxiv.org/abs/2005.00687"
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function "
            "itself. Generative AI might have been used to construct tests. This statement "
            "was written by hand. "
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting  systems of interacting objects, "
            "such as social networks (Easley et al., 2010), knowledge graphs (Nickel et al., 2015), "
            "molecular graphs (Wu et al., 2018), and biological networks (Barabasi & Oltvai, 2004), "
            "as well as for modeling 3D objects (Simonovsky & Komodakis, 2017), "
            "manifolds (Bronstein et al., 2017), and source code (Allamanis et al., "
            "2017). Machine learning (ML), especially deep learning, "
            "on graphs is an emerging field (Hamilton et al., 2017b; Bronstein et al., 2017). "
            "W. Hu et al. To represent a graph, an adjaceny matrix is used, which is inherently sparse."
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
        adjacency, adjacency_T, features, targets, weights1, bias1, weights2, bias2 = data
        num_iterations=meta["num_iterations"]
        learning_rate=meta["learning_rate"]

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
            xp.to_binsparse(weights1_out),
            xp.to_binsparse(bias1_out),
            xp.to_binsparse(weights2_out),
            xp.to_binsparse(bias2_out),
        )


def generate_gcn_backward_data(source, hidden_dim=4):
    """Generate degree prediction training data from a SuiteSparse matrix.

    Creates a GCN training task where the network learns to predict normalized
    node degrees from graph structure using constant features (all 1s).

    Args:
        source: Name of matrix in SuiteSparse collection
        hidden_dim: Hidden layer dimension (default 4)

    Returns:
        Tuple of (A, A_T, features, W1, b1, W2, b2, targets) in BinsparseFormat
    """
    matrices = ssgetpy.search(name=source)
    if not matrices:
        raise ValueError(f"No matrix found with name '{source}'")
    matrix = matrices[0]
    (path, _) = matrix.download(extract=True)
    matrix_path = os.path.join(path, matrix.name + ".mtx")
    if not (matrix_path and os.path.exists(matrix_path)):
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

    A = mmread(matrix_path)
    A = A.tocoo()

    # Make adjacency binary (unweighted graph)
    A_binary = A.copy()
    A_binary.data = np.ones_like(A_binary.data)

    # Compute node degrees and normalize as targets
    n = A.shape[0]
    degrees = np.array(A_binary.sum(axis=1)).flatten()
    max_degree = degrees.max() if degrees.max() > 0 else 1
    targets = (degrees / max_degree).reshape(-1, 1)

    # Constant features (force learning from structure)
    features = np.ones((n, 1))

    # Initialize weights deterministically
    rng = np.random.default_rng(42)
    weights1 = rng.standard_normal((1, hidden_dim)) * 0.5
    bias1 = np.zeros(hidden_dim)
    weights2 = rng.standard_normal((hidden_dim, 1)) * 0.5
    bias2 = np.zeros(1)

    # Create transpose
    A_T = A_binary.T.tocoo()

    # Convert to BinsparseFormat
    A_bin = BinsparseFormat.from_coo(
        (A_binary.row, A_binary.col), A_binary.data.astype(np.float64), A_binary.shape
    )
    A_T_bin = BinsparseFormat.from_coo(
        (A_T.row, A_T.col), A_T.data.astype(np.float64), A_T.shape
    )
    features_bin = BinsparseFormat.from_numpy(features)
    weights1_bin = BinsparseFormat.from_numpy(weights1)
    bias1_bin = BinsparseFormat.from_numpy(bias1)
    weights2_bin = BinsparseFormat.from_numpy(weights2)
    bias2_bin = BinsparseFormat.from_numpy(bias2)
    targets_bin = BinsparseFormat.from_numpy(targets)

    return (
        A_bin,
        A_T_bin,
        features_bin,
        weights1_bin,
        bias1_bin,
        weights2_bin,
        bias2_bin,
        targets_bin,
    )


def dg_gcn_backward_small_1():
    """Small graph - Zachary's karate club (34 nodes)."""
    return generate_gcn_backward_data("karate", hidden_dim=4)


def dg_gcn_backward_small_2():
    """Small graph - Dolphins social network (62 nodes)."""
    return generate_gcn_backward_data("dolphins", hidden_dim=4)


def dg_gcn_backward_small_3():
    """Small graph - Les Miserables character network (77 nodes)."""
    return generate_gcn_backward_data("lesmis", hidden_dim=4)


def dg_gcn_backward_medium_1():
    """Medium graph - Football network (115 nodes)."""
    return generate_gcn_backward_data("football", hidden_dim=8)


def dg_gcn_backward_medium_2():
    """Medium graph - Political books network (105 nodes)."""
    return generate_gcn_backward_data("polbooks", hidden_dim=8)


def dg_gcn_backward_large_1():
    """Larger graph - Email network (~1K nodes)."""
    return generate_gcn_backward_data("email", hidden_dim=16)


def dg_gcn_backward_large_2():
    """Larger graph - ca-GrQc collaboration network (~5K nodes)."""
    return generate_gcn_backward_data("ca-GrQc", hidden_dim=16)

"""
Name: Graph Convolutional Network Training (Backward Pass)
Author: Tarun Devi
Email: tdevi3@gatech.edu
Motivation: "Graphs are widely used for abstracting systems of interacting objects,
such as social networks (Easley et al., 2010), knowledge graphs (Nickel et al., 2015),
molecular graphs (Wu et al., 2018), and biological networks (Barabasi & Oltvai, 2004),
as well as for modeling 3D objects (Simonovsky & Komodakis, 2017),
manifolds (Bronstein et al., 2017), and source code (Allamanis et al.,
2017). Machine learning (ML), especially deep learning,
on graphs is an emerging field (Hamilton et al., 2017b; Bronstein et al., 2017)."
W. Hu et al., "Open Graph Benchmark: Datasets for Machine Learning on Graphs,"
arXiv, vol. 2005.00687, pp. 1–15, Feb. 2021, doi: 10.48550/arXiv.2005.00687.
Role of Sparsity:
To represent a graph, an adjacency matrix is used, which is inherently sparse.
The backward pass requires both the adjacency matrix A and its transpose A.T
for efficient gradient computation through the graph structure.
Implementation Details:
Backpropagation derivation based on:
Y. Hsiao, R. Yue, and A. Dutta, "Derivation of Back-propagation for Graph
Convolutional Networks using Matrix Calculus and its Application to Explainable
Artificial Intelligence," arXiv, vol. 2408.01408, Aug. 2024,
doi: 10.48550/arXiv.2408.01408.
Data Generation:
Data generators create degree prediction tasks where the GCN learns to predict
node degrees from graph structure using constant features.
Statement on the use of Generative AI:
No generative AI was used to construct the benchmark function itself.
Generative AI might have been used to construct tests. This statement
was written by hand.
"""

