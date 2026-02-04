"""
Graph Convolutional Network Training (Backward Pass)
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
Data generators have not been implemented yet - using random weights for the matrix
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""

import os

import numpy as np
from scipy.io import mmread

import ssgetpy

from ..binsparse_format import BinsparseFormat


def benchmark_gcn_backward(
    xp,
    adjacency_bench,
    adjacency_T_bench,
    features_bench,
    weights1_bench,
    bias1_bench,
    weights2_bench,
    bias2_bench,
    targets_bench,
    num_iterations=10,
    learning_rate=0.01,
):
    """Benchmarks 2-layer GCN training loop (forward, backward, weight updates).

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
        (final_loss, final_W1, final_b1, final_W2, final_b2) - loss and weights after training
    """
    adjacency = xp.lazy(xp.from_benchmark(adjacency_bench))
    adjacency_T = xp.lazy(xp.from_benchmark(adjacency_T_bench))
    features = xp.lazy(xp.from_benchmark(features_bench))
    targets = xp.lazy(xp.from_benchmark(targets_bench))

    # Initialize weights
    weights1 = xp.lazy(xp.from_benchmark(weights1_bench))
    bias1 = xp.lazy(xp.from_benchmark(bias1_bench))
    weights2 = xp.lazy(xp.from_benchmark(weights2_bench))
    bias2 = xp.lazy(xp.from_benchmark(bias2_bench))

    for _ in range(num_iterations):
        # Forward pass
        Z1 = adjacency @ features              # (N × F)
        H1_pre = Z1 @ weights1 + bias1         # (N × H)
        H1 = xp.maximum(H1_pre, 0)             # (N × H) ReLU
        Z2 = adjacency @ H1                    # (N × H)
        Y = Z2 @ weights2 + bias2              # (N × O)

        # MSE loss
        N = Y.shape[0]
        diff = Y - targets
        loss = xp.sum(diff * diff) / N

        # Backward pass
        dY = (2 / N) * diff

        # Layer 2 gradients
        dW2 = Z2.T @ dY                        # (H × O)
        db2 = xp.sum(dY, axis=0)               # (O,)
        dZ2 = dY @ weights2.T                  # (N × H)

        # Backprop through adjacency
        dH1 = adjacency_T @ dZ2                # (N × H)

        # Backprop through ReLU
        dH1_pre = dH1 * (H1_pre > 0)           # (N × H)

        # Layer 1 gradients
        dW1 = Z1.T @ dH1_pre                   # (F × H)
        db1 = xp.sum(dH1_pre, axis=0)          # (H,)

        weights1 = weights1 - learning_rate * dW1
        bias1 = bias1 - learning_rate * db1
        weights2 = weights2 - learning_rate * dW2
        bias2 = bias2 - learning_rate * db2

    # Compute final outputs
    loss_out = xp.compute(loss)
    weights1_out = xp.compute(weights1)
    bias1_out = xp.compute(bias1)
    weights2_out = xp.compute(weights2)
    bias2_out = xp.compute(bias2)

    return (
        loss_out,
        xp.to_benchmark(weights1_out),
        xp.to_benchmark(bias1_out),
        xp.to_benchmark(weights2_out),
        xp.to_benchmark(bias2_out),
    )
