
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
    learning_rate=0.01,
):
    """Computes a 2-layer GCN forward pass followed by backward pass (gradient computation).

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
        dH1 = A.T @ dZ2         # KEY: SpMM on transpose
        dH1_pre = dH1 * (H1_pre > 0)  # ReLU gradient
        dW1 = Z1.T @ dH1_pre
        db1 = sum(dH1_pre, axis=0)
        dZ1 = dH1_pre @ W1.T
        dX = A.T @ dZ1          # KEY: SpMM on transpose

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
        Weights for first GCN layer W1 (F x H)
    bias1_bench : BinsparseFormat
        Bias for first GCN layer b1 (H,)
    weights2_bench : BinsparseFormat
        Weights for second GCN layer W2 (H x O)
    bias2_bench : BinsparseFormat
        Bias for second GCN layer b2 (O,)
    targets_bench : BinsparseFormat
        Target values T (N x O) for MSE loss
    learning_rate : float
        Learning rate for gradient descent (default 0.01)

    Returns:
    -------
    tuple
        (loss, dW1, db1, dW2, db2, dX) - MSE loss scalar and gradients
    """
    adjacency = xp.lazy(xp.from_benchmark(adjacency_bench))
    adjacency_T = xp.lazy(xp.from_benchmark(adjacency_T_bench))
    features = xp.lazy(xp.from_benchmark(features_bench))
    weights1 = xp.lazy(xp.from_benchmark(weights1_bench))
    bias1 = xp.lazy(xp.from_benchmark(bias1_bench))
    weights2 = xp.lazy(xp.from_benchmark(weights2_bench))
    bias2 = xp.lazy(xp.from_benchmark(bias2_bench))
    targets = xp.lazy(xp.from_benchmark(targets_bench))

    # Forward pass
    Z1 = adjacency @ features              # (N × F)  features
    H1_pre = Z1 @ weights1 + bias1         # (N × H) layer 1
    H1 = xp.maximum(H1_pre, 0)             # (N × H) ReLU 
    Z2 = adjacency @ H1                    # (N × H) aggregate hidden
    Y = Z2 @ weights2 + bias2              # (N × O) output 


    N = Y.shape[0]
    diff = Y - targets
    loss = xp.sum(diff * diff) / N

    # Backward (MSE)
    dY = (2 / N) * diff  #derivative of x^2 

    # Layer 2 gradients
    dW2 = Z2.T @ dY                         # (H × O)
    db2 = xp.sum(dY, axis=0)                # (O,)
    dZ2 = dY @ weights2.T                   # (N × H)

    dH1 = adjacency_T @ dZ2                 # (N × H)

    # Backprop through ReLU
    dH1_pre = dH1 * (H1_pre > 0)            # (N × H)

    # Layer 1 gradients
    dW1 = Z1.T @ dH1_pre                    # (F × H)
    db1 = xp.sum(dH1_pre, axis=0)           # (H,)
    dZ1 = dH1_pre @ weights1.T              # (N × F)


    dX = adjacency_T @ dZ1                  # (N × F)

    # Compute outputs
    loss_out = xp.compute(loss)
    dW1_out = xp.compute(dW1)
    db1_out = xp.compute(db1)
    dW2_out = xp.compute(dW2)
    db2_out = xp.compute(db2)
    dX_out = xp.compute(dX)


    #TODO: WRAP IN LOOP
    return (
        loss_out,
        xp.to_benchmark(dW1_out),
        xp.to_benchmark(db1_out),
        xp.to_benchmark(dW2_out),
        xp.to_benchmark(db2_out),
        xp.to_benchmark(dX_out),
    )

