"""
Name: diBELLA Transitive Reduction Algorithm
Author: Jaehun Baek
Email: jbaek90@gatech.edu
Motivation:
This algorithm implements the iterative transitive reduction
(TO-DO: Add significance of algorithm)
step from the diBELLA 2D paper
G. Guidi et al., “Parallel String Graph Construction and Transitive Reduction
for De Novo Genome Assembly,”
in Proc. IEEE Int. Parallel & Distributed Processing Symposium (IPDPS), vol. 2021,
pp. 517-526, May 2021,
doi: 10.1109/IPDPS49936.2021.00060.
Role of Sparsity:
The overlap graph R is a sparse matrix where R[i, j]
represents the suffix length of an overlap between read i and read j.
The computation is a sparse matrix-matrix multiplication (SpGEMM)
over the (min, +) semiring, N = R^2, to find shortest 2-hop paths.
Statement on the Use of Generative AI:
No generative AI was used to construct the benchmark function.
This statement was written by hand.
"""

import numpy as np


def transitive_reduction(xp, R_bench, x=1, max_iters=10):
    """
    Performs iterative transitive reduction on a sparse overlap graph R

    Parameters:
    -----------
    xp : The Array API module
    R_bench : A binsparse tensor representing the overlap matrix R
    max_iters : The maximum number of reduction iterations

    Returns:
    --------
    A binsparse tensor of the transitively reduced graph S
    """

    R = xp.from_binsparse(R_bench)
    R_nnz_prev_tensor = xp.sum(np.inf != R)
    R_nnz_prev = R_nnz_prev_tensor[()]

    for _i in range(max_iters):
        # R_plus = xp.with_fill_value(R, np.inf)

        # handle dense arrays (Numpy) where 0 must be converted to inf
        # without this, 0s act as valid edges with 0 weight
        # R_plus = xp.where(R == 0, np.inf, R_plus)

        # N <- R ^ 2 in Algo 2 that uses custom MinPlus semiring
        # expressed through einsum- R[i, k] + R[k, j] iterates
        # over all intermediate nodes k,
        # finds all 2-hop paths, and adds their lengths
        N = xp.einsum("N[i, j] min= R[i, k] + R[k, j]", R=R)

        R_for_max = xp.where(np.inf == R, -1.0, R)

        # max(r) not max(n)
        v = xp.max(R_for_max, axis=1)

        # reason to add this scalar value across all nonzero value is
        # to make algorithm 'robust to sequencing error' (page 5)
        v = v + x

        # Build M matrix
        v_expanded = xp.expand_dims(v, axis=1)
        M = v_expanded

        is_transitive = M >= N
        common_sparsity = xp.logical_and(np.inf != R, np.inf != N)
        edges_to_remove = xp.logical_and(common_sparsity, is_transitive)

        R = xp.where(edges_to_remove, np.inf, R)
        R_nnz_new_tensor = xp.sum(np.inf != R)

        # Compute R and its nnz at the same time
        R, R_nnz_new_scalar = (R, R_nnz_new_tensor)
        R_nnz_new = R_nnz_new_scalar[()]

        if R_nnz_new == R_nnz_prev:
            # R = R_computed
            break

        R_nnz_prev = R_nnz_new
        # R = R_computed

    return R


# TO-DO: add data generator functions
# BEGIN COPIED TEST FILE: tests/test_transitive_reduction.py
# import numpy as np
#
# from frameworks.saps_numpy import NumpyFramework
# from saps.benchmarks.transitive_reduction import (
#     transitive_reduction,
# )
#
#
# def create_graph(xp, edges, n):
#     """Helper to build a Binsparse graph from a list of (row, col, val) tuples."""
#     rows = [e[0] for e in edges]
#     cols = [e[1] for e in edges]
#     vals = [e[2] for e in edges]
#
#     dense = np.full((n, n), np.inf)
#     dense[np.arange(n), np.arange(n)] = np.inf
#     dense[rows, cols] = vals
#
#     return xp.to_binsparse(dense)
#
#
# def to_dense(xp, bench_matrix):
#     """Helper to convert output back to dense array for easy assertion."""
#     return bench_matrix
#
#
# def test_case_1():
#     """
#     Test that direct edge is removed when indirect path (2-hop) is shorter (better)
#     than the direct edge.
#     """
#     xp = NumpyFramework()
#     n = 3
#     edges = [(0, 1, 10.0), (1, 2, 10.0), (0, 2, 30.0)]
#     R_input = create_graph(xp, edges, n)
#
#     # Run reduction
#     R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
#     output_dense = to_dense(xp, R_output)
#
#     # Assertions
#     assert output_dense[0, 1] == 10.0, "Edge 0->1 should be kept"
#     assert output_dense[1, 2] == 10.0, "Edge 1->2 should be kept"
#     assert output_dense[0, 2] == np.inf, "Edge 0->2 should be removed (Transitive)"
#
#
# def test_case_2():
#     """
#     Test that direct edge is kept when direct edge is shorter (better)
#     than the 2-hop path (or indirect is longer)
#     """
#     xp = NumpyFramework()
#     n = 3
#     edges = [(0, 1, 10.0), (1, 2, 10.0), (0, 2, 15.0)]
#     R_input = create_graph(xp, edges, n)
#
#     R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
#     output_dense = to_dense(xp, R_output)
#
#     assert output_dense[0, 2] == 15.0, (
#         "Edge 0->2 should be kept (since its the shortest path)"
#     )
#
#
# def test_case_3():
#     """
#     Test that direct edge is kept whenthe 2-hop path is very long (poor overlap)
#     """
#     xp = NumpyFramework()
#     n = 3
#     edges = [(0, 1, 40.0), (1, 2, 40.0), (0, 2, 30.0)]
#     R_input = create_graph(xp, edges, n)
#
#     R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
#     output_dense = to_dense(xp, R_output)
#
#     assert output_dense[0, 2] == 30.0, (
#         "Edge 0->2 should be kept (since 2-hop path is worse)"
#     )
#
#
# def test_case_4():
#     """
#     Test that direct edge is removed when it is exactly equal in weight (length)
#     to an indirect path.
#     """
#     xp = NumpyFramework()
#     n = 3
#     edges = [(0, 1, 10.0), (1, 2, 10.0), (0, 2, 20.0)]
#     R_input = create_graph(xp, edges, n)
#
#     R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
#     output_dense = to_dense(xp, R_output)
#
#     assert output_dense[0, 2] == np.inf, (
#         "Edge 0->2 should be removed (since equal weight, we should remove redundancy)"
#     )
# END COPIED TEST FILE: tests/test_transitive_reduction.py

