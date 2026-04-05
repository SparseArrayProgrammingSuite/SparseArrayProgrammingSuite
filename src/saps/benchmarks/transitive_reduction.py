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

    R = xp.from_benchmark(R_bench)
    R = xp.lazy(R)
    R_nnz_prev_tensor = xp.sum(np.inf != R)
    R, R_nnz_prev_tensor = xp.compute((R, R_nnz_prev_tensor))
    R_nnz_prev = R_nnz_prev_tensor[()]

    for _i in range(max_iters):
        # R_plus = xp.with_fill_value(R, np.inf)
        R = xp.lazy(R)

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
        R, R_nnz_new_scalar = xp.compute((R, R_nnz_new_tensor))
        R_nnz_new = R_nnz_new_scalar[()]

        if R_nnz_new == R_nnz_prev:
            # R = R_computed
            break

        R_nnz_prev = R_nnz_new
        # R = xp.lazy(R_computed)

    return xp.to_benchmark(R)


# TO-DO: add data generator functions
