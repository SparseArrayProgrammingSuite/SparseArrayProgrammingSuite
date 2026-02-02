"""
Name: Markov Clustering Algorithm
Co-Authors: Prateek Hanumappanahalli, Joel Mathew Cherian
Email: phanumap3@gatech.edu
Motivation: "The Markov Clustering (MCL) algorithm relies heavily on repeated matrix
operations, particularly matrix multiplication during the expansion step. Since the
efficient execution of matrix-based kernels has been extensively studied in linear
algebra, MCL serves as an effective benchmark for evaluating the performance of
iterative numerical methods."
Gene H. Golub and Charles F. Van Loan, Matrix Computations (4th ed., SIAM, 2013),
doi: 10.1137/1.9780898719918.
Role of sparsity: The input is a sparse adjacency matrix.
The algorithm uses sparse matrix multiplication and element-wise operations repeatedly,
so it depends heavily on efficient sparse matrix functions.
Implementation: Handwritten code based on the implementation from
https://github.com/GuyAllard/markov_clustering.
Data Generation: Data collected from SuiteSparse Matrix Collection consisting of
sparse adjacency matrices used to evaluate graph clustering performance.
Statement on Generative AI: No generative AI was used to write the benchmark function
itself. Generative AI was used to debug code. This statement was written by hand.
"""

import os

from scipy.io import mmread

import ssgetpy

from ..binsparse_format import BinsparseFormat


def _normalize(array_api, matrix):
    col_sums = array_api.sum(matrix, axis=0)
    col_sums = array_api.maximum(col_sums, array_api.finfo(matrix.dtype).eps)
    return matrix / col_sums


def _sparse_allclose(array_api, matrix_a, matrix_b, rtol=1e-5, atol=1e-8):
    matrix_a = array_api.lazy(matrix_a)
    matrix_b = array_api.lazy(matrix_b)
    c = array_api.all(
        array_api.abs(matrix_a - matrix_b) <= atol + rtol * array_api.abs(matrix_b)
    )
    return array_api.compute(c)


def _prune(array_api, matrix, threshold):
    max_vals = array_api.max(matrix, axis=0)

    mask = (matrix >= threshold) | ((matrix == max_vals) & (matrix > 0))

    return matrix * mask


"""

benchmark_mcl(array_api, graph_binsparse, expansion=2, inflation=2, loop_value=1,
              iterations=100, pruning_threshold=1e-5, pruning_frequency=1,
              convergence_check_frequency=1)

Computes Markov Clustering on a given sparse adjacency matrix

Args:
----
array_api: The array API module to utilize
graph_binsparse: The sparse adjacency matrix of the graph in binsparse format.
expansion: The cluster expansion factor.
inflation: The cluster inflation factor.
loop_value: The value to add to the diagonal for self loops.
iterations: The maximum number of iterations.
pruning_threshold: Threshold below which matrix elements will be set to 0.
pruning_frequency: Perform pruning every 'pruning_frequency' iterations.
convergence_check_frequency: Perform convergence check every
                             'convergence_check_frequency' iterations.

Returns
-------
The final converged matrix in binsparse format

"""


def benchmark_mcl(
    array_api,
    graph_binsparse,
    expansion=2,
    inflation=2,
    loop_value=1,
    iterations=100,
    pruning_threshold=1e-5,
    pruning_frequency=1,
    convergence_check_frequency=1,
):
    # begin region 1
    graph_lazy = array_api.lazy(array_api.from_benchmark(graph_binsparse))

    loops_matrix = array_api.eye(graph_lazy.shape[0], dtype=graph_lazy.dtype)
    current_matrix = graph_lazy + loop_value * loops_matrix
    current_matrix = _normalize(array_api, current_matrix)
    # end region 1
    current_matrix = array_api.compute(current_matrix)

    for i in range(iterations):
        previous_matrix = current_matrix

        # begin region 2
        current_matrix = array_api.lazy(current_matrix)

        expanded_matrix = current_matrix
        for _ in range(expansion - 1):
            expanded_matrix = array_api.matmul(expanded_matrix, current_matrix)

        inflated_matrix = expanded_matrix**inflation
        current_matrix = _normalize(array_api, inflated_matrix)

        if pruning_threshold > 0 and i % pruning_frequency == (pruning_frequency - 1):
            current_matrix = _prune(array_api, current_matrix, pruning_threshold)

        # end region 2
        current_matrix = array_api.compute(current_matrix)

        if i % convergence_check_frequency == (
            convergence_check_frequency - 1
        ) and _sparse_allclose(array_api, current_matrix, previous_matrix):
            break

    return array_api.to_benchmark(current_matrix)


def generate_mcl_data(source):
    matrices = ssgetpy.search(name=source)
    if not matrices:
        raise ValueError(f"No matrix found with name '{source}'")
    matrix = matrices[0]
    path, archive = matrix.download(extract=True)
    matrix_path = os.path.join(path, matrix.name + ".mtx")
    if matrix_path and os.path.exists(matrix_path):
        A = mmread(matrix_path)
    else:
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    A = A.tocoo()
    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    return (A_bin,)


def dg_mcl_sparse_1():
    return generate_mcl_data("Trefethen_200")


def dg_mcl_sparse_2():
    return generate_mcl_data("mesh3em5")


def dg_mcl_sparse_3():
    return generate_mcl_data("fv1")


def dg_mcl_sparse_4():
    return generate_mcl_data("bcsstk05")


def dg_mcl_sparse_5():
    return generate_mcl_data("nos1")


def dg_mcl_sparse_6():
    return generate_mcl_data("nos2")


def dg_mcl_sparse_7():
    return generate_mcl_data("nos3")


def dg_mcl_sparse_8():
    return generate_mcl_data("dwt_59")
