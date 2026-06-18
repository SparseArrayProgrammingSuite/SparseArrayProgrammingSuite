import os
from typing import Any

from scipy.io import mmread

import ssgetpy

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


def _normalize(array_api, matrix):
    col_sums = array_api.sum(matrix, axis=0)
    col_sums = array_api.maximum(col_sums, array_api.finfo(matrix.dtype).eps)
    return matrix / col_sums


def _sparse_allclose(array_api, matrix_a, matrix_b, rtol=1e-5, atol=1e-8):
    return array_api.all(
        array_api.abs(matrix_a - matrix_b) <= atol + rtol * array_api.abs(matrix_b)
    )


def _prune(array_api, matrix, threshold):
    max_vals = array_api.max(matrix, axis=0)

    mask = (matrix >= threshold) | ((matrix == max_vals) & (matrix > 0))

    return matrix * mask


class MCLDataset(Dataset):
    def __init__(self, source_name: str):
        self._tags = []
        self.source_name = source_name

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"MCL {self.source_name}"

    @property
    def description(self) -> str:
        return f"SuiteSparse adjacency matrix {self.source_name}."

    @property
    def tags(self) -> list[str]:
        return self._tags


class MCLGenerator(Generator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_inputs"

    @property
    def pretty_name(self) -> str:
        return "MCL SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of "
            "sparse adjacency matrices used to evaluate graph clustering performance."
        )

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def authors(self) -> list[Contributor]:
        return MCLBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return MCLBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return MCLBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return MCLBenchmark().motivation

    @property
    def datasets(self) -> list[MCLDataset]:
        return [
            MCLDataset("Trefethen_200"),
            MCLDataset("mesh3em5"),
            MCLDataset("fv1"),
            MCLDataset("bcsstk05"),
            MCLDataset("nos1"),
            MCLDataset("nos2"),
            MCLDataset("nos3"),
            MCLDataset("dwt_59"),
        ]

    def generate(self, dataset: MCLDataset):
        matrices = ssgetpy.search(name=dataset.source_name)
        if not matrices:
            raise ValueError(f"No matrix found with name '{dataset.source_name}'")
        matrix = matrices[0]
        path, archive = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        A = A.tocoo()
        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        return [A_bin], {}


class MCLBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mcl"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering Algorithm"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Prateek Hanumappanahalli", "phanumap3@gatech.edu"),
            Contributor("Joel Mathew Cherian", "jcherian32@gatech.edu"),
        ]

    @property
    def description(self) -> str:
        return (
            "Computes Markov Clustering on a given sparse adjacency matrix. "
            "Handwritten code based on the implementation from GuyAllard on github"
        )

    @property
    def motivation(self) -> str:
        return (
            '"The Markov Clustering (MCL) algorithm relies heavily on repeated '
            "matrix operations, particularly matrix multiplication during the "
            "expansion step. Since the efficient execution of matrix-based "
            "kernels has been extensively studied in linear algebra, MCL "
            "serves as an effective benchmark for evaluating the performance "
            'of iterative numerical methods." The input is a sparse adjacency '
            "matrix. The algorithm uses sparse matrix multiplication and element-wise"
            " operations repeatedly, so it depends heavily on efficient sparse matrix"
            " functions."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Matrix Computations",
                authors=[Author("Gene H. Golub"), Author("Charles F. Van Loan")],
                publisher="SIAM",
                year=2013,
                doi="10.1137/1.9780898719918",
            ),
            Ref(
                title="markov_clustering",
                authors=[Author("Guy Allard")],
                url="https://github.com/GuyAllard/markov_clustering",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def generators(self):
        return [MCLGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]):
        """
                benchmark(data, meta)

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
                The final converged matrix.

        """
        array_api = xp
        graph = data[0]
        expansion = meta.get("expansion", 2)
        inflation = meta.get("inflation", 2)
        loop_value = meta.get("loop_value", 1)
        iterations = meta.get("iterations", 100)
        pruning_threshold = meta.get("pruning_threshold", 1e-5)
        pruning_frequency = meta.get("pruning_frequency", 1)
        convergence_check_frequency = meta.get("convergence_check_frequency", 1)

        loops_matrix = array_api.eye(graph.shape[0], dtype=graph.dtype)
        current_matrix = graph + loop_value * loops_matrix
        current_matrix = _normalize(array_api, current_matrix)

        for i in range(iterations):
            previous_matrix = current_matrix

            expanded_matrix = current_matrix
            for _ in range(expansion - 1):
                expanded_matrix = array_api.matmul(expanded_matrix, current_matrix)

            inflated_matrix = expanded_matrix**inflation
            current_matrix = _normalize(array_api, inflated_matrix)

            if pruning_threshold > 0 and i % pruning_frequency == (
                pruning_frequency - 1
            ):
                current_matrix = _prune(array_api, current_matrix, pruning_threshold)


            if i % convergence_check_frequency == (
                convergence_check_frequency - 1
            ) and _sparse_allclose(array_api, current_matrix, previous_matrix):
                break

        return [current_matrix]
