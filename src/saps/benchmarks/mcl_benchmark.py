from typing import Any

import numpy as np

from binsparse import BinsparseTensor

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import SuiteSparseDataset, fetch_suitesparse_matrix


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


class MCLDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        suites: list[str] | None = None,
        A: Any | None = None,
        expected_count: int | None = None,
    ):
        super().__init__(
            source_name,
            pretty_name=f"MCL {source_name}",
            description=f"SuiteSparse adjacency matrix {source_name}.",
            suites=suites,
        )
        self.A = A
        self.expected_count = expected_count


class MCLTestGenerator(Generator[MCLDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "mcl_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "MCL Test Data Generator"

    @property
    def description(self) -> str:
        return "Small MCL examples with expected cluster counts."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705</concept_id>
<concept_desc>Mathematics of computing~Mathematical software</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003705.10011686</concept_id>
<concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10010917</concept_id>
<concept_desc>Mathematics of computing~Graph algorithms</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10003640</concept_id>
<concept_desc>Mathematics of computing~Paths and connectivity problems</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

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
        planted_clique = np.zeros((10, 10), dtype=np.float32)
        planted_clique[:4, :4] = 1.0
        np.fill_diagonal(planted_clique, 0)
        return [
            MCLDataset(
                "two_star_components",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                expected_count=2,
            ),
            MCLDataset(
                "three_block_pairs",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                    dtype=np.float32,
                ),
                expected_count=3,
            ),
            MCLDataset(
                "planted_clique",
                suites=["test", "trace"],
                A=planted_clique,
                expected_count=7,
            ),
        ]

    def generate(self, dataset: MCLDataset):
        A = np.asarray(dataset.A)
        rows, cols = np.nonzero(A)
        A_bin = BinsparseTensor.from_coo((rows, cols), A[rows, cols], A.shape)
        return DataInstance(
            inputs=[A_bin],
            meta={"expansion": 2, "inflation": 2, "loop_value": 1},
            ref_meta={"expected_count": dataset.expected_count},
        )


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
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
    def cacheable(self) -> bool:
        return False

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
            MCLDataset("gap-road"),
            MCLDataset("gap-twitter"),
            MCLDataset("gap-web"),
            MCLDataset("gap-kron"),
            MCLDataset("gap-urand"),
        ]

    def generate(self, dataset: MCLDataset):
        raw = fetch_suitesparse_matrix(dataset.source_name)
        return DataInstance(inputs=[raw.inputs[0]], meta={})


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
                title="Graph Algorithms in the Language of Linear Algebra",
                authors=[],
                publisher="Society for Industrial and Applied Mathematics",
                year=2011,
                url="https://doi.org/10.1137/1.9780898719918",
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
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def generators(self):
        return [MCLTestGenerator(), MCLGenerator()]

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
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

    def check(self, param):
        super().check(param)
        if not self._ref_meta or "expected_count" not in self._ref_meta:
            return
        expected_count = self._ref_meta["expected_count"]

        output = BinsparseTensor.to_coo(self._output[0])
        rows = output.data["indices_0"]
        cols = output.data["indices_1"]
        values = output.data["values"]
        present = values != 0
        rows = rows[present]
        cols = cols[present]
        attractors = rows[rows == cols]
        clusters = {
            tuple(np.sort(cols[rows == attractor]).tolist()) for attractor in attractors
        }
        assert len(clusters) == expected_count
