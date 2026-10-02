# ruff: noqa: E501
from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

from binsparse import COORMatrix
from binsparse.conversions import to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.adjacency import zero_one_adjacency
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


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
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: Any | None = None,
        expected_count: int | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"MCL input {name}."
        self._suites = list(suites or [])
        self.A = A
        self.expected_count = expected_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


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
        return ["test"]

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
                suites=["test"],
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
                suites=["test"],
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
                suites=["test"],
                A=planted_clique,
                expected_count=7,
            ),
        ]

    def generate(self, dataset: MCLDataset):
        A = np.asarray(dataset.A)
        rows, cols = np.nonzero(A)
        A_bin = COORMatrix(
            A.shape,
            len(rows),
            indices_0=rows,
            indices_1=cols,
            values=A[rows, cols],
        )
        return DataInstance(
            inputs=[A_bin],
            meta={"expansion": 2, "inflation": 2, "loop_value": 1},
            ref_meta={"expected_count": dataset.expected_count},
        )


class MCLSNAPGenerator(Generator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "MCL SNAP Input Generator"

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
        # fmt: off
        return [
            MCLDataset("SNAP/soc-Epinions1", suites=["standard"]),
            MCLDataset("SNAP/soc-LiveJournal1", suites=["standard"]),
            MCLDataset("SNAP/soc-Pokec", suites=["standard"]),
            MCLDataset("SNAP/soc-Slashdot0811", suites=["standard"]),
            MCLDataset("SNAP/soc-Slashdot0902", suites=["standard"]),
            MCLDataset("SNAP/wiki-Vote", suites=["standard"]),
            MCLDataset("SNAP/wiki-RfA", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-bitcoin-otc", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-bitcoin-alpha", suites=["standard"]),
            MCLDataset("SNAP/com-LiveJournal", suites=["standard"]),
            MCLDataset("SNAP/com-Friendster", suites=["standard"]),
            MCLDataset("SNAP/com-Orkut", suites=["standard"]),
            MCLDataset("SNAP/com-Youtube", suites=["standard"]),
            MCLDataset("SNAP/com-DBLP", suites=["standard"]),
            MCLDataset("SNAP/com-Amazon", suites=["standard"]),
            MCLDataset("SNAP/email-Eu-core", suites=["standard"]),
            MCLDataset("SNAP/wiki-topcats", suites=["standard"]),
            MCLDataset("SNAP/email-EuAll", suites=["standard"]),
            MCLDataset("SNAP/email-Enron", suites=["standard"]),
            MCLDataset("SNAP/wiki-Talk", suites=["standard"]),
            MCLDataset("SNAP/cit-HepPh", suites=["standard"]),
            MCLDataset("SNAP/cit-HepTh", suites=["standard"]),
            MCLDataset("SNAP/cit-Patents", suites=["standard"]),
            MCLDataset("SNAP/ca-AstroPh", suites=["standard"]),
            MCLDataset("SNAP/ca-CondMat", suites=["standard"]),
            MCLDataset("SNAP/ca-GrQc", suites=["standard"]),
            MCLDataset("SNAP/ca-HepPh", suites=["standard"]),
            MCLDataset("SNAP/ca-HepTh", suites=["standard"]),
            MCLDataset("SNAP/web-BerkStan", suites=["standard"]),
            MCLDataset("SNAP/web-Google", suites=["standard"]),
            MCLDataset("SNAP/web-NotreDame", suites=["standard"]),
            MCLDataset("SNAP/web-Stanford", suites=["standard"]),
            MCLDataset("SNAP/amazon0302", suites=["standard"]),
            MCLDataset("SNAP/amazon0312", suites=["standard"]),
            MCLDataset("SNAP/amazon0505", suites=["standard"]),
            MCLDataset("SNAP/amazon0601", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella04", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella05", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella06", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella08", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella09", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella24", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella25", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella30", suites=["standard"]),
            MCLDataset("SNAP/p2p-Gnutella31", suites=["standard"]),
            MCLDataset("SNAP/roadNet-CA", suites=["standard"]),
            MCLDataset("SNAP/roadNet-PA", suites=["standard"]),
            MCLDataset("SNAP/roadNet-TX", suites=["standard"]),
            MCLDataset("SNAP/as-735", suites=["standard"]),
            MCLDataset("SNAP/as-Skitter", suites=["standard"]),
            MCLDataset("SNAP/as-caida", suites=["standard"]),
            MCLDataset("SNAP/Oregon-1", suites=["standard"]),
            MCLDataset("SNAP/Oregon-2", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-epinions", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-Slashdot081106", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-Slashdot090216", suites=["standard"]),
            MCLDataset("SNAP/soc-sign-Slashdot090221", suites=["standard"]),
            MCLDataset("SNAP/loc-Gowalla", suites=["standard"]),
            MCLDataset("SNAP/loc-Brightkite", suites=["standard"]),
            MCLDataset("SNAP/sx-stackoverflow", suites=["standard"]),
            MCLDataset("SNAP/sx-mathoverflow", suites=["standard"]),
            MCLDataset("SNAP/sx-superuser", suites=["standard"]),
            MCLDataset("SNAP/sx-askubuntu", suites=["standard"]),
            MCLDataset("SNAP/wiki-talk-temporal", suites=["standard"]),
            MCLDataset("SNAP/email-Eu-core-temporal", suites=["standard"]),
            MCLDataset("SNAP/CollegeMsg", suites=["standard"]),
            MCLDataset("SNAP/twitter7", suites=["standard"]),
            MCLDataset("SNAP/higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: MCLDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name.removeprefix("SNAP/"))
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.float32)], meta={}
        )


class MCLGAPGenerator(Generator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "MCL GAP Input Generator"

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
        # fmt: off
        return [
            MCLDataset("GAP/GAP-road", suites=["standard"]),
            MCLDataset("GAP/GAP-twitter", suites=["standard"]),
            MCLDataset("GAP/GAP-web", suites=["standard"]),
            MCLDataset("GAP/GAP-kron", suites=["standard"]),
            MCLDataset("GAP/GAP-urand", suites=["standard"]),
        ]
        # fmt: on
        # fmt: on

    def generate(self, dataset: MCLDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name.removeprefix("GAP/"))
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.float32)], meta={}
        )


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
            Ref(
                title=(
                    "HipMCL: a high-performance parallel implementation of the "
                    "Markov clustering algorithm for large-scale networks"
                ),
                authors=[
                    Author("Ariful Azad"),
                    Author("Georgios A. Pavlopoulos"),
                    Author("Christos A. Ouzounis"),
                    Author("Nikos C. Kyrpides"),
                    Author("Aydin Buluç"),
                ],
                journal="Nucleic Acids Research",
                volume=46,
                number=6,
                year=2018,
                url="https://doi.org/10.1093/nar/gkx1313",
                doi="10.1093/nar/gkx1313",
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
        return ["group-graphs-iterative"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def generators(self):
        return [MCLTestGenerator(), MCLSNAPGenerator(), MCLGAPGenerator()]

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
        # MCL works with transition probabilities, whatever the input's dtype.
        graph = array_api.astype(data[0], array_api.float64)
        expansion = meta.get("expansion", 2)
        inflation = meta.get("inflation", 2)
        loop_value = meta.get("loop_value", 1)
        iterations = meta.get("iterations", 100)
        # HipMCL's prune limit. A pruned column-stochastic column keeps at most
        # 1 / pruning_threshold entries, which bounds fill-in from expansion.
        pruning_threshold = meta.get("pruning_threshold", 1e-4)
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

            # As in HipMCL, prune the expanded matrix before inflating it.
            if pruning_threshold > 0 and i % pruning_frequency == (
                pruning_frequency - 1
            ):
                expanded_matrix = _prune(array_api, expanded_matrix, pruning_threshold)

            inflated_matrix = expanded_matrix**inflation
            current_matrix = _normalize(array_api, inflated_matrix)

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

        try:
            output = to_scipy(self._output[0]).tocoo()
        except TypeError:
            output = scipy_sparse.coo_array(to_numpy(self._output[0]))
        rows = output.row
        cols = output.col
        values = output.data
        present = values != 0
        rows = rows[present]
        cols = cols[present]
        attractors = rows[rows == cols]
        clusters = {
            tuple(np.sort(cols[rows == attractor]).tolist()) for attractor in attractors
        }
        assert len(clusters) == expected_count
