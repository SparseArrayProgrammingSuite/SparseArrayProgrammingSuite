# ruff: noqa: E501
from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph
from saps.util.adjacency import graph_csr, zero_one_adjacency
from saps.util.graphblas_process import call_graphblas

# Bound on the multiplications, and so the unpruned nonzeros, of one column
# block of an MCL expansion.
_BLOCK_WORK = 1 << 28


def _inverse_degrees(graph: BinsparseTensor, loop_value: float) -> BinsparseTensor:
    """Each column's 1 / (sum + loop_value), rounded down to a multiple of 2^-26.

    MCL's first step divides each column of graph + loop_value * I by its sum.
    Computing the reciprocals once, here, leaves every implementation the same
    starting matrix.
    """
    sums = np.asarray(to_scipy(graph).sum(axis=0), dtype=np.float64).ravel()
    return from_numpy(np.floor(2.0**26 / (sums + loop_value)) / 2.0**26)


def _clusters(matrix: Any) -> set[bytes]:
    """The clusters of a converged MCL matrix, given as a SciPy COO array.

    Attractors keep weight on their own diagonal, and an attractor's cluster is
    the set of columns with weight in its row. Each cluster is its sorted
    column indices, as bytes.
    """
    present = matrix.data != 0
    rows = matrix.row[present]
    cols = matrix.col[present].astype(np.int64)
    attractor = np.zeros(matrix.shape[0], dtype=bool)
    attractor[rows[rows == cols]] = True
    keep = attractor[rows]
    rows, cols = rows[keep], cols[keep]
    if not rows.size:
        return set()
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    starts = np.flatnonzero(np.diff(rows)) + 1
    return {cluster.tobytes() for cluster in np.split(cols, starts)}


def _mcl_reference(
    graph: Any, inverse_degrees: np.ndarray, meta: dict[str, Any]
) -> Any:
    """Run MCL on a SciPy adjacency matrix to convergence, as a COO array.

    It rounds where the benchmark does, so its sums are exact and it computes
    the same bits as any implementation of the benchmark.

    SuiteSparse:GraphBLAS runs each step on every core. It is imported here so
    that the benchmarks themselves do not depend on it. Call it through
    `call_graphblas`.
    """
    import graphblas as gb

    eps = np.finfo(float).eps

    def rounded(matrix):
        scaled = matrix.apply(gb.binary.times, right=2.0**26).new()
        scaled = scaled.apply(gb.unary.floor).new()
        return scaled.apply(gb.binary.truediv, right=2.0**26).new()

    def normalized(matrix):
        sums = matrix.reduce_columnwise(gb.monoid.plus).new()
        sums = sums.apply(gb.binary.max, right=eps).new()
        # Divide each entry by its column's sum, as the benchmark does, rather
        # than multiply by a rounded reciprocal.
        divisors = matrix.mxm(sums.diag(), gb.semiring.any_second).new()
        return rounded(matrix.ewise_mult(divisors, gb.binary.truediv).new())

    n = graph.shape[0]
    current = (
        gb.io.from_scipy_sparse(
            graph.astype(float) + meta.get("loop_value", 1) * scipy_sparse.eye(n)
        )
        .mxm(gb.Vector.from_dense(inverse_degrees).diag(), gb.semiring.plus_times)
        .new()
    )
    threshold = meta.get("pruning_threshold", 1e-4)
    frequency = meta.get("pruning_frequency", 1)
    for iteration in range(meta.get("iterations", 100)):
        previous = current
        converged = (iteration + 1) % meta.get("convergence_check_frequency", 1) == 0
        # As in HipMCL, expand a block of columns at a time and prune it right
        # away, so the unpruned product never exists in full. Column j of the
        # product depends only on column j of its right factor, and pruning,
        # inflation, normalization, and the convergence test act on columns
        # alone, so the blocks give the same result as the whole matrix.
        counts = previous.reduce_columnwise(gb.agg.count).new()
        column_work = counts.vxm(previous, gb.semiring.plus_first).new(dtype=int)
        work = np.concatenate([[0], np.cumsum(column_work.to_dense(fill_value=0))])
        blocks = []
        start = 0
        while start < n:
            stop = max(
                start + 1,
                np.searchsorted(work, work[start] + _BLOCK_WORK, side="right") - 1,
            )
            old = previous[:, start:stop].new()
            expanded = old
            for _ in range(meta.get("expansion", 2) - 1):
                expanded = rounded(previous.mxm(expanded, gb.semiring.plus_times).new())
            if threshold > 0 and (iteration + 1) % frequency == 0:
                maxima = expanded.reduce_columnwise(gb.monoid.max).new()
                column_maxima = expanded.mxm(
                    maxima.diag(), gb.semiring.any_second
                ).new()
                keep = expanded.apply(gb.binary.ge, right=threshold).new()
                keep(gb.binary.lor) << expanded.ewise_mult(column_maxima, gb.binary.eq)
                expanded = expanded.dup(mask=keep.V)
            inflated = expanded
            for _ in range(meta.get("inflation", 2) - 1):
                inflated = inflated.ewise_mult(expanded, gb.binary.times).new()
            block = normalized(inflated)
            if converged:
                # |block - old| <= 2^-26 + 2^-17 * |old| entrywise, on the
                # union of both patterns. Every step here is exact.
                difference = block.ewise_add(old, gb.binary.minus).new()
                difference = difference.apply(gb.unary.abs).new()
                scaled = old.apply(gb.unary.abs).new()
                scaled = scaled.apply(gb.binary.times, right=2.0**-17).new()
                excess = difference.ewise_add(scaled, gb.binary.minus).new()
                converged = (
                    excess.nvals == 0
                    or excess.reduce_scalar(gb.monoid.max).new().value <= 2.0**-26
                )
            blocks.append(block)
            start = stop
        current = gb.ss.concat([blocks])
        if converged:
            break
    return gb.io.to_scipy_sparse(current, format="coo")


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
        return "mcl_test"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL) Test"

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
                pretty_name="Two Star Components",
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
                pretty_name="Three Block Pairs",
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
                pretty_name="Planted Clique",
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
        meta = {"expansion": 2, "inflation": 2, "loop_value": 1}
        problem = DataInstance(
            inputs=[A_bin, _inverse_degrees(A_bin, meta["loop_value"])],
            meta=meta,
            ref_meta={"expected_count": dataset.expected_count},
        )
        meta = problem.meta
        expected = from_scipy(
            call_graphblas(
                _mcl_reference,
                graph_csr(problem.inputs[0]),
                to_numpy(problem.inputs[1]),
                meta,
            )
        )
        problem.ref_outputs = [expected]
        problem.ref_meta = {
            **(problem.ref_meta or {}),
            "reference_backend": "graphblas",
            "reference_algorithm": "mcl",
        }
        return problem


class MCLSNAPGenerator(Generator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_snap"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL) SNAP"

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
            MCLDataset("soc-Epinions1", suites=["standard"]),
            MCLDataset("soc-LiveJournal1", suites=["standard"]),
            MCLDataset("soc-Pokec", suites=["standard"]),
            MCLDataset("soc-Slashdot0811", suites=["standard"]),
            MCLDataset("soc-Slashdot0902", suites=["standard"]),
            MCLDataset("wiki-Vote", suites=["standard"]),
            MCLDataset("wiki-RfA", suites=["standard"]),
            MCLDataset("soc-sign-bitcoin-otc", suites=["standard"]),
            MCLDataset("soc-sign-bitcoin-alpha", suites=["standard"]),
            MCLDataset("com-LiveJournal", suites=["standard"]),
            MCLDataset("com-Friendster", suites=["standard"]),
            MCLDataset("com-Orkut", suites=["standard"]),
            MCLDataset("com-Youtube", suites=["standard"]),
            MCLDataset("com-DBLP", suites=["standard"]),
            MCLDataset("com-Amazon", suites=["standard"]),
            MCLDataset("email-Eu-core", suites=["standard", "trace", "train"]),
            MCLDataset("wiki-topcats", suites=["standard"]),
            MCLDataset("email-EuAll", suites=["standard"]),
            MCLDataset("email-Enron", suites=["standard"]),
            MCLDataset("wiki-Talk", suites=["standard"]),
            MCLDataset("cit-HepPh", suites=["standard"]),
            MCLDataset("cit-HepTh", suites=["standard"]),
            MCLDataset("cit-Patents", suites=["standard"]),
            MCLDataset("ca-AstroPh", suites=["standard"]),
            MCLDataset("ca-CondMat", suites=["standard"]),
            MCLDataset("ca-GrQc", suites=["standard"]),
            MCLDataset("ca-HepPh", suites=["standard"]),
            MCLDataset("ca-HepTh", suites=["standard"]),
            MCLDataset("web-BerkStan", suites=["standard"]),
            MCLDataset("web-Google", suites=["standard"]),
            MCLDataset("web-NotreDame", suites=["standard"]),
            MCLDataset("web-Stanford", suites=["standard"]),
            MCLDataset("amazon0302", suites=["standard"]),
            MCLDataset("amazon0312", suites=["standard"]),
            MCLDataset("amazon0505", suites=["standard"]),
            MCLDataset("amazon0601", suites=["standard"]),
            MCLDataset("p2p-Gnutella04", suites=["standard"]),
            MCLDataset("p2p-Gnutella05", suites=["standard"]),
            MCLDataset("p2p-Gnutella06", suites=["standard"]),
            MCLDataset("p2p-Gnutella08", suites=["standard"]),
            MCLDataset("p2p-Gnutella09", suites=["standard"]),
            MCLDataset("p2p-Gnutella24", suites=["standard"]),
            MCLDataset("p2p-Gnutella25", suites=["standard"]),
            MCLDataset("p2p-Gnutella30", suites=["standard"]),
            MCLDataset("p2p-Gnutella31", suites=["standard"]),
            MCLDataset("roadNet-CA", suites=["standard"]),
            MCLDataset("roadNet-PA", suites=["standard"]),
            MCLDataset("roadNet-TX", suites=["standard"]),
            MCLDataset("as-735", suites=["standard"]),
            MCLDataset("as-Skitter", suites=["standard"]),
            MCLDataset("as-caida", suites=["standard"]),
            MCLDataset("Oregon-1", suites=["standard"]),
            MCLDataset("Oregon-2", suites=["standard"]),
            MCLDataset("soc-sign-epinions", suites=["standard"]),
            MCLDataset("soc-sign-Slashdot081106", suites=["standard"]),
            MCLDataset("soc-sign-Slashdot090216", suites=["standard"]),
            MCLDataset("soc-sign-Slashdot090221", suites=["standard"]),
            MCLDataset("loc-Gowalla", suites=["standard"]),
            MCLDataset("loc-Brightkite", suites=["standard"]),
            MCLDataset("sx-stackoverflow", suites=["standard"]),
            MCLDataset("sx-mathoverflow", suites=["standard"]),
            MCLDataset("sx-superuser", suites=["standard"]),
            MCLDataset("sx-askubuntu", suites=["standard"]),
            MCLDataset("wiki-talk-temporal", suites=["standard"]),
            MCLDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            MCLDataset("CollegeMsg", suites=["standard", "trace"]),
            MCLDataset("twitter7", suites=["standard"]),
            MCLDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: MCLDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = MCLSNAPReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(self, dataset: MCLDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        graph = zero_one_adjacency(raw.inputs[0])
        return DataInstance(inputs=[graph, _inverse_degrees(graph, 1)], meta={})


class MCLSNAPReferenceGenerator(ReferenceGenerator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_snap_reference"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL) SNAP Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Markov Clustering (MCL) SNAP, cached apart from"
            " its inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[MCLDataset]]:
        return [ReferenceDataset(dataset) for dataset in MCLSNAPGenerator().datasets]

    def generate(self, dataset: ReferenceDataset[MCLDataset]) -> DataInstance:
        problem = MCLSNAPGenerator().generate_inputs(dataset.dataset)
        meta = problem.meta
        expected = from_scipy(
            call_graphblas(
                _mcl_reference,
                graph_csr(problem.inputs[0]),
                to_numpy(problem.inputs[1]),
                meta,
            )
        )
        problem.ref_outputs = [expected]
        problem.ref_meta = {
            **(problem.ref_meta or {}),
            "reference_backend": "graphblas",
            "reference_algorithm": "mcl",
        }
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=problem.ref_outputs,
            ref_meta=problem.ref_meta,
        )


class MCLSNAPReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return MCLSNAPReferenceGenerator()


class MCLGAPGenerator(Generator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_gap"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL) GAP"

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
            MCLDataset("GAP-road", suites=["standard"]),
            MCLDataset("GAP-twitter", suites=["standard"]),
            MCLDataset("GAP-web", suites=["standard"]),
            MCLDataset("GAP-kron", suites=["standard"]),
            MCLDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on
        # fmt: on

    def generate(self, dataset: MCLDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = MCLGAPReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(self, dataset: MCLDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        graph = zero_one_adjacency(raw.inputs[0])
        return DataInstance(inputs=[graph, _inverse_degrees(graph, 1)], meta={})


class MCLGAPReferenceGenerator(ReferenceGenerator[MCLDataset]):
    @property
    def name(self) -> str:
        return "mcl_gap_reference"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL) GAP Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Markov Clustering (MCL) GAP, cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[MCLDataset]]:
        return [ReferenceDataset(dataset) for dataset in MCLGAPGenerator().datasets]

    def generate(self, dataset: ReferenceDataset[MCLDataset]) -> DataInstance:
        problem = MCLGAPGenerator().generate_inputs(dataset.dataset)
        meta = problem.meta
        expected = from_scipy(
            call_graphblas(
                _mcl_reference,
                graph_csr(problem.inputs[0]),
                to_numpy(problem.inputs[1]),
                meta,
            )
        )
        problem.ref_outputs = [expected]
        problem.ref_meta = {
            **(problem.ref_meta or {}),
            "reference_backend": "graphblas",
            "reference_algorithm": "mcl",
        }
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=problem.ref_outputs,
            ref_meta=problem.ref_meta,
        )


# class MCLGAPReferenceShellBenchmark(ShellBenchmark):
#     @property
#     def generator(self) -> Generator:
#         return MCLGAPReferenceGenerator()


class MCLBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mcl"

    @property
    def pretty_name(self) -> str:
        return "Markov Clustering (MCL)"

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
        return ["standard-graphs-iterative"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def generators(self):
        return [MCLTestGenerator(), MCLSNAPGenerator()]  # MCLGAPGenerator()

    def benchmark(self, xp, meta: dict[str, Any], graph, inverse_degrees):
        """
                benchmark(data, meta)

                Computes Markov Clustering on a given sparse adjacency matrix

        Args:
        ----
        array_api: The array API module to utilize
        graph_binsparse: The sparse adjacency matrix of the graph in binsparse format.
        inverse_degrees: 1 / (column sum + loop_value) for each column of the
                         graph, rounded down to a multiple of 2^-26.
        expansion: The cluster expansion factor.
        inflation: The cluster inflation factor, a positive integer.
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
        graph = array_api.astype(graph, array_api.float64)
        expansion = meta.get("expansion", 2)
        inflation = meta.get("inflation", 2)
        loop_value = meta.get("loop_value", 1)
        iterations = meta.get("iterations", 100)
        # HipMCL's prune limit. A pruned column-stochastic column keeps at most
        # 1 / pruning_threshold entries, which bounds fill-in from expansion.
        pruning_threshold = meta.get("pruning_threshold", 1e-4)
        pruning_frequency = meta.get("pruning_frequency", 1)
        convergence_check_frequency = meta.get("convergence_check_frequency", 1)

        # Matrices that get multiplied hold multiples of 2^-26 in [0, 1], so
        # every product is a multiple of 2^-52 and every partial sum stays
        # below 2. Float64 then adds them exactly, in any order, and every
        # implementation computes the same bits. Rounding down keeps zeros zero.
        loops_matrix = array_api.eye(graph.shape[0], dtype=graph.dtype)
        current_matrix = (graph + loop_value * loops_matrix) * inverse_degrees

        for i in range(iterations):
            previous_matrix = current_matrix

            expanded_matrix = current_matrix
            for _ in range(expansion - 1):
                expanded_matrix = array_api.matmul(current_matrix, expanded_matrix)
                expanded_matrix = array_api.floor(expanded_matrix * 2.0**26) / 2.0**26

            # As in HipMCL, prune the expanded matrix before inflating it.
            if pruning_threshold > 0 and i % pruning_frequency == (
                pruning_frequency - 1
            ):
                max_vals = array_api.max(expanded_matrix, axis=0)
                mask = (expanded_matrix >= pruning_threshold) | (
                    (expanded_matrix == max_vals) & (expanded_matrix > 0)
                )
                expanded_matrix = expanded_matrix * mask

            # Multiplying rather than calling pow keeps squares of grid values
            # exact, so the column sums below are exact at inflation 2.
            inflated_matrix = expanded_matrix
            for _ in range(inflation - 1):
                inflated_matrix = inflated_matrix * expanded_matrix
            col_sums = array_api.sum(inflated_matrix, axis=0)
            col_sums = array_api.maximum(
                col_sums, array_api.finfo(inflated_matrix.dtype).eps
            )
            current_matrix = (
                array_api.floor(inflated_matrix / col_sums * 2.0**26) / 2.0**26
            )

            # Both tolerances are powers of two, so this test is exact too.
            if i % convergence_check_frequency == (
                convergence_check_frequency - 1
            ) and array_api.all(
                array_api.abs(current_matrix - previous_matrix)
                <= 2.0**-26 + 2.0**-17 * array_api.abs(previous_matrix)
            ):
                break

        return current_matrix

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        try:
            output = to_scipy(self._output[0]).tocoo()
        except TypeError:
            output = scipy_sparse.coo_array(to_numpy(self._output[0]))
        output.sum_duplicates()
        assert output.data.min(initial=0) >= 0, "MCL output has negative entries"
        # Rounding down to multiples of 2^-26 leaves columns a little under 1.
        np.testing.assert_allclose(output.sum(axis=0), 1, atol=1e-3)
        clusters = _clusters(output)
        if self._ref_outputs is not None:
            expected = _clusters(to_scipy(self._ref_outputs[0]).tocoo())
            assert clusters == expected, "MCL clusters differ from the reference"
        if self._ref_meta and "expected_count" in self._ref_meta:
            assert len(clusters) == self._ref_meta["expected_count"]
