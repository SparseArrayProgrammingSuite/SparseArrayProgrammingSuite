import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import fetch_suitesparse_matrix
from saps.downloaders.snap import download_snap_dataset


class GraphCountingDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        expected: np.ndarray | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Graph counting input {name}."
        self._suites = suites or []
        self.A = A
        self.expected = expected

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


class TriangleCountTestGenerator(Generator[GraphCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_count_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Triangle Count Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic triangle-count examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI might have been used to construct tests. This statement "
            "was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Provide small graph examples for triangle-count correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[GraphCountingDataset]:
        return [
            GraphCountingDataset(
                "test_triangle_count_single_triangle",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(1),
            ),
            GraphCountingDataset(
                "test_triangle_count_path",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(0),
            ),
            GraphCountingDataset(
                "test_triangle_count_4_clique",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(4),
            ),
            GraphCountingDataset(
                "test_triangle_snap_toy",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(0),
            ),
        ]

    def generate(self, dataset: GraphCountingDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("Triangle-count test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class TriangleCountGenerator(Generator[GraphCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_count_inputs"

    @property
    def pretty_name(self) -> str:
        return "Triangle Count Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for triangle counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "SNAP: A General Purpose Network Analysis and Graph Mining Library"
                ),
                authors=[
                    Author("Leskovec, Jure"),
                    Author("Sosič, Rok"),
                ],
                journal="ACM Transactions on Intelligent Systems and Technology",
                volume=8,
                number=1,
                year=2016,
                url="https://snap.stanford.edu/index.html",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for triangle counting."

    @property
    def datasets(self) -> list[GraphCountingDataset]:
        return [
            GraphCountingDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: GraphCountingDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported triangle count dataset: {dataset.name}")


class TriangleCountGAPGenerator(Generator[GraphCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_count_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Triangle Count GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for triangle counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="The GAP Benchmark Suite",
                authors=[
                    Author("Scott Beamer"),
                    Author("Krste Asanović"),
                    Author("David Patterson"),
                ],
                url="https://arxiv.org/abs/1508.03619",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate GAP graph inputs for triangle counting."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[GraphCountingDataset]:
        return [
            GraphCountingDataset(
                name="gap-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="gap-twitter",
                pretty_name="GAP Twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="gap-web",
                pretty_name="GAP Web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="gap-kron",
                pretty_name="GAP Kron",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Kronecker synthetic graph generator with parameters"
                    " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                    " edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="gap-urand",
                pretty_name="GAP Urand",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                    " 2,147.4M edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: GraphCountingDataset) -> DataInstance:
        if dataset.name.startswith("gap"):
            raw = fetch_suitesparse_matrix(dataset.name)
            return DataInstance(inputs=[raw.inputs[0]], meta=raw.meta)
        raise ValueError(f"Unsupported triangle count dataset: {dataset.name}")


class TriangleCountBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "triangle_count"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting"

    @property
    def motivation(self) -> str:
        return (
            "Adjacency matrices are often sparse, and are used as input in this"
            " problem."
            "'It is generally known that counting the exact number of"
            "triangles in a graph G can be described using the language of"
            "linear algebra as 1/6 Γ(A3),"
            "where A is the adjacency matrix of the graph G, and Γ(X)"
            "is the trace of the square matrix X [1]. Other linear algebra"
            "approaches [2], [3] also require a sparse-matrix multiplication"
            "of A or parts of A as part of their computation. Alternative"
            "approaches that are not based on linear algebra leverage other"
            "formats for describing graphs such as the adjacency list to"
            "design their algorithms [4], [5].'"
            "'...the shortcut method of computing a power of a [adjacency] matrix,"
            "is isomorphic to a similar shortcut for ﬁnding all shortest paths.'"
        )

    @property
    def description(self) -> str:
        return (
            "Triangle Counting: Given adjacency matrix A, # triangles = trace(A^3) //"
            " 6. This counts the number of walks of length 3 that start at vertex i and"
            " end at vertex i, which is exactly a triangle. Divide by 6 to avoid"
            " overcounting. These methods are implemented using the property that"
            " multiplying a graph's adjacency matrix by itself n times yields the"
            " number of walks of length n that begin at the vertex denoted by the row"
            " label and end at the vertex denoted by the column label."
        )

    @property
    def suites(self) -> list[str]:
        return []

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
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "First look: Linear algebra-based triangle counting"
                    " without matrix multiplication"
                ),
                authors=[
                    Author("T. M. Low"),
                    Author("V. N. Rao"),
                    Author("M. Lee"),
                    Author("D. Popovici"),
                    Author("F. Franchetti"),
                    Author("S. McMillan"),
                ],
                journal="IEEE High Performance Extreme Computing Conference (HPEC)",
                year=2017,
                url="https://doi.org/10.1109/HPEC.2017.8091046",
            ),
            Ref(
                title="Graph Algorithms in the Language of Linear Algebra",
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics",
                year=2011,
                url="https://doi.org/10.1137/1.9780898719918",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator[GraphCountingDataset]]:
        return [
            TriangleCountTestGenerator(),
            TriangleCountGenerator(),
            TriangleCountGAPGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A = data[0]
        triangles = xp.einsum("S[] += A[i,j] * A[j,k] * A[k,i]", A=A) / 6
        return [xp.asarray(triangles)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        result = to_numpy(self._output[0])
        expected = (
            to_numpy(self._ref_outputs[0])
        )
        assert np.allclose(result, expected)
