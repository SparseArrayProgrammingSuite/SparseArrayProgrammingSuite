# ruff: noqa: E501

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
from saps.benchmarks.adjacency import zero_one_adjacency
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


class TriangleCountDataset(Dataset):
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
        self._suites = list(suites or [])
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


class TriangleCountTestGenerator(Generator[TriangleCountDataset]):
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
        return ["test"]

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
    def datasets(self) -> list[TriangleCountDataset]:
        return [
            TriangleCountDataset(
                "test_triangle_count_single_triangle",
                suites=["test"],
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
            TriangleCountDataset(
                "test_triangle_count_path",
                suites=["test"],
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
            TriangleCountDataset(
                "test_triangle_count_4_clique",
                suites=["test"],
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
            TriangleCountDataset(
                "test_triangle_snap_toy",
                suites=["test"],
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

    def generate(self, dataset: TriangleCountDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("Triangle-count test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class TriangleCountSNAPGenerator(Generator[TriangleCountDataset]):
    @property
    def name(self) -> str:
        return "triangle_count_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Triangle Count SNAP Input Generator"

    @property
    def description(self) -> str:
        return "SNAP input generator for triangle counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return ["standard"]

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
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[TriangleCountDataset]:
        # Trace selects successful Smart runs < 30s in competition/run_13803684.
        # fmt: off
        return [
            TriangleCountDataset("soc-Epinions1", suites=["standard", "trace"]),
            TriangleCountDataset("soc-LiveJournal1", suites=["standard"]),
            TriangleCountDataset("soc-Pokec", suites=["standard"]),
            TriangleCountDataset("soc-Slashdot0811", suites=["standard"]),
            TriangleCountDataset("soc-Slashdot0902", suites=["standard"]),
            TriangleCountDataset("wiki-Vote", suites=["standard", "trace"]),
            TriangleCountDataset("wiki-RfA", suites=["standard", "trace"]),
            TriangleCountDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            TriangleCountDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace"]),
            TriangleCountDataset("com-LiveJournal", suites=["standard"]),
            TriangleCountDataset("com-Friendster", suites=["standard"]),
            TriangleCountDataset("com-Orkut", suites=["standard"]),
            TriangleCountDataset("com-Youtube", suites=["standard"]),
            TriangleCountDataset("com-DBLP", suites=["standard", "trace"]),
            TriangleCountDataset("com-Amazon", suites=["standard", "trace"]),
            TriangleCountDataset("email-Eu-core", suites=["standard", "trace"]),
            TriangleCountDataset("wiki-topcats", suites=["standard"]),
            TriangleCountDataset("email-EuAll", suites=["standard", "trace"]),
            TriangleCountDataset("email-Enron", suites=["standard", "trace"]),
            TriangleCountDataset("wiki-Talk", suites=["standard"]),
            TriangleCountDataset("cit-HepPh", suites=["standard", "trace"]),
            TriangleCountDataset("cit-HepTh", suites=["standard", "trace"]),
            TriangleCountDataset("cit-Patents", suites=["standard"]),
            TriangleCountDataset("ca-AstroPh", suites=["standard", "trace"]),
            TriangleCountDataset("ca-CondMat", suites=["standard", "trace"]),
            TriangleCountDataset("ca-GrQc", suites=["standard", "trace"]),
            TriangleCountDataset("ca-HepPh", suites=["standard", "trace"]),
            TriangleCountDataset("ca-HepTh", suites=["standard", "trace"]),
            TriangleCountDataset("web-BerkStan", suites=["standard"]),
            TriangleCountDataset("web-Google", suites=["standard", "trace"]),
            TriangleCountDataset("web-NotreDame", suites=["standard"]),
            TriangleCountDataset("web-Stanford", suites=["standard", "trace"]),
            TriangleCountDataset("amazon0302", suites=["standard", "trace"]),
            TriangleCountDataset("amazon0312", suites=["standard", "trace"]),
            TriangleCountDataset("amazon0505", suites=["standard", "trace"]),
            TriangleCountDataset("amazon0601", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            TriangleCountDataset("p2p-Gnutella31", suites=["standard", "trace"]),
            TriangleCountDataset("roadNet-CA", suites=["standard", "trace"]),
            TriangleCountDataset("roadNet-PA", suites=["standard", "trace"]),
            TriangleCountDataset("roadNet-TX", suites=["standard", "trace"]),
            TriangleCountDataset("as-735", suites=["standard", "trace"]),
            TriangleCountDataset("as-Skitter", suites=["standard"]),
            TriangleCountDataset("as-caida", suites=["standard", "trace"]),
            TriangleCountDataset("Oregon-1", suites=["standard", "trace"]),
            TriangleCountDataset("Oregon-2", suites=["standard", "trace"]),
            TriangleCountDataset("soc-sign-epinions", suites=["standard"]),
            TriangleCountDataset("soc-sign-Slashdot081106", suites=["standard", "trace"]),
            TriangleCountDataset("soc-sign-Slashdot090216", suites=["standard", "trace"]),
            TriangleCountDataset("soc-sign-Slashdot090221", suites=["standard", "trace"]),
            TriangleCountDataset("loc-Gowalla", suites=["standard"]),
            TriangleCountDataset("loc-Brightkite", suites=["standard", "trace"]),
            TriangleCountDataset("sx-stackoverflow", suites=["standard"]),
            TriangleCountDataset("sx-mathoverflow", suites=["standard", "trace"]),
            TriangleCountDataset("sx-superuser", suites=["standard"]),
            TriangleCountDataset("sx-askubuntu", suites=["standard", "trace"]),
            TriangleCountDataset("wiki-talk-temporal", suites=["standard"]),
            TriangleCountDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            TriangleCountDataset("CollegeMsg", suites=["standard", "trace"]),
            TriangleCountDataset("twitter7", suites=["standard"]),
            TriangleCountDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: TriangleCountDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)], meta=dict(raw.meta)
        )


class TriangleCountGAPGenerator(Generator[TriangleCountDataset]):
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
    def datasets(self) -> list[TriangleCountDataset]:
        # fmt: off
        return [
            TriangleCountDataset("GAP-road", suites=["standard"]),
            TriangleCountDataset("GAP-twitter", suites=["standard"]),
            TriangleCountDataset("GAP-web", suites=["standard"]),
            TriangleCountDataset("GAP-kron", suites=["standard"]),
            TriangleCountDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: TriangleCountDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)], meta=dict(raw.meta)
        )


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
        return ["standard-graphs-query"]

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
    def generators(self) -> list[Generator]:
        return [
            TriangleCountTestGenerator(),
            TriangleCountSNAPGenerator(),
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
        expected = to_numpy(self._ref_outputs[0])
        assert np.allclose(result, expected)
