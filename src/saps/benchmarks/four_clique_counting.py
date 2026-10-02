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


class FourCliqueCountDataset(Dataset):
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


class FourCliqueCountTestGenerator(Generator[FourCliqueCountDataset]):
    @property
    def name(self) -> str:
        return "four_clique_count_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Count Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic 4-clique-count examples."

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
        return "Provide small graph examples for 4-clique-count correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FourCliqueCountDataset]:
        return [
            FourCliqueCountDataset(
                "test_4clique_count_complete_k3",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(0),
            ),
            FourCliqueCountDataset(
                "test_4clique_count_single_k4",
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
                expected=np.array(1),
            ),
            FourCliqueCountDataset(
                "test_4clique_count_overlapping",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 1, 1, 0],
                        [1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1],
                        [0, 1, 1, 1, 0],
                    ],
                    dtype=int,
                ),
                expected=np.array(2),
            ),
            FourCliqueCountDataset(
                "test_4clique_snap_toy",
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

    def generate(self, dataset: FourCliqueCountDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("4-clique test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class FourCliqueCountSNAPGenerator(Generator[FourCliqueCountDataset]):
    @property
    def name(self) -> str:
        return "four_clique_count_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Count SNAP Input Generator"

    @property
    def description(self) -> str:
        return "SNAP input generator for 4-clique counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return ["standard"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

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
        return "Generate sparse graph inputs for 4-clique counting."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FourCliqueCountDataset]:
        # Successful standard-suite Smart runs <= 30s in competition/run_13662472.
        # fmt: off
        return [
            FourCliqueCountDataset("soc-Epinions1", suites=["standard"]),
            FourCliqueCountDataset("soc-LiveJournal1", suites=["standard"]),
            FourCliqueCountDataset("soc-Pokec", suites=["standard"]),
            FourCliqueCountDataset("soc-Slashdot0811", suites=["standard"]),
            FourCliqueCountDataset("soc-Slashdot0902", suites=["standard"]),
            FourCliqueCountDataset("wiki-Vote", suites=["standard"]),
            FourCliqueCountDataset("wiki-RfA", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-bitcoin-otc", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-bitcoin-alpha", suites=["standard"]),
            FourCliqueCountDataset("com-LiveJournal", suites=["standard"]),
            FourCliqueCountDataset("com-Friendster", suites=["standard"]),
            FourCliqueCountDataset("com-Orkut", suites=["standard"]),
            FourCliqueCountDataset("com-Youtube", suites=["standard"]),
            FourCliqueCountDataset("com-DBLP", suites=["standard"]),
            FourCliqueCountDataset("com-Amazon", suites=["standard"]),
            FourCliqueCountDataset("email-Eu-core", suites=["standard"]),
            FourCliqueCountDataset("wiki-topcats", suites=["standard"]),
            FourCliqueCountDataset("email-EuAll", suites=["standard"]),
            FourCliqueCountDataset("email-Enron", suites=["standard"]),
            FourCliqueCountDataset("wiki-Talk", suites=["standard"]),
            FourCliqueCountDataset("cit-HepPh", suites=["standard"]),
            FourCliqueCountDataset("cit-HepTh", suites=["standard"]),
            FourCliqueCountDataset("cit-Patents", suites=["standard"]),
            FourCliqueCountDataset("ca-AstroPh", suites=["standard"]),
            FourCliqueCountDataset("ca-CondMat", suites=["standard"]),
            FourCliqueCountDataset("ca-GrQc", suites=["standard", "trace"]),
            FourCliqueCountDataset("ca-HepPh", suites=["standard"]),
            FourCliqueCountDataset("ca-HepTh", suites=["standard", "trace"]),
            FourCliqueCountDataset("web-BerkStan", suites=["standard"]),
            FourCliqueCountDataset("web-Google", suites=["standard"]),
            FourCliqueCountDataset("web-NotreDame", suites=["standard"]),
            FourCliqueCountDataset("web-Stanford", suites=["standard"]),
            FourCliqueCountDataset("amazon0302", suites=["standard"]),
            FourCliqueCountDataset("amazon0312", suites=["standard"]),
            FourCliqueCountDataset("amazon0505", suites=["standard"]),
            FourCliqueCountDataset("amazon0601", suites=["standard"]),
            FourCliqueCountDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            FourCliqueCountDataset("p2p-Gnutella31", suites=["standard"]),
            FourCliqueCountDataset("roadNet-CA", suites=["standard"]),
            FourCliqueCountDataset("roadNet-PA", suites=["standard"]),
            FourCliqueCountDataset("roadNet-TX", suites=["standard"]),
            FourCliqueCountDataset("as-735", suites=["standard"]),
            FourCliqueCountDataset("as-Skitter", suites=["standard"]),
            FourCliqueCountDataset("as-caida", suites=["standard"]),
            FourCliqueCountDataset("Oregon-1", suites=["standard"]),
            FourCliqueCountDataset("Oregon-2", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-epinions", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-Slashdot081106", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-Slashdot090216", suites=["standard"]),
            FourCliqueCountDataset("soc-sign-Slashdot090221", suites=["standard"]),
            FourCliqueCountDataset("loc-Gowalla", suites=["standard"]),
            FourCliqueCountDataset("loc-Brightkite", suites=["standard"]),
            FourCliqueCountDataset("sx-stackoverflow", suites=["standard"]),
            FourCliqueCountDataset("sx-mathoverflow", suites=["standard"]),
            FourCliqueCountDataset("sx-superuser", suites=["standard"]),
            FourCliqueCountDataset("sx-askubuntu", suites=["standard"]),
            FourCliqueCountDataset("wiki-talk-temporal", suites=["standard"]),
            FourCliqueCountDataset("email-Eu-core-temporal", suites=["standard"]),
            FourCliqueCountDataset("CollegeMsg", suites=["standard"]),
            FourCliqueCountDataset("twitter7", suites=["standard"]),
            FourCliqueCountDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: FourCliqueCountDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)], meta=dict(raw.meta)
        )


class FourCliqueCountGAPGenerator(Generator[FourCliqueCountDataset]):
    @property
    def name(self) -> str:
        return "four_clique_count_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Count GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for 4-clique counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Willow Ahrens", "ahrens@gatech.edu"),
        ]

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
        return "Generate GAP graph inputs for 4-clique counting."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FourCliqueCountDataset]:
        # fmt: off
        return [
            FourCliqueCountDataset("GAP-road", suites=["standard"]),
            FourCliqueCountDataset("GAP-twitter", suites=["standard"]),
            FourCliqueCountDataset("GAP-web", suites=["standard"]),
            FourCliqueCountDataset("GAP-kron", suites=["standard"]),
            FourCliqueCountDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: FourCliqueCountDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)], meta=dict(raw.meta)
        )


class FourCliqueCountBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "four_clique_count"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Counting"

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
            "4-clique Counting: A 4-clique must contain 6 edges that connect all 4"
            " vertices. The einsum does the following: for a given vertex i, checks for"
            " existence of 3 edges to 3 other vertices, then checks for existence of 3"
            " edges between those 3 vertices. This constitutes a 4-clique. Divide by 24"
            " to avoid overcounting. These methods are implemented using the property"
            " that multiplying a graph's adjacency matrix by itself n times yields the"
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
            FourCliqueCountTestGenerator(),
            FourCliqueCountSNAPGenerator(),
            FourCliqueCountGAPGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A = data[0]
        cliq_4 = (
            xp.einsum(
                "S[] += A[i,j] * A[i,k] * A[i,l] * A[j,k] * A[j,l] * A[k,l]",
                A=A,
            )
            / 24
        )
        return [xp.asarray(cliq_4)]

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
