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


class TriangleCountingDataset(Dataset):
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


class TriangleCountingTestGenerator(Generator[TriangleCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_counting_test"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting Test"

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
    def datasets(self) -> list[TriangleCountingDataset]:
        return [
            TriangleCountingDataset(
                "single_triangle",
                pretty_name="Single Triangle",
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
            TriangleCountingDataset(
                "path",
                pretty_name="Path",
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
            TriangleCountingDataset(
                "four_clique",
                pretty_name="4-Clique",
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
            TriangleCountingDataset(
                "snap_toy",
                pretty_name="SNAP Toy",
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

    def generate(self, dataset: TriangleCountingDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("Triangle-count test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


# Reference values of sum A[i,j] A[j,k] A[k,i] on each graph's 0-1 adjacency, before the
# division by 6, computed by scripts/compute_graph_counts.py. Graphs with
# over a billion edges have none, so their outputs are not checked.
_TRIANGLE_SUMS: dict[str, int] = {
    "soc-Epinions1": 2220930,
    "soc-LiveJournal1": 730982880,
    "soc-Pokec": 79611987,
    "soc-Slashdot0811": 4885971,
    "soc-Slashdot0902": 4970748,
    "wiki-Vote": 131925,
    "wiki-RfA": 401466,
    "soc-sign-bitcoin-otc": 115743,
    "soc-sign-bitcoin-alpha": 84453,
    "com-LiveJournal": 1066920780,
    "com-Orkut": 3765505086,
    "com-Youtube": 18338316,
    "com-DBLP": 13346310,
    "com-Amazon": 4002774,
    "email-Eu-core": 395667,
    "wiki-topcats": 27691482,
    "email-EuAll": 727266,
    "email-Enron": 4362264,
    "wiki-Talk": 15416112,
    "cit-HepPh": 1667,
    "cit-HepTh": 1716,
    "cit-Patents": 1,
    "ca-AstroPh": 8117964,
    "ca-CondMat": 1048156,
    "ca-GrQc": 289779,
    "ca-HepPh": 20154623,
    "ca-HepTh": 171238,
    "web-BerkStan": 41421291,
    "web-Google": 11669313,
    "web-NotreDame": 42866732,
    "web-Stanford": 3166875,
    "amazon0302": 1338180,
    "amazon0312": 6788373,
    "amazon0505": 7651068,
    "amazon0601": 7875786,
    "p2p-Gnutella04": 99,
    "p2p-Gnutella05": 102,
    "p2p-Gnutella06": 120,
    "p2p-Gnutella08": 159,
    "p2p-Gnutella09": 135,
    "p2p-Gnutella24": 90,
    "p2p-Gnutella25": 48,
    "p2p-Gnutella30": 177,
    "p2p-Gnutella31": 171,
    "roadNet-CA": 724056,
    "roadNet-PA": 402900,
    "roadNet-TX": 497214,
    "as-735": 72096,
    "as-Skitter": 172619208,
    "as-caida": 218190,
    "Oregon-1": 119364,
    "Oregon-2": 537246,
    "soc-sign-epinions": 7099728,
    "soc-sign-Slashdot081106": 765204,
    "soc-sign-Slashdot090216": 754944,
    "soc-sign-Slashdot090221": 768540,
    "loc-Gowalla": 13638828,
    "loc-Brightkite": 2968368,
    "sx-stackoverflow": 188501666,
    "sx-mathoverflow": 2520164,
    "sx-superuser": 2788107,
    "sx-askubuntu": 1450657,
    "wiki-talk-temporal": 14009570,
    "email-Eu-core-temporal": 347700,
    "CollegeMsg": 32796,
    "higgs-twitter": 70278125,
    "GAP-road": 2632824,
}


def _reference_outputs(name: str) -> list | None:
    total = _TRIANGLE_SUMS.get(name)
    return None if total is None else [from_numpy(np.array(total / 6))]


class TriangleCountingSNAPGenerator(Generator[TriangleCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_counting_snap"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting SNAP"

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
    def datasets(self) -> list[TriangleCountingDataset]:
        # Trace selects successful Smart runs < 60s in competition/run_13803684.
        # fmt: off
        return [
            TriangleCountingDataset("soc-Epinions1", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-LiveJournal1", suites=["standard"]),
            TriangleCountingDataset("soc-Pokec", suites=["standard"]),
            TriangleCountingDataset("soc-Slashdot0811", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-Slashdot0902", suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-Vote", suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-RfA", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace"]),
            TriangleCountingDataset("com-LiveJournal", suites=["standard"]),
            TriangleCountingDataset("com-Friendster", suites=["standard"]),
            TriangleCountingDataset("com-Orkut", suites=["standard"]),
            TriangleCountingDataset("com-Youtube", suites=["standard"]),
            TriangleCountingDataset("com-DBLP", suites=["standard", "trace"]),
            TriangleCountingDataset("com-Amazon", suites=["standard", "trace"]),
            TriangleCountingDataset("email-Eu-core", suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-topcats", suites=["standard"]),
            TriangleCountingDataset("email-EuAll", suites=["standard", "trace"]),
            TriangleCountingDataset("email-Enron", suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-Talk", suites=["standard"]),
            TriangleCountingDataset("cit-HepPh", suites=["standard", "trace"]),
            TriangleCountingDataset("cit-HepTh", suites=["standard", "trace"]),
            TriangleCountingDataset("cit-Patents", suites=["standard"]),
            TriangleCountingDataset("ca-AstroPh", suites=["standard", "trace"]),
            TriangleCountingDataset("ca-CondMat", suites=["standard", "trace"]),
            TriangleCountingDataset("ca-GrQc", suites=["standard", "trace"]),
            TriangleCountingDataset("ca-HepPh", suites=["standard", "trace"]),
            TriangleCountingDataset("ca-HepTh", suites=["standard", "trace"]),
            TriangleCountingDataset("web-BerkStan", suites=["standard"]),
            TriangleCountingDataset("web-Google", suites=["standard", "trace"]),
            TriangleCountingDataset("web-NotreDame", suites=["standard", "trace", "train"]),
            TriangleCountingDataset("web-Stanford", suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0302", suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0312", suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0505", suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0601", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella31", suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-CA", suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-PA", suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-TX", suites=["standard", "trace"]),
            TriangleCountingDataset("as-735", suites=["standard", "trace"]),
            TriangleCountingDataset("as-Skitter", suites=["standard"]),
            TriangleCountingDataset("as-caida", suites=["standard", "trace"]),
            TriangleCountingDataset("Oregon-1", suites=["standard", "trace"]),
            TriangleCountingDataset("Oregon-2", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-epinions", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot081106", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot090216", suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot090221", suites=["standard", "trace"]),
            TriangleCountingDataset("loc-Gowalla", suites=["standard"]),
            TriangleCountingDataset("loc-Brightkite", suites=["standard", "trace"]),
            TriangleCountingDataset("sx-stackoverflow", suites=["standard"]),
            TriangleCountingDataset("sx-mathoverflow", suites=["standard", "trace"]),
            TriangleCountingDataset("sx-superuser", suites=["standard", "trace"]),
            TriangleCountingDataset("sx-askubuntu", suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-talk-temporal", suites=["standard"]),
            TriangleCountingDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            TriangleCountingDataset("CollegeMsg", suites=["standard", "trace"]),
            TriangleCountingDataset("twitter7", suites=["standard"]),
            TriangleCountingDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: TriangleCountingDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)],
            meta=dict(raw.meta),
            ref_outputs=_reference_outputs(dataset.name),
        )


class TriangleCountingGAPGenerator(Generator[TriangleCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_counting_gap"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting GAP"

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
    def datasets(self) -> list[TriangleCountingDataset]:
        # fmt: off
        return [
            TriangleCountingDataset("GAP-road", suites=["standard"]),
            TriangleCountingDataset("GAP-twitter", suites=["standard"]),
            TriangleCountingDataset("GAP-web", suites=["standard"]),
            TriangleCountingDataset("GAP-kron", suites=["standard"]),
            TriangleCountingDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: TriangleCountingDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.int64)],
            meta=dict(raw.meta),
            ref_outputs=_reference_outputs(dataset.name),
        )


class TriangleCountingBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "triangle_counting"

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
            TriangleCountingTestGenerator(),
            TriangleCountingSNAPGenerator(),
            TriangleCountingGAPGenerator(),
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
