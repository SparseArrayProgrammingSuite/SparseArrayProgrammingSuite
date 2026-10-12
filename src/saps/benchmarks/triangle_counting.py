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
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph
from saps.util.adjacency import zero_one_adjacency


class TriangleCountingDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        expected: np.ndarray | None = None,
        simple_undirected: bool = True,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Graph counting input {name}."
        if simple_undirected:
            self._description += " Symmetrized, with self-loops removed; only the strict upper triangle is stored."
        self._suites = list(suites or [])
        self.A = A
        self.expected = expected
        self.simple_undirected = simple_undirected

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
                        [1, 0, 1],
                        [0, 1, 0],
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
            inputs=[from_numpy(np.triu(dataset.A, k=1))],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


# Reference counts for simple undirected graphs. The generators store only
# the strict upper triangle, so each triangle is counted once.
# Published SNAP counts: https://snap.stanford.edu/data/<name>.html
# Entries marked computed use scripts/compute_graph_counts.py on the stored
# SuiteSparse graph with the same simple-undirected convention.
_TRIANGLE_COUNTS: dict[str, int] = {
    "soc-Epinions1": 1_624_481,
    "soc-LiveJournal1": 285_730_264,
    "soc-Pokec": 32_557_458,
    "soc-Slashdot0811": 551_724,
    "soc-Slashdot0902": 602_592,
    "wiki-Vote": 608_389,
    # Computed: SuiteSparse includes neutral votes, unlike SNAP's published total.
    # https://sparse.tamu.edu/SNAP/wiki-RfA
    "wiki-RfA": 1_416_059,
    # Computed: these SNAP pages do not publish triangle totals.
    "soc-sign-bitcoin-otc": 33_493,
    "soc-sign-bitcoin-alpha": 22_153,
    "com-LiveJournal": 177_820_130,
    "com-Orkut": 627_584_181,
    "com-Youtube": 3_056_386,
    "com-DBLP": 2_224_385,
    "com-Amazon": 667_129,
    "email-Eu-core": 105_461,
    "wiki-topcats": 52_106_893,
    "email-EuAll": 267_313,
    "email-Enron": 727_044,
    "wiki-Talk": 9_203_519,
    "cit-HepPh": 1_276_868,
    "cit-HepTh": 1_478_735,
    "cit-Patents": 7_515_023,
    "ca-AstroPh": 1_351_441,
    "ca-CondMat": 173_361,
    "ca-GrQc": 48_260,
    "ca-HepPh": 3_358_499,
    "ca-HepTh": 28_339,
    "web-BerkStan": 64_690_980,
    "web-Google": 13_391_903,
    "web-NotreDame": 8_910_005,
    "web-Stanford": 11_329_473,
    "amazon0302": 717_719,
    "amazon0312": 3_686_467,
    "amazon0505": 3_951_063,
    "amazon0601": 3_986_507,
    "p2p-Gnutella04": 934,
    "p2p-Gnutella05": 1_112,
    "p2p-Gnutella06": 1_142,
    "p2p-Gnutella08": 2_383,
    "p2p-Gnutella09": 2_354,
    "p2p-Gnutella24": 986,
    "p2p-Gnutella25": 806,
    "p2p-Gnutella30": 1_590,
    "p2p-Gnutella31": 2_024,
    "roadNet-CA": 120_676,
    "roadNet-PA": 67_150,
    "roadNet-TX": 82_869,
    # https://snap.stanford.edu/data/as-733.html (January 02 2000)
    "as-735": 6_584,
    "as-Skitter": 28_769_868,
    # https://snap.stanford.edu/data/as-Caida.html (November 05 2007)
    "as-caida": 36_365,
    # https://snap.stanford.edu/data/oregon1.html (May 26 2001)
    "Oregon-1": 19_894,
    # https://snap.stanford.edu/data/oregon2.html (May 26 2001)
    "Oregon-2": 89_541,
    "soc-sign-epinions": 4_910_076,
    "soc-sign-Slashdot081106": 548_054,
    "soc-sign-Slashdot090216": 570_569,
    "soc-sign-Slashdot090221": 579_565,
    "loc-Gowalla": 2_273_138,
    "loc-Brightkite": 494_728,
    # Computed: these SNAP pages do not publish triangle totals.
    "sx-stackoverflow": 114_206_974,
    "sx-mathoverflow": 1_403_919,
    "sx-superuser": 1_543_161,
    "sx-askubuntu": 680_407,
    "wiki-talk-temporal": 8_113_676,
    "email-Eu-core-temporal": 105_461,
    "CollegeMsg": 14_319,
    "higgs-twitter": 83_023_401,
    "GAP-road": 438_804,
    # SNAP / GraphMiner, with matching node and edge counts:
    # https://snap.stanford.edu/data/com-Friendster.html
    # https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/clique/README.md
    "com-Friendster": 4_173_724_142,
    "twitter7": 34_824_916_864,
    # LAGraph's known counts for the SuiteSparse GAP matrices:
    # https://github.com/GraphBLAS/LAGraph/blob/7543de1f310371c540c23ff5da82705aa4e94ea0/src/benchmark/tc_demo.c#L22-L27
    # twitter7 and GAP-twitter agree after removing loops and ignoring weights:
    # https://sparse.tamu.edu/GAP/GAP-twitter
    "GAP-twitter": 34_824_916_864,
    "GAP-web": 84_907_041_475,
    "GAP-kron": 106_873_365_648,
    "GAP-urand": 5_378,
}


def _reference_outputs(name: str) -> list | None:
    total = _TRIANGLE_COUNTS.get(name)
    return None if total is None else [from_numpy(np.array(total, dtype=np.int64))]


class TriangleCountingSNAPGenerator(Generator[TriangleCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_counting_snap"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting SNAP"

    @property
    def description(self) -> str:
        return "Simple undirected SNAP input generator for triangle counting benchmarks."

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
            ),
            Ref(
                title="Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU",
                authors=[
                    Author("Xuhao Chen"),
                    Author("Roshan Dathathri"),
                    Author("Gurbinder Gill"),
                    Author("Keshav Pingali"),
                ],
                journal="Proc. VLDB Endow.",
                volume=13,
                number=8,
                year=2020,
                publisher="VLDB Endowment",
                url="https://arxiv.org/abs/1911.06969",
            ),
            Ref(
                title="Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining",
                authors=[
                    Author("Xuhao Chen"),
                    Author("Roshan Dathathri"),
                    Author("Gurbinder Gill"),
                    Author("Loc Hoang"),
                    Author("Keshav Pingali"),
                ],
                booktitle="Proceedings of the 35th ACM International Conference on Supercomputing",
                year=2021,
                url="https://doi.org/10.1145/3447818.3460359",
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
            inputs=[
                zero_one_adjacency(
                    raw.inputs[0],
                    np.int64,
                    simple_undirected=dataset.simple_undirected,
                    upper_triangle=True,
                )
            ],
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
            Ref(
                title="Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU",
                authors=[
                    Author("Xuhao Chen"),
                    Author("Roshan Dathathri"),
                    Author("Gurbinder Gill"),
                    Author("Keshav Pingali"),
                ],
                journal="Proc. VLDB Endow.",
                volume=13,
                number=8,
                year=2020,
                publisher="VLDB Endowment",
                url="https://arxiv.org/abs/1911.06969",
            ),
            Ref(
                title="Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining",
                authors=[
                    Author("Xuhao Chen"),
                    Author("Roshan Dathathri"),
                    Author("Gurbinder Gill"),
                    Author("Loc Hoang"),
                    Author("Keshav Pingali"),
                ],
                booktitle="Proceedings of the 35th ACM International Conference on Supercomputing",
                year=2021,
                url="https://doi.org/10.1145/3447818.3460359",
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
            inputs=[
                zero_one_adjacency(
                    raw.inputs[0],
                    np.int64,
                    simple_undirected=dataset.simple_undirected,
                    upper_triangle=True,
                )
            ],
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
            "Triangle Counting: Given the strict upper triangle A of a simple"
            " undirected adjacency matrix, sum A[i,j] A[j,k] A[i,k]. Each triangle"
            " has exactly one ordering i < j < k, so it is counted once."
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

    def benchmark(self, xp, meta: dict, A):
        triangles = xp.einsum("S[] += A[i,j] * A[j,k] * A[i,k]", A=A)
        return xp.asarray(triangles)

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
