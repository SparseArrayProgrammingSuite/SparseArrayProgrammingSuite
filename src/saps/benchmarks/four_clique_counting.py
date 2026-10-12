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


class FourCliqueCountingDataset(Dataset):
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


class FourCliqueCountingTestGenerator(Generator[FourCliqueCountingDataset]):
    @property
    def name(self) -> str:
        return "four_clique_counting_test"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Counting Test"

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
    def datasets(self) -> list[FourCliqueCountingDataset]:
        return [
            FourCliqueCountingDataset(
                "complete_k3",
                pretty_name="Complete K3",
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
            FourCliqueCountingDataset(
                "single_k4",
                pretty_name="Single K4",
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
            FourCliqueCountingDataset(
                "overlapping",
                pretty_name="Overlapping",
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
            FourCliqueCountingDataset(
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

    def generate(self, dataset: FourCliqueCountingDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("4-clique test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(np.triu(dataset.A, k=1))],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


# Reference counts for simple undirected graphs, computed by
# scripts/compute_graph_counts.py unless cited below. The generators store only
# the strict upper triangle, so each four-clique is counted once.
# GAP-web, GAP-kron, and GAP-urand still have no verified reference counts.
_FOUR_CLIQUE_COUNTS: dict[str, int] = {
    "soc-Epinions1": 5_803_397,
    # GraphMiner (livej): same symmetrized graph and triangle count as SNAP.
    # https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/clique/README.md
    "soc-LiveJournal1": 9_933_532_019,
    "soc-Pokec": 42_947_031,
    "soc-Slashdot0811": 1_989_958,
    "soc-Slashdot0902": 2_260_269,
    "wiki-Vote": 2_077_903,
    "wiki-RfA": 6_305_134,
    "soc-sign-bitcoin-otc": 48_390,
    "soc-sign-bitcoin-alpha": 30_948,
    "com-LiveJournal": 5_216_918_441,
    "com-Orkut": 3_221_946_137,
    "com-Youtube": 4_986_965,
    "com-DBLP": 16_713_192,
    "com-Amazon": 275_961,
    "email-Eu-core": 423_750,
    "wiki-topcats": 65_479_271,
    "email-EuAll": 581_032,
    "email-Enron": 2_341_639,
    "wiki-Talk": 64_940_189,
    "cit-HepPh": 2_562_245,
    "cit-HepTh": 4_113_289,
    "cit-Patents": 3_501_071,
    "ca-AstroPh": 9_580_415,
    "ca-CondMat": 294_008,
    "ca-GrQc": 329_297,
    "ca-HepPh": 150_281_372,
    "ca-HepTh": 65_592,
    "web-BerkStan": 1_065_796_916,
    "web-Google": 39_881_472,
    "web-NotreDame": 231_911_102,
    "web-Stanford": 78_757_781,
    "amazon0302": 304_441,
    "amazon0312": 3_975_712,
    "amazon0505": 4_361_282,
    "amazon0601": 4_420_994,
    "p2p-Gnutella04": 3,
    "p2p-Gnutella05": 55,
    "p2p-Gnutella06": 45,
    "p2p-Gnutella08": 175,
    "p2p-Gnutella09": 160,
    "p2p-Gnutella24": 11,
    "p2p-Gnutella25": 7,
    "p2p-Gnutella30": 13,
    "p2p-Gnutella31": 16,
    "roadNet-CA": 42,
    "roadNet-PA": 21,
    "roadNet-TX": 32,
    "as-735": 5_636,
    "as-Skitter": 148_834_439,
    "as-caida": 53_875,
    "Oregon-1": 30_479,
    "Oregon-2": 399_013,
    "soc-sign-epinions": 58_567_924,
    "soc-sign-Slashdot081106": 1_968_248,
    "soc-sign-Slashdot090216": 2_078_864,
    "soc-sign-Slashdot090221": 2_109_090,
    "loc-Gowalla": 6_086_852,
    "loc-Brightkite": 2_851_308,
    "sx-stackoverflow": 787_147_026,
    "sx-mathoverflow": 10_714_812,
    "sx-superuser": 6_399_900,
    "sx-askubuntu": 2_260_000,
    "wiki-talk-temporal": 53_439_733,
    "email-Eu-core-temporal": 423_750,
    "CollegeMsg": 5_389,
    "higgs-twitter": 429_733_013,
    "GAP-road": 90,
    # GraphMiner's simple undirected graphs, with matching node and edge counts:
    # https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/clique/README.md
    # twitter7 and GAP-twitter agree after removing loops and ignoring weights:
    # https://sparse.tamu.edu/GAP/GAP-twitter
    "com-Friendster": 8_963_503_263,
    "twitter7": 6_622_234_180_319,
    "GAP-twitter": 6_622_234_180_319,
}


def _reference_outputs(name: str) -> list | None:
    total = _FOUR_CLIQUE_COUNTS.get(name)
    return None if total is None else [from_numpy(np.array(total, dtype=np.int64))]


class FourCliqueCountingSNAPGenerator(Generator[FourCliqueCountingDataset]):
    @property
    def name(self) -> str:
        return "four_clique_counting_snap"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Counting SNAP"

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
        return "Generate sparse graph inputs for 4-clique counting."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FourCliqueCountingDataset]:
        # Trace selects successful Smart runs < 60s in competition/run_13803684.
        # fmt: off
        return [
            FourCliqueCountingDataset("soc-Epinions1", suites=["standard"]),
            FourCliqueCountingDataset("soc-LiveJournal1", suites=["standard"]),
            FourCliqueCountingDataset("soc-Pokec", suites=["standard"]),
            FourCliqueCountingDataset("soc-Slashdot0811", suites=["standard"]),
            FourCliqueCountingDataset("soc-Slashdot0902", suites=["standard"]),
            FourCliqueCountingDataset("wiki-Vote", suites=["standard"]),
            FourCliqueCountingDataset("wiki-RfA", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-bitcoin-otc", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-bitcoin-alpha", suites=["standard"]),
            FourCliqueCountingDataset("com-LiveJournal", suites=["standard"]),
            FourCliqueCountingDataset("com-Friendster", suites=["standard"]),
            FourCliqueCountingDataset("com-Orkut", suites=["standard"]),
            FourCliqueCountingDataset("com-Youtube", suites=["standard"]),
            FourCliqueCountingDataset("com-DBLP", suites=["standard"]),
            FourCliqueCountingDataset("com-Amazon", suites=["standard"]),
            FourCliqueCountingDataset("email-Eu-core", suites=["standard"]),
            FourCliqueCountingDataset("wiki-topcats", suites=["standard"]),
            FourCliqueCountingDataset("email-EuAll", suites=["standard"]),
            FourCliqueCountingDataset("email-Enron", suites=["standard"]),
            FourCliqueCountingDataset("wiki-Talk", suites=["standard"]),
            FourCliqueCountingDataset("cit-HepPh", suites=["standard"]),
            FourCliqueCountingDataset("cit-HepTh", suites=["standard"]),
            FourCliqueCountingDataset("cit-Patents", suites=["standard"]),
            FourCliqueCountingDataset("ca-AstroPh", suites=["standard"]),
            FourCliqueCountingDataset("ca-CondMat", suites=["standard"]),
            FourCliqueCountingDataset("ca-GrQc", suites=["standard", "trace"]),
            FourCliqueCountingDataset("ca-HepPh", suites=["standard"]),
            FourCliqueCountingDataset("ca-HepTh", suites=["standard", "trace"]),
            FourCliqueCountingDataset("web-BerkStan", suites=["standard"]),
            FourCliqueCountingDataset("web-Google", suites=["standard"]),
            FourCliqueCountingDataset("web-NotreDame", suites=["standard"]),
            FourCliqueCountingDataset("web-Stanford", suites=["standard"]),
            FourCliqueCountingDataset("amazon0302", suites=["standard"]),
            FourCliqueCountingDataset("amazon0312", suites=["standard"]),
            FourCliqueCountingDataset("amazon0505", suites=["standard"]),
            FourCliqueCountingDataset("amazon0601", suites=["standard"]),
            FourCliqueCountingDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella31", suites=["standard"]),
            FourCliqueCountingDataset("roadNet-CA", suites=["standard"]),
            FourCliqueCountingDataset("roadNet-PA", suites=["standard"]),
            FourCliqueCountingDataset("roadNet-TX", suites=["standard"]),
            FourCliqueCountingDataset("as-735", suites=["standard"]),
            FourCliqueCountingDataset("as-Skitter", suites=["standard"]),
            FourCliqueCountingDataset("as-caida", suites=["standard"]),
            FourCliqueCountingDataset("Oregon-1", suites=["standard"]),
            FourCliqueCountingDataset("Oregon-2", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-epinions", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot081106", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot090216", suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot090221", suites=["standard"]),
            FourCliqueCountingDataset("loc-Gowalla", suites=["standard"]),
            FourCliqueCountingDataset("loc-Brightkite", suites=["standard"]),
            FourCliqueCountingDataset("sx-stackoverflow", suites=["standard"]),
            FourCliqueCountingDataset("sx-mathoverflow", suites=["standard"]),
            FourCliqueCountingDataset("sx-superuser", suites=["standard"]),
            FourCliqueCountingDataset("sx-askubuntu", suites=["standard"]),
            FourCliqueCountingDataset("wiki-talk-temporal", suites=["standard"]),
            FourCliqueCountingDataset("email-Eu-core-temporal", suites=["standard"]),
            FourCliqueCountingDataset("CollegeMsg", suites=["standard", "trace", "train"]),
            FourCliqueCountingDataset("twitter7", suites=["standard"]),
            FourCliqueCountingDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: FourCliqueCountingDataset) -> DataInstance:
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


class FourCliqueCountingGAPGenerator(Generator[FourCliqueCountingDataset]):
    @property
    def name(self) -> str:
        return "four_clique_counting_gap"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Counting GAP"

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
        return "Generate GAP graph inputs for 4-clique counting."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FourCliqueCountingDataset]:
        # fmt: off
        return [
            FourCliqueCountingDataset("GAP-road", suites=["standard"]),
            FourCliqueCountingDataset("GAP-twitter", suites=["standard"]),
            FourCliqueCountingDataset("GAP-web", suites=["standard"]),
            FourCliqueCountingDataset("GAP-kron", suites=["standard"]),
            FourCliqueCountingDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: FourCliqueCountingDataset) -> DataInstance:
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


class FourCliqueCountingBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "four_clique_counting"

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
            " edges between those 3 vertices. This constitutes a 4-clique. Only the strict"
            " upper triangle of the simple undirected adjacency is stored, so each"
            " clique is counted once, with i < j < k < l. These methods are implemented using the property"
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
            FourCliqueCountingTestGenerator(),
            FourCliqueCountingSNAPGenerator(),
            FourCliqueCountingGAPGenerator(),
        ]

    def benchmark(self, xp, meta: dict, A):
        cliq_4 = xp.einsum(
            "S[] += A[i,j] * A[i,k] * A[i,l] * A[j,k] * A[j,l] * A[k,l]",
            A=A,
        )
        return xp.asarray(cliq_4)

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
