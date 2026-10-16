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
from saps.benchmarks.snap import fetch_snap_graph
from saps.util.adjacency import zero_one_adjacency


class FourCliqueCountingDataset(Dataset):
    def __init__(
        self,
        name: str,
        expected: int,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
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
                    dtype=np.int64,
                ),
                expected=0,
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
                    dtype=np.int64,
                ),
                expected=1,
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
                    dtype=np.int64,
                ),
                expected=2,
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
                    dtype=np.int64,
                ),
                expected=0,
            ),
        ]

    def generate(self, dataset: FourCliqueCountingDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("4-clique test datasets must define A.")
        return DataInstance(
            inputs=[from_numpy(np.triu(dataset.A, k=1).astype(np.int64))],
            meta={},
            ref_outputs=[from_numpy(np.asarray(dataset.expected, dtype=np.int64))],
        )


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
                title="GraphMiner: k-Clique Listing reference counts",
                authors=[Author("Xuhao Chen")],
                url="https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/clique/README.md",
            ),
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
                url="https://www.vldb.org/pvldb/vol13/p1190-chen.pdf",
            ),
            Ref(
                title="Sandslash",
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
        # Simple undirected counts: prefer GraphMiner for matching graphs.
        # GraphMiner source and aliases are documented in references and README.md.
        # All other counts were computed with scripts/compute_graph_counts.py.
        # fmt: off
        return [
            FourCliqueCountingDataset("soc-Epinions1", expected=5_803_397, suites=["standard"]),
            # GraphMiner: livej (not com-LiveJournal).
            FourCliqueCountingDataset("soc-LiveJournal1", expected=9_933_532_019, suites=["standard"]),
            FourCliqueCountingDataset("soc-Pokec", expected=42_947_031, suites=["standard"]),
            FourCliqueCountingDataset("soc-Slashdot0811", expected=1_989_958, suites=["standard"]),
            FourCliqueCountingDataset("soc-Slashdot0902", expected=2_260_269, suites=["standard"]),
            FourCliqueCountingDataset("wiki-Vote", expected=2_077_903, suites=["standard"]),
            FourCliqueCountingDataset("wiki-RfA", expected=6_305_134, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-bitcoin-otc", expected=48_390, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-bitcoin-alpha", expected=30_948, suites=["standard"]),
            FourCliqueCountingDataset("com-LiveJournal", expected=5_216_918_441, suites=["standard"]),
            # GraphMiner: friendster.
            FourCliqueCountingDataset("com-Friendster", expected=8_963_503_263, suites=["standard"]),
            # GraphMiner: com-Orkut.
            FourCliqueCountingDataset("com-Orkut", expected=3_221_946_137, suites=["standard"]),
            # GraphMiner's youtube is a different graph.
            FourCliqueCountingDataset("com-Youtube", expected=4_986_965, suites=["standard"]),
            FourCliqueCountingDataset("com-DBLP", expected=16_713_192, suites=["standard"]),
            FourCliqueCountingDataset("com-Amazon", expected=275_961, suites=["standard"]),
            FourCliqueCountingDataset("email-Eu-core", expected=423_750, suites=["standard"]),
            FourCliqueCountingDataset("wiki-topcats", expected=65_479_271, suites=["standard"]),
            FourCliqueCountingDataset("email-EuAll", expected=581_032, suites=["standard"]),
            FourCliqueCountingDataset("email-Enron", expected=2_341_639, suites=["standard"]),
            FourCliqueCountingDataset("wiki-Talk", expected=64_940_189, suites=["standard"]),
            FourCliqueCountingDataset("cit-HepPh", expected=2_562_245, suites=["standard"]),
            FourCliqueCountingDataset("cit-HepTh", expected=4_113_289, suites=["standard"]),
            # GraphMiner: cit-Patents (not patent_citations).
            FourCliqueCountingDataset("cit-Patents", expected=3_501_071, suites=["standard"]),
            FourCliqueCountingDataset("ca-AstroPh", expected=9_580_415, suites=["standard"]),
            FourCliqueCountingDataset("ca-CondMat", expected=294_008, suites=["standard"]),
            FourCliqueCountingDataset("ca-GrQc", expected=329_297, suites=["standard", "trace"]),
            FourCliqueCountingDataset("ca-HepPh", expected=150_281_372, suites=["standard"]),
            FourCliqueCountingDataset("ca-HepTh", expected=65_592, suites=["standard", "trace"]),
            FourCliqueCountingDataset("web-BerkStan", expected=1_065_796_916, suites=["standard"]),
            FourCliqueCountingDataset("web-Google", expected=39_881_472, suites=["standard"]),
            FourCliqueCountingDataset("web-NotreDame", expected=231_911_102, suites=["standard"]),
            FourCliqueCountingDataset("web-Stanford", expected=78_757_781, suites=["standard"]),
            FourCliqueCountingDataset("amazon0302", expected=304_441, suites=["standard"]),
            FourCliqueCountingDataset("amazon0312", expected=3_975_712, suites=["standard"]),
            FourCliqueCountingDataset("amazon0505", expected=4_361_282, suites=["standard"]),
            FourCliqueCountingDataset("amazon0601", expected=4_420_994, suites=["standard"]),
            FourCliqueCountingDataset("p2p-Gnutella04", expected=3, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella05", expected=55, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella06", expected=45, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella08", expected=175, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella09", expected=160, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella24", expected=11, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella25", expected=7, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella30", expected=13, suites=["standard", "trace"]),
            FourCliqueCountingDataset("p2p-Gnutella31", expected=16, suites=["standard"]),
            FourCliqueCountingDataset("roadNet-CA", expected=42, suites=["standard"]),
            FourCliqueCountingDataset("roadNet-PA", expected=21, suites=["standard"]),
            FourCliqueCountingDataset("roadNet-TX", expected=32, suites=["standard"]),
            FourCliqueCountingDataset("as-735", expected=5_636, suites=["standard"]),
            FourCliqueCountingDataset("as-Skitter", expected=148_834_439, suites=["standard"]),
            FourCliqueCountingDataset("as-caida", expected=53_875, suites=["standard"]),
            FourCliqueCountingDataset("Oregon-1", expected=30_479, suites=["standard"]),
            FourCliqueCountingDataset("Oregon-2", expected=399_013, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-epinions", expected=58_567_924, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot081106", expected=1_968_248, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot090216", expected=2_078_864, suites=["standard"]),
            FourCliqueCountingDataset("soc-sign-Slashdot090221", expected=2_109_090, suites=["standard"]),
            FourCliqueCountingDataset("loc-Gowalla", expected=6_086_852, suites=["standard"]),
            FourCliqueCountingDataset("loc-Brightkite", expected=2_851_308, suites=["standard"]),
            FourCliqueCountingDataset("sx-stackoverflow", expected=787_147_026, suites=["standard"]),
            FourCliqueCountingDataset("sx-mathoverflow", expected=10_714_812, suites=["standard"]),
            FourCliqueCountingDataset("sx-superuser", expected=6_399_900, suites=["standard"]),
            FourCliqueCountingDataset("sx-askubuntu", expected=2_260_000, suites=["standard"]),
            FourCliqueCountingDataset("wiki-talk-temporal", expected=53_439_733, suites=["standard"]),
            FourCliqueCountingDataset("email-Eu-core-temporal", expected=423_750, suites=["standard"]),
            FourCliqueCountingDataset("CollegeMsg", expected=5_389, suites=["standard", "trace", "train"]),
            # GraphMiner: twitter40.
            FourCliqueCountingDataset("twitter7", expected=6_622_234_180_319, suites=["standard"]),
            FourCliqueCountingDataset("higgs-twitter", expected=429_733_013, suites=["standard"]),
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
            ref_outputs=[from_numpy(np.asarray(dataset.expected, dtype=np.int64))],
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
        assert self._ref_outputs is not None, "No reference output"
        result = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])
        assert result.shape == expected.shape == (), "Count must be a scalar"
        assert np.issubdtype(result.dtype, np.integer), (
            "Count must have an integer dtype"
        )
        assert result.item() == expected.item(), (
            f"Count mismatch: got {result.item()}, expected {expected.item()}"
        )
