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
                    dtype=np.int64,
                ),
                expected=1,
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
                    dtype=np.int64,
                ),
                expected=0,
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
                    dtype=np.int64,
                ),
                expected=4,
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
                    dtype=np.int64,
                ),
                expected=0,
            ),
        ]

    def generate(self, dataset: TriangleCountingDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("Triangle-count test datasets must define A.")
        return DataInstance(
            inputs=[from_numpy(np.triu(dataset.A, k=1).astype(np.int64))],
            meta={},
            ref_outputs=[from_numpy(np.asarray(dataset.expected, dtype=np.int64))],
        )


class TriangleCountingSNAPGenerator(Generator[TriangleCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_counting_snap"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting SNAP"

    @property
    def description(self) -> str:
        return (
            "Simple undirected SNAP input generator for triangle counting benchmarks."
        )

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
                title="GraphMiner: Triangle Counting reference counts",
                authors=[Author("Xuhao Chen")],
                url="https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/triangle/README.md",
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
        # Simple undirected counts: prefer GraphMiner for matching graphs.
        # GraphMiner source and aliases are documented in references and README.md.
        # Other counts are from https://snap.stanford.edu/data/<name>.html,
        # except the local computations marked below.
        # fmt: off
        return [
            TriangleCountingDataset("soc-Epinions1", expected=1_624_481, suites=["standard", "trace"]),
            # GraphMiner: livej (not com-LiveJournal).
            TriangleCountingDataset("soc-LiveJournal1", expected=285_730_264, suites=["standard"]),
            TriangleCountingDataset("soc-Pokec", expected=32_557_458, suites=["standard"]),
            TriangleCountingDataset("soc-Slashdot0811", expected=551_724, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-Slashdot0902", expected=602_592, suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-Vote", expected=608_389, suites=["standard", "trace"]),
            # Computed: SuiteSparse includes neutral votes; SNAP's total excludes them.
            # https://sparse.tamu.edu/SNAP/wiki-RfA
            TriangleCountingDataset("wiki-RfA", expected=1_416_059, suites=["standard", "trace"]),
            # Computed with scripts/compute_graph_counts.py (both Bitcoin graphs).
            TriangleCountingDataset("soc-sign-bitcoin-otc", expected=33_493, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-bitcoin-alpha", expected=22_153, suites=["standard", "trace"]),
            TriangleCountingDataset("com-LiveJournal", expected=177_820_130, suites=["standard"]),
            # GraphMiner: friendster.
            TriangleCountingDataset("com-Friendster", expected=4_173_724_142, suites=["standard"]),
            # GraphMiner: com-Orkut.
            TriangleCountingDataset("com-Orkut", expected=627_584_181, suites=["standard"]),
            # GraphMiner's youtube is a different graph.
            TriangleCountingDataset("com-Youtube", expected=3_056_386, suites=["standard"]),
            TriangleCountingDataset("com-DBLP", expected=2_224_385, suites=["standard", "trace"]),
            TriangleCountingDataset("com-Amazon", expected=667_129, suites=["standard", "trace"]),
            TriangleCountingDataset("email-Eu-core", expected=105_461, suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-topcats", expected=52_106_893, suites=["standard"]),
            TriangleCountingDataset("email-EuAll", expected=267_313, suites=["standard", "trace"]),
            TriangleCountingDataset("email-Enron", expected=727_044, suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-Talk", expected=9_203_519, suites=["standard"]),
            TriangleCountingDataset("cit-HepPh", expected=1_276_868, suites=["standard", "trace"]),
            TriangleCountingDataset("cit-HepTh", expected=1_478_735, suites=["standard", "trace"]),
            # GraphMiner: cit-Patents (not patent_citations).
            TriangleCountingDataset("cit-Patents", expected=7_515_023, suites=["standard"]),
            TriangleCountingDataset("ca-AstroPh", expected=1_351_441, suites=["standard", "trace"]),
            TriangleCountingDataset("ca-CondMat", expected=173_361, suites=["standard", "trace"]),
            TriangleCountingDataset("ca-GrQc", expected=48_260, suites=["standard", "trace"]),
            TriangleCountingDataset("ca-HepPh", expected=3_358_499, suites=["standard", "trace"]),
            TriangleCountingDataset("ca-HepTh", expected=28_339, suites=["standard", "trace"]),
            TriangleCountingDataset("web-BerkStan", expected=64_690_980, suites=["standard"]),
            TriangleCountingDataset("web-Google", expected=13_391_903, suites=["standard", "trace"]),
            TriangleCountingDataset("web-NotreDame", expected=8_910_005, suites=["standard", "trace", "train"]),
            TriangleCountingDataset("web-Stanford", expected=11_329_473, suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0302", expected=717_719, suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0312", expected=3_686_467, suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0505", expected=3_951_063, suites=["standard", "trace"]),
            TriangleCountingDataset("amazon0601", expected=3_986_507, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella04", expected=934, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella05", expected=1_112, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella06", expected=1_142, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella08", expected=2_383, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella09", expected=2_354, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella24", expected=986, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella25", expected=806, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella30", expected=1_590, suites=["standard", "trace"]),
            TriangleCountingDataset("p2p-Gnutella31", expected=2_024, suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-CA", expected=120_676, suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-PA", expected=67_150, suites=["standard", "trace"]),
            TriangleCountingDataset("roadNet-TX", expected=82_869, suites=["standard", "trace"]),
            # https://snap.stanford.edu/data/as-733.html (January 02 2000)
            TriangleCountingDataset("as-735", expected=6_584, suites=["standard", "trace"]),
            TriangleCountingDataset("as-Skitter", expected=28_769_868, suites=["standard"]),
            # https://snap.stanford.edu/data/as-Caida.html (November 05 2007)
            TriangleCountingDataset("as-caida", expected=36_365, suites=["standard", "trace"]),
            # https://snap.stanford.edu/data/oregon1.html (May 26 2001)
            TriangleCountingDataset("Oregon-1", expected=19_894, suites=["standard", "trace"]),
            # https://snap.stanford.edu/data/oregon2.html (May 26 2001)
            TriangleCountingDataset("Oregon-2", expected=89_541, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-epinions", expected=4_910_076, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot081106", expected=548_054, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot090216", expected=570_569, suites=["standard", "trace"]),
            TriangleCountingDataset("soc-sign-Slashdot090221", expected=579_565, suites=["standard", "trace"]),
            TriangleCountingDataset("loc-Gowalla", expected=2_273_138, suites=["standard"]),
            TriangleCountingDataset("loc-Brightkite", expected=494_728, suites=["standard", "trace"]),
            # Remaining SNAP counts computed with scripts/compute_graph_counts.py,
            # except twitter7, which uses GraphMiner.
            TriangleCountingDataset("sx-stackoverflow", expected=114_206_974, suites=["standard"]),
            TriangleCountingDataset("sx-mathoverflow", expected=1_403_919, suites=["standard", "trace"]),
            TriangleCountingDataset("sx-superuser", expected=1_543_161, suites=["standard", "trace"]),
            TriangleCountingDataset("sx-askubuntu", expected=680_407, suites=["standard", "trace"]),
            TriangleCountingDataset("wiki-talk-temporal", expected=8_113_676, suites=["standard"]),
            TriangleCountingDataset("email-Eu-core-temporal", expected=105_461, suites=["standard", "trace"]),
            TriangleCountingDataset("CollegeMsg", expected=14_319, suites=["standard", "trace"]),
            # GraphMiner: twitter40.
            TriangleCountingDataset("twitter7", expected=34_824_916_864, suites=["standard"]),
            TriangleCountingDataset("higgs-twitter", expected=83_023_401, suites=["standard"]),
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
            ref_outputs=[from_numpy(np.asarray(dataset.expected, dtype=np.int64))],
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
                title="GraphMiner: Triangle Counting reference counts",
                authors=[Author("Xuhao Chen")],
                url="https://github.com/chenxuhao/GraphMiner/blob/2a76e3f612e40e46a821d603ca11d10fcbc63ddd/src/triangle/README.md",
            ),
            Ref(
                title="LAGraph: Known triangle counts for GAP graphs",
                authors=[Author("Timothy A. Davis")],
                url="https://github.com/GraphBLAS/LAGraph/blob/7543de1f310371c540c23ff5da82705aa4e94ea0/src/benchmark/tc_demo.c#L22-L27",
            ),
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
                url="https://www.vldb.org/pvldb/vol13/p1190-chen.pdf",
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
        # LAGraph counts except GAP-twitter, sourced from GraphMiner below:
        # https://github.com/GraphBLAS/LAGraph/blob/7543de1f310371c540c23ff5da82705aa4e94ea0/src/benchmark/tc_demo.c#L22-L27
        # fmt: off
        return [
            TriangleCountingDataset("GAP-road", expected=438_804, suites=["standard"]),
            # GraphMiner: twitter40, equivalent after normalization:
            # https://sparse.tamu.edu/GAP/GAP-twitter
            TriangleCountingDataset("GAP-twitter", expected=34_824_916_864, suites=["standard"]),
            TriangleCountingDataset("GAP-web", expected=84_907_041_475, suites=["standard"]),
            TriangleCountingDataset("GAP-kron", expected=106_873_365_648, suites=["standard"]),
            TriangleCountingDataset("GAP-urand", expected=5_378, suites=["standard"]),
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
            ref_outputs=[from_numpy(np.asarray(dataset.expected, dtype=np.int64))],
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
