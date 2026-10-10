# ruff: noqa: E501
import numpy as np

import sparse as sp
from binsparse import BinsparseTensor, COORMatrix
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
from saps.util.adjacency import distance_matrix
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


class MSSPDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        symmetrize: bool = False,
        A=None,
        sources: list[int] | None = None,
        expected=None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Multi-source shortest paths input {name}."
        self._suites = list(suites or [])
        self.symmetrize = symmetrize
        self.A = A
        self.sources = sources
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


def initial_distances(n: int, sources) -> COORMatrix:
    """Distance matrix with row i zero at sources[i] and infinite elsewhere."""
    sources = np.asarray(sources, dtype=np.int64)
    return COORMatrix(
        (sources.size, n),
        sources.size,
        fill=True,
        fill_value=np.inf,
        indices_0=np.arange(sources.size, dtype=np.int64),
        indices_1=sources,
        values=np.zeros(sources.size, dtype=np.float64),
    )


def multi_source_instance(
    adjacency: BinsparseTensor, sources, *, symmetrize: bool = False
) -> DataInstance:
    """Unweighted distance graph plus initial distances from each source."""
    n = adjacency.shape[0]
    return DataInstance(
        inputs=[
            distance_matrix(adjacency, symmetrize=symmetrize),
            initial_distances(n, sources),
        ],
        meta={"sources": list(sources)},
    )


def all_pairs_reference(A):
    if isinstance(A, sp.SparseArray):
        A = A.todense()
    expected = A.copy()
    for k in range(expected.shape[0]):
        expected = np.minimum(
            expected,
            np.expand_dims(expected[:, k], axis=1)
            + np.expand_dims(expected[k, :], axis=0),
        )
    return expected


def shortest_paths_input_from_edges(
    n: int, edges: list[tuple[int, int]], *, symmetric: bool = False
):
    rows = [*range(n)]
    cols = [*range(n)]
    values = [0.0] * n
    for u, v in edges:
        rows.append(u)
        cols.append(v)
        values.append(1.0)
        if symmetric:
            rows.append(v)
            cols.append(u)
            values.append(1.0)
    return sp.COO(
        coords=np.array([rows, cols], dtype=np.int64),
        data=np.array(values, dtype=np.float64),
        shape=(n, n),
        fill_value=np.inf,
    )


class MSSPTestGenerator(Generator[MSSPDataset]):
    @property
    def name(self) -> str:
        return "mssp_test"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths (MSSP) Test"

    @property
    def description(self) -> str:
        return "Small deterministic Multi-Source Shortest Paths examples."

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
        return "Provide small graph examples for shortest-path correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MSSPDataset]:
        return [
            MSSPDataset(
                name="single_node",
                pretty_name="Single Node",
                description="Multi-Source Shortest Paths test case single-node.",
                suites=["test"],
                A=np.array([[0.0]]),
                expected=np.array([[0.0]]),
            ),
            MSSPDataset(
                name="two_node_directed",
                pretty_name="Two Node Directed",
                description="Multi-Source Shortest Paths test case two-node-directed.",
                suites=["test"],
                A=np.array([[0.0, 1.0], [np.inf, 0.0]]),
                expected=np.array([[0.0, 1.0], [np.inf, 0.0]]),
            ),
            MSSPDataset(
                name="three_node_chain",
                pretty_name="Three Node Chain",
                description="Multi-Source Shortest Paths test case three-node-chain.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, np.inf],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            MSSPDataset(
                name="three_node_shortcut",
                pretty_name="Three Node Shortcut",
                description="Multi-Source Shortest Paths test case three-node-shortcut.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, 5.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            MSSPDataset(
                name="two_components",
                pretty_name="Two Components",
                description="Multi-Source Shortest Paths test case two-components.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
            ),
            MSSPDataset(
                name="large_symmetric",
                pretty_name="Large Symmetric",
                description="Multi-Source Shortest Paths test case large-symmetric.",
                suites=["test"],
                A=shortest_paths_input_from_edges(
                    39,
                    [
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (0, 7),
                        (0, 8),
                        (1, 9),
                        (1, 10),
                        (1, 11),
                        (1, 12),
                        (1, 13),
                        (1, 14),
                        (1, 15),
                        (1, 16),
                        (2, 9),
                        (2, 10),
                        (2, 17),
                        (2, 18),
                        (3, 11),
                        (3, 12),
                        (3, 19),
                        (3, 20),
                        (3, 21),
                        (4, 13),
                        (4, 22),
                        (4, 23),
                        (4, 24),
                        (5, 14),
                        (5, 22),
                        (5, 25),
                        (5, 26),
                        (6, 15),
                        (6, 23),
                        (6, 27),
                        (6, 28),
                        (7, 16),
                        (7, 24),
                        (7, 29),
                        (7, 30),
                        (8, 17),
                        (8, 18),
                        (8, 19),
                        (8, 20),
                        (8, 21),
                        (9, 22),
                        (9, 31),
                        (9, 32),
                        (10, 23),
                        (10, 31),
                        (10, 33),
                        (11, 24),
                        (11, 32),
                        (11, 34),
                        (12, 25),
                        (12, 26),
                        (12, 35),
                        (13, 27),
                        (13, 36),
                        (14, 28),
                        (14, 37),
                        (15, 29),
                        (15, 38),
                        (16, 30),
                        (17, 31),
                        (18, 32),
                        (19, 33),
                        (20, 34),
                        (21, 35),
                        (22, 36),
                        (23, 37),
                        (24, 38),
                        (25, 27),
                        (25, 29),
                        (26, 28),
                        (26, 30),
                        (27, 31),
                        (27, 33),
                        (28, 32),
                        (28, 34),
                        (29, 35),
                        (30, 36),
                        (31, 37),
                        (32, 38),
                        (33, 35),
                        (34, 36),
                        (35, 37),
                        (36, 38),
                        (37, 38),
                    ],
                    symmetric=True,
                ),
                sources=[0, 7, 21, 38, 21],
            ),
        ]

    def generate(self, dataset: MSSPDataset):
        inputs = (
            dataset.A.todense() if isinstance(dataset.A, sp.SparseArray) else dataset.A
        )
        n = inputs.shape[0]
        sources = dataset.sources if dataset.sources is not None else list(range(n))
        expected = dataset.expected
        if expected is None:
            expected = all_pairs_reference(inputs)[sources, :]
        return DataInstance(
            inputs=[from_numpy(inputs), initial_distances(n, sources)],
            meta={"sources": list(sources)},
            ref_outputs=[from_numpy(expected)],
        )


class MSSPGAPGenerator(Generator[MSSPDataset]):
    @property
    def name(self) -> str:
        return "mssp_gap"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths (MSSP) GAP"

    @property
    def description(self) -> str:
        return (
            "Data is collected from the SuiteSparse Matrix Collection and standard"
            " benchmark graph datasets, with sparse adjacency matrices converted into"
            " unweighted all-pairs shortest path inputs. This generator uses real-world"
            " networks, including the Chesapeake road network and soc-tribes network"
            " from the Network Repository."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Aarav Joglekar", "ajoglekar32@gatech.edu"),
            Contributor("Joel Mathew Cherian", "jcherian32@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=("Graph Algorithms in the Language of Linear Algebra"),
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics (SIAM)",
                year=2011,
            ),
            Ref(
                title=(
                    "The Network Data Repository with Interactive"
                    " Graph Analytics and Visualization"
                ),
                authors=[
                    Author("Ryan A. Rossi"),
                    Author("Nesreen K. Ahmed"),
                ],
                journal="AAAI",
                url="https://networkrepository.com",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[MSSPDataset]:
        # fmt: off
        return [
            MSSPDataset("GAP-road", symmetrize=False, suites=["standard"]),
            MSSPDataset("GAP-twitter", symmetrize=True, suites=["standard"]),
            MSSPDataset("GAP-web", symmetrize=True, suites=["standard"]),
            MSSPDataset("GAP-kron", symmetrize=False, suites=["standard"]),
            MSSPDataset("GAP-urand", symmetrize=False, suites=["standard"]),
        ]
        # fmt: on

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: MSSPDataset):
        raw = fetch_gap_graph(dataset.name)
        n, m = raw.inputs[0].shape
        if n != m:
            raise ValueError(
                f"Multi-Source Shortest Paths requires a square matrix, got {(n, m)}"
            )

        return multi_source_instance(
            raw.inputs[0], raw.meta["sources"], symmetrize=dataset.symmetrize
        )


class MSSPSNAPGenerator(Generator[MSSPDataset]):
    @property
    def name(self) -> str:
        return "mssp_snap"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths (MSSP) SNAP"

    @property
    def description(self) -> str:
        return (
            "SNAP input generator for multi-source shortest paths, with"
            " the SNAP shell's seeded sources for each graph, deduplicated."
        )

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate unweighted SNAP graph inputs for multi-source shortest paths."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MSSPDataset]:
        # fmt: off
        return [
            MSSPDataset("soc-Epinions1", suites=["standard"]),
            MSSPDataset("soc-LiveJournal1", suites=["standard"]),
            MSSPDataset("soc-Pokec", suites=["standard"]),
            MSSPDataset("soc-Slashdot0811", suites=["standard"]),
            MSSPDataset("soc-Slashdot0902", suites=["standard"]),
            MSSPDataset("wiki-Vote", suites=["standard"]),
            MSSPDataset("wiki-RfA", suites=["standard"]),
            MSSPDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            MSSPDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace", "train"]),
            MSSPDataset("com-LiveJournal", suites=["standard"]),
            MSSPDataset("com-Friendster", suites=["standard"]),
            MSSPDataset("com-Orkut", suites=["standard"]),
            MSSPDataset("com-Youtube", suites=["standard"]),
            MSSPDataset("com-DBLP", suites=["standard"]),
            MSSPDataset("com-Amazon", suites=["standard"]),
            MSSPDataset("email-Eu-core", suites=["standard", "trace"]),
            MSSPDataset("wiki-topcats", suites=["standard"]),
            MSSPDataset("email-EuAll", suites=["standard"]),
            MSSPDataset("email-Enron", suites=["standard"]),
            MSSPDataset("wiki-Talk", suites=["standard"]),
            MSSPDataset("cit-HepPh", suites=["standard"]),
            MSSPDataset("cit-HepTh", suites=["standard"]),
            MSSPDataset("cit-Patents", suites=["standard"]),
            MSSPDataset("ca-AstroPh", suites=["standard"]),
            MSSPDataset("ca-CondMat", suites=["standard"]),
            MSSPDataset("ca-GrQc", suites=["standard"]),
            MSSPDataset("ca-HepPh", suites=["standard"]),
            MSSPDataset("ca-HepTh", suites=["standard"]),
            MSSPDataset("web-BerkStan", suites=["standard"]),
            MSSPDataset("web-Google", suites=["standard"]),
            MSSPDataset("web-NotreDame", suites=["standard"]),
            MSSPDataset("web-Stanford", suites=["standard"]),
            MSSPDataset("amazon0302", suites=["standard"]),
            MSSPDataset("amazon0312", suites=["standard"]),
            MSSPDataset("amazon0505", suites=["standard"]),
            MSSPDataset("amazon0601", suites=["standard"]),
            MSSPDataset("p2p-Gnutella04", suites=["standard"]),
            MSSPDataset("p2p-Gnutella05", suites=["standard"]),
            MSSPDataset("p2p-Gnutella06", suites=["standard"]),
            MSSPDataset("p2p-Gnutella08", suites=["standard"]),
            MSSPDataset("p2p-Gnutella09", suites=["standard"]),
            MSSPDataset("p2p-Gnutella24", suites=["standard"]),
            MSSPDataset("p2p-Gnutella25", suites=["standard"]),
            MSSPDataset("p2p-Gnutella30", suites=["standard"]),
            MSSPDataset("p2p-Gnutella31", suites=["standard"]),
            MSSPDataset("roadNet-CA", suites=["standard"]),
            MSSPDataset("roadNet-PA", suites=["standard"]),
            MSSPDataset("roadNet-TX", suites=["standard"]),
            MSSPDataset("as-735", suites=["standard"]),
            MSSPDataset("as-Skitter", suites=["standard"]),
            MSSPDataset("as-caida", suites=["standard"]),
            MSSPDataset("Oregon-1", suites=["standard"]),
            MSSPDataset("Oregon-2", suites=["standard"]),
            MSSPDataset("soc-sign-epinions", suites=["standard"]),
            MSSPDataset("soc-sign-Slashdot081106", suites=["standard"]),
            MSSPDataset("soc-sign-Slashdot090216", suites=["standard"]),
            MSSPDataset("soc-sign-Slashdot090221", suites=["standard"]),
            MSSPDataset("loc-Gowalla", suites=["standard"]),
            MSSPDataset("loc-Brightkite", suites=["standard"]),
            MSSPDataset("sx-stackoverflow", suites=["standard"]),
            MSSPDataset("sx-mathoverflow", suites=["standard"]),
            MSSPDataset("sx-superuser", suites=["standard"]),
            MSSPDataset("sx-askubuntu", suites=["standard"]),
            MSSPDataset("wiki-talk-temporal", suites=["standard"]),
            MSSPDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            MSSPDataset("CollegeMsg", suites=["standard", "trace"]),
            MSSPDataset("twitter7", suites=["standard"]),
            MSSPDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: MSSPDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        sources = np.unique(raw.meta["sources"]).tolist()
        return multi_source_instance(raw.inputs[0], sources)


class MSSPBenchmark(Benchmark):
    @property
    def name(self):
        return "mssp"

    @property
    def pretty_name(self):
        return "Multi-Source Shortest Paths (MSSP)"

    @property
    def description(self):
        return (
            "Computes shortest paths from a list of source vertices to every vertex"
            " in a weighted directed graph by repeated min-plus relaxation of a"
            " source-by-vertex distance matrix."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard-graphs-iterative"]

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
        return [
            Contributor("Aarav Joglekar", "ajoglekar32@gatech.edu"),
            Contributor("Joel Mathew Cherian", "jcherian32@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=("Graph Algorithms in the Language of Linear Algebra"),
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics (SIAM)",
                publisher="SIAM",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "Sparse graphs reduce unnecessary computation, as most entries in the"
            " adjacency matrix represent non-edges and begin as inifinity. Efficient"
            " sparse representations allow the backend framework to skip work and"
            " minimize memory movement during the relaxation steps of the algorithm."
        )

    @property
    def generators(self):
        return [
            MSSPTestGenerator(),
            MSSPGAPGenerator(),
            MSSPSNAPGenerator(),
        ]

    def benchmark(self, xp, meta, G, D):
        """
        Returns multi-source shortest paths, i.e. D[s, j] is the shortest path
        from meta["sources"][s] to j
        """
        n, m = G.shape
        assert n == m
        for _ in range(n):
            D_new = xp.einsum("D[s, j] min= D[s, k] + G[k, j]", D=D, G=G)
            stop = xp.all(D_new == D)
            D = D_new
            if stop:
                break
        return D

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        output = to_numpy(self._output[0])
        if self._ref_outputs is not None:
            expected = to_numpy(self._ref_outputs[0])
            both_inf = np.isinf(output) & np.isinf(expected)
            both_finite = np.isfinite(output) & np.isfinite(expected)
            assert np.all(both_inf | (both_finite & (output == expected))), (
                f"Multi-Source Shortest Paths output mismatch for {param.dataset.name}"
            )
