import numpy as np
import scipy.sparse as sps

import sparse as sp
from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Generator,
    Ref,
)
from saps.benchmarks.bellmanford import _adjacency_to_distance
from saps.benchmarks.snap import (
    SNAPDataset,
    SNAPGraphGenerator,
    fetch_snap_graph,
    select_source_vertices,
)
from saps.benchmarks.suitesparse import (
    _GAP_KRON_SOURCES,
    _GAP_ROAD_SOURCES,
    _GAP_TWITTER_SOURCES,
    _GAP_URAND_SOURCES,
    _GAP_WEB_SOURCES,
    SuiteSparseDataset,
    fetch_suitesparse_matrix,
)

# Number of seeded sources sampled for graphs without a published source list.
_NUM_SAMPLED_SOURCES = 64


class MultiSourceShortestPathsDataset(SuiteSparseDataset):
    def __init__(
        self,
        name,
        pretty_name,
        description,
        suites,
        source,
        symmetrize=False,
        A=None,
        sources=None,
        expected=None,
    ):
        super().__init__(
            name,
            source_name=source,
            pretty_name=pretty_name,
            description=description,
            suites=suites,
        )
        self.symmetrize = symmetrize
        self.A = A
        if sources is None and A is not None:
            sources = list(range(A.shape[0]))
        self.sources = sources
        if expected is None and A is not None:
            expected = all_pairs_reference(A)[sources, :]
        self.expected = expected


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


def sample_sources(adjacency: BinsparseTensor) -> list[int]:
    """Seeded, deduplicated sources for graphs without a published source list."""
    return np.unique(
        select_source_vertices(adjacency, _NUM_SAMPLED_SOURCES, seed=0)
    ).tolist()


def multi_source_instance(adjacency: BinsparseTensor, sources=None) -> DataInstance:
    """Unweighted distance graph plus initial distances from each source."""
    n = adjacency.shape[0]
    if sources is None:
        sources = sample_sources(adjacency)
    return DataInstance(
        inputs=[_adjacency_to_distance(adjacency), initial_distances(n, sources)],
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


class MultiSourceShortestPathsTestGenerator(Generator[MultiSourceShortestPathsDataset]):
    @property
    def name(self) -> str:
        return "multi_source_shortest_paths_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths Test Input Generator"

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
    def datasets(self) -> list[MultiSourceShortestPathsDataset]:
        return [
            MultiSourceShortestPathsDataset(
                name="single-node",
                pretty_name="single-node",
                description="Multi-Source Shortest Paths test case single-node.",
                suites=["test"],
                source="single-node",
                A=np.array([[0.0]]),
                expected=np.array([[0.0]]),
            ),
            MultiSourceShortestPathsDataset(
                name="two-node-directed",
                pretty_name="two-node-directed",
                description="Multi-Source Shortest Paths test case two-node-directed.",
                suites=["test"],
                source="two-node-directed",
                A=np.array([[0.0, 1.0], [np.inf, 0.0]]),
                expected=np.array([[0.0, 1.0], [np.inf, 0.0]]),
            ),
            MultiSourceShortestPathsDataset(
                name="three-node-chain",
                pretty_name="three-node-chain",
                description="Multi-Source Shortest Paths test case three-node-chain.",
                suites=["test"],
                source="three-node-chain",
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
            MultiSourceShortestPathsDataset(
                name="three-node-shortcut",
                pretty_name="three-node-shortcut",
                description="Multi-Source Shortest Paths test case three-node-shortcut.",
                suites=["test"],
                source="three-node-shortcut",
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
            MultiSourceShortestPathsDataset(
                name="two-components",
                pretty_name="two-components",
                description="Multi-Source Shortest Paths test case two-components.",
                suites=["test"],
                source="two-components",
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
            MultiSourceShortestPathsDataset(
                name="large-symmetric",
                pretty_name="large-symmetric",
                description="Multi-Source Shortest Paths test case large-symmetric.",
                suites=["test"],
                source="large-symmetric",
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

    def generate(self, dataset: MultiSourceShortestPathsDataset):
        inputs = (
            dataset.A.todense() if isinstance(dataset.A, sp.SparseArray) else dataset.A
        )
        n = inputs.shape[0]
        return DataInstance(
            inputs=[from_numpy(inputs), initial_distances(n, dataset.sources)],
            meta={"sources": list(dataset.sources)},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class MultiSourceShortestPathsGenerator(Generator[MultiSourceShortestPathsDataset]):
    @property
    def name(self) -> str:
        return "multi_source_shortest_paths_inputs"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths Input Generator"

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
    def datasets(self) -> list[MultiSourceShortestPathsDataset]:
        return [
            MultiSourceShortestPathsDataset(
                name="HB/bcspwr01",
                pretty_name="BCS Power Grid 01",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/bcspwr01",
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/bcspwr02",
                pretty_name="BCS Power Grid 02",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/bcspwr02",
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/bcspwr03",
                pretty_name="BCS Power Grid 03",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/bcspwr03",
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="DIMACS10/chesapeake",
                pretty_name="Chesapeake",
                description="Sparse road network input for Multi-Source Shortest Paths.",
                suites=[],
                source="DIMACS10/chesapeake",
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/ash85",
                pretty_name="ASH 85",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/ash85",
                symmetrize=False,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/arc130",
                pretty_name="ARC 130",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/arc130",
                symmetrize=False,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/bcspwr04",
                pretty_name="BCS Power Grid 04",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=[],
                source="HB/bcspwr04",
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="HB/ash292",
                pretty_name="ASH 292",
                description="Sparse SuiteSparse graph input for Multi-Source Shortest Paths.",
                suites=["trace"],
                source="HB/ash292",
                symmetrize=False,
            ),
            MultiSourceShortestPathsDataset(
                name="GAP/GAP-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=["standard"],
                source="GAP/GAP-road",
                sources=_GAP_ROAD_SOURCES,
                symmetrize=False,
            ),
            MultiSourceShortestPathsDataset(
                name="GAP/GAP-twitter",
                pretty_name="GAP Twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=["standard"],
                source="GAP/GAP-twitter",
                sources=_GAP_TWITTER_SOURCES,
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="GAP/GAP-web",
                pretty_name="GAP Web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=["standard"],
                source="GAP/GAP-web",
                sources=_GAP_WEB_SOURCES,
                symmetrize=True,
            ),
            MultiSourceShortestPathsDataset(
                name="GAP/GAP-kron",
                pretty_name="GAP Kron",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Kronecker synthetic graph generator with parameters"
                    " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                    " edges."
                ),
                suites=["standard"],
                source="GAP/GAP-kron",
                sources=_GAP_KRON_SOURCES,
                symmetrize=False,
            ),
            MultiSourceShortestPathsDataset(
                name="GAP/GAP-urand",
                pretty_name="GAP Urand",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                    " 2,147.4M edges."
                ),
                suites=["standard"],
                source="GAP/GAP-urand",
                sources=_GAP_URAND_SOURCES,
                symmetrize=False,
            ),
        ]

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: MultiSourceShortestPathsDataset):
        raw = fetch_suitesparse_matrix(dataset.source_name)
        n, m = raw.meta["shape"]
        if n != m:
            raise ValueError(f"Multi-Source Shortest Paths requires a square matrix, got {(n, m)}")

        adjacency = abs(to_scipy(raw.inputs[0]).tocoo())
        if dataset.symmetrize:
            adjacency = sps.coo_array(adjacency + adjacency.T)
        return multi_source_instance(from_scipy(adjacency), dataset.sources)


class MultiSourceShortestPathsSNAPGenerator(Generator[SNAPDataset]):
    @property
    def name(self) -> str:
        return "multi_source_shortest_paths_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Shortest Paths SNAP Input Generator"

    @property
    def description(self) -> str:
        return (
            "SNAP input generator for multi-source shortest paths, with"
            f" {_NUM_SAMPLED_SOURCES} seeded sources sampled per graph and"
            " deduplicated."
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
    def datasets(self) -> list[SNAPDataset]:
        return [
            graph.with_suites(["standard"]) for graph in SNAPGraphGenerator().datasets
        ]

    def generate(self, dataset: SNAPDataset) -> DataInstance:
        if dataset.name in self.dataset_names:
            return multi_source_instance(fetch_snap_graph(dataset.name).inputs[0])
        raise ValueError(f"Unsupported Multi-Source Shortest Paths dataset: {dataset.name}")


class MultiSourceShortestPathsBenchmark(Benchmark):
    @property
    def name(self):
        return "multi_source_shortest_paths"

    @property
    def pretty_name(self):
        return "Multi-Source Shortest Paths"

    @property
    def description(self):
        return (
            "Computes shortest paths from a list of source vertices to every vertex"
            " in a weighted directed graph by repeated min-plus relaxation of a"
            " source-by-vertex distance matrix."
        )

    @property
    def suites(self) -> list[str]:
        return ["group-graphs-iterative"]

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
            MultiSourceShortestPathsTestGenerator(),
            MultiSourceShortestPathsGenerator(),
            MultiSourceShortestPathsSNAPGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        """
        Returns multi-source shortest paths, i.e. D[s, j] is the shortest path
        from meta["sources"][s] to j
        """
        G = data[0]
        D = data[1]
        n, m = G.shape
        assert n == m
        for _ in range(n):
            D_new = xp.einsum("D[s, j] min= D[s, k] + G[k, j]", D=D, G=G)
            stop = xp.all(D_new == D)
            D = D_new
            if stop:
                break
        return [D]

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
