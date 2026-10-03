# ruff: noqa: E501
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import shortest_path

from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, from_scipy

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
from saps.benchmarks.snap import (
    NUM_SNAP_SOURCES,
    fetch_snap_graph,
    seeded_source_vertices,
)
from saps_framework.binsparse_utils import binsparse_equal


class MultiSourceBreadthFirstSearchDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        sources: list[int] | None = None,
        expected: np.ndarray | None = None,
        source_name: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = (
            description or f"Multi-source breadth-first search input {name}."
        )
        self._suites = list(suites or [])
        self.A = A
        self.sources = sources
        self.source_name = source_name or name
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


def initial_frontier(n: int, sources) -> COORMatrix:
    """Boolean frontier matrix with row i set only at sources[i]."""
    sources = np.asarray(sources, dtype=np.int64)
    return COORMatrix(
        (sources.size, n),
        sources.size,
        fill=True,
        fill_value=False,
        indices_0=np.arange(sources.size, dtype=np.int64),
        indices_1=sources,
        values=np.ones(sources.size, dtype=bool),
    )


def multi_source_bfs_instance(adjacency: BinsparseTensor, sources) -> DataInstance:
    """Boolean edge pattern plus a one-hot frontier row for each source."""
    n = adjacency.shape[0]
    return DataInstance(
        inputs=[zero_one_adjacency(adjacency), initial_frontier(n, sources)],
        meta={"sources": list(sources)},
    )


def reference_levels(A, sources) -> np.ndarray:
    """levels[s, v] is one more than the hops from sources[s] to v, or 0 if unreachable."""
    distances = shortest_path(A, directed=True, unweighted=True, indices=list(sources))
    return np.where(np.isfinite(distances), distances + 1, 0).astype(int)


class MultiSourceBreadthFirstSearchTestGenerator(
    Generator[MultiSourceBreadthFirstSearchDataset]
):
    @property
    def name(self) -> str:
        return "msbfs_test"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Breadth-First Search Test"

    @property
    def description(self) -> str:
        return "Small deterministic multi-source BFS examples with reference outputs."

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
        return "These tests were written with assistance from Claude."

    @property
    def motivation(self) -> str:
        return (
            "Provide small multi-source BFS examples for benchmark correctness checks."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MultiSourceBreadthFirstSearchDataset]:
        return [
            MultiSourceBreadthFirstSearchDataset(
                "basic",
                pretty_name="Basic",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0],
                    ],
                    dtype=bool,
                ),
                sources=[0, 2, 5],
                expected=np.array(
                    [
                        [1, 2, 2, 3, 3, 4],
                        [0, 0, 1, 4, 2, 3],
                        [0, 0, 0, 2, 0, 1],
                    ],
                    dtype=int,
                ),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "single_node",
                pretty_name="Single Node",
                suites=["test"],
                A=np.array([[0]], dtype=bool),
                sources=[0],
                expected=np.array([[1]], dtype=int),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "disconnected",
                pretty_name="Disconnected",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                sources=[0, 2],
                expected=np.array([[1, 2, 0, 0], [0, 0, 1, 2]], dtype=int),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "undirected",
                pretty_name="Undirected",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=bool,
                ),
                sources=[0, 3, 1],
                expected=np.array(
                    [[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3]], dtype=int
                ),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "cycle_all_sources",
                pretty_name="Cycle All Sources",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                sources=[0, 1, 2, 3],
                expected=np.array(
                    [[1, 2, 3, 4], [4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1]],
                    dtype=int,
                ),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "repeated_source",
                pretty_name="Repeated Source",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                sources=[1, 1, 0],
                expected=np.array([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dtype=int),
            ),
            MultiSourceBreadthFirstSearchDataset(
                "signed_edges",
                pretty_name="Signed Edges",
                description=(
                    "Opposite-signed edges into vertex 3 must not cancel when"
                    " 1 and 2 share a frontier."
                ),
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, -1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, -1],
                        [0, 0, 0, 0],
                    ],
                    dtype=int,
                ),
                sources=[0],
                expected=np.array([[1, 2, 2, 3]], dtype=int),
            ),
            # Sources are picked as the SNAP generator picks them.
            MultiSourceBreadthFirstSearchDataset(
                "snap_sources",
                pretty_name="SNAP Sources",
                suites=["test"],
                A=np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
            ),
        ]

    def generate(self, dataset: MultiSourceBreadthFirstSearchDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("Multi-source BFS test datasets must define A.")
        adjacency = from_scipy(coo_array(dataset.A))
        sources = dataset.sources
        if sources is None:
            sources = np.unique(
                seeded_source_vertices(adjacency, NUM_SNAP_SOURCES)
            ).tolist()
        expected = dataset.expected
        if expected is None:
            expected = reference_levels(dataset.A, sources)
        problem = multi_source_bfs_instance(adjacency, sources)
        problem.ref_outputs = [from_numpy(expected)]
        return problem


class MultiSourceBreadthFirstSearchSNAPGenerator(
    Generator[MultiSourceBreadthFirstSearchDataset]
):
    @property
    def name(self) -> str:
        return "msbfs_snap"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Breadth-First Search SNAP"

    @property
    def description(self) -> str:
        return (
            "SNAP input generator for multi-source breadth-first search, searching"
            " from the SNAP shell's seeded sources for each graph, deduplicated."
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
        return []

    @property
    def ai_disclosure(self) -> str:
        return "This generator was written with assistance from Claude."

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for multi-source breadth-first search."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MultiSourceBreadthFirstSearchDataset]:
        # fmt: off
        return [
            MultiSourceBreadthFirstSearchDataset("soc-Epinions1", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-LiveJournal1", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("soc-Pokec", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("soc-Slashdot0811", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-Slashdot0902", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("wiki-Vote", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("wiki-RfA", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("com-LiveJournal", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("com-Friendster", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("com-Orkut", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("com-Youtube", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("com-DBLP", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("com-Amazon", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("email-Eu-core", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("wiki-topcats", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("email-EuAll", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("email-Enron", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("wiki-Talk", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("cit-HepPh", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("cit-HepTh", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("cit-Patents", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("ca-AstroPh", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("ca-CondMat", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("ca-GrQc", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("ca-HepPh", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("ca-HepTh", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("web-BerkStan", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("web-Google", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("web-NotreDame", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("web-Stanford", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("amazon0302", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("amazon0312", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("amazon0505", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("amazon0601", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("p2p-Gnutella31", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("roadNet-CA", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("roadNet-PA", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("roadNet-TX", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("as-735", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("as-Skitter", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("as-caida", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("Oregon-1", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("Oregon-2", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-epinions", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-Slashdot081106", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-Slashdot090216", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("soc-sign-Slashdot090221", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("loc-Gowalla", suites=["standard", "trace", "train"]),
            MultiSourceBreadthFirstSearchDataset("loc-Brightkite", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("sx-stackoverflow", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("sx-mathoverflow", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("sx-superuser", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("sx-askubuntu", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("wiki-talk-temporal", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("CollegeMsg", suites=["standard", "trace"]),
            MultiSourceBreadthFirstSearchDataset("twitter7", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: MultiSourceBreadthFirstSearchDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.source_name)
        sources = np.unique(raw.meta["sources"]).tolist()
        return multi_source_bfs_instance(raw.inputs[0], sources)


class MultiSourceBreadthFirstSearchGAPGenerator(
    Generator[MultiSourceBreadthFirstSearchDataset]
):
    @property
    def name(self) -> str:
        return "msbfs_gap"

    @property
    def pretty_name(self) -> str:
        return "Multi-Source Breadth-First Search GAP"

    @property
    def description(self) -> str:
        return (
            "GAP input generator for multi-source breadth-first search, searching"
            " from every published GAP source of each graph at once."
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
        return "This generator was written with assistance from Claude."

    @property
    def motivation(self) -> str:
        return "Generate GAP graph inputs for multi-source breadth-first search."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MultiSourceBreadthFirstSearchDataset]:
        # fmt: off
        return [
            MultiSourceBreadthFirstSearchDataset("GAP-road", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("GAP-twitter", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("GAP-web", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("GAP-kron", suites=["standard"]),
            MultiSourceBreadthFirstSearchDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: MultiSourceBreadthFirstSearchDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.source_name)
        return multi_source_bfs_instance(raw.inputs[0], raw.meta["sources"])


class MultiSourceBreadthFirstSearchBenchmark(Benchmark):
    @property
    def name(self):
        return "msbfs"

    @property
    def pretty_name(self):
        return "Multi-Source Breadth-First Search"

    @property
    def description(self):
        return (
            "Runs one breadth-first search from each of several source vertices at"
            " once. The frontiers of all searches are stacked into a sparse"
            " source-by-vertex boolean matrix, so each step of every search is"
            " advanced together by one sparse matrix-matrix product with the"
            " adjacency matrix."
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
            Contributor("Willow Ahrens", "willow.marie.ahrens@gmail.com"),
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
                city="Philadelphia",
                year=2011,
            ),
            Ref(
                title="The More the Merrier",
                authors=[
                    Author("Manuel Then"),
                    Author("Moritz Kaufmann"),
                    Author("Fernando Chirigati"),
                    Author("Tuan-Anh Hoang-Vu"),
                    Author("Kien Pham"),
                    Author("Alfons Kemper"),
                    Author("Thomas Neumann"),
                    Author("Huy T. Vo"),
                ],
                journal="Proceedings of the VLDB Endowment",
                volume=8,
                number=4,
                pages="449-460",
                year=2014,
                doi="10.14778/2735496.2735507",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI was used to construct tests and harnesses. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "Batching many searches into one traversal is how graph analytics such"
            " as betweenness centrality, closeness centrality, and all-pairs"
            " reachability amortize work over sources. The frontier becomes a sparse"
            " matrix whose rows can have very different densities, and each step is"
            " a masked sparse matrix-matrix product, so performance depends on"
            " choosing formats and loop orders that exploit sparsity in both the"
            " frontier and the graph."
        )

    @property
    def generators(self):
        return [
            MultiSourceBreadthFirstSearchSNAPGenerator(),
            MultiSourceBreadthFirstSearchTestGenerator(),
            MultiSourceBreadthFirstSearchGAPGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        """
        Returns levels, where level[s, v] is one more than the number of hops
        from meta["sources"][s] to v, or 0 if v is unreachable from it.
        """
        edges = data[0]
        frontier = data[1]

        (n, m) = edges.shape
        assert n == m
        (k, m) = frontier.shape
        assert n == m
        visited = xp.zeros((k, n), dtype=bool)
        level = xp.zeros((k, n), dtype=int)
        level_idx = 1
        frontier_count = xp.sum(frontier)
        while frontier_count > 0:
            level = xp.where(frontier, level_idx, level)
            visited = xp.logical_or(visited, frontier)
            frontier = xp.einsum(
                "frontier[s, j] or= frontier[s, i] & edges[i, j]",
                edges=edges,
                frontier=frontier,
            )
            frontier = xp.logical_and(frontier, xp.logical_not(visited))
            frontier_count = xp.sum(frontier)

            level_idx += 1

        return [level]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert binsparse_equal(self._output[0], self._ref_outputs[0]), (
            f"Multi-source BFS output mismatch for {param.dataset.name}"
        )
