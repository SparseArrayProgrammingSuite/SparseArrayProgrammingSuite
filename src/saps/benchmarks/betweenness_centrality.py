# ruff: noqa: E501

import numpy as np

import igraph as ig
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.benchmarks.gap import fetch_gap_graph, gap_graph
from saps.benchmarks.snap import fetch_snap_graph, snap_graph
from saps.util.adjacency import graph_csr, zero_one_adjacency
from saps_framework.binsparse_utils import assert_binsparse_allclose


class BetweennessCentralityDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        expected: np.ndarray | None = None,
        meta: dict | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Betweenness centrality input {name}."
        self._suites = list(suites or [])
        self.A = A
        self.expected = expected
        self.meta = dict(meta or {})

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


class BetweennessCentralityGAPDataset(BetweennessCentralityDataset):
    def __init__(
        self, source_name: str, trial: int, *, suites: list[str] | None = None
    ):
        if not 0 <= trial < 16:
            raise ValueError("GAP betweenness trial must be between 0 and 15.")
        graph = gap_graph(source_name)
        self.source_name = graph.name
        self.trial = trial
        self.sources = graph.sources[4 * trial : 4 * (trial + 1)]
        super().__init__(
            name=f"{graph.name}_trial{trial}",
            pretty_name=f"{graph.name} (Trial {trial})",
            description=(
                f"GAP betweenness trial {trial} on {graph.name}, using the four"
                " published sources in order, including source dependencies and"
                " normalizing by the maximum score."
            ),
            suites=suites,
            meta={
                "bc_sources": list(self.sources),
                "include_source": True,
                "normalize": True,
            },
        )

    @property
    def metadata(self):
        return {
            **super().metadata,
            "source_name": self.source_name,
            "trial": self.trial,
            "sources": list(self.sources),
        }


class BetweennessCentralitySNAPDataset(BetweennessCentralityDataset):
    def __init__(
        self, source_name: str, trial: int, *, suites: list[str] | None = None
    ):
        if not 0 <= trial < 16:
            raise ValueError("SNAP betweenness trial must be between 0 and 15.")
        graph = snap_graph(source_name)
        self.source_name = graph.name
        self.trial = trial
        self.source_seeds = list(range(4 * trial, 4 * (trial + 1)))
        super().__init__(
            name=f"{graph.name}_trial{trial}",
            pretty_name=f"{graph.name} (Trial {trial})",
            description=(
                f"GAP-style betweenness trial {trial} on {graph.name}, using four"
                " seeded SNAP sources in order, including source dependencies and"
                " normalizing by the maximum score."
            ),
            suites=suites,
            meta={"include_source": True, "normalize": True},
        )

    @property
    def metadata(self):
        return {
            **super().metadata,
            "source_name": self.source_name,
            "trial": self.trial,
            "source_seeds": list(self.source_seeds),
        }


def random_centrality_matrix():
    rng = np.random.default_rng(42)
    n = 10
    A = (rng.random((n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0)
    return A


def undirected_path_matrix():
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


class BetweennessCentralityTestGenerator(Generator[BetweennessCentralityDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_test"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality Test"

    @property
    def description(self) -> str:
        return "Small deterministic betweenness centrality examples."

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
        return "Provide small graph examples for centrality correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BetweennessCentralityDataset]:
        random_A = random_centrality_matrix()
        undirected_A = undirected_path_matrix()
        igraph_A = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
            ],
            dtype=float,
        )
        return [
            BetweennessCentralityDataset(
                name="joels_case",
                pretty_name="Joel's Case",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=float,
                ),
                expected=np.array([0.0, 1.0, 1.0, 3.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="empty",
                pretty_name="Empty",
                suites=["test"],
                A=np.zeros((3, 3)),
                expected=np.array([0.0, 0.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="chain",
                pretty_name="Chain",
                suites=["test"],
                A=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
                expected=np.array([0.0, 1.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="two_components",
                pretty_name="Two Components",
                suites=["test"],
                A=np.array(
                    [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                    dtype=float,
                ),
                expected=np.array([0.0, 0.0, 0.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="matrix_vertex_comparison",
                pretty_name="Matrix Vertex Comparison",
                suites=["test"],
                A=random_A,
            ),
            BetweennessCentralityDataset(
                name="undirected",
                pretty_name="Undirected",
                suites=["test"],
                A=undirected_A,
            ),
            BetweennessCentralityDataset(
                name="igraph",
                pretty_name="igraph Comparison",
                suites=["test"],
                A=igraph_A,
            ),
            BetweennessCentralityDataset(
                name="snap_toy",
                pretty_name="SNAP Toy",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=float,
                ),
                expected=np.array([0.0, 1.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="gap_four_sources",
                pretty_name="GAP Four Sources",
                suites=["test"],
                A=np.diag(np.ones(4), k=1),
                meta={
                    "bc_sources": [0, 1, 3, 4],
                    "include_source": True,
                    "normalize": True,
                },
                expected=np.array([4.0, 6.0, 4.0, 3.0, 0.0]) / 6,
            ),
        ]

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("Centrality test datasets must define A.")
        expected = dataset.expected
        if expected is None:
            graph = ig.Graph(
                n=len(dataset.A),
                edges=np.column_stack(np.nonzero(dataset.A)),
                directed=True,
            )
            expected = np.asarray(graph.betweenness(directed=True))
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta=dict(dataset.meta),
            ref_outputs=[from_numpy(expected)],
        )


class BetweennessCentralitySNAPGenerator(Generator[BetweennessCentralitySNAPDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_snap"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality SNAP"

    @property
    def description(self) -> str:
        return (
            "GAP-style betweenness centrality on SNAP graphs: 16 trials per graph,"
            " each using four consecutive entries from the shell's 64 seeded"
            " sources, preserving repeated vertices. Scores include source"
            " dependencies and are normalized by their maximum."
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
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse directed graph inputs for betweenness centrality."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BetweennessCentralitySNAPDataset]:
        # fmt: off
        graphs = [
            ("soc-Epinions1", ["standard"]),
            ("soc-LiveJournal1", ["standard"]),
            ("soc-Pokec", ["standard"]),
            ("soc-Slashdot0811", ["standard"]),
            ("soc-Slashdot0902", ["standard"]),
            ("wiki-Vote", ["standard"]),
            ("wiki-RfA", ["standard"]),
            ("soc-sign-bitcoin-otc", ["standard"]),
            ("soc-sign-bitcoin-alpha", ["standard"]),
            ("com-LiveJournal", ["standard"]),
            ("com-Friendster", ["standard"]),
            ("com-Orkut", ["standard"]),
            ("com-Youtube", ["standard"]),
            ("com-DBLP", ["standard"]),
            ("com-Amazon", ["standard"]),
            ("email-Eu-core", ["standard", "trace"]),
            ("wiki-topcats", ["standard"]),
            ("email-EuAll", ["standard"]),
            ("email-Enron", ["standard"]),
            ("wiki-Talk", ["standard"]),
            ("cit-HepPh", ["standard"]),
            ("cit-HepTh", ["standard"]),
            ("cit-Patents", ["standard"]),
            ("ca-AstroPh", ["standard"]),
            ("ca-CondMat", ["standard"]),
            ("ca-GrQc", ["standard"]),
            ("ca-HepPh", ["standard"]),
            ("ca-HepTh", ["standard"]),
            ("web-BerkStan", ["standard"]),
            ("web-Google", ["standard"]),
            ("web-NotreDame", ["standard"]),
            ("web-Stanford", ["standard"]),
            ("amazon0302", ["standard"]),
            ("amazon0312", ["standard"]),
            ("amazon0505", ["standard"]),
            ("amazon0601", ["standard"]),
            ("p2p-Gnutella04", ["standard"]),
            ("p2p-Gnutella05", ["standard"]),
            ("p2p-Gnutella06", ["standard"]),
            ("p2p-Gnutella08", ["standard"]),
            ("p2p-Gnutella09", ["standard"]),
            ("p2p-Gnutella24", ["standard"]),
            ("p2p-Gnutella25", ["standard"]),
            ("p2p-Gnutella30", ["standard"]),
            ("p2p-Gnutella31", ["standard"]),
            ("roadNet-CA", ["standard"]),
            ("roadNet-PA", ["standard"]),
            ("roadNet-TX", ["standard"]),
            ("as-735", ["standard"]),
            ("as-Skitter", ["standard"]),
            ("as-caida", ["standard"]),
            ("Oregon-1", ["standard"]),
            ("Oregon-2", ["standard"]),
            ("soc-sign-epinions", ["standard"]),
            ("soc-sign-Slashdot081106", ["standard"]),
            ("soc-sign-Slashdot090216", ["standard"]),
            ("soc-sign-Slashdot090221", ["standard"]),
            ("loc-Gowalla", ["standard"]),
            ("loc-Brightkite", ["standard"]),
            ("sx-stackoverflow", ["standard"]),
            ("sx-mathoverflow", ["standard"]),
            ("sx-superuser", ["standard"]),
            ("sx-askubuntu", ["standard"]),
            ("wiki-talk-temporal", ["standard"]),
            ("email-Eu-core-temporal", ["standard", "trace", "train"]),
            ("CollegeMsg", ["standard"]),
            ("twitter7", ["standard"]),
            ("higgs-twitter", ["standard"]),
        ]
        # fmt: on
        return [
            BetweennessCentralitySNAPDataset(name, trial, suites=suites)
            for name, suites in graphs
            for trial in range(16)
        ]

    def generate(self, dataset: BetweennessCentralitySNAPDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = BetweennessCentralitySNAPReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(
        self, dataset: BetweennessCentralitySNAPDataset
    ) -> DataInstance:
        raw = fetch_snap_graph(dataset.source_name)
        start = 4 * dataset.trial
        sources = list(raw.meta["sources"][start : start + 4])
        if len(sources) != 4:
            raise ValueError("SNAP betweenness trials require four source vertices.")
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])],
            meta={
                **raw.meta,
                **dataset.meta,
                "bc_sources": sources,
                "trial": dataset.trial,
            },
        )


class BetweennessCentralitySNAPReferenceGenerator(
    ReferenceGenerator[BetweennessCentralitySNAPDataset]
):
    @property
    def name(self) -> str:
        return "betweenness_centrality_snap_reference"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality SNAP Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Betweenness Centrality SNAP, cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[BetweennessCentralitySNAPDataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in BetweennessCentralitySNAPGenerator().datasets
        ]

    def generate(
        self, dataset: ReferenceDataset[BetweennessCentralitySNAPDataset]
    ) -> DataInstance:
        problem = BetweennessCentralitySNAPGenerator().generate_inputs(dataset.dataset)
        meta = problem.meta
        adjacency = graph_csr(problem.inputs[0], boolean=True).tocoo()
        graph = ig.Graph(
            n=adjacency.shape[0],
            edges=np.column_stack((adjacency.row, adjacency.col)),
            directed=True,
        )
        sources = meta.get("bc_sources", range(graph.vcount()))
        scores = np.asarray(graph.betweenness(sources=sources, directed=True))
        if meta.get("include_source", False):
            for source in sources:
                scores[source] += (
                    graph.neighborhood_size(
                        int(source), order=graph.vcount(), mode="out"
                    )
                    - 1
                )
        if meta.get("normalize", False) and scores.max(initial=0) > 0:
            scores /= scores.max()
        expected = from_numpy(scores)
        problem.ref_outputs = [expected]
        problem.ref_meta = {
            **(problem.ref_meta or {}),
            "reference_backend": "igraph",
            "reference_algorithm": "betweenness_centrality",
        }
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=problem.ref_outputs,
            ref_meta=problem.ref_meta,
        )


class BetweennessCentralitySNAPReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return BetweennessCentralitySNAPReferenceGenerator()


class BetweennessCentralityGAPGenerator(Generator[BetweennessCentralityGAPDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_gap"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality GAP"

    @property
    def description(self) -> str:
        return (
            "GAP betweenness centrality: 16 trials per graph, with four published"
            " sources per trial. Scores include source dependencies and are"
            " normalized by their maximum, as in the GAP reference implementation."
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
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate GAP graph inputs for betweenness centrality."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BetweennessCentralityGAPDataset]:
        return [
            BetweennessCentralityGAPDataset(name, trial, suites=["standard"])
            for name in ["GAP-road", "GAP-twitter", "GAP-web", "GAP-kron", "GAP-urand"]
            for trial in range(16)
        ]

    def generate(self, dataset: BetweennessCentralityGAPDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = BetweennessCentralityGAPReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(self, dataset: BetweennessCentralityGAPDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.source_name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])],
            meta={
                **raw.meta,
                **dataset.meta,
                "bc_sources": list(dataset.sources),
                "trial": dataset.trial,
            },
        )


class BetweennessCentralityGAPReferenceGenerator(
    ReferenceGenerator[BetweennessCentralityGAPDataset]
):
    @property
    def name(self) -> str:
        return "betweenness_centrality_gap_reference"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality GAP Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Betweenness Centrality GAP, cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[BetweennessCentralityGAPDataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in BetweennessCentralityGAPGenerator().datasets
        ]

    def generate(
        self, dataset: ReferenceDataset[BetweennessCentralityGAPDataset]
    ) -> DataInstance:
        problem = BetweennessCentralityGAPGenerator().generate_inputs(dataset.dataset)
        meta = problem.meta
        adjacency = graph_csr(problem.inputs[0], boolean=True).tocoo()
        graph = ig.Graph(
            n=adjacency.shape[0],
            edges=np.column_stack((adjacency.row, adjacency.col)),
            directed=True,
        )
        sources = meta.get("bc_sources", range(graph.vcount()))
        scores = np.asarray(graph.betweenness(sources=sources, directed=True))
        if meta.get("include_source", False):
            for source in sources:
                scores[source] += (
                    graph.neighborhood_size(
                        int(source), order=graph.vcount(), mode="out"
                    )
                    - 1
                )
        if meta.get("normalize", False) and scores.max(initial=0) > 0:
            scores /= scores.max()
        expected = from_numpy(scores)
        problem.ref_outputs = [expected]
        problem.ref_meta = {
            **(problem.ref_meta or {}),
            "reference_backend": "igraph",
            "reference_algorithm": "betweenness_centrality",
        }
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=problem.ref_outputs,
            ref_meta=problem.ref_meta,
        )


class BetweennessCentralityGAPReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return BetweennessCentralityGAPReferenceGenerator()


class BetweennessCentralityBenchmark(Benchmark):
    @property
    def name(self):
        return "betweenness_centrality"

    @property
    def pretty_name(self):
        return "Betweenness Centrality"

    @property
    def description(self):
        return (
            "This code is based on the Brandes betweenness centrality algorithm. The "
            "current code for the benchmark takes a two step approach. The first step "
            "involves going layer by layer from each selected starting node to find "
            "the total amount of shortest paths that lead to a node. So for example "
            "4 -> 6 could have 3 diff shortest paths and 4 -> 2 could have only 1 "
            "shortest path. The second step is for tracing backwards to see how many "
            "times a node appears in other shortest paths. The number of times this "
            "node is in one of the shortest path divided by total shortest paths "
            "between the two edge nodes gets added to the intermediate nodes bc score. "
            "GAP and SNAP cases use four sources per trial, include source "
            "dependencies, and normalize by the maximum score. Small test cases "
            "use all sources unless bc_sources is supplied."
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
            Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=(
                    "Comparing the speed and accuracy of approaches to betweenness"
                    " centrality approximation"
                ),
                authors=[
                    Author("John Matta"),
                    Author("Gunes Ercal"),
                    Author("Koushik Sinha"),
                ],
                journal="Computational Social Networks",
                publisher="Springer Science and Business Media LLC",
                volume="6",
                number="1",
                year=2019,
                url="https://doi.org/10.1186/s40649-019-0062-5",
                doi="10.1186/s40649-019-0062-5",
            ),
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
        ]

    @property
    def ai_disclosure(self):
        return (
            "The original benchmark function was written without generative AI. "
            "Generative AI assisted with GAP and SNAP source sampling, source "
            "dependencies, normalization, self-loop handling, and tests."
        )

    @property
    def motivation(self):
        return ""

    @property
    def generators(self) -> list[Generator]:
        return [
            BetweennessCentralityTestGenerator(),
            BetweennessCentralitySNAPGenerator(),
            BetweennessCentralityGAPGenerator(),
        ]

    def benchmark(self, xp, meta, G):
        n = G.shape[0]
        bc_scores = xp.zeros((n,), dtype=float)

        for v in meta.get("bc_sources", range(n)):
            number_of_paths = xp.zeros((n,), dtype=float)
            self_dist = xp.zeros((n,), dtype=float)
            self_dist = self_dist + xp.asarray(
                [1.0 if i == v else 0.0 for i in range(n)]
            )
            number_of_paths = number_of_paths + self_dist

            # A self-loop is not part of a shortest path to another vertex.
            neighbors = xp.asarray(G[v], dtype=float) * (1 - self_dist)
            layer_traversal = []
            depth = 0

            node_count = xp.sum(neighbors)

            while node_count != 0:
                depth += 1

                layer_traversal.append(neighbors != 0)

                number_of_paths = number_of_paths + neighbors

                not_neighbors = xp.equal(number_of_paths, 0)
                next_neighbors = xp.matmul(neighbors, G) * not_neighbors

                node_count = xp.sum(next_neighbors)

                neighbors = next_neighbors

            score_update = xp.zeros((n,), dtype=float)

            while depth >= 2:
                neighbors_layer = xp.astype(layer_traversal[depth - 1], xp.float64)

                prev_layer = xp.astype(layer_traversal[depth - 2], xp.float64)

                denom = xp.maximum(number_of_paths, 1e-10)
                update_val = neighbors_layer * (1.0 + score_update) / denom

                update_val = xp.matmul(G, update_val)

                update_val = update_val * prev_layer * number_of_paths

                score_update = score_update + update_val

                depth -= 1

            if meta.get("include_source", False):
                # GAP includes the source's dependency: one per reachable target.
                score_update = score_update + self_dist * (
                    xp.sum(number_of_paths != 0) - 1
                )

            bc_scores = bc_scores + score_update

        if meta.get("normalize", False):
            biggest_score = xp.max(bc_scores)
            if biggest_score > 0:
                bc_scores = bc_scores / biggest_score

        return bc_scores

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is not None:
            assert_binsparse_allclose(
                self._ref_outputs[0], self._output[0], rtol=1e-5, atol=1e-12
            )
