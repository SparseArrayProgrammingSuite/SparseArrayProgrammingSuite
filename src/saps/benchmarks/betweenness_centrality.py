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
from saps.util.adjacency import zero_one_adjacency
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


class BetweennessCentralityDataset(Dataset):
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
        self._description = description or f"Betweenness centrality input {name}."
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


def reference_bc_alg_6_4(A):
    n = A.shape[0]
    BC = np.zeros(n)
    for s in range(n):
        stack = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1
        d = -np.ones(n)
        d[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            stack.append(v)
            for w in np.where(A[v, :] > 0)[0]:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                BC[w] += delta[w]
    return BC


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
        networkx_A = np.array(
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
                expected=reference_bc_alg_6_4(random_A),
            ),
            BetweennessCentralityDataset(
                name="undirected",
                pretty_name="Undirected",
                suites=["test"],
                A=undirected_A,
                expected=reference_bc_alg_6_4(undirected_A),
            ),
            BetweennessCentralityDataset(
                name="networkx",
                pretty_name="NetworkX Comparison",
                suites=["test"],
                A=networkx_A,
                expected=reference_bc_alg_6_4(networkx_A),
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
        ]

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("Centrality test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class BetweennessCentralitySNAPGenerator(Generator[BetweennessCentralityDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_snap"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality SNAP"

    @property
    def description(self) -> str:
        return "SNAP input generator for betweenness centrality benchmarks."

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
    def datasets(self) -> list[BetweennessCentralityDataset]:
        # fmt: off
        return [
            BetweennessCentralityDataset("soc-Epinions1", suites=["standard"]),
            BetweennessCentralityDataset("soc-LiveJournal1", suites=["standard"]),
            BetweennessCentralityDataset("soc-Pokec", suites=["standard"]),
            BetweennessCentralityDataset("soc-Slashdot0811", suites=["standard"]),
            BetweennessCentralityDataset("soc-Slashdot0902", suites=["standard"]),
            BetweennessCentralityDataset("wiki-Vote", suites=["standard"]),
            BetweennessCentralityDataset("wiki-RfA", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-bitcoin-otc", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-bitcoin-alpha", suites=["standard"]),
            BetweennessCentralityDataset("com-LiveJournal", suites=["standard"]),
            BetweennessCentralityDataset("com-Friendster", suites=["standard"]),
            BetweennessCentralityDataset("com-Orkut", suites=["standard"]),
            BetweennessCentralityDataset("com-Youtube", suites=["standard"]),
            BetweennessCentralityDataset("com-DBLP", suites=["standard"]),
            BetweennessCentralityDataset("com-Amazon", suites=["standard"]),
            BetweennessCentralityDataset("email-Eu-core", suites=["standard", "trace"]),
            BetweennessCentralityDataset("wiki-topcats", suites=["standard"]),
            BetweennessCentralityDataset("email-EuAll", suites=["standard"]),
            BetweennessCentralityDataset("email-Enron", suites=["standard"]),
            BetweennessCentralityDataset("wiki-Talk", suites=["standard"]),
            BetweennessCentralityDataset("cit-HepPh", suites=["standard"]),
            BetweennessCentralityDataset("cit-HepTh", suites=["standard"]),
            BetweennessCentralityDataset("cit-Patents", suites=["standard"]),
            BetweennessCentralityDataset("ca-AstroPh", suites=["standard"]),
            BetweennessCentralityDataset("ca-CondMat", suites=["standard"]),
            BetweennessCentralityDataset("ca-GrQc", suites=["standard"]),
            BetweennessCentralityDataset("ca-HepPh", suites=["standard"]),
            BetweennessCentralityDataset("ca-HepTh", suites=["standard"]),
            BetweennessCentralityDataset("web-BerkStan", suites=["standard"]),
            BetweennessCentralityDataset("web-Google", suites=["standard"]),
            BetweennessCentralityDataset("web-NotreDame", suites=["standard"]),
            BetweennessCentralityDataset("web-Stanford", suites=["standard"]),
            BetweennessCentralityDataset("amazon0302", suites=["standard"]),
            BetweennessCentralityDataset("amazon0312", suites=["standard"]),
            BetweennessCentralityDataset("amazon0505", suites=["standard"]),
            BetweennessCentralityDataset("amazon0601", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella04", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella05", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella06", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella08", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella09", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella24", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella25", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella30", suites=["standard"]),
            BetweennessCentralityDataset("p2p-Gnutella31", suites=["standard"]),
            BetweennessCentralityDataset("roadNet-CA", suites=["standard"]),
            BetweennessCentralityDataset("roadNet-PA", suites=["standard"]),
            BetweennessCentralityDataset("roadNet-TX", suites=["standard"]),
            BetweennessCentralityDataset("as-735", suites=["standard"]),
            BetweennessCentralityDataset("as-Skitter", suites=["standard"]),
            BetweennessCentralityDataset("as-caida", suites=["standard"]),
            BetweennessCentralityDataset("Oregon-1", suites=["standard"]),
            BetweennessCentralityDataset("Oregon-2", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-epinions", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-Slashdot081106", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-Slashdot090216", suites=["standard"]),
            BetweennessCentralityDataset("soc-sign-Slashdot090221", suites=["standard"]),
            BetweennessCentralityDataset("loc-Gowalla", suites=["standard"]),
            BetweennessCentralityDataset("loc-Brightkite", suites=["standard"]),
            BetweennessCentralityDataset("sx-stackoverflow", suites=["standard"]),
            BetweennessCentralityDataset("sx-mathoverflow", suites=["standard"]),
            BetweennessCentralityDataset("sx-superuser", suites=["standard"]),
            BetweennessCentralityDataset("sx-askubuntu", suites=["standard"]),
            BetweennessCentralityDataset("wiki-talk-temporal", suites=["standard"]),
            BetweennessCentralityDataset("email-Eu-core-temporal", suites=["standard", "trace", "train"]),
            BetweennessCentralityDataset("CollegeMsg", suites=["standard"]),
            BetweennessCentralityDataset("twitter7", suites=["standard"]),
            BetweennessCentralityDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.float64)], meta=dict(raw.meta)
        )


class BetweennessCentralityGAPGenerator(Generator[BetweennessCentralityDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_gap"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality GAP"

    @property
    def description(self) -> str:
        return "Input GAP generator for betweenness centrality benchmarks."

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
    def datasets(self) -> list[BetweennessCentralityDataset]:
        # fmt: off
        return [
            BetweennessCentralityDataset("GAP-road", suites=["standard"]),
            BetweennessCentralityDataset("GAP-twitter", suites=["standard"]),
            BetweennessCentralityDataset("GAP-web", suites=["standard"]),
            BetweennessCentralityDataset("GAP-kron", suites=["standard"]),
            BetweennessCentralityDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0], np.float64)], meta=dict(raw.meta)
        )


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
            "involves going layer by layer from each potential starting node to find "
            "the total amount of shortest paths that lead to a node. So for example "
            "4 -> 6 could have 3 diff shortest paths and 4 -> 2 could have only 1 "
            "shortest path. The second step is for tracing backwards to see how many "
            "times a node appears in other shortest paths. The number of times this "
            "node is in one of the shortest path divided by total shortest paths "
            "between the two edge nodes gets added to the intermediate nodes bc score."
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
            "No generative AI was used to construct the benchmark function. This"
            " statement is written by hand."
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

        for v in range(n):
            number_of_paths = xp.zeros((n,), dtype=float)
            self_dist = xp.zeros((n,), dtype=float)
            self_dist = self_dist + xp.asarray(
                [1.0 if i == v else 0.0 for i in range(n)]
            )
            number_of_paths = number_of_paths + self_dist

            neighbors = xp.asarray(G[v], dtype=float)
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
                neighbors_layer = layer_traversal[depth - 1].astype(float)

                prev_layer = layer_traversal[depth - 2].astype(float)

                denom = xp.maximum(number_of_paths, 1e-10)
                update_val = neighbors_layer * (1.0 + score_update) / denom

                update_val = xp.matmul(G, update_val)

                update_val = update_val * prev_layer * number_of_paths

                score_update = score_update + update_val

                depth -= 1

            bc_scores = bc_scores + score_update

        return bc_scores

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        result = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])
        assert np.allclose(result, expected, atol=1e-6), (
            f"Betweenness centrality output mismatch for {param.dataset.name}"
        )
