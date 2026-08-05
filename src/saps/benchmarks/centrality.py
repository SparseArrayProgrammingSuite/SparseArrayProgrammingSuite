import numpy as np

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.gap import GAP_REFERENCE, fetch_gap_graph
from saps.downloaders.snap import download_snap_dataset
from saps_framework import BinsparseFormat


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
        self._suites = suites or []
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
        return "betweenness_centrality_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic betweenness centrality examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

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
                name="test_joels_case",
                suites=["test", "trace"],
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
                name="test_basic_empty",
                suites=["test", "trace"],
                A=np.zeros((3, 3)),
                expected=np.array([0.0, 0.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="test_basic_chain",
                suites=["test", "trace"],
                A=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
                expected=np.array([0.0, 1.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="test_basic_two_components",
                suites=["test", "trace"],
                A=np.array(
                    [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                    dtype=float,
                ),
                expected=np.array([0.0, 0.0, 0.0, 0.0]),
            ),
            BetweennessCentralityDataset(
                name="test_matrix_vertex_algorithm_comparison",
                suites=["test", "trace"],
                A=random_A,
                expected=reference_bc_alg_6_4(random_A),
            ),
            BetweennessCentralityDataset(
                name="test_undirected_graph",
                suites=["test", "trace"],
                A=undirected_A,
                expected=reference_bc_alg_6_4(undirected_A),
            ),
            BetweennessCentralityDataset(
                name="test_networkx",
                suites=["test", "trace"],
                A=networkx_A,
                expected=reference_bc_alg_6_4(networkx_A),
            ),
            BetweennessCentralityDataset(
                name="test_centrality_snap_toy",
                suites=["test", "trace"],
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
            inputs=[BinsparseFormat.from_numpy(dataset.A)],
            meta={},
            ref_outputs=[BinsparseFormat.from_numpy(dataset.expected)],
        )


class BetweennessCentralityGenerator(Generator[BetweennessCentralityDataset]):
    @property
    def name(self) -> str:
        return "betweenness_centrality_inputs"

    @property
    def pretty_name(self) -> str:
        return "Betweenness Centrality Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for betweenness centrality benchmarks."

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
        return [GAP_REFERENCE]

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
    def datasets(self) -> list[BetweennessCentralityDataset]:
        return [
            BetweennessCentralityDataset(
                name="snap-email-Eu-core-temporal-Dept3",
                pretty_name="SNAP email-Eu-core temporal Dept3",
                description=(
                    "Department 3 email network from the SNAP email-Eu-core"
                    " temporal dataset, with 89 nodes and 1,506 static edges."
                ),
                suites=[],
            ),
            BetweennessCentralityDataset(
                name="snap-email-Eu-core-temporal-Dept4",
                pretty_name="SNAP email-Eu-core temporal Dept4",
                description=(
                    "Department 4 email network from the SNAP email-Eu-core"
                    " temporal dataset, with 142 nodes and 1,375 static edges."
                ),
                suites=[],
            ),
            BetweennessCentralityDataset(
                name="gap-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with distances in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        if dataset.name.startswith("gap"):
            return fetch_gap_graph(dataset.name)
        raise ValueError(f"Unsupported betweenness centrality dataset: {dataset.name}")


class BetweennessCentralityBenchmark(Benchmark):
    @property
    def name(self):
        return "betweenness_centrality"

    @property
    def pretty_name(self):
        return "Betweenness Centrality Algorithm"

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
        return []

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
    def generators(self) -> list[Generator[BetweennessCentralityDataset]]:
        return [BetweennessCentralityTestGenerator(), BetweennessCentralityGenerator()]

    def benchmark(self, xp, data, meta):
        G = data[0]
        n = G.shape[0]
        bc_scores = xp.zeros((n,), dtype=float)

        for v in range(n):
            number_of_paths = xp.zeros((n,), dtype=float)
            self_dist = xp.zeros((n,), dtype=float)
            self_dist = self_dist + xp.array([1.0 if i == v else 0.0 for i in range(n)])
            number_of_paths = number_of_paths + self_dist

            neighbors = xp.array(G[v], dtype=float)
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

        return [bc_scores]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        result = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        expected = (
            self._ref_outputs[0]
            .data["values"]
            .reshape(self._ref_outputs[0].data["shape"])
        )
        assert np.allclose(result, expected, atol=1e-6), (
            f"Betweenness centrality output mismatch for {param.dataset.name}"
        )
