import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.downloaders.snap import download_snap_dataset

xp = saps.xp


class BetweennessCentralityDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Betweenness centrality input {name}."
        self._suites = suites or []

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
        ]

    def generate(self, dataset: BetweennessCentralityDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
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
        return "<ccs2012></ccs2012>"

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
        return [BetweennessCentralityGenerator()]

    def benchmark(self, data, meta):
        G = xp.from_binsparse(data[0])
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
