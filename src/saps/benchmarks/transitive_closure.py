from typing import Any

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

from saps.downloaders.snap import download_snap_dataset
from saps_framework.binsparse_format import BinsparseFormat

xp = saps.xp


class TransitiveClosureDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Transitive closure input {name}."
        self._tags = tags or ["graph", "sparse"]

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
    def tags(self) -> list[str]:
        return self._tags


class TransitiveClosureGenerator(Generator[TransitiveClosureDataset]):
    @property
    def name(self) -> str:
        return "transitive_closure_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transitive Closure Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for transitive closure benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["graph", "transitive-closure", "sparse"]

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
        return "Generate sparse directed graph inputs for transitive closure."

    @property
    def datasets(self) -> list[TransitiveClosureDataset]:
        return [
            TransitiveClosureDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                tags=["graph", "transitive-closure", "sparse", "snap", "directed"],
            ),
            TransitiveClosureDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                tags=[
                    "graph",
                    "transitive-closure",
                    "sparse",
                    "snap",
                    "collaboration-network",
                ],
            ),
        ]

    def generate(
        self, dataset: TransitiveClosureDataset
    ) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            return download_snap_dataset(dataset.name)
        raise ValueError(f"Unsupported transitive closure dataset: {dataset.name}")


class TransitiveClosureBenchmark(Benchmark):
    @property
    def name(self):
        return "transitive_closure"

    @property
    def pretty_name(self):
        return "Transitive Closure"

    @property
    def description(self):
        return (
            "Computes the transitive closure of a directed graph using fixed-point"
            " iteration. The algorithm initializes the adjacency matrix with the"
            " identity, then iteratively applies the closure operation using sparse"
            " matrix operations until convergence. This enables reachability queries."
        )

    @property
    def tags(self):
        return ["graph", "reachability", "transitive-closure", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Rithvik Reddygari", "rreddygari3@gatech.edu"),
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
                city="Philadelphia",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self):
        return ""

    @property
    def generators(self) -> list[Generator[TransitiveClosureDataset]]:
        return [TransitiveClosureGenerator()]

    def benchmark(self, data, meta):
        edges = data[0]
        (n, m) = edges.shape
        assert m == n

        # create identity matrix with edges
        graph = xp.array(edges, dtype=bool)
        identity_matrix = xp.eye(n, dtype=bool)
        graph = xp.logical_or(identity_matrix, graph)

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextGraph = xp.einsum(
                "nextGraph[i,j] or= graph[i,k] & graph[k,j]", graph=graph
            )

            if xp.all(xp.equal(graph, nextGraph)):
                break
            graph = nextGraph
        return [graph]
