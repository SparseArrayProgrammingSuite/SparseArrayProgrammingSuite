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


class ConnectedComponentsDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Connected components input {name}."
        self._tags = tags or []

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


class ConnectedComponentsGenerator(Generator[ConnectedComponentsDataset]):
    @property
    def name(self) -> str:
        return "connected_components_inputs"

    @property
    def pretty_name(self) -> str:
        return "Connected Components Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for connected components benchmarks."

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
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
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for connected components."

    @property
    def datasets(self) -> list[ConnectedComponentsDataset]:
        return [
            ConnectedComponentsDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                tags=[],
            ),
            ConnectedComponentsDataset(
                name="snap-facebook_combined",
                pretty_name="SNAP facebook_combined",
                description=(
                    "Combined Facebook social-circle network, with 4,039 nodes and"
                    " 88,234 edges."
                ),
                tags=[],
            ),
            ConnectedComponentsDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                tags=[],
            ),
        ]

    def generate(
        self, dataset: ConnectedComponentsDataset
    ) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            return download_snap_dataset(dataset.name)
        raise ValueError(f"Unsupported connected components dataset: {dataset.name}")


class SimplyConnectedComponentsBenchmark(Benchmark):
    @property
    def name(self):
        return "simply_connected_components"

    @property
    def pretty_name(self):
        return "Simply Connected Components"

    @property
    def description(self):
        return (
            "Computes the simply connected components of a directed graph using label"
            " propagation."
        )

    @property
    def tags(self):
        return []

    @property
    def authors(self):
        return [
            Contributor("Willow Ahrens", "ahrens@gatech.edu"),
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
            )
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
    def generators(self) -> list[Generator[ConnectedComponentsDataset]]:
        return [ConnectedComponentsGenerator()]

    def benchmark(self, data, meta):
        edges = xp.from_binsparse(data[0])
        (n, m) = edges.shape
        assert m == n

        # create identity matrix with edges
        graph = xp.array(edges, dtype=bool)
        graph = xp.logical_or(graph, graph.T)
        identity_matrix = xp.eye(n, dtype=bool)
        graph = xp.logical_or(identity_matrix, graph)
        labels = xp.arange(n)
        int_max = xp.iinfo(labels.dtype).max

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextLabels = xp.min(xp.where(graph, labels, int_max), axis=1)

            if xp.all(xp.equal(labels, nextLabels)):
                break
            labels = nextLabels
        return [labels]
