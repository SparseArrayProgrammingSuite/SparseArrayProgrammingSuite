from typing import Any

import numpy as np

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


class BellmanFordDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Bellman-Ford input {name}."
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


class BellmanFordGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for Bellman-Ford shortest-path benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["graph", "sparse", "shortest-path"]

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ilisha Gupta", "igupta90@gatech.edu"),
            Contributor("Joel Mathew Cherian", "jcherian32@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self) -> str:
        return "Generate weighted graph inputs for Bellman-Ford."

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        return [
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept3",
                pretty_name="SNAP email-Eu-core temporal Dept3",
                description=(
                    "Department 3 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 89"
                    " nodes and 1,506 static edges."
                ),
                tags=["graph", "sparse", "snap", "directed", "shortest-path"],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept4",
                pretty_name="SNAP email-Eu-core temporal Dept4",
                description=(
                    "Department 4 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 142"
                    " nodes and 1,375 static edges."
                ),
                tags=["graph", "sparse", "snap", "directed", "shortest-path"],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept2",
                pretty_name="SNAP email-Eu-core temporal Dept2",
                description=(
                    "Department 2 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 162"
                    " nodes and 1,772 static edges."
                ),
                tags=["graph", "sparse", "snap", "directed", "shortest-path"],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept1",
                pretty_name="SNAP email-Eu-core temporal Dept1",
                description=(
                    "Department 1 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 309"
                    " nodes and 3,031 static edges."
                ),
                tags=["graph", "sparse", "snap", "directed", "shortest-path"],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, projected to unit-weight edges, with 1,005"
                    " nodes and 25,571 edges."
                ),
                tags=["graph", "sparse", "snap", "directed", "shortest-path"],
            ),
        ]

    def generate(
        self, dataset: BellmanFordDataset
    ) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            data, meta = download_snap_dataset(dataset.name)
            return [_adjacency_to_unit_distance(data[0])], meta
        raise ValueError(f"Unsupported Bellman-Ford dataset: {dataset.name}")


def _adjacency_to_unit_distance(adjacency: BinsparseFormat) -> BinsparseFormat:
    shape = adjacency.data["shape"]
    distances = np.full(shape, np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)

    if adjacency.data["format"] == "COO":
        rows = adjacency.data["indices_0"]
        cols = adjacency.data["indices_1"]
        distances[rows, cols] = 1.0
        return BinsparseFormat.from_numpy(distances)

    if adjacency.data["format"] == "dense":
        values = adjacency.data["values"].reshape(shape)
        distances[values.astype(bool)] = 1.0
        np.fill_diagonal(distances, 0.0)
        return BinsparseFormat.from_numpy(distances)

    raise ValueError(f"Unsupported format: {adjacency.data['format']}")


class BellmanFordBenchmark(Benchmark):
    @property
    def name(self):
        return "bellman_ford"

    @property
    def pretty_name(self):
        return "Bellman Ford Algorithm"

    @property
    def description(self):
        return (
            "This code implements an Array-API compatible version of Bellman Ford"
            " Algorithm to find the shortest distance from a src node to all edges"
            " across a graph. It takes in an adjacency matrix as an input and then"
            " slowly relaxes each vector by broadcasting it and then determining the"
            " minimum distances iteratively."
        )

    @property
    def tags(self) -> list[str]:
        return ["graph", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ilisha Gupta", "igupta90@gatech.edu"),
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
        return (
            "Linear algebraic graph algorithms use sparsity to avoid unnecessary"
            " computations by focusing only on non-zero elements. Optimizing the use of"
            " sparse data structures and algorithms is key to achieving high"
            " performance, as it reduces memory footprint and leads to faster"
            " traversals."
        )

    @property
    def generators(self):
        return [BellmanFordGenerator()]

    def benchmark(self, data, meta):
        (edges,) = data
        src = meta["src"]

        n = edges.shape[0]

        G = xp.asarray(edges, dtype=float)
        D = xp.full((n,), xp.inf)
        D[src] = 0

        for _ in range(n):
            D_prev = D
            candidates = xp.expand_dims(D, 1) + G
            D = xp.minimum(D, candidates.min(axis=0))
            stop = xp.all(D_prev == D)
            if stop:
                break

        return [D]
