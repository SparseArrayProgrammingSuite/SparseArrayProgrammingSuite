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


class BreadthFirstSearchDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Breadth-first search input {name}."
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


class BreadthFirstSearchGenerator(Generator[BreadthFirstSearchDataset]):
    @property
    def name(self) -> str:
        return "bfs_inputs"

    @property
    def pretty_name(self) -> str:
        return "Breadth-First Search Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for breadth-first search benchmarks."

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
        return "Generate sparse graph inputs for breadth-first search."

    @property
    def datasets(self) -> list[BreadthFirstSearchDataset]:
        return [
            BreadthFirstSearchDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            BreadthFirstSearchDataset(
                name="snap-facebook_combined",
                pretty_name="SNAP facebook_combined",
                description=(
                    "Combined Facebook social-circle network, with 4,039 nodes and"
                    " 88,234 edges."
                ),
                suites=[],
            ),
            BreadthFirstSearchDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
            BreadthFirstSearchDataset(
                name="snap-p2p-Gnutella04",
                pretty_name="SNAP p2p-Gnutella04",
                description=(
                    "Directed Gnutella peer-to-peer network snapshot from August 4,"
                    " 2002, with 10,876 nodes and 39,994 edges."
                ),
                suites=[],
            ),
        ]

    def generate(
        self, dataset: BreadthFirstSearchDataset
    ) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            return download_snap_dataset(dataset.name)
        raise ValueError(f"Unsupported BFS dataset: {dataset.name}")


class BreadthFirstSearchBenchmark(Benchmark):
    @property
    def name(self):
        return "bfs"

    @property
    def pretty_name(self):
        return "Breadth-First Search Algorithm"

    @property
    def description(self):
        return (
            "The Breadth-First Search algorithm is an important graph traversal"
            " technique used to explore vertices by layers. It is a fundamental"
            " building block for more complex graph algorithms, especially in areas"
            " like parallel processing and high-performance computing."
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
                city="Philadelphia",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "In standard BFS, algorithms on sparse graphs are faster because they"
            " process fewer edges, and specialized algebraic methods use sparsity to"
            " avoid unnecessary computations by focusing only on non-zero elements."
            " Optimizing the use of sparse data structures and algorithms is key to"
            " achieving high performance, as it reduces memory footprint and leads to"
            " faster traversals."
        )

    @property
    def generators(self):
        return [BreadthFirstSearchGenerator()]

    def benchmark(self, data: list[BinsparseFormat], meta: dict):
        edges = xp.from_binsparse(data[0])
        src = meta["src"]

        (n, m) = edges.shape
        assert n == m
        visited = xp.zeros((n,), dtype=bool)
        frontier = xp.zeros((n,), dtype=bool)
        frontier[src] = True
        level = xp.zeros((n,), dtype=int)
        level_idx = 1
        frontier_count = 1
        while frontier_count > 0:
            level = xp.where(frontier, level_idx, level)
            visited = xp.logical_or(visited, frontier)
            frontier = xp.einsum(
                "frontier[j] += edges[i,j] * frontier[i]",
                edges=edges,
                frontier=frontier,
            )
            frontier = xp.logical_and(frontier, xp.logical_not(visited))
            frontier_count = xp.sum(frontier)

            level_idx += 1

        return [level]
