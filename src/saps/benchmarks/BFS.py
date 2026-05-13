import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Ref,
)

xp = saps.xp


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
    def tags(self) -> list[str]:
        return ["graph", "sparse"]

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
        return []

    def benchmark(self, data, meta):
        (edges,) = data
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
            visited, frontier, level = [visited, frontier, level]
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
