import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Ref,
)

xp = saps.xp


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
            "This code implements an Array-API compatible version of Bellman Ford Algorithm"
            "to find the shortest distance from a src node to all edges across a graph."
            "It takes in an adjacency matrix as an input and then slowly relaxes each vector"
            "by broadcasting it and then determining the minimum distances iteratively."
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
            "Linear algebraic graph algorithms use sparsity to avoid unnecessary computations "
            "by focusing only on non-zero elements. Optimizing the use of sparse data structures and "
            "algorithms is key to achieving high performance, as it reduces memory footprint and "
            "leads to faster traversals."
        )

    @property
    def generators(self):
        return []

    def benchmark(self, data, meta):
        (edges,) = data
        src = meta["src"]

        n = edges.shape[0]

        G = xp.asarray(edges, dtype=float)
        D = xp.full((n,), xp.inf)
        D[src] = 0

        for _ in range(n):
            D_prev = D
            D = D
            candidates = xp.expand_dims(D, 1) + G
            D = xp.minimum(D, candidates.min(axis=0))
            stop = xp.all(D_prev == D)
            D, stop = (D, stop)
            if stop:
                break

        return [D]
