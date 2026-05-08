import saps
from saps.benchmark import (
    Benchmark,
    Contributor,
    Ref,
    Author,
)

xp = saps.xp


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
            "Computes the transitive closure of a directed graph using fixed-point iteration. "
            "The algorithm initializes the adjacency matrix with the identity, then iteratively "
            "applies the closure operation using sparse matrix operations until convergence. "
            "This enables reachability queries."
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
                title=(
                    "Graph Algorithms in the Language of Linear Algebra"
                ),
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
            ""
        )

    @property
    def generators(self):
        return []

    def benchmark(self, data, meta):
        edges_b = data[0]
        edges = xp.from_binsparse(edges_b)
        (n, m) = edges.shape
        assert m == n

        # create identity matrix with edges
        graph = xp.array(edges, dtype=bool)
        graph = graph
        identity_matrix = xp.eye(n, dtype=bool)
        graph = xp.logical_or(identity_matrix, graph)

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextGraph = xp.einsum("nextGraph[i,j] or= graph[i,k] & graph[k,j]", graph=graph)

            if xp.all(xp.equal(graph, nextGraph)):
                break
            graph = nextGraph
        return [graph]
