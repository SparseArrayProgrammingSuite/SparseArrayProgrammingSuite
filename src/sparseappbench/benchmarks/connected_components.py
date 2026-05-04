import sparseappbench
from sparseappbench.benchmark import (
    Benchmark,
    Contributor,
    Ref,
    Author,
)

xp = sparseappbench.xp


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
            "Computes the simply connected components of a directed graph using label propagation. "
        )

    @property
    def tags(self):
        return ["graph", "connected-components", "sparse"]

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
        labels = xp.arange(n)
        int_max = xp.iinfo(labels.dtype).max

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextLabels = xp.min(xp.where(graph, labels, int_max), axis=1)

            if xp.all(xp.equal(labels, nextLabels)):
                break
            labels = nextLabels
        return [xp.to_binsparse(labels)]
