import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Ref,
)

xp = saps.xp


class PageRankBenchmark(Benchmark):
    @property
    def name(self):
        return "pagerank"

    @property
    def pretty_name(self):
        return "Google Page Rank Algorithm"

    @property
    def description(self):
        return (
            "First the code calls from_binsparse on the wrapper to translate from binsparse COO. "
            "Once that is done the out-degree of the adjacency is found by summing columns, giving "
            "us the number of outbound links per page. If out-degree is not 0, we divide by k "
            "(the number of outbound links). If out-degree is 0, that means the node had no links, "
            "so we distribute it evenly among all nodes to preserve probability mass. We then run "
            "iteration multiple times so that the PageRank vector converges to its theoretical "
            "stationary value."
        )

    @property
    def tags(self):
        return ["graph", "pagerank", "sparse"]

    @property
    def authors(self):
        return [Contributor("Aarav Joglekar", "ajoglekar32@gatech.edu")]

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
            Ref(
                title="Page Rank Algorithm and Implementation",
                authors=[Author("GeeksforGeeks contributors")],
                url="https://www.geeksforgeeks.org/python/page-rank-algorithm-implementation/",
                year="2025"
            ),
            Ref(
                title="The anatomy of a large-scale hypertextual Web search engine",
                authors=[Author("Brin, S."), Author("Page, L.")],
                year=1998,
                journal="Computer Networks and ISDN Systems",
                pages="107-117",
                url="https://doi.org/10.1016/S1389-1286(98)00110-X",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests. This statement "
            "was written by hand."
        )

    @property
    def motivation(self):
        return (
            "TODO"
        )

    @property
    def generators(self):
        return []

    def benchmark(self, data, meta):
        A_binsparse = data[0]
        alpha = meta.get("alpha", 0.85)
        max_iter = meta.get("max_iter", 100)
        tol = meta.get("tol", 1e-8)
        
        A = xp.from_binsparse(A_binsparse)
        A = A
        out_degree = xp.sum(A, axis=0)
        M = xp.array(A, dtype=float)
        N = A.shape[0]

        zero_deg = xp.equal(out_degree, 0)
        safe_out = xp.where(zero_deg, N, out_degree)
        M = M / safe_out
        M = M * (1 - zero_deg) + (1.0 / N) * zero_deg

        M = M
        x = xp.full((N,), 1.0 / N)
        u = xp.full((N,), 1.0 / N)
        for _ in range(max_iter):
            x_new = alpha * xp.matmul(M, x) + (1 - alpha) * u
            diff = xp.sqrt(xp.sum(xp.multiply(x_new - x, x_new - x)))
            (x_new, diff) = (x_new, diff)
            if diff < tol:
                break
            x = x_new
        return [x]
