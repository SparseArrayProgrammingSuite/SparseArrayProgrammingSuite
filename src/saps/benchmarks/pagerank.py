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


class PageRankDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"PageRank input {name}."
        self._tags = tags or ["graph", "pagerank", "sparse"]

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


class PageRankGenerator(Generator[PageRankDataset]):
    @property
    def name(self) -> str:
        return "pagerank_inputs"

    @property
    def pretty_name(self) -> str:
        return "PageRank Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for PageRank benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["graph", "pagerank", "sparse"]

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
        return "Generate sparse graph inputs for PageRank."

    @property
    def datasets(self) -> list[PageRankDataset]:
        return [
            PageRankDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                tags=["graph", "pagerank", "sparse", "snap", "directed"],
            ),
            PageRankDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                tags=["graph", "pagerank", "sparse", "snap", "collaboration-network"],
            ),
            PageRankDataset(
                name="snap-p2p-Gnutella04",
                pretty_name="SNAP p2p-Gnutella04",
                description=(
                    "Directed Gnutella peer-to-peer network snapshot from August 4,"
                    " 2002, with 10,876 nodes and 39,994 edges."
                ),
                tags=["graph", "pagerank", "sparse", "snap", "directed", "peer-to-peer"],
            ),
        ]

    def generate(self, dataset: PageRankDataset) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            return download_snap_dataset(dataset.name)
        raise ValueError(f"Unsupported PageRank dataset: {dataset.name}")


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
            "The out-degree of the adjacency is found by summing columns, giving "
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
                title=("Graph Algorithms in the Language of Linear Algebra"),
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
                year=2025,
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
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return "TODO"

    @property
    def generators(self) -> list[Generator[PageRankDataset]]:
        return [PageRankGenerator()]

    def benchmark(self, data, meta):
        alpha = meta.get("alpha", 0.85)
        max_iter = meta.get("max_iter", 100)
        tol = meta.get("tol", 1e-8)

        A = xp.from_binsparse(data[0])
        out_degree = xp.sum(A, axis=0)
        M = xp.array(A, dtype=float)
        N = A.shape[0]

        zero_deg = xp.equal(out_degree, 0)
        safe_out = xp.where(zero_deg, N, out_degree)
        M = M / safe_out
        M = M * (1 - zero_deg) + (1.0 / N) * zero_deg

        x = xp.full((N,), 1.0 / N)
        u = xp.full((N,), 1.0 / N)
        for _ in range(max_iter):
            x_new = alpha * xp.matmul(M, x) + (1 - alpha) * u
            diff = xp.sqrt(xp.sum(xp.multiply(x_new - x, x_new - x)))
            if diff < tol:
                break
            x = x_new
        return [x]
