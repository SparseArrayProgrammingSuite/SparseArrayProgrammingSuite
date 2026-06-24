import numpy as np

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.downloaders.snap import download_snap_dataset
from saps_framework.binsparse_format import BinsparseFormat



class PageRankDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        expected: np.ndarray | None = None,
        ref_meta: dict | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"PageRank input {name}."
        self._suites = suites or []
        self.A = A
        self.expected = expected
        self.ref_meta = ref_meta

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


def pagerank_networkx_reference(A: np.ndarray) -> np.ndarray:
    import networkx as nx

    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    expected_dict = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6)
    return np.array([expected_dict[i] for i in range(len(graph))])


class PageRankTestGenerator(Generator[PageRankDataset]):
    @property
    def name(self) -> str:
        return "pagerank_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "PageRank Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic PageRank examples."

    @property
    def suites(self) -> list[str]:
        return ["test"]

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
            "Generative AI might have been used to construct tests. This statement "
            "was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Provide small graph examples for PageRank correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[PageRankDataset]:
        return [
            PageRankDataset(
                name="test_pagerank_two_node_cycle",
                suites=["test"],
                A=np.array([[0, 1], [1, 0]], dtype=float),
                expected=np.array([0.5, 0.5], dtype=float),
            ),
            PageRankDataset(
                name="test_pagerank_three_node_chain",
                suites=["test"],
                A=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                ref_meta={"rank_order": [0, 1, 2]},
            ),
            PageRankDataset(
                name="test_pagerank_two_node_sink",
                suites=["test"],
                A=np.array([[0, 0], [1, 0]], dtype=float),
                ref_meta={"rank_order": [0, 1]},
            ),
            PageRankDataset(
                name="test_pagerank_against_networkx",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0],
                    ],
                    dtype=float,
                ),
                expected=pagerank_networkx_reference(
                    np.array(
                        [
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0],
                        ],
                        dtype=float,
                    )
                ),
            ),
            PageRankDataset(
                name="test_pagerank_snap_toy",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=float,
                ),
                expected=pagerank_networkx_reference(
                    np.array(
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                        ],
                        dtype=float,
                    )
                ),
            ),
        ]

    def generate(self, dataset: PageRankDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("PageRank test datasets must define A.")
        ref_outputs = None
        if dataset.expected is not None:
            ref_outputs = [BinsparseFormat.from_numpy(dataset.expected)]
        return DataInstance(
            inputs=[BinsparseFormat.from_numpy(dataset.A)],
            meta={},
            ref_outputs=ref_outputs,
            ref_meta=dataset.ref_meta,
        )


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
                suites=[],
            ),
            PageRankDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
            PageRankDataset(
                name="snap-p2p-Gnutella04",
                pretty_name="SNAP p2p-Gnutella04",
                description=(
                    "Directed Gnutella peer-to-peer network snapshot from August 4,"
                    " 2002, with 10,876 nodes and 39,994 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: PageRankDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
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
            "us the number of outbound links per page. If out-degree is not 0, "
            "we divide by k (the number of outbound links). If out-degree is 0, "
            "that means the node had no links, so we distribute it evenly among "
            "all nodes to preserve probability mass. We then run iteration "
            "multiple times so that the PageRank vector converges to its "
            "theoretical stationary value."
        )

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
                authors=[Author("Sergey Brin"), Author("Lawrence Page")],
                year=1998,
                journal="Computer Networks and ISDN Systems",
                pages="107-117",
                url="https://doi.org/10.1016/S0169-7552(98)00110-X",
                doi="10.1016/S0169-7552(98)00110-X",
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
        return [PageRankTestGenerator(), PageRankGenerator()]

    def benchmark(self, xp, data, meta):
        alpha = meta.get("alpha", 0.85)
        max_iter = meta.get("max_iter", 100)
        tol = meta.get("tol", 1e-8)

        A = data[0]
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )

        result = self._output[0].data["values"].reshape(self._output[0].data["shape"])

        if self._ref_outputs is not None:
            expected = self._ref_outputs[0].data["values"].reshape(
                self._ref_outputs[0].data["shape"]
            )
            assert np.allclose(result, expected, atol=1e-2), (
                f"PageRank output mismatch for {param.dataset.name}"
            )

        if self._ref_meta is None:
            return
        assert np.isclose(np.sum(result), 1.0, atol=1e-6)
        assert np.all(result >= 0)

        eps = 1e-6
        rank_order = self._ref_meta.get("rank_order")
        if rank_order is not None:
            for lower, higher in zip(rank_order, rank_order[1:], strict=False):
                assert result[lower] < result[higher] - eps
