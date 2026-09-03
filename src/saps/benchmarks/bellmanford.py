import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import (
    _GAP_KRON_SOURCES,
    _GAP_ROAD_SOURCES,
    _GAP_TWITTER_SOURCES,
    _GAP_URAND_SOURCES,
    _GAP_WEB_SOURCES,
    fetch_suitesparse_matrix,
)
from saps.downloaders.snap import download_snap_dataset


class BellmanFordDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        src: int = 0,
        expected: np.ndarray | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Bellman-Ford input {name}."
        self._suites = suites or []
        self.A = A
        self.src = src
        if expected is None and A is not None:
            expected = bellman_ford_reference(A, src)
        self.expected = expected

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


def bellman_ford_reference(A, src):
    n = A.shape[0]
    D = np.full((n,), np.inf)
    D[src] = 0
    for _ in range(n):
        for v in range(n):
            for u in range(n):
                if A[u, v] + D[u] < D[v]:
                    D[v] = A[u, v] + D[u]
    return D


def bellman_ford_matrix(n, edges, *, symmetric=False):
    A = np.full((n, n), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in edges:
        A[u, v] = 1.0
        if symmetric:
            A[v, u] = 1.0
    return A


class BellmanFordTestGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic Bellman-Ford examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

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
        return "Provide small graph examples for Bellman-Ford correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        tribes = bellman_ford_matrix(
            16,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (2, 3),
                (3, 2),
                (0, 4),
                (4, 0),
                (1, 4),
                (4, 1),
                (0, 5),
                (5, 0),
                (1, 5),
                (5, 1),
                (2, 5),
                (5, 2),
                (2, 6),
                (6, 2),
                (4, 6),
                (6, 4),
                (5, 6),
                (6, 5),
                (2, 7),
                (7, 2),
                (3, 7),
                (7, 3),
                (5, 7),
                (7, 5),
                (6, 7),
                (7, 6),
                (1, 8),
                (8, 1),
                (4, 8),
                (8, 4),
                (7, 8),
                (8, 7),
                (3, 9),
                (9, 3),
                (8, 9),
                (9, 8),
                (9, 10),
                (10, 9),
                (10, 11),
                (11, 10),
                (8, 11),
                (11, 8),
                (11, 12),
                (12, 11),
                (7, 12),
                (12, 7),
                (12, 13),
                (13, 12),
                (13, 14),
                (14, 13),
                (9, 14),
                (14, 9),
                (14, 15),
                (15, 14),
                (12, 15),
                (15, 12),
            ],
        )
        chesapeake = bellman_ford_matrix(
            39,
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (0, 7),
                (0, 8),
                (1, 9),
                (1, 10),
                (1, 11),
                (1, 12),
                (1, 13),
                (1, 14),
                (1, 15),
                (1, 16),
                (2, 9),
                (2, 10),
                (2, 17),
                (2, 18),
                (3, 11),
                (3, 12),
                (3, 19),
                (3, 20),
                (3, 21),
                (4, 13),
                (4, 22),
                (4, 23),
                (4, 24),
                (5, 14),
                (5, 22),
                (5, 25),
                (5, 26),
                (6, 15),
                (6, 23),
                (6, 27),
                (6, 28),
                (7, 16),
                (7, 24),
                (7, 29),
                (7, 30),
                (8, 17),
                (8, 18),
                (8, 19),
                (8, 20),
                (8, 21),
                (9, 22),
                (9, 31),
                (9, 32),
                (10, 23),
                (10, 31),
                (10, 33),
                (11, 24),
                (11, 32),
                (11, 34),
                (12, 25),
                (12, 26),
                (12, 35),
                (13, 27),
                (13, 36),
                (14, 28),
                (14, 37),
                (15, 29),
                (15, 38),
                (16, 30),
                (17, 31),
                (18, 32),
                (19, 33),
                (20, 34),
                (21, 35),
                (22, 36),
                (23, 37),
                (24, 38),
                (25, 27),
                (25, 29),
                (26, 28),
                (26, 30),
                (27, 31),
                (27, 33),
                (28, 32),
                (28, 34),
                (29, 35),
                (30, 36),
                (31, 37),
                (32, 38),
                (33, 35),
                (34, 36),
                (35, 37),
                (36, 38),
                (37, 38),
            ],
            symmetric=True,
        )
        return [
            BellmanFordDataset(
                name="test_bellman_ford_tribes_src_0",
                suites=["test", "trace"],
                A=tribes,
                src=0,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_0",
                suites=["test", "trace"],
                A=chesapeake,
                src=0,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_10",
                suites=["test", "trace"],
                A=chesapeake,
                src=10,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_38",
                suites=["test", "trace"],
                A=chesapeake,
                src=38,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_snap_toy",
                suites=["test", "trace"],
                A=bellman_ford_matrix(3, [(0, 1), (1, 2)]),
                src=0,
                expected=np.array([0.0, 1.0, 2.0]),
            ),
        ]

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        if dataset.A is None or dataset.expected is None:
            raise ValueError("Bellman-Ford test datasets must define A and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={"src": dataset.src},
            ref_outputs=[from_numpy(dataset.expected)],
        )


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
                suites=[],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept4",
                pretty_name="SNAP email-Eu-core temporal Dept4",
                description=(
                    "Department 4 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 142"
                    " nodes and 1,375 static edges."
                ),
                suites=[],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept2",
                pretty_name="SNAP email-Eu-core temporal Dept2",
                description=(
                    "Department 2 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 162"
                    " nodes and 1,772 static edges."
                ),
                suites=[],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core-temporal-Dept1",
                pretty_name="SNAP email-Eu-core temporal Dept1",
                description=(
                    "Department 1 email network from the SNAP email-Eu-core"
                    " temporal dataset, projected to unit-weight edges, with 309"
                    " nodes and 3,031 static edges."
                ),
                suites=[],
            ),
            BellmanFordDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, projected to unit-weight edges, with 1,005"
                    " nodes and 25,571 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            data, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=[_adjacency_to_distance(data[0])], meta=meta)
        raise ValueError(f"Unsupported Bellman-Ford dataset: {dataset.name}")


class BellmanFordGAPGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for Bellman-Ford shortest-path benchmarks."

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
        return [
            Ref(
                title="The GAP Benchmark Suite",
                authors=[
                    Author("Scott Beamer"),
                    Author("Krste Asanović"),
                    Author("David Patterson"),
                ],
                url="https://arxiv.org/abs/1508.03619",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate weighted GAP graph inputs for Bellman-Ford."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        return (
            [
                BellmanFordDataset(
                    name=f"GAP/GAP-road_{src}",
                    pretty_name="GAP Road",
                    description=(
                        "Directed roads with weights in the US, with 23.9M nodes and"
                        " 58.3M edges."
                    ),
                    suites=[],
                    src=src,
                )
                for src in _GAP_ROAD_SOURCES
            ]
            + [
                BellmanFordDataset(
                    name=f"GAP/GAP-twitter_{src}",
                    pretty_name="GAP Twitter",
                    description=(
                        "Directed weighted social network topology of Twitter, with"
                        " 61.6M nodes and 1,468.4M edges."
                    ),
                    suites=[],
                    src=src,
                )
                for src in _GAP_TWITTER_SOURCES
            ]
            + [
                BellmanFordDataset(
                    name=f"GAP/GAP-web_{src}",
                    pretty_name="GAP Web",
                    description=(
                        "A web-crawl of the .sk domain, directed and weighted, with"
                        " 50.6M nodes and 1,949.4M edges."
                    ),
                    suites=[],
                    src=src,
                )
                for src in _GAP_WEB_SOURCES
            ]
            + [
                BellmanFordDataset(
                    name=f"GAP/GAP-kron_{src}",
                    pretty_name="GAP Kron",
                    description=(
                        "Symmetric random undirected weighted graph generated by"
                        " Kronecker synthetic graph generator with parameters"
                        " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                        " edges."
                    ),
                    suites=[],
                    src=src,
                )
                for src in _GAP_KRON_SOURCES
            ]
            + [
                BellmanFordDataset(
                    name=f"GAP/GAP-urand_{src}",
                    pretty_name="GAP Urand",
                    description=(
                        "Symmetric random undirected weighted graph generated by"
                        " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                        " 2,147.4M edges."
                    ),
                    suites=[],
                    src=src,
                )
                for src in _GAP_URAND_SOURCES
            ]
        )

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        if dataset.name.startswith("GAP/"):
            raw = fetch_suitesparse_matrix(dataset.name.rsplit("_", 1)[0])
            meta = raw.meta
            meta["src"] = dataset.src
            return DataInstance(
                inputs=[_adjacency_to_distance(raw.inputs[0], keep_weights=True)],
                meta=meta,
            )
        raise ValueError(f"Unsupported Bellman-Ford dataset: {dataset.name}")


def _adjacency_to_distance(
    adjacency: BinsparseTensor, keep_weights=False
) -> BinsparseTensor:
    shape = adjacency.shape
    distances = np.full(shape, np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)

    try:
        values = to_numpy(adjacency)
        distances[values.astype(bool)] = (
            values[values.astype(bool)] if keep_weights else 1.0
        )
    except TypeError:
        adjacency_coo = to_scipy(adjacency).tocoo()
        distances[adjacency_coo.row, adjacency_coo.col] = (
            adjacency_coo.data if keep_weights else 1.0
        )

    np.fill_diagonal(distances, 0.0)
    return from_numpy(distances)


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
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705</concept_id>
<concept_desc>Mathematics of computing~Mathematical software</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003705.10011686</concept_id>
<concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10010917</concept_id>
<concept_desc>Mathematics of computing~Graph algorithms</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10003640</concept_id>
<concept_desc>Mathematics of computing~Paths and connectivity problems</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

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
        return [
            BellmanFordTestGenerator(),
            BellmanFordGenerator(),
            BellmanFordGAPGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        edges = data[0]
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        result = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])
        assert np.allclose(result, expected, equal_nan=True), (
            f"Bellman-Ford output mismatch for {param.dataset.name}"
        )
