import numpy as np

import sparse as sp
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import SuiteSparseDataset, fetch_suitesparse_matrix
from saps_framework.binsparse_utils import to_coo


class FloydWarshallDataset(SuiteSparseDataset):
    def __init__(
        self,
        name,
        pretty_name,
        description,
        suites,
        source,
        symmetrize=False,
        A=None,
        expected=None,
        ref_meta=None,
    ):
        super().__init__(
            name,
            source_name=source,
            pretty_name=pretty_name,
            description=description,
            suites=suites,
        )
        self.symmetrize = symmetrize
        self.A = A
        if expected is None and A is not None:
            expected = floyd_warshall_reference(A)
        self.expected = expected
        self.ref_meta = ref_meta


def floyd_warshall_reference(A):
    if isinstance(A, sp.SparseArray):
        A = A.todense()
    expected = A.copy()
    for k in range(expected.shape[0]):
        expected = np.minimum(
            expected,
            np.expand_dims(expected[:, k], axis=1)
            + np.expand_dims(expected[k, :], axis=0),
        )
    return expected


def floyd_warshall_input_from_edges(
    n: int, edges: list[tuple[int, int]], *, symmetric: bool = False
):
    rows = [*range(n)]
    cols = [*range(n)]
    values = [0.0] * n
    for u, v in edges:
        rows.append(u)
        cols.append(v)
        values.append(1.0)
        if symmetric:
            rows.append(v)
            cols.append(u)
            values.append(1.0)
    return sp.COO(
        coords=np.array([rows, cols], dtype=np.int64),
        data=np.array(values, dtype=np.float64),
        shape=(n, n),
        fill_value=np.inf,
    )


class FloydWarshallTestGenerator(Generator[FloydWarshallDataset]):
    @property
    def name(self) -> str:
        return "floyd_warshall_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Floyd-Warshall Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic Floyd-Warshall examples."

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
        return "Provide small graph examples for shortest-path correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FloydWarshallDataset]:
        return [
            FloydWarshallDataset(
                name="single-node",
                pretty_name="single-node",
                description="Floyd-Warshall test case single-node.",
                suites=["test", "trace"],
                source="single-node",
                A=np.array([[0.0]]),
                expected=np.array([[0.0]]),
            ),
            FloydWarshallDataset(
                name="two-node-directed",
                pretty_name="two-node-directed",
                description="Floyd-Warshall test case two-node-directed.",
                suites=["test", "trace"],
                source="two-node-directed",
                A=np.array([[0.0, 1.0], [np.inf, 0.0]]),
                expected=np.array([[0.0, 1.0], [np.inf, 0.0]]),
            ),
            FloydWarshallDataset(
                name="three-node-chain",
                pretty_name="three-node-chain",
                description="Floyd-Warshall test case three-node-chain.",
                suites=["test", "trace"],
                source="three-node-chain",
                A=np.array(
                    [
                        [0.0, 1.0, np.inf],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="three-node-shortcut",
                pretty_name="three-node-shortcut",
                description="Floyd-Warshall test case three-node-shortcut.",
                suites=["test", "trace"],
                source="three-node-shortcut",
                A=np.array(
                    [
                        [0.0, 1.0, 5.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="two-components",
                pretty_name="two-components",
                description="Floyd-Warshall test case two-components.",
                suites=["test", "trace"],
                source="two-components",
                A=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="large-symmetric",
                pretty_name="large-symmetric",
                description="Floyd-Warshall test case large-symmetric.",
                suites=["test", "trace"],
                source="large-symmetric",
                A=floyd_warshall_input_from_edges(
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
                ),
                ref_meta={"large_symmetric": True},
            ),
        ]

    def generate(self, dataset: FloydWarshallDataset):
        inputs = (
            dataset.A.todense() if isinstance(dataset.A, sp.SparseArray) else dataset.A
        )
        return DataInstance(
            inputs=[from_numpy(inputs)],
            meta={},
            ref_outputs=[from_numpy(dataset.expected)],
            ref_meta=dataset.ref_meta,
        )


class FloydWarshallGenerator(Generator[FloydWarshallDataset]):
    @property
    def name(self) -> str:
        return "floyd_warshall_inputs"

    @property
    def pretty_name(self) -> str:
        return "Floyd-Warshall Input Generator"

    @property
    def description(self) -> str:
        return (
            "Data is collected from the SuiteSparse Matrix Collection and standard"
            " benchmark graph datasets, with sparse adjacency matrices converted into"
            " unweighted all-pairs shortest path inputs. This generator uses real-world"
            " networks, including the Chesapeake road network and soc-tribes network"
            " from the Network Repository."
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
                year=2011,
            ),
            Ref(
                title=(
                    "The Network Data Repository with Interactive"
                    " Graph Analytics and Visualization"
                ),
                authors=[
                    Author("Ryan A. Rossi"),
                    Author("Nesreen K. Ahmed"),
                ],
                journal="AAAI",
                url="https://networkrepository.com",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[FloydWarshallDataset]:
        return [
            FloydWarshallDataset(
                name="bcspwr01",
                pretty_name="BCS Power Grid 01",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="bcspwr01",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="bcspwr02",
                pretty_name="BCS Power Grid 02",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="bcspwr02",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="bcspwr03",
                pretty_name="BCS Power Grid 03",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="bcspwr03",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="chesapeake",
                pretty_name="Chesapeake",
                description="Sparse road network input for Floyd-Warshall.",
                suites=[],
                source="chesapeake",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="ash85",
                pretty_name="ASH 85",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="ash85",
                symmetrize=False,
            ),
            FloydWarshallDataset(
                name="arc130",
                pretty_name="ARC 130",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="arc130",
                symmetrize=False,
            ),
            FloydWarshallDataset(
                name="bcspwr04",
                pretty_name="BCS Power Grid 04",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="bcspwr04",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="ash292",
                pretty_name="ASH 292",
                description="Sparse SuiteSparse graph input for Floyd-Warshall.",
                suites=[],
                source="ash292",
                symmetrize=False,
            ),
            FloydWarshallDataset(
                name="gap-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=[],
                source="gap-road",
                symmetrize=False,
            ),
            FloydWarshallDataset(
                name="gap-twitter",
                pretty_name="GAP Twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=[],
                source="gap-twitter",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="gap-web",
                pretty_name="GAP Web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=[],
                source="gap-web",
                symmetrize=True,
            ),
            FloydWarshallDataset(
                name="gap-kron",
                pretty_name="GAP Kron",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Kronecker synthetic graph generator with parameters"
                    " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                    " edges."
                ),
                suites=[],
                source="gap-kron",
                symmetrize=False,
            ),
            FloydWarshallDataset(
                name="gap-urand",
                pretty_name="GAP Urand",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                    " 2,147.4M edges."
                ),
                suites=[],
                source="gap-urand",
                symmetrize=False,
            ),
        ]

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: FloydWarshallDataset):
        raw = fetch_suitesparse_matrix(dataset.source_name)
        n, m = raw.meta["shape"]
        if n != m:
            raise ValueError(f"Floyd-Warshall requires a square matrix, got {(n, m)}")

        coo = to_coo(raw.inputs[0])
        G = np.full((n, n), np.inf, dtype=np.float64)
        if raw.meta["nnz"] > 0:
            G[coo.indices_0, coo.indices_1] = 1.0
        np.fill_diagonal(G, 0.0)

        if dataset.symmetrize:
            G = np.minimum(G, G.T)

        G_bin = from_numpy(G)
        return DataInstance(inputs=[G_bin], meta={})


class FloydWarshallBenchmark(Benchmark):
    @property
    def name(self):
        return "floyd_warshall"

    @property
    def pretty_name(self):
        return "Floyd-Warshall"

    @property
    def description(self):
        return (
            "The Floyd-Warshall algorithm computes the shortest paths between every "
            "pair of vertices in a weighted directed graph."
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
                publisher="SIAM",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "Sparse graphs reduce unnecessary computation, as most entries in the"
            " adjacency matrix represent non-edges and begin as inifinity. Efficient"
            " sparse representations allow the backend framework to skip work and"
            " minimize memory movement during the relaxation steps of the algorithm."
        )

    @property
    def generators(self):
        return [FloydWarshallTestGenerator(), FloydWarshallGenerator()]

    def benchmark(self, xp, data, meta):
        """
        Returns the all pair shortest path i.e. A[i,j] is the shortest
        path from i to j
        """
        G = data[0]
        n, m = G.shape
        assert n == m
        for k in range(n):
            G_k = xp.expand_dims(G[:, k], axis=1) + xp.expand_dims(G[k, :], axis=0)
            G = xp.minimum(G, G_k)
        return [G]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        output = to_numpy(self._output[0])
        if self._ref_outputs is not None:
            expected = (
                to_numpy(self._ref_outputs[0])
            )
            both_inf = np.isinf(output) & np.isinf(expected)
            both_finite = np.isfinite(output) & np.isfinite(expected)
            assert np.all(both_inf | (both_finite & (output == expected))), (
                f"Floyd-Warshall output mismatch for {param.dataset.name}"
            )
        if self._ref_meta and self._ref_meta.get("large_symmetric"):
            assert output.shape[0] == output.shape[1]
            assert np.all(np.diag(output) == 0.0)
            assert np.all(output >= 0.0)
            assert np.all(output == output.T)
            rng = np.random.default_rng(0)
            for _ in range(50):
                i, j, k = rng.integers(0, output.shape[0], size=3)
                assert output[i, j] <= output[i, k] + output[k, j]
