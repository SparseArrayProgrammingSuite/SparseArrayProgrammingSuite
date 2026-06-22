import os

import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework.binsparse_format import BinsparseFormat

xp = saps.xp


class FloydWarshallDataset(Dataset):
    def __init__(
        self, name, pretty_name, description, suites, source, symmetrize=False
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.source = source
        self.symmetrize = symmetrize

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


# BEGIN COPIED TEST FILE: tests/test_floyd_warshall.py
# import numpy as np
#
# import saps.benchmarks.floyd_warshall as floyd_warshall
# from frameworks.saps_numpy import NumpyFramework
# from saps_framework.binsparse_format import BinsparseFormat
#
#
# def _run_fw_case(xp, A, expected):
#     """Run Floyd–Warshall and compare to an expected APSP distance matrix."""
#     floyd_warshall.xp = xp
#
#     A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
#
#     out = floyd_warshall.FloydWarshallBenchmark().benchmark([A_bin], {})[0]
#
#     both_inf = np.isinf(out) & np.isinf(expected)
#     both_finite = np.isfinite(out) & np.isfinite(expected)
#     assert np.all(both_inf | (both_finite & (out == expected))), (
#         f"Floyd–Warshall output mismatch.\nGot:\n{out}\nExpected:\n{expected}"
#     )
#
#
# def test_fw_tiny_cases():
#     xp = NumpyFramework()
#
#     A = np.array([[0.0]])
#     expected = np.array([[0.0]])
#     _run_fw_case(xp, A, expected)
#
#     A = np.array([[0.0, 1.0], [np.inf, 0.0]])
#     expected = np.array([[0.0, 1.0], [np.inf, 0.0]])
#     _run_fw_case(xp, A, expected)
#
#     A = np.array(
#         [
#             [0.0, 1.0, np.inf],
#             [np.inf, 0.0, 1.0],
#             [np.inf, np.inf, 0.0],
#         ]
#     )
#     expected = np.array(
#         [
#             [0.0, 1.0, 2.0],
#             [np.inf, 0.0, 1.0],
#             [np.inf, np.inf, 0.0],
#         ]
#     )
#     _run_fw_case(xp, A, expected)
#
#     A = np.array(
#         [
#             [0.0, 1.0, 5.0],
#             [np.inf, 0.0, 1.0],
#             [np.inf, np.inf, 0.0],
#         ]
#     )
#     expected = np.array(
#         [
#             [0.0, 1.0, 2.0],
#             [np.inf, 0.0, 1.0],
#             [np.inf, np.inf, 0.0],
#         ]
#     )
#     _run_fw_case(NumpyFramework(), A, expected)
#
#     A = np.array(
#         [
#             [0.0, 1.0, np.inf, np.inf],
#             [1.0, 0.0, np.inf, np.inf],
#             [np.inf, np.inf, 0.0, 1.0],
#             [np.inf, np.inf, 1.0, 0.0],
#         ]
#     )
#     expected = A.copy()
#     _run_fw_case(NumpyFramework(), A, expected)
#
#
# def test_fw_larger_symmetric_graph():
#     """FW on a 39-node symmetric graph: verify APSP properties."""
#     xp = NumpyFramework()
#     floyd_warshall.xp = xp
#
#     n = 39
#     edges = [
#         (0, 1),
#         (0, 2),
#         (0, 3),
#         (0, 4),
#         (0, 5),
#         (0, 6),
#         (0, 7),
#         (0, 8),
#         (1, 9),
#         (1, 10),
#         (1, 11),
#         (1, 12),
#         (1, 13),
#         (1, 14),
#         (1, 15),
#         (1, 16),
#         (2, 9),
#         (2, 10),
#         (2, 17),
#         (2, 18),
#         (3, 11),
#         (3, 12),
#         (3, 19),
#         (3, 20),
#         (3, 21),
#         (4, 13),
#         (4, 22),
#         (4, 23),
#         (4, 24),
#         (5, 14),
#         (5, 22),
#         (5, 25),
#         (5, 26),
#         (6, 15),
#         (6, 23),
#         (6, 27),
#         (6, 28),
#         (7, 16),
#         (7, 24),
#         (7, 29),
#         (7, 30),
#         (8, 17),
#         (8, 18),
#         (8, 19),
#         (8, 20),
#         (8, 21),
#         (9, 22),
#         (9, 31),
#         (9, 32),
#         (10, 23),
#         (10, 31),
#         (10, 33),
#         (11, 24),
#         (11, 32),
#         (11, 34),
#         (12, 25),
#         (12, 26),
#         (12, 35),
#         (13, 27),
#         (13, 36),
#         (14, 28),
#         (14, 37),
#         (15, 29),
#         (15, 38),
#         (16, 30),
#         (17, 31),
#         (18, 32),
#         (19, 33),
#         (20, 34),
#         (21, 35),
#         (22, 36),
#         (23, 37),
#         (24, 38),
#         (25, 27),
#         (25, 29),
#         (26, 28),
#         (26, 30),
#         (27, 31),
#         (27, 33),
#         (28, 32),
#         (28, 34),
#         (29, 35),
#         (30, 36),
#         (31, 37),
#         (32, 38),
#         (33, 35),
#         (34, 36),
#         (35, 37),
#         (36, 38),
#         (37, 38),
#     ]
#
#     A = np.full((n, n), np.inf)
#     np.fill_diagonal(A, 0.0)
#     for u, v in edges:
#         A[u, v] = 1.0
#         A[v, u] = 1.0
#     A_bin = BinsparseFormat.from_numpy(A)
#     out = floyd_warshall.FloydWarshallBenchmark().benchmark([A_bin], {})[0]
#
#     assert out.shape[0] == out.shape[1]
#     assert np.all(np.diag(out) == 0.0)
#     assert np.all(out >= 0.0)
#     # symmetric input → symmetric output
#     assert np.all(out == out.T)
#     # triangle inequality (spot-checked)
#     rng = np.random.default_rng(0)
#     for _ in range(50):
#         i, j, k = rng.integers(0, n, size=3)
#         assert out[i, j] <= out[i, k] + out[k, j]
# END COPIED TEST FILE: tests/test_floyd_warshall.py

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
        ]

    def generate(self, dataset: FloydWarshallDataset):
        from scipy.io import mmread

        import ssgetpy

        matrices = ssgetpy.search(name=dataset.source)
        if not matrices:
            raise ValueError(f"No matrix found with name '{dataset.source}'")
        matrix = matrices[0]
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

        A = A.tocoo()
        n, m = A.shape
        if n != m:
            raise ValueError(f"Floyd-Warshall requires a square matrix, got {A.shape}")

        G = np.full((n, n), np.inf, dtype=np.float64)
        if A.nnz > 0:
            G[A.row, A.col] = 1.0
        np.fill_diagonal(G, 0.0)

        if dataset.symmetrize:
            G = np.minimum(G, G.T)

        G_bin = BinsparseFormat.from_numpy(G)
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
        return [FloydWarshallGenerator()]

    def benchmark(self, data, meta):
        """
        Returns the all pair shortest path i.e. A[i,j] is the shortest
        path from i to j
        """
        G = xp.from_binsparse(data[0])
        n, m = G.shape
        assert n == m
        for k in range(n):
            G_k = xp.expand_dims(G[:, k], axis=1) + xp.expand_dims(G[k, :], axis=0)
            G = xp.minimum(G, G_k)
        return [G]
