import os

import numpy as np

import saps
from saps.benchmark import Author, Benchmark, Contributor, Dataset, Generator, Ref
from saps_framework.binsparse_format import BinsparseFormat

xp = saps.xp


class FloydWarshallDataset(Dataset):
    def __init__(self, name, pretty_name, description, suites, source, symmetrize=False):
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
        return [G_bin], {}


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
