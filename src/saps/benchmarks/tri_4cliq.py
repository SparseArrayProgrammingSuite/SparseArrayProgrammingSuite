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
from saps.downloaders.snap import download_snap_dataset

xp = saps.xp


class GraphCountingDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Graph counting input {name}."
        self._suites = suites or []

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


# BEGIN COPIED TEST FILE: tests/test_tri_4cliq.py
# import pytest
#
# import numpy as np
#
# import saps.benchmarks.tri_4cliq as tri_4cliq
# from frameworks.saps_numpy import NumpyFramework
# from saps.downloaders.snap import load_toy_dataset
# from saps_framework import BinsparseFormat
#
#
# def run_count_benchmark(benchmark, xp, A):
#     prev_xp = getattr(tri_4cliq, "xp", None)
#     tri_4cliq.xp = xp
#     A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
#     try:
#         (result,) = benchmark.benchmark([A_bin], {})
#     finally:
#         tri_4cliq.xp = prev_xp
#     return result
#
#
# @pytest.mark.parametrize(
#     "A, expected",
#     [
#         # Single triangle
#         (
#             np.array(
#                 [
#                     [0, 1, 1],
#                     [1, 0, 1],
#                     [1, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             1,
#         ),
#         # Path - no triangles
#         (
#             np.array(
#                 [
#                     [0, 1, 0, 0],
#                     [1, 0, 1, 0],
#                     [0, 1, 0, 1],
#                     [0, 0, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             0,
#         ),
#         # 4 clique - contains 4c3 = 4 triangles
#         (
#             np.array(
#                 [
#                     [0, 1, 1, 1],
#                     [1, 0, 1, 1],
#                     [1, 1, 0, 1],
#                     [1, 1, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             4,
#         ),
#     ],
# )
# def test_triangle_count(A, expected):
#     xp = NumpyFramework()
#     result = run_count_benchmark(tri_4cliq.TriangleCountBenchmark(), xp, A).item()
#     assert np.allclose(result, expected)
#
#
# @pytest.mark.parametrize(
#     "A, expected",
#     [
#         # Complete graph K3 - no 4-cliques
#         (
#             np.array(
#                 [
#                     [0, 1, 1],
#                     [1, 0, 1],
#                     [1, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             0,
#         ),
#         # Single 4-clique (K4)
#         (
#             np.array(
#                 [
#                     [0, 1, 1, 1],
#                     [1, 0, 1, 1],
#                     [1, 1, 0, 1],
#                     [1, 1, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             1,
#         ),
#         # Two overlapping 4-cliques sharing an edge, only 2 4-cliques should return 2.
#         # Nodes {0,1,2,3} and {1,2,3,4}
#         (
#             np.array(
#                 [
#                     [0, 1, 1, 1, 0],
#                     [1, 0, 1, 1, 1],
#                     [1, 1, 0, 1, 1],
#                     [1, 1, 1, 0, 1],
#                     [0, 1, 1, 1, 0],
#                 ],
#                 dtype=int,
#             ),
#             2,
#         ),
#     ],
# )
# def test_4clique_count(A, expected):
#     xp = NumpyFramework()
#     result = run_count_benchmark(tri_4cliq.FourCliqueCountBenchmark(), xp, A).item()
#     assert np.allclose(result, expected)
#
#
# def test_triangle_snap_toy():
#     data, _ = load_toy_dataset()
#     xp = NumpyFramework()
#     result = run_count_benchmark(tri_4cliq.TriangleCountBenchmark(), xp, data[0]).item()  # noqa: E501
#     assert np.allclose(result, 0)
#
#
# def test_4clique_snap_toy():
#     data, _ = load_toy_dataset()
#     xp = NumpyFramework()
#     result = run_count_benchmark(
#         tri_4cliq.FourCliqueCountBenchmark(), xp, data[0]
#     ).item()
#     assert np.allclose(result, 0)
# END COPIED TEST FILE: tests/test_tri_4cliq.py

class TriangleCountGenerator(Generator[GraphCountingDataset]):
    @property
    def name(self) -> str:
        return "triangle_count_inputs"

    @property
    def pretty_name(self) -> str:
        return "Triangle Count Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for triangle counting benchmarks."

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
                title=(
                    "SNAP: A General Purpose Network Analysis and Graph Mining Library"
                ),
                authors=[
                    Author("Leskovec, Jure"),
                    Author("Sosič, Rok"),
                ],
                journal="ACM Transactions on Intelligent Systems and Technology",
                volume=8,
                number=1,
                year=2016,
                url="https://snap.stanford.edu/index.html",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for triangle counting."

    @property
    def datasets(self) -> list[GraphCountingDataset]:
        return [
            GraphCountingDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: GraphCountingDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported triangle count dataset: {dataset.name}")


class FourCliqueCountGenerator(Generator[GraphCountingDataset]):
    @property
    def name(self) -> str:
        return "four_clique_count_inputs"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Count Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for 4-clique counting benchmarks."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for 4-clique counting."

    @property
    def datasets(self) -> list[GraphCountingDataset]:
        # 4-clique counting is very expensive (6-way einsum); use small graphs only.
        return [
            GraphCountingDataset(
                name="snap-email-Eu-core-temporal-Dept3",
                pretty_name="SNAP email-Eu-core temporal Dept3",
                description=(
                    "Department 3 email network from the SNAP email-Eu-core"
                    " temporal dataset, with 89 nodes and 1,506 static edges."
                ),
                suites=[],
            ),
            GraphCountingDataset(
                name="snap-email-Eu-core-temporal-Dept4",
                pretty_name="SNAP email-Eu-core temporal Dept4",
                description=(
                    "Department 4 email network from the SNAP email-Eu-core"
                    " temporal dataset, with 142 nodes and 1,375 static edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: GraphCountingDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported 4-clique count dataset: {dataset.name}")


class TriangleCountBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "triangle_count"

    @property
    def pretty_name(self) -> str:
        return "Triangle Counting"

    @property
    def motivation(self) -> str:
        return (
            "Adjacency matrices are often sparse, and are used as input in this"
            " problem."
            "'It is generally known that counting the exact number of"
            "triangles in a graph G can be described using the language of"
            "linear algebra as 1/6 Γ(A3),"
            "where A is the adjacency matrix of the graph G, and Γ(X)"
            "is the trace of the square matrix X [1]. Other linear algebra"
            "approaches [2], [3] also require a sparse-matrix multiplication"
            "of A or parts of A as part of their computation. Alternative"
            "approaches that are not based on linear algebra leverage other"
            "formats for describing graphs such as the adjacency list to"
            "design their algorithms [4], [5].'"
            "'...the shortcut method of computing a power of a [adjacency] matrix,"
            "is isomorphic to a similar shortcut for ﬁnding all shortest paths.'"
        )

    @property
    def description(self) -> str:
        return (
            "Triangle Counting: Given adjacency matrix A, # triangles = trace(A^3) //"
            " 6. This counts the number of walks of length 3 that start at vertex i and"
            " end at vertex i, which is exactly a triangle. Divide by 6 to avoid"
            " overcounting. These methods are implemented using the property that"
            " multiplying a graph's adjacency matrix by itself n times yields the"
            " number of walks of length n that begin at the vertex denoted by the row"
            " label and end at the vertex denoted by the column label."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "First look: Linear algebra-based triangle counting"
                    " without matrix multiplication"
                ),
                authors=[
                    Author("T. M. Low"),
                    Author("V. N. Rao"),
                    Author("M. Lee"),
                    Author("D. Popovici"),
                    Author("F. Franchetti"),
                    Author("S. McMillan"),
                ],
                journal="IEEE High Performance Extreme Computing Conference (HPEC)",
                year=2017,
                url="https://doi.org/10.1109/HPEC.2017.8091046",
            ),
            Ref(
                title="Graph Algorithms in the Language of Linear Algebra",
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics",
                year=2011,
                url="https://doi.org/10.1137/1.9780898719918",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator[GraphCountingDataset]]:
        return [TriangleCountGenerator()]

    def benchmark(self, data: list, meta: dict):
        A = xp.from_binsparse(data[0])
        triangles = xp.einsum("S[] += A[i,j] * A[j,k] * A[k,i]", A=A) / 6
        return [xp.asarray(triangles)]


class FourCliqueCountBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "four_clique_count"

    @property
    def pretty_name(self) -> str:
        return "4-Clique Counting"

    @property
    def motivation(self) -> str:
        return (
            "Adjacency matrices are often sparse, and are used as input in this"
            " problem."
            "'It is generally known that counting the exact number of"
            "triangles in a graph G can be described using the language of"
            "linear algebra as 1/6 Γ(A3),"
            "where A is the adjacency matrix of the graph G, and Γ(X)"
            "is the trace of the square matrix X [1]. Other linear algebra"
            "approaches [2], [3] also require a sparse-matrix multiplication"
            "of A or parts of A as part of their computation. Alternative"
            "approaches that are not based on linear algebra leverage other"
            "formats for describing graphs such as the adjacency list to"
            "design their algorithms [4], [5].'"
            "'...the shortcut method of computing a power of a [adjacency] matrix,"
            "is isomorphic to a similar shortcut for ﬁnding all shortest paths.'"
        )

    @property
    def description(self) -> str:
        return (
            "4-clique Counting: A 4-clique must contain 6 edges that connect all 4"
            " vertices. The einsum does the following: for a given vertex i, checks for"
            " existence of 3 edges to 3 other vertices, then checks for existence of 3"
            " edges between those 3 vertices. This constitutes a 4-clique. Divide by 24"
            " to avoid overcounting. These methods are implemented using the property"
            " that multiplying a graph's adjacency matrix by itself n times yields the"
            " number of walks of length n that begin at the vertex denoted by the row"
            " label and end at the vertex denoted by the column label."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "First look: Linear algebra-based triangle counting"
                    " without matrix multiplication"
                ),
                authors=[
                    Author("T. M. Low"),
                    Author("V. N. Rao"),
                    Author("M. Lee"),
                    Author("D. Popovici"),
                    Author("F. Franchetti"),
                    Author("S. McMillan"),
                ],
                journal="IEEE High Performance Extreme Computing Conference (HPEC)",
                year=2017,
                url="https://doi.org/10.1109/HPEC.2017.8091046",
            ),
            Ref(
                title="Graph Algorithms in the Language of Linear Algebra",
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics",
                year=2011,
                url="https://doi.org/10.1137/1.9780898719918",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator[GraphCountingDataset]]:
        return [FourCliqueCountGenerator()]

    def benchmark(self, data: list, meta: dict):
        A = xp.from_binsparse(data[0])
        cliq_4 = (
            xp.einsum(
                "S[] += A[i,j] * A[i,k] * A[i,l] * A[j,k] * A[j,l] * A[k,l]",
                A=A,
            )
            / 24
        )
        return [xp.asarray(cliq_4)]
