import sparseappbench
from sparseappbench.benchmark import Benchmark, Contributor, Generator, Ref, Author

xp = sparseappbench.xp

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
            "Adjacency matrices are often sparse, and are used as input in this problem."
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
            "Triangle Counting: Given adjacency matrix A, # triangles = trace(A^3) // 6."
            "This counts the number of walks of length 3 that start at vertex i"
            "and end at vertex i, which is exactly a triangle. Divide by 6 to avoid overcounting."
            "These methods are implemented using the property that"
            "multiplying a graph's adjacency matrix by itself n times"
            "yields the number of walks of length n that begin at the vertex denoted by the row label"
            "and end at the vertex denoted by the column label."
        )

    @property
    def tags(self) -> list[str]:
        return ["graph", "triangle-counting", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title = "First look: Linear algebra-based triangle counting without matrix multiplication",
                authors = [
                    Author("T. M. Low"),
                    Author("V. N. Rao"),
                    Author("M. Lee"),
                    Author("D. Popovici"),
                    Author("F. Franchetti"),
                    Author("S. McMillan")
                ],
                journal = "IEEE High Performance Extreme Computing Conference (HPEC)",
                year = 2017,
                url="https://doi.org/10.1109/HPEC.2017.8091046"
            ),
            Ref(
                title = "Graph Algorithms in the Language of Linear Algebra",
                authors = [
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal = "Society for Industrial and Applied Mathematics",
                year = 2011,
                url = "https://doi.org/10.1137/1.9780898719918",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator]:
        return []

    def benchmark(self, data: list, meta: dict):
        A = data[0]
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
            "Adjacency matrices are often sparse, and are used as input in this problem."
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
            "4-clique Counting: A 4-clique must contain 6 edges that connect all 4 vertices."
            "The einsum does the following: for a given vertex i, checks for existence"
            "of 3 edges to 3 other vertices, then checks for existence"
            "of 3 edges between those 3 vertices."
            "This constitutes a 4-clique. Divide by 24 to avoid overcounting."
            "These methods are implemented using the property that"
            "multiplying a graph's adjacency matrix by itself n times"
            "yields the number of walks of length n that begin at the vertex denoted by the row label"
            "and end at the vertex denoted by the column label."
        )

    @property
    def tags(self) -> list[str]:
        return ["graph", "clique-counting", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jeffrey Xu", "jxu743@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title = "First look: Linear algebra-based triangle counting without matrix multiplication",
                authors = [
                    Author("T. M. Low"),
                    Author("V. N. Rao"),
                    Author("M. Lee"),
                    Author("D. Popovici"),
                    Author("F. Franchetti"),
                    Author("S. McMillan")
                ],
                journal = "IEEE High Performance Extreme Computing Conference (HPEC)",
                year = 2017,
                url="https://doi.org/10.1109/HPEC.2017.8091046"
            ),
            Ref(
                title = "Graph Algorithms in the Language of Linear Algebra",
                authors = [
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal = "Society for Industrial and Applied Mathematics",
                year = 2011,
                url = "https://doi.org/10.1137/1.9780898719918",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator]:
        return []

    def benchmark(self, data: list, meta: dict):
        A = data[0]
        cliq_4 = (
            xp.einsum(
                "S[] += A[i,j] * A[i,k] * A[i,l] * A[j,k] * A[j,l] * A[k,l]",
                A=A,
            )
            / 24
        )
        return [xp.asarray(cliq_4)]
