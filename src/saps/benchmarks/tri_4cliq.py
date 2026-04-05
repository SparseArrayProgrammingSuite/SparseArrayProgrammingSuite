from ..frameworks.einsum import einsum

"""
Name: Triangle, 4-Clique Counting
Author: Jeffrey Xu
Email: jxu743@gatech.edu

Motivation:
"It is generally known that counting the exact number of
triangles in a graph G can be described using the language of
linear algebra as 1/6 Γ(A3),
where A is the adjacency matrix of the graph G, and Γ(X)
is the trace of the square matrix X [1]. Other linear algebra
approaches [2], [3] also require a sparse-matrix multiplication
of A or parts of A as part of their computation. Alternative
approaches that are not based on linear algebra leverage other
formats for describing graphs such as the adjacency list to
design their algorithms [4], [5]."
T. M. Low, V. N. Rao, M. Lee, D. Popovici, F. Franchetti and S. McMillan,
"First look: Linear algebra-based triangle counting without matrix multiplication,"
2017 IEEE High Performance Extreme Computing Conference (HPEC), Waltham, MA, USA, 2017,
pp. 1-6, doi: 10.1109/HPEC.2017.8091046.


Additional Citation for Motivation:
"...the shortcut method of computing a power of a [adjacency] matrix,
is isomorphic to a similar shortcut for ﬁnding all shortest paths."
Kepner, Jeremy and Gilbert, John, "Graph Algorithms in the Language of Linear Algebra,"
Society for Industrial and Applied Mathematics,
2011, pp. 27, doi: 10.1137/1.9780898719918.


Role of Sparsity:
Adjacency matrices are often sparse, and are used as input in this problem.

Implementation:
These methods are implemented using the property that
multiplying a graph's adjacency matrix by itself n times
yields the number of walks of length n that begin at the vertex denoted by the row label
and end at the vertex denoted by the column label.

Triangle Counting: Given adjacency matrix A, # triangles = trace(A^3) // 6.
This counts the number of walks of length 3 that start at vertex i
and end at vertex i, which is exactly a triangle. Divide by 6 to avoid overcounting.

4-clique Counting: A 4-clique must contain 6 edges that connect all 4 vertices.
The einsum does the following: for a given vertex i, checks for existence
of 3 edges to 3 other vertices, then checks for existence
of 3 edges between those 3 vertices.
This constitutes a 4-clique. Divide by 24 to avoid overcounting.

Data Generation:

Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


def benchmark_triangle_count(xp, A_bench):
    A = xp.lazy(xp.from_benchmark(A_bench))
    triangles = einsum(xp, "S[] += A[i,j] * A[j,k] * A[k,i]", A=A) / 6
    return xp.to_benchmark(xp.compute(triangles))


def benchmark_4clique_count(xp, A_bench):
    A = xp.lazy(xp.from_benchmark(A_bench))
    cliq_4 = (
        einsum(xp, "S[] += A[i,j] * A[i,k] * A[i,l] * A[j,k] * A[j,l] * A[k,l]", A=A)
        / 24
    )
    return xp.to_benchmark(xp.compute(cliq_4))
