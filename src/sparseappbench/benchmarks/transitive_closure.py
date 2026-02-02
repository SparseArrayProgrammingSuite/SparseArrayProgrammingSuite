"""
Name: Transitive Closure + Simply Connected Components
Author: Rithvik Reddygari
Email: rreddygari3@gatech.edu
Motivation (Importance of problem with citation):
“Connected component labeling is a key step in a wide-range of applications,
such as community detection in social networks and coherent region
tracking in image analysis.”
J. Iverson, C. Kamath, and G. Karypis,
“Evaluation of connected-component labeling algorithms
 for distributed-memory systems,” Parallel Computing, vol. 44,
 pp. 53–68, May 2015, doi: 10.1016/j.parco.2015.02.005.

“Reachability queries, which ask whether there exists
 a path between two vertices in a directed graph,
are fundamental operations in graph databases and are widely used in applications
such as XML querying, program analysis, social networks, and biological networks.
Computing the transitive closure is a classical approach to answering
reachability queries.”

Y. Chen, W. Wang, Z. Liu, and X. Lin,
“Efficient Reachability Query Evaluation in Large Directed Graphs,”
ACM Trans. Database Syst., vol. 35, no. 4, pp. 1–45, Oct. 2010,
doi: 10.1145/1862919.1862920.

Role of sparsity (How sparsity is used in the problem):
The input graphs are sparse, meaning the number of edges < number of vertex pairs.
 When represented as matrices, optimized algorithms can focus on nonzero entries.
This allows SCC to operate near linear time scaling with the number of edges,
and limits the work done during the transitive closure algorithm.

Implementation (Where did the reference algorithm come from? With citation.):

Came from Github issue.

Author: Joel Mathew Cherian
Email: jcherian32@gatech.edu
Reference: Kepner, Jeremy, and John Gilbert, eds. Graph algorithms in the language of
linear algebra

Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""


def benchmark_transitive_closure(xp, edges_b):
    graph = _transitive_closure_computation(xp, edges_b)
    return xp.to_benchmark(graph)


def benchmark_simple_connected_components(xp, edges_b):
    graph = _transitive_closure_computation(xp, edges_b)
    # final computation
    res = xp.einsum("res[i,j] = graph[i,j] & graph[j,i]", graph=graph)
    res = xp.compute(res)
    return xp.to_benchmark(res)


def _transitive_closure_computation(xp, edges_b):
    edges = xp.from_benchmark(edges_b)
    (n, m) = edges.shape
    assert m == n

    # create identity matrix with edges
    graph = xp.array(edges, dtype=bool)
    graph = xp.lazy(graph)
    identity_matrix = xp.eye(n, dtype=bool)
    graph = xp.logical_or(identity_matrix, graph)

    # do fixed-point iteration
    max_iterations = n
    for _iteration in range(max_iterations):
        graph = xp.lazy(graph)
        nextGraph = xp.compute(
            xp.einsum("nextGraph[i,j] or= graph[i,k] & graph[k,j]", graph=graph)
        )
        if xp.compute(xp.all(xp.equal(xp.compute(graph), nextGraph))):
            break
        graph = nextGraph
    return graph
