"""
Name: Google Page Rank Algorithm

Author: Aarav Joglekar

Email: ajoglekar32@gatech.edu

What does this code do:
First the code calls from_binsparse on the wrapper to translate from binsparse COO.
Once that is done the out-degree of the adjacency is found by summing columns, giving
us the number of outbound links per page. If out-degree is not 0, we divide by k
(the number of outbound links). If out-degree is 0, that means the node had no links,
so we distribute it evenly among all nodes to preserve probability mass. We then run
iteration multiple times so that the PageRank vector converges to its theoretical
stationary value.

Citation for reference implementation:
“Page Rank Algorithm and Implementation.” GeeksforGeeks,
GeeksforGeeks, 15 Apr. 2025,
www.geeksforgeeks.org/python/page-rank-algorithm-implementation/.

Citation corroborating the choice of data & importance of problem:
Brin, S., & Page, L. (1998). "The anatomy of a large-scale hypertextual
 Web search engine." *Computer Networks and ISDN Systems*, 30(1-7), 107-117.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. Generative AI might have been used to construct
tests. This statement was written by hand.
"""


def pagerank(xp, A_binsparse, alpha=0.85, max_iter=100, tol=1e-8):
    A = xp.from_benchmark(A_binsparse)
    A = xp.lazy(A)
    out_degree = xp.sum(A, axis=0)
    M = xp.array(A, dtype=float)
    N = A.shape[0]

    zero_deg = xp.equal(out_degree, 0)
    safe_out = xp.where(zero_deg, N, out_degree)
    M = M / safe_out
    M = M * (1 - zero_deg) + (1.0 / N) * zero_deg

    M = xp.compute(M)
    x = xp.full((N,), 1.0 / N)
    u = xp.full((N,), 1.0 / N)
    for _ in range(max_iter):
        x_new = alpha * xp.matmul(M, x) + (1 - alpha) * u
        diff = norm(xp, x_new - x)
        (x_new, diff) = xp.compute((x_new, diff))
        if diff < tol:
            break
        x = x_new
    return xp.to_benchmark(x)


def norm(xp, v):
    return xp.sqrt(xp.sum(xp.multiply(v, v)))
