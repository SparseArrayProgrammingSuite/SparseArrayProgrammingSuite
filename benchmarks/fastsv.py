"""
Name: FastSV Algorithm
Author: Richard Wan
Email: rwan41@gatech.edu

Motivation:
The FastSV algorithm is a graph algorithm used to find the connected components
for a simple graph. This algorithm introduces several optimizations that allow
for faster convergence to a solution compared to the SV algorithm it is based on,
specifically through modifications to the tree hooking and termination condition.

Citation for reference implementation:
Zhang, Y., Azad, A., & Hu, Z. (2020). FastSV: A distributed-memory connected
component algorithm with fast convergence. In Proceedings of the 2020 SIAM Conference on
Parallel Processing for Scientific Computing (pp. 46-57). Society for Industrial and
Applied Mathematics.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. Generative AI was used for debugging. Generative
AI might have been used to construct tests. This statement was written by hand.
"""


def benchmark_fastsv(xp, adjacency_matrix):
    A = xp.from_benchmark(adjacency_matrix)
    A = A != 0

    (n, m) = A.shape
    assert n == m

    f = xp.arange(n)
    gf = xp.asarray(f, copy=True)

    int_max = xp.iinfo(f.dtype).max

    while True:
        dup = gf

        A, f, gf = xp.lazy([A, f, gf])

        # step 1: stochastic hooking
        mngf = xp.min(xp.where(A, xp.expand_dims(gf, 0), int_max), axis=1)
        B = xp.zeros((n, n), dtype=bool)
        B[f, xp.arange(n)] = True
        f = xp.min(xp.where(B, xp.expand_dims(mngf, 0), int_max), axis=1)

        # step 2: aggressive hooking
        f = xp.minimum(f, mngf)

        # step 3: shortcutting
        f = xp.minimum(f, gf)

        # step 4: calculate grandparents
        gf = xp.take(f, f)

        # step 5: check termination
        stop = xp.all(dup == gf)

        f, gf, stop = xp.compute([f, gf, stop])

        if stop:
            break

    return xp.to_benchmark(f)
