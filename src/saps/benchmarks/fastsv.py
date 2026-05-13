import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Ref,
)

xp = saps.xp


class FastSVBenchmark(Benchmark):
    @property
    def name(self):
        return "fastsv"

    @property
    def pretty_name(self):
        return "FastSV Algorithm"

    @property
    def description(self):
        return (
            "The FastSV algorithm is a graph algorithm used to find the connected"
            " components for a simple graph. This algorithm introduces several"
            " optimizations that allow for faster convergence to a solution compared to"
            " the SV algorithm it is based on, specifically through modifications to"
            " the tree hooking and termination condition."
        )

    @property
    def tags(self):
        return ["graph", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Richard Wan", "rwan41@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=(
                    "FastSV: A distributed-memory connected component"
                    " algorithm with fast convergence."
                ),
                authors=[
                    Author("Zhang, Y."),
                    Author("Azad, A."),
                    Author("Hu, Z."),
                ],
                journal=(
                    "Proceedings of the 2020 SIAM Conference on Parallel"
                    " Processing for Scientific Computing"
                ),
                pages="46-57",
                publisher="Society for Industrial and Applied Mathematics",
                year=2020,
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
        return ""

    @property
    def generators(self):
        return []

    def benchmark(self, data, meta):
        (adjacency_matrix,) = data

        # Inlined benchmark_fastsv helper
        A = xp.from_binsparse(adjacency_matrix)
        A = A != 0

        (n, m) = A.shape
        assert n == m

        f = xp.arange(n)
        gf = xp.asarray(f, copy=True)

        int_max = xp.iinfo(f.dtype).max

        while True:
            dup = gf

            A, f, gf = [A, f, gf]

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

            f, gf, stop = [f, gf, stop]

            if stop:
                break

        result = xp.to_binsparse(f)
        return [result]
