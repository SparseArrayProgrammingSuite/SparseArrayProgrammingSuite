from typing import Any

import scipy.sparse.linalg as scipy_spla

from saps.benchmark import Author, Benchmark, Contributor, Ref
from saps.benchmarks.cg import CGBenchmark, CGGenerator, CGTestGenerator


class SciPyCGBenchmark(Benchmark):
    """SciPy's own CG solver, for comparison against the SAPS implementation.

    Uses the datasets and correctness checks of `CGBenchmark`. It subclasses
    `Benchmark` rather than `CGBenchmark` so that it does not also inherit the
    latter's generated `time_cg_solver` entry point.
    """

    @property
    def tag(self) -> str:
        return "scipy_cg_solver"

    @property
    def name(self) -> str:
        return "scipy_cg_solver"

    @property
    def pretty_name(self) -> str:
        return "Conjugate Gradient Iterative Solver (scipy.sparse.linalg.cg)"

    @property
    def description(self) -> str:
        return (
            "Solves sparse symmetric positive definite linear systems using "
            "scipy.sparse.linalg.cg."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return CGBenchmark().concepts

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Iterative Methods for Sparse Linear Systems",
                authors=[Author("Yousef Saad")],
                publisher="SIAM",
                year=2003,
            )
        ]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jash Ambaliya", "jash.ambaliya@juliahub.com")]

    @property
    def ai_disclosure(self) -> str:
        return "TODO: to be written by hand before submission."

    @property
    def motivation(self) -> str:
        return (
            "SciPy ships its own CG implementation. Running it on the same"
            " datasets as the array-programming CG shows what the framework"
            " abstraction costs relative to a library solver."
        )

    @property
    def generators(self):
        return [CGTestGenerator(), CGGenerator()]

    def check(self, param):
        CGBenchmark.check(self, param)

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        A, b, x = data
        # scipy stops on norm(b - A @ x) <= max(rtol*norm(b), atol), matching the
        # max(rel_tol*norm(b), abs_tol) test the array-programming CG applies.
        x_solution, _info = scipy_spla.cg(
            A,
            b,
            x0=x,
            rtol=meta.get("rel_tol", 1e-6),
            atol=meta.get("abs_tol", 1e-20),
            maxiter=meta.get("max_iter", 100),
        )
        return [x_solution]
