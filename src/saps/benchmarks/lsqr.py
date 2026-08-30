from typing import Any

import numpy as np

from binsparse import BinsparseTensor

from saps.benchmark import (
    Benchmark,
    Contributor,
    DataInstance,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import (
    SuiteSparseDataset,
    fetch_suitesparse_linear_system,
)


def normof2(xp, x, y):
    return xp.sqrt(xp.sum(xp.multiply(x, y)))


class LSQRDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        nnz: int | None = None,
        noise_amt: float = 0.1,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        convergence: str | None = None,
    ):
        super().__init__(
            source_name,
            pretty_name=f"LSQR {source_name}",
            suites=suites,
            nnz=nnz,
        )
        self.noise_amt = noise_amt
        self.A = A
        self.b = b
        self.convergence = convergence

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["noise_amt"] = self.noise_amt
        return data


class LSQRTestGenerator(Generator[LSQRDataset]):
    @property
    def name(self) -> str:
        return "lsqr_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "LSQR Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined matrices from the LSQR pytest examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return LSQRBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return LSQRBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Uses small inlined least-squares systems to verify convergence."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LSQRDataset]:
        return [
            LSQRDataset(
                "test_lsqr_underdetermined_3",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0]]),
                b=np.array([4.1, 10.1]),
                convergence="residual",
            ),
            LSQRDataset(
                "test_lsqr_overdetermined_3",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [7.0, 2.0, 1.0],
                        [2.0, 6.0, -1.0],
                        [1.0, -1.0, 5.0],
                        [4.0, -3.0, 1.0],
                    ]
                ),
                b=np.array([13.2, -3.3, 8.1, 12.4]),
                convergence="gradient",
            ),
            LSQRDataset(
                "test_lsqr_exact_3",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                b=np.array([4.0, 8.0, 16.0]),
                convergence="residual",
            ),
            LSQRDataset(
                "test_lsqr_underdetermined_4",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [8.0, -1.0, 0.0, 0.0],
                        [-1.0, 8.0, -1.0, 0.0],
                        [0.0, -1.0, 8.0, -1.0],
                    ]
                ),
                b=np.array([8.1, -2.2, 6.3]),
                convergence="residual",
            ),
            LSQRDataset(
                "test_lsqr_overdetermined_sparse",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [12.0, 2.0, -1.0],
                        [2.0, 10.0, 3.0],
                        [-1.0, 3.0, 9.0],
                        [5.0, 1.0, 2.0],
                    ]
                ),
                b=np.array([40.1, 10.2, -18.3, 15.4]),
                convergence="gradient",
            ),
            LSQRDataset(
                "test_lsqr_exact_4",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [8.0, -1.0, 0.0, 0.0],
                        [-1.0, 8.0, -1.0, 0.0],
                        [0.0, -1.0, 8.0, -1.0],
                        [0.0, 0.0, -1.0, 8.0],
                    ]
                ),
                b=np.array([8.0, -2.0, 6.0, 15.0]),
                convergence="residual",
            ),
            LSQRDataset(
                "test_lsqr_scaled_underdetermined",
                suites=["test", "trace"],
                A=np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0]]),
                b=np.array([118.1, 116.1]),
                convergence="residual",
            ),
            LSQRDataset(
                "test_lsqr_overdetermined_dense",
                suites=["test", "trace"],
                A=np.array(
                    [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0], [2.0, 1.0, 3.0]]
                ),
                b=np.array([5.1, 7.2, 11.3, 12.4]),
                convergence="gradient",
            ),
            LSQRDataset(
                "test_lsqr_exact_5",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [15.0, -2.0, 0.0, 0.0, -1.0],
                        [-2.0, 14.0, -3.0, 0.0, 0.0],
                        [0.0, -3.0, 16.0, -2.0, 0.0],
                        [0.0, 0.0, -2.0, 15.0, -3.0],
                        [-1.0, 0.0, 0.0, -3.0, 17.0],
                    ]
                ),
                b=np.array([27.0, -1.0, -18.0, 8.0, 46.0]),
                convergence="residual",
            ),
        ]

    def generate(self, dataset: LSQRDataset) -> DataInstance:
        if dataset.A is None or dataset.b is None or dataset.convergence is None:
            raise ValueError("LSQR test datasets must define A, b, and convergence.")
        return DataInstance(
            inputs=[
                BinsparseTensor.from_numpy(dataset.A),
                BinsparseTensor.from_numpy(dataset.b),
            ],
            meta={},
            ref_meta={"convergence": dataset.convergence},
        )


class LSQRGenerator(Generator[LSQRDataset]):
    @property
    def name(self) -> str:
        return "lsqr_inputs"

    @property
    def pretty_name(self) -> str:
        return "LSQR SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of "
            "square and rectangular matrices. Following the methodology of "
            "Paige and Saunders the problems span a range of convergence "
            "criteria from well-conditioned to extremely ill-conditioned."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return LSQRBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return LSQRBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return self.description

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LSQRDataset]:
        return [
            LSQRDataset("abb313"),
            LSQRDataset("ash958"),
            LSQRDataset("well1033"),
            LSQRDataset("Maragal_5"),
            LSQRDataset("illc1850"),
            LSQRDataset("bayer06"),
        ]

    def generate(self, dataset: LSQRDataset):
        A_bin, b, has_real_rhs = fetch_suitesparse_linear_system(dataset.source_name)
        if not has_real_rhs:
            # Adds a small amount of noise so that Ax != b
            rng = np.random.default_rng(0)
            noise_level = dataset.noise_amt * np.linalg.norm(b)
            noise = rng.standard_normal(b.shape) * noise_level
            b = b + noise

        b_bin = BinsparseTensor.from_numpy(b)
        return DataInstance(inputs=[A_bin, b_bin], meta={})


class LSQRBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "lsqr"

    @property
    def pretty_name(self) -> str:
        return "LSQR Iterative Solver"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def description(self) -> str:
        return (
            "Hand written code based on the algorithm defined in Paige and "
            "Saunders' LSQR paper. Implementation structure was also based "
            "around the work of Michael Friedlander and Dominique Orban."
        )

    @property
    def motivation(self) -> str:
        return (
            '"[LSQR] is analytically equivalent to the standard method of '
            "conjugate gradients, but possesses more favorable numerical "
            "properties...  Numerical tests are described comparing LSQR with "
            "several other conjugate-gradient algorithms, indicating that LSQR "
            'is the most reliable algorithm when A is ill-conditioned." '
            "C. C. Paige and M. A. Saunders, "
            '"LSQR: An Algorithm for Sparse Linear Equations and Sparse Least '
            'Squares," ACM Transactions on Mathematical Software, vol. 8, '
            "no. 1, 1982, p. 43. The main computation of the algorithm is 2 SpMVs per"
            " iteration Av and ATu. Through sparsity the computation is lowered from"
            " two operations of O(n^2) to two O(nnz). This efficiency allows the"
            " algorithm to handle massive, ill-conditioned systems with very low"
            " storage requirements."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="",
                authors=[],
                url="https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf",
            ),
            Ref(
                title="",
                authors=[],
                url="https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py#L293",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return (
            """
        <ccs2012>
        <concept>
        <concept_id>10002950.10003705.10003707</concept_id>
        <concept_desc>Mathematics of computing~Solvers</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10002950.10003705.10011686</concept_id>
        <concept_desc>Mathematics of computing~"""
            "Mathematical software performance"
            """</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10002950.10003714.10003715</concept_id>
        <concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        </ccs2012>
        """
        )

    @property
    def generators(self):
        return [LSQRTestGenerator(), LSQRGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or "convergence" not in self._ref_meta:
            return

        A_bin, b_bin = self._input
        A = A_bin.data["values"].reshape(A_bin.data["shape"])
        b = b_bin.data["values"].reshape(b_bin.data["shape"])
        x_sol = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        residual = b - A @ x_sol

        if self._ref_meta["convergence"] == "residual":
            assert np.linalg.norm(residual) < 1e-5 * np.linalg.norm(b) + 1e-5
        elif self._ref_meta["convergence"] == "gradient":
            assert np.linalg.norm(A.T @ residual) < (
                1e-5 * np.linalg.norm(A.T @ b) + 1e-5
            )

    def benchmark(self, xp, data: list, meta: dict):
        A, b = data
        atol = meta.get("atol", 1e-9)
        btol = meta.get("btol", 1e-9)
        conlim = meta.get("conlim", 1.0e8)
        max_iters = meta.get("max_iters", 10000)
        exit = 0

        u = b
        beta = normof2(xp, u, u)
        u = u / beta

        v = A.T @ u
        alpha = normof2(xp, v, v)
        v = v / alpha

        solution_is_zero = False
        bnorm = beta
        ctol = 1 / conlim

        Arnorm = alpha * beta
        if Arnorm == 0:
            solution_is_zero = True

        w = v
        x = xp.zeros(A.shape[1])
        phi_bar = beta
        rho_bar = alpha
        it = 0

        # An approximation of the Frobenius norm of A squared using an
        # iterative update by summing the squares of the scalars alpha and beta
        Anorm_sq = beta**2

        # An approximation of the vector norm of x squared based on the
        # step size contributing each iteration
        xnorm_sq = 0

        # The Fronbenius norm squared of the matrix of search directions
        # updated by adding the squared norm of each search direction
        dnorm_sq = 0

        # An approximation of the condition number of A found by multiplying
        # Anorm by sqrt(ddnorm)
        Acond = 0

        while it < max_iters and not solution_is_zero:
            it += 1

            u = A @ v - alpha * u

            beta = normof2(xp, u, u)
            u = u / beta

            v = A.T @ u - beta * v
            alpha = normof2(xp, v, v)
            v = v / alpha

            rho = xp.sqrt(rho_bar**2 + beta**2)
            c = rho_bar / rho
            s = beta / rho
            theta = s * alpha
            rho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar *= s
            step = phi / rho

            x += step * w

            dk = 1.0 / rho * w
            dnorm_sq += xp.sum(xp.multiply(dk, dk))

            w = v - (theta / rho) * w

            # Estimate for the size of the residual r = b - Ax
            rnorm = abs(phi_bar)

            # Estimate of the norm of the gradient ATr
            Arnorm = alpha * abs(phi_bar * c)

            Anorm_sq += alpha**2 + beta**2
            Anorm = xp.sqrt(Anorm_sq)

            xnorm_sq += step**2
            xnorm = xp.sqrt(xnorm_sq)

            Acond = Anorm * xp.sqrt(dnorm_sq)

            test1 = rnorm / bnorm
            test2 = Arnorm / (Anorm * rnorm)
            test3 = 1 / Acond

            reltol = atol * Anorm * xnorm / bnorm + btol

            # Exits if the condition number grows too high
            if test3 <= ctol:
                exit = 3
            # Exits if the gradient is small so the min has been found
            if test2 <= atol:
                exit = 2
            # Exits if the residual is small so we have found the solution
            if test1 <= reltol:
                exit = 1

            if exit > 0:
                break

        return [x]
