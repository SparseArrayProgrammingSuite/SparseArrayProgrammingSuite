from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import sparse as pydata_sparse

from saps.benchmark import (
    Author,
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
from saps.downloaders.suitesparse import random_rhs_for_matrix
from saps_framework import BinsparseFormat


def _generate_cg_data(source, A=None):
    if A is not None:
        import scipy.sparse as sp

        A = sp.coo_matrix(A)
        b = random_rhs_for_matrix(A)
        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    else:
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(source)
    x0 = np.zeros(A_bin.data["shape"][1])
    return (A_bin, b, x0)


class PreconditionedCGDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        condition_number: str,
        A=None,
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
    ):
        super().__init__(
            source_name,
            pretty_name=f"Preconditioned CG {source_name}",
            suites=suites,
        )
        self.condition_number = condition_number
        self.A = A
        self.ref_meta = ref_meta

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["condition_number"] = self.condition_number
        return data


class BlockJacobiCGGenerator(Generator[PreconditionedCGDataset]):
    @property
    def name(self) -> str:
        return "block_jacobi_cg_inputs"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi CG SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of symmetric"
            " positive definite matrices, particularly those with a low convergence"
            " criteria."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return PreconditionedCGBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return PreconditionedCGBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return PreconditionedCGBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return PreconditionedCGBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset(
                "test_A0",
                "",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A1",
                "",
                suites=["test", "trace"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A2",
                "",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [8.0, -1.0, 0.0, 0.0],
                        [-1.0, 8.0, -1.0, 0.0],
                        [0.0, -1.0, 8.0, -1.0],
                        [0.0, 0.0, -1.0, 8.0],
                    ]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A3",
                "",
                suites=["test", "trace"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A4",
                "",
                suites=["test", "trace"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A5",
                "",
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
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset("mhdb416", "3994223509->6.24"),
            PreconditionedCGDataset("lund_b", "30036->36.3"),
            PreconditionedCGDataset("Chem97ZtZ", "247->8.48"),
            PreconditionedCGDataset("bcsstm12", "633194->7.14"),
            PreconditionedCGDataset("mesh1em1", "19->11.4"),
        ]

    def generate(self, dataset: PreconditionedCGDataset) -> DataInstance:
        import scipy.sparse as sp

        A_bin, b, x0 = _generate_cg_data(dataset.source_name, dataset.A)
        A_csr = A_bin.to_scipy_coo().tocsr()
        n = A_csr.shape[0]
        # Create one block for every processor modelled after
        # this example: https://petsc.org/main/src/ksp/ksp/tutorials/ex7.c.html
        p = min(10, n)
        block_size = n // p
        blocks = []
        i = 0
        while i < n:
            j = min(i + block_size, n)
            A_ii = A_csr[i:j, i:j].toarray()
            L_i = np.linalg.cholesky(A_ii)
            blocks.append(L_i)
            i = j
        M = sp.block_diag(blocks).tocoo()
        M_bin = BinsparseFormat.from_coo((M.row, M.col), M.data, M.shape)
        b_bin = BinsparseFormat.from_numpy(b)
        x0_bin = BinsparseFormat.from_numpy(x0)
        return DataInstance(
            inputs=[A_bin, b_bin, x0_bin, M_bin],
            meta={},
            ref_meta=dataset.ref_meta,
        )


class JacobiCGGenerator(Generator[PreconditionedCGDataset]):
    @property
    def name(self) -> str:
        return "jacobi_cg_inputs"

    @property
    def pretty_name(self) -> str:
        return "Jacobi CG SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of symmetric"
            " positive definite matrices, particularly those with a low convergence"
            " criteria."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return PreconditionedCGBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return PreconditionedCGBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return PreconditionedCGBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return PreconditionedCGBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset(
                "test_A0",
                "",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A1",
                "",
                suites=["test", "trace"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A2",
                "",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [8.0, -1.0, 0.0, 0.0],
                        [-1.0, 8.0, -1.0, 0.0],
                        [0.0, -1.0, 8.0, -1.0],
                        [0.0, 0.0, -1.0, 8.0],
                    ]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A3",
                "",
                suites=["test", "trace"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A4",
                "",
                suites=["test", "trace"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A5",
                "",
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
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset("mhdb416", "3994223509->69.7"),
            PreconditionedCGDataset("lund_b", "30036->144"),
            PreconditionedCGDataset("Chem97ZtZ", "247->8.48"),
            PreconditionedCGDataset("bcsstm12", "633194->3160"),
            PreconditionedCGDataset("mesh1em1", "19->11.6"),
        ]

    def generate(self, dataset: PreconditionedCGDataset) -> DataInstance:
        A_bin, b, x0 = _generate_cg_data(dataset.source_name, dataset.A)
        M = A_bin.diagonal()
        M_bin = BinsparseFormat.from_numpy(M)
        b_bin = BinsparseFormat.from_numpy(b)
        x0_bin = BinsparseFormat.from_numpy(x0)
        return DataInstance(
            inputs=[A_bin, b_bin, x0_bin, M_bin],
            meta={},
            ref_meta=dataset.ref_meta,
        )


class _PreconditionedCGBase(Benchmark, ABC):
    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def description(self) -> str:
        return (
            "Hand-written code modelling the algorithm structure outlined in "
            "https://www.netlib.org/templates/templates.pdf Page 13."
        )

    @property
    def motivation(self) -> str:
        return (
            '"The preconditioned conjugate gradient method is well established '
            "for solving linear systems of equations that arise from the "
            "discretization of partial differential equations. Point and block "
            'Jacobi preconditioning are both common preconditioning techniques." '
            "Sparsity enhances the functionality of both the solver and the "
            "preconditioner. Similar to normal conjugate gradient, the SpMV "
            "done once per iteration reduces complexity from O(n^2) to O(nnz). "
            "Furthermore, the sparse block Jacobi preconditioner avoids filling "
            "in all the 0s around the blocks, which prevents memory overhead "
            "and keeps the per-iteration block solve cost proportional to the "
            "block size instead of the full matrix dimension."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Block Jacobi Preconditioning of the Conjugate Gradient Method on"
                    " a Vector Processor"
                ),
                authors=[Author("M. Hegland"), Author("P. E. Saylor")],
                journal="International Journal of Computer Mathematics",
                volume=44,
                number="1-4",
                pages="71-89",
                year=1992,
            ),
            Ref(
                title="",
                authors=[],
                url="https://www.netlib.org/templates/templates.pdf",
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
        return """
        <ccs2012>
        <concept>
        <concept_id>10002950.10003705.10003707</concept_id>
        <concept_desc>Mathematics of computing~Solvers</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10002950.10003705.10011686</concept_id>
        <concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10002950.10003714.10003715</concept_id>
        <concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        </ccs2012>
        """

    @abstractmethod
    def _solve_cg(self, xp, M, r):
        raise NotImplementedError

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or not self._ref_meta.get("check_residual"):
            return

        A_bin, b_bin, _x0_bin, _M_bin = self._input
        A_coo = BinsparseFormat.to_coo(A_bin)
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.data["indices_0"], A_coo.data["indices_1"])),
            data=A_coo.data["values"],
            shape=A_coo.data["shape"],
        )
        b = b_bin.data["values"].reshape(b_bin.data["shape"])
        x_sol = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        residual = b - A @ x_sol
        assert np.linalg.norm(residual) < 1e-6 * np.linalg.norm(b) + 1e-6, (
            f"Preconditioned CG residual too high for {param.dataset.name}"
        )

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        A, b, x0, M = data
        rel_tol = meta.get("rel_tol", 1e-8)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iters = meta.get("max_iters", 10000)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        # tol_sq used to avoid having to sqrt dot products when checking tolerance
        tol_sq = tolerance * tolerance

        x = x0
        r = b - A @ x
        z = self._solve_cg(xp, M, r)
        rho = xp.vecdot(r, z)
        p = z
        it = 0
        rr = xp.vecdot(r, r)[()]

        if rr >= tol_sq:
            while it < max_iters:
                Ap = A @ p
                alpha = rho / xp.vecdot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap

                new_rr = xp.vecdot(r, r)[()]

                it += 1

                if new_rr < tol_sq:
                    break

                z = self._solve_cg(xp, M, r)
                new_rho = xp.vecdot(r, z)
                beta = new_rho / rho
                p = z + beta * p
                rho = new_rho
                rr = new_rr

        x_solution = x
        return [x_solution]


class _BlockJacobiCGMixin:
    @property
    def generators(self):
        return [BlockJacobiCGGenerator()]

    def _solve_cg(self, xp, M, r):
        y = xp.linalg.solve(M, r)
        return xp.linalg.solve(M.T, y)


class _JacobiCGMixin:
    @property
    def generators(self):
        return [JacobiCGGenerator()]

    def _solve_cg(self, xp, M, r):
        output = r / M
        if hasattr(xp, "with_fill_value"):
            return xp.with_fill_value(output, 0)
        return output


class PreconditionedCGBenchmark(_BlockJacobiCGMixin, _PreconditionedCGBase):
    @property
    def name(self) -> str:
        return "preconditioned_cg"

    @property
    def pretty_name(self) -> str:
        return "Preconditioned Conjugate Gradient (Block Jacobi)"


class JacobiPreconditionedCGBenchmark(_JacobiCGMixin, _PreconditionedCGBase):
    @property
    def name(self) -> str:
        return "jacobi_preconditioned_cg"

    @property
    def pretty_name(self) -> str:
        return "Preconditioned Conjugate Gradient (Jacobi)"
