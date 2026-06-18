import os
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

import ssgetpy

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat


def _generate_cg_data(source, has_b_file, A=None):
    if A is not None:
        A = sp.coo_matrix(A)
    else:
        matrices = ssgetpy.search(name=source)
        if not matrices:
            raise ValueError(f"No matrix found with name '{source}'")
        matrix = matrices[0]
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    A = A.tocoo()
    rng = np.random.default_rng(0)

    if has_b_file:
        matrix_path = os.path.join(path, matrix.name + "_b.mtx")
        if matrix_path and os.path.exists(matrix_path):
            b = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        if not isinstance(b, np.ndarray):
            b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        b = b.flatten()
    else:
        x = sp.random(
            A.shape[1], 1, density=0.1, format="coo", dtype=np.float64, random_state=rng
        )
        b = A @ x
        b = b.toarray().flatten()
    x = np.zeros(A.shape[1])
    return (A, b, x)


xp = saps.xp


class PreconditionedCGDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        condition_number: str,
        has_b_file=False,
        A=None,
    ):
        self._tags = []
        self.source_name = source_name
        self.condition_number = condition_number
        self.has_b_file = has_b_file
        self.A = A

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"Preconditioned CG {self.source_name}"

    @property
    def description(self) -> str:
        return f"SuiteSparse matrix {self.source_name}."

    @property
    def tags(self) -> list[str]:
        return self._tags

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["condition_number"] = self.condition_number
        data["has_b_file"] = self.has_b_file
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
    def tags(self) -> list[str]:
        return []

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
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset("mhdb416", "3994223509->6.24"),
            PreconditionedCGDataset("lund_b", "30036->36.3"),
            PreconditionedCGDataset("Chem97ZtZ", "247->8.48"),
            PreconditionedCGDataset("bcsstm12", "633194->7.14"),
            PreconditionedCGDataset("mesh1em1", "19->11.4"),
        ]

    def generate(
        self, dataset: PreconditionedCGDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        A, b, x = _generate_cg_data(dataset.source_name, dataset.has_b_file, dataset.A)
        A_csr = A.tocsr()
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
        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        b_bin = BinsparseFormat.from_numpy(b)
        x_bin = BinsparseFormat.from_numpy(x)
        return [A_bin, b_bin, x_bin, M_bin], {"solve": solve_block_jacobi_cg}


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
    def tags(self) -> list[str]:
        return []

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
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset("mhdb416", "3994223509->69.7"),
            PreconditionedCGDataset("lund_b", "30036->144"),
            PreconditionedCGDataset("Chem97ZtZ", "247->8.48"),
            PreconditionedCGDataset("bcsstm12", "633194->3160"),
            PreconditionedCGDataset("mesh1em1", "19->11.6"),
        ]

    def generate(
        self, dataset: PreconditionedCGDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        A, b, x = _generate_cg_data(dataset.source_name, dataset.has_b_file, dataset.A)
        M = A.diagonal()
        M_bin = BinsparseFormat.from_numpy(M)
        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        b_bin = BinsparseFormat.from_numpy(b)
        x_bin = BinsparseFormat.from_numpy(x)
        return [A_bin, b_bin, x_bin, M_bin], {"solve": solve_jacobi_cg}


class PreconditionedCGBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "preconditioned_cg"

    @property
    def pretty_name(self) -> str:
        return "Preconditioned Conjugate Gradient (Block Jacobi)"

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
    def tags(self) -> list[str]:
        return []

    @property
    def generators(self):
        return [BlockJacobiCGGenerator(), JacobiCGGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]):
        A, b, x, M = data
        solve_cg = meta["solve"]
        rel_tol = meta.get("rel_tol", 1e-8)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iters = meta.get("max_iters", 10000)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        # tol_sq used to avoid having to sqrt dot products when checking tolerance
        tol_sq = tolerance * tolerance

        r = b - A @ x
        z = solve_cg(xp, M, r)
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

                z = solve_cg(xp, M, r)
                new_rho = xp.vecdot(r, z)
                beta = new_rho / rho
                p = z + beta * p
                rho = new_rho
                rr = new_rr

        x_solution = x
        return [x_solution]


def solve_block_jacobi_cg(xp, M, r):
    y = xp.linalg.solve(M, r)
    return xp.linalg.solve(M.T, y)


def solve_jacobi_cg(xp, M, r):
    output = r / M
    return xp.with_fill_value(output, 0) if hasattr(xp, "with_fill_value") else output
