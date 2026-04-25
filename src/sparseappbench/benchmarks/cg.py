import os
from typing import Any

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from sparseappbench.binsparse_format import BinsparseFormat

xp = sparseappbench.xp


class CGDataset(Dataset):
    def __init__(self, source_name: str, has_b_file: bool = False, nnz: int | None = None):
        self.source_name = source_name
        self.has_b_file = has_b_file
        self.nnz = nnz

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"CG {self.source_name}"

    @property
    def description(self) -> str:
        return f"SuiteSparse matrix {self.source_name}."

    @property
    def tags(self) -> list[str]:
        return ["suitesparse", "sparse"]

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        data["has_b_file"] = self.has_b_file
        return data


class CGGenerator(Generator[CGDataset]):
    @property
    def name(self) -> str:
        return "cg_inputs"

    @property
    def pretty_name(self) -> str:
        return "Conjugate Gradient SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Accesses and prepares symmetric positive definite matrices from "
            "SuiteSparse for conjugate gradient."
        )

    @property
    def tags(self) -> list[str]:
        return ["iterative", "solver", "suitesparse", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of "
            "symmetric positive definite matrices, particularly those "
            "with a low convergence criteria"
        )

    @property
    def datasets(self) -> list[CGDataset]:
        return [
            CGDataset("mesh3em5", nnz=1889),
            CGDataset("bcsstm02", nnz=66),
            CGDataset("fv1", nnz=85264),
            CGDataset("Muu", nnz=170134),
            CGDataset("Chem97ZtZ", nnz=7361),
            CGDataset("Dubcova1", nnz=253009),
            CGDataset("t3dl_e", nnz=20360),
            CGDataset("bcsstk09", nnz=18437),
        ]

    def generate(
        self, dataset: CGDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        matrices = ssgetpy.search(name=dataset.source_name)
        if not matrices:
            raise ValueError(f"No matrix found with name '{dataset.source_name}'")
        matrix = matrices[0]
        path, _archive = matrix.download(extract=True)
        matrix_path = os.path.join(path, f"{matrix.name}.mtx")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

        A = mmread(matrix_path).tocoo()
        rng = np.random.default_rng(0)

        if dataset.has_b_file:
            b_path = os.path.join(path, f"{matrix.name}_b.mtx")
            if not os.path.exists(b_path):
                raise FileNotFoundError(f"Matrix file not found at {b_path}")
            b = mmread(b_path)
            if not isinstance(b, np.ndarray):
                b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
            b = b.flatten()
        else:
            x_rand = random(
                A.shape[1],
                1,
                density=0.1,
                format="coo",
                dtype=np.float64,
                random_state=rng,
            )
            b = A @ x_rand
            b = b.toarray().flatten()

        x = np.zeros(A.shape[1])

        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        b_bin = BinsparseFormat.from_numpy(b)
        x_bin = BinsparseFormat.from_numpy(x)

        return [A_bin, b_bin, x_bin], {}


class CGBenchmark(Benchmark):
    @property
    def tag(self) -> str:
        return "cg_solver"

    @property
    def name(self) -> str:
        return "cg_solver"

    @property
    def pretty_name(self) -> str:
        return "Conjugate Gradient Iterative Solver"

    @property
    def description(self) -> str:
        return "Solves sparse symmetric positive definite linear systems with CG."

    @property
    def tags(self) -> list[str]:
        return ["iterative", "solver", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

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
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Each iteration of the conjugate gradient requires a SpMV to compute Ap"
            "which can be done in O(nnz) time rather than O(n^2) time "
            "for dense matrices."
        )

    @property
    def generators(self):
        return [CGGenerator()]

    def benchmark(self, data: list, meta: dict):
        A, b, x = data
        rel_tol = 1e-8
        abs_tol = 1e-20
        max_iters = 10_000

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        tol_sq = tolerance * tolerance

        r = b - A @ x
        p = r
        rr = xp.vecdot(r, r)[()]
        it = 0

        if rr >= tol_sq:
            while it < max_iters:
                x = x
                r = r
                p = p

                Ap = A @ p
                alpha = rr / xp.vecdot(p, Ap)[()]
                x += alpha * p
                r -= alpha * Ap

                x = x
                r = r
                p = p
                old_rr = rr
                new_rr = xp.vecdot(r, r)[()]
                rr = new_rr

                it += 1

                if rr < tol_sq:
                    break

                beta = new_rr / old_rr
                p = r + beta * p

        if rr >= tol_sq:
            raise RuntimeError(
                "Conjugate gradient did not converge "
                "within the maximum number of iterations"
            )

        return [x]
