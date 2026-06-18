import os
from typing import Any

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

import saps
from saps.benchmark import (
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


class GMRESDataset(Dataset):
    def __init__(
        self, source_name: str, has_b_file: bool = False, nnz: int | None = None
    ):
        self._tags = []
        self.source_name = source_name
        self.has_b_file = has_b_file
        self.nnz = nnz

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"GMRES {self.source_name}"

    @property
    def description(self) -> str:
        return f"SuiteSparse matrix {self.source_name}."

    @property
    def tags(self) -> list[str]:
        return self._tags

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        data["has_b_file"] = self.has_b_file
        return data


class GMRESGenerator(Generator[GMRESDataset]):
    @property
    def name(self) -> str:
        return "gmres_inputs"

    @property
    def pretty_name(self) -> str:
        return "GMRES SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return "Accesses and prepares sparse matrices from SuiteSparse for GMRES."

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="",
                authors=[],
                url="https://www.netlib.org/templates/templates.pdf",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function. Generative"
            " AI might have been used to construct tests. This statement is written by"
            " hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "GMRES is the most widely used and effective method for solving "
            "linear systems that are indefinite, non-symmetric, and sparse."
        )

    @property
    def datasets(self) -> list[GMRESDataset]:
        return [
            GMRESDataset("mesh3em5", nnz=1889),
            GMRESDataset("bcsstm02", nnz=66),
            GMRESDataset("fv1", nnz=85264),
            GMRESDataset("Muu", nnz=170134),
            GMRESDataset("Chem97ZtZ", nnz=7361),
            GMRESDataset("Dubcova1", nnz=253009),
            GMRESDataset("t3dl_e", nnz=20360),
            GMRESDataset("bcsstk09", nnz=18437),
        ]

    def generate(self, dataset: GMRESDataset):
        source = dataset.source_name
        has_b_file = dataset.metadata.get("has_b_file", False)
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
        rng = np.random.default_rng(0)
        A = A.tocoo()

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
            x = random(
                A.shape[1],
                1,
                density=0.1,
                format="coo",
                dtype=np.float64,
                random_state=rng,
            )
            b = A @ x
            b = b.toarray().flatten()
        x = np.zeros(A.shape[1])

        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        b_bin = BinsparseFormat.from_numpy(b)
        x_bin = BinsparseFormat.from_numpy(x)
        return [A_bin, b_bin, x_bin], {}


class GMRESBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gmres"

    @property
    def pretty_name(self) -> str:
        return "GMRES Iterative Solver"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def description(self) -> str:
        return (
            "This code is implements the GMRES algorithm for solving indefinite and"
            " non-symmetric linear systems. The algorithm follows the Arnoldi iteration"
            " process where a Krylov matrix is maintained at each iteration. Starting"
            " with an initial guess and the residual for that guess, the matrix A is"
            " dot producted with the previous residual to obtain the next basis vector."
            " This algorithm also uses a similar method to Gram-Schmidt to ensure that"
            " the Kyrlov matrix is orthogonal. I also maintain an upper Hessenberg"
            " matrix which keeps track of the dot products between different basis"
            " vectors and the norm of the new basis vector. The Hessenberg matrix"
            " follows the property: Q_n * A = Q_(n+1) * H_n where Q is the Krylov"
            " matrix. This matrix allows for a simplified least squares problem to be"
            " solved at each iteration so that the residual is minimized at each step."
            " My implementation restarts the Kyrlov matrix every 50 iterations and will"
            " end when the current residual / initial residual is less than the"
            " tolerance level."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="",
                authors=[],
                url="https://www.netlib.org/templates/templates.pdf",
            ),
            Ref(
                title="",
                authors=[],
                url="https://www.netlib.org/utk/people/JackDongarra/PAPERS/sparse-bench.pdf",
            ),
        ]

    @property
    def motivation(self) -> str:
        return (
            "GMRES is the most widely used and effective method for solving linear"
            " systems that are indefinite, non-symmetric, and are sparse in nature."
        )

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function. Generative"
            " AI might have been used to construct tests. This statement is written by"
            " hand."
        )

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def generators(self):
        return [GMRESGenerator()]

    def benchmark(self, data: list, meta: dict):
        A, b, x0 = data
        restart = meta.get("restart", 50)
        tol = meta.get("tol", 1e-8)
        max_iter = meta.get("max_iter", 1000)

        itcount = 0
        r0 = b - A @ x0
        initial_beta = xp.linalg.norm(r0)[()]
        if initial_beta < tol:
            return [x0]

        rcurr = r0 / initial_beta
        beta = initial_beta

        while itcount < max_iter:
            Q = xp.zeros((A.shape[0], restart + 1), dtype=float)
            H = xp.zeros((restart + 1, restart), dtype=float)
            Q[:, 0] = rcurr

            x_cycle_start = x0
            for i in range(restart):
                x0 = (x0, rcurr)
                rcurr = A @ Q[:, i]

                H[: i + 1, i] = xp.vecdot(Q[:, : i + 1].T, rcurr)
                rcurr = rcurr - Q[:, : i + 1] @ H[: i + 1, i]
                H[i + 1, i] = xp.linalg.norm(rcurr)[()]
                Q[:, i + 1] = rcurr / H[i + 1, i]

                e1 = xp.zeros((i + 2,), dtype=float)
                e1[0] = beta

                H_reduced = H[: i + 2, : i + 1]
                coeffs, _, _, _ = xp.linalg.lstsq(H_reduced, e1, rcond=None)
                x0 = x_cycle_start + Q[:, : i + 1] @ coeffs

                r0 = b - A @ x0
                r0_norm = xp.linalg.norm(r0)[()]
                rcurr = r0 / r0_norm
                if r0_norm / initial_beta < tol:
                    return [x0]

                itcount += 1
                if itcount >= max_iter:
                    break

            beta = r0_norm

        xsol = x0
        return [xsol]
