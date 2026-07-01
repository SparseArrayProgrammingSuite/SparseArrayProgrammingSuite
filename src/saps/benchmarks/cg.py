import os
from typing import Any

import numpy as np

import sparse as pydata_sparse

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework.binsparse_format import BinsparseFormat

DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-20
DEFAULT_MAX_ITERS = 1000


class CGDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        has_b_file: bool = False,
        nnz: int | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        x: np.ndarray | None = None,
        rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        self._suites = suites or []
        self.source_name = source_name
        self.has_b_file = has_b_file
        self.nnz = nnz
        self.A = A
        self.b = b
        self.x = x
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.max_iters = max_iters

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
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        data["has_b_file"] = self.has_b_file
        data["rel_tol"] = self.rel_tol
        data["abs_tol"] = self.abs_tol
        data["max_iters"] = self.max_iters
        return data


class CGTestGenerator(Generator[CGDataset]):
    @property
    def name(self) -> str:
        return "cg_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Conjugate Gradient Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined matrices from the CG pytest examples."

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Uses small inlined linear systems to verify solver correctness."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[CGDataset]:
        return [
            CGDataset(
                "test_3x3_tridiagonal",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                b=np.array([4.0, 8.0, 16.0]),
                x=np.zeros((3,)),
            ),
            CGDataset(
                "test_3x3_dense",
                suites=["test", "trace"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                b=np.array([13.0, -3.0, 8.0]),
                x=np.zeros((3,)),
            ),
            CGDataset(
                "test_4x4_tridiagonal",
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
                x=np.zeros((4,)),
            ),
            CGDataset(
                "test_3x3_indefinite_sparse",
                suites=["test", "trace"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                b=np.array([40.0, 10.0, -18.0]),
                x=np.zeros((3,)),
            ),
            CGDataset(
                "test_3x3_scaled_tridiagonal",
                suites=["test", "trace"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                b=np.array([118.0, 116.0, 118.0]),
                x=np.zeros((3,)),
            ),
            CGDataset(
                "test_5x5_sparse",
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
                x=np.zeros((5,)),
            ),
        ]

    def generate(self, dataset: CGDataset) -> DataInstance:
        if dataset.A is None or dataset.b is None or dataset.x is None:
            raise ValueError("CG test datasets must define A, b, and x.")

        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(dataset.A),
                BinsparseFormat.from_numpy(dataset.b),
                BinsparseFormat.from_numpy(dataset.x),
            ],
            meta={
                "rel_tol": dataset.rel_tol,
                "abs_tol": dataset.abs_tol,
                "max_iters": dataset.max_iters,
            },
            ref_meta={"check_rounded_residual": True, "round_decimals": 4},
        )


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
            "Accesses and prepares symmetric positive definite matrices from"
            " SuiteSparse for conjugate gradient."
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
            "Data collected from SuiteSparse Matrix Collection consisting of symmetric"
            " positive definite matrices, particularly those with a low convergence"
            " criteria"
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

    def generate(self, dataset: CGDataset) -> DataInstance:
        from scipy.io import mmread
        from scipy.sparse import random

        import ssgetpy

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

        return DataInstance(
            inputs=[A_bin, b_bin, x_bin],
            meta={
                "rel_tol": dataset.rel_tol,
                "abs_tol": dataset.abs_tol,
                "max_iters": dataset.max_iters,
            },
        )


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
            " which can be done in O(nnz) time rather than O(n^2) time for dense"
            " matrices."
        )

    @property
    def generators(self):
        return [CGTestGenerator(), CGGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or not self._ref_meta.get("check_rounded_residual"):
            return

        A_bin, b_bin, _x_bin = self._input
        A_coo = BinsparseFormat.to_coo(A_bin)
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.data["indices_0"], A_coo.data["indices_1"])),
            data=A_coo.data["values"],
            shape=A_coo.data["shape"],
        )
        decimals = self._ref_meta["round_decimals"]
        x_sol = np.round(
            self._output[0].data["values"].reshape(self._output[0].data["shape"]),
            decimals=decimals,
        )

        actual_b = BinsparseFormat.to_coo(
            BinsparseFormat.from_numpy(np.asarray(A @ x_sol))
        )
        expected_b = BinsparseFormat.to_coo(b_bin)
        assert expected_b == actual_b, f"CG residual mismatch for {param.dataset.name}"

    def benchmark(self, xp, data: list, meta: dict):
        A, b, x = data
        rel_tol = meta["rel_tol"]
        abs_tol = meta["abs_tol"]
        max_iters = meta["max_iters"]

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        tol_sq = tolerance * tolerance

        r = b - A @ x
        p = r
        rr = xp.vecdot(r, r)[()]
        it = 0

        if rr >= tol_sq:
            while it < max_iters:
                Ap = A @ p
                alpha = rr / xp.vecdot(p, Ap)[()]
                x = x + alpha * p
                r = r - alpha * Ap

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
