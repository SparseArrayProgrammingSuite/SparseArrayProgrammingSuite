import os
from typing import Any

import numpy as np

import sparse as pydata_sparse

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


class JacobiDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        has_b_file: bool = False,
        nnz: int | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ):
        self._suites = suites or []
        self.source_name = source_name
        self.has_b_file = has_b_file
        self.nnz = nnz
        self.A = A
        self.b = b
        self.x = x

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"Jacobi {self.source_name}"

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
        return data


class JacobiTestGenerator(Generator[JacobiDataset]):
    @property
    def name(self) -> str:
        return "jacobi_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Jacobi Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined matrices from the Jacobi pytest examples."

    @property
    def suites(self) -> list[str]:
        return ["test"]

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
    def datasets(self) -> list[JacobiDataset]:
        return [
            JacobiDataset(
                "test_3x3",
                suites=["test"],
                A=np.array(
                    [[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]
                ),
                b=np.array([5.0, 8.0, 8.0]),
                x=np.zeros((3,)),
            ),
            JacobiDataset(
                "test_4x4",
                suites=["test"],
                A=np.array(
                    [
                        [10.0, 1.0, 0.0, 2.0],
                        [1.0, 8.0, 1.0, 0.0],
                        [0.0, 2.0, 9.0, 1.0],
                        [1.0, 0.0, 1.0, 7.0],
                    ]
                ),
                b=np.array([16.0, 18.0, 15.0, 16.0]),
                x=np.zeros((4,)),
            ),
            JacobiDataset(
                "test_3x3_dominant",
                suites=["test"],
                A=np.array(
                    [[20.0, 3.0, 1.0], [2.0, 15.0, 4.0], [1.0, 2.0, 18.0]]
                ),
                b=np.array([24.0, 21.0, 21.0]),
                x=np.zeros((3,)),
            ),
        ]

    def generate(self, dataset: JacobiDataset):
        if dataset.A is None or dataset.b is None or dataset.x is None:
            raise ValueError("Jacobi test datasets must define A, b, and x.")

        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(dataset.A),
                BinsparseFormat.from_numpy(dataset.b),
                BinsparseFormat.from_numpy(dataset.x),
            ],
            meta={},
            ref_meta={"check_rounded_residual": True, "round_decimals": 4},
        )

class JacobiGenerator(Generator[JacobiDataset]):
    @property
    def name(self) -> str:
        return "jacobi_inputs"

    @property
    def pretty_name(self) -> str:
        return "Jacobi SuiteSparse Data Generator"

    @property
    def description(self) -> str:
        return (
            "Accesses and prepares symmetric "
            "positive definite matrices from SuiteSparse."
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
            " positive definite matrices whose Jacobi iteration matrices have spectral"
            " radius < 1."
        )

    @property
    def datasets(self) -> list[JacobiDataset]:
        return [
            JacobiDataset("mesh3em5", nnz=1889),
            JacobiDataset("Trefethen_200", nnz=2873),
            JacobiDataset("Chem97ZtZ", nnz=7361),
            JacobiDataset("Trefethen_500", nnz=8478),
            JacobiDataset("Trefethen_700", nnz=12654),
            JacobiDataset("fv1", nnz=85264),
            JacobiDataset("fv2", nnz=87025),
            JacobiDataset("Trefethen_20000", nnz=554466),
        ]

    def generate(self, dataset: JacobiDataset):
        from scipy.io import mmread
        from scipy.sparse import random

        import ssgetpy

        matrices = ssgetpy.search(name=dataset.source_name)
        if not matrices:
            raise ValueError(f"No matrix found with name '{dataset.source_name}'")
        matrix = matrices[0]
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")

        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")

        rng = np.random.default_rng(0)
        A = A.tocoo()

        if dataset.has_b_file:
            matrix_path_b = os.path.join(path, matrix.name + "_b.mtx")
            if matrix_path_b and os.path.exists(matrix_path_b):
                b = mmread(matrix_path_b)
            else:
                raise FileNotFoundError(f"Matrix file not found at {matrix_path_b}")
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

        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta={})


class JacobiBenchmark(Benchmark):
    @property
    def tag(self) -> str:
        return "jacobi_solver"

    @property
    def name(self) -> str:
        return "jacobi_solver"

    @property
    def pretty_name(self) -> str:
        return "Jacobi Iterative Solver"

    @property
    def description(self) -> str:
        return "Solves linear systems using the Jacobi iterative method."

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
                title="Probabilistic iterative methods for linear systems",
                authors=[
                    Author("J. Cockayne"),
                    Author("I. C. F. Ipsen"),
                    Author("C. J. Oates"),
                    Author("T. W. Reid"),
                ],
                journal="J. Mach. Learn. Res.",
                volume="22",
                pages="1-34",
                year=2021,
                url="https://www.jmlr.org/papers/volume22/21-0031/21-0031.pdf",
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
            "Sparsity makes the Jacobi method efficient because each update only needs "
            "to access the nonzero entries, reducing complexity from O(mn) to O(nnz)"
        )

    @property
    def generators(self):
        return [JacobiTestGenerator(), JacobiGenerator()]

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
        assert expected_b == actual_b, (
            f"Jacobi residual mismatch for {param.dataset.name}"
        )

    def _norm(self, xp, v):
        return xp.sqrt(xp.sum(xp.multiply(v, v)))

    def benchmark(self, data: list, meta: dict):
        A, b, x = data

        rel_tol = 1e-6
        abs_tol = 1e-20
        max_iters = 1000

        tolerance = max(rel_tol * self._norm(xp, b)[()], abs_tol)
        d = xp.with_fill_value(xp.diagonal(A), 1)
        if xp.any(d == 0):
            raise ValueError("Jacobi requires nonzero diagonal entries.")

        r = b - A @ x
        it = 0

        while self._norm(xp, r)[()] >= tolerance and it < max_iters:
            x = x + r / d

            r = b - A @ x
            it += 1
        if it >= max_iters:
            raise RuntimeError(
                "Jacobi did not converge within the maximum number of iterations"
            )
        return [x]


# Matrices below run extremely slowly on numpy framework (>1 minutes per convergence):

# def dg_jacobi_sparse_9():
#     return generate_jacobi_data("obstclae")  # nnz = 197,608

# def dg_jacobi_sparse_10():
#     return generate_jacobi_data("minsurfo")  # nnz = 203,622

# def dg_jacobi_sparse_11():
#     return generate_jacobi_data("jnlbrng1") #nnz = 199,200

# def dg_jacobi_sparse_12():
#     return generate_jacobi_data("shallow_water1") #nnz = 327,680

# def dg_jacobi_sparse_13():
#     return generate_jacobi_data("shallow_water2") #nnz = 327,680

# def dg_jacobi_sparse_14():
#     return generate_jacobi_data("finan512") #nnz = 596,992
