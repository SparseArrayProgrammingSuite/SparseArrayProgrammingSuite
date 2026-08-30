import numpy as np
import scipy.sparse as scipy_sparse

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy, to_scipy

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
from saps_framework.binsparse_utils import binsparse_equal


class CGDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        nnz: int | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ):
        super().__init__(
            source_name,
            pretty_name=f"CG {source_name}",
            suites=suites,
            nnz=nnz,
        )
        self.A = A
        self.b = b
        self.x = x


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
                from_numpy(dataset.A),
                from_numpy(dataset.b),
                from_numpy(dataset.x),
            ],
            meta={},
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
    def cacheable(self) -> bool:
        return False

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
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(dataset.source_name)
        x_bin = from_numpy(np.zeros(A_bin.shape[1]))
        b_bin = from_numpy(b)

        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta={})


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
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or not self._ref_meta.get("check_rounded_residual"):
            return

        A_bin, b_bin, _x_bin = self._input
        try:
            A_coo = to_scipy(A_bin).tocoo()
        except TypeError:
            A_coo = scipy_sparse.coo_array(to_numpy(A_bin))
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.row, A_coo.col)),
            data=A_coo.data,
            shape=A_coo.shape,
        )
        decimals = self._ref_meta["round_decimals"]
        x_sol = np.round(
            to_numpy(self._output[0]),
            decimals=decimals,
        )

        actual_b = from_numpy(np.asarray(A @ x_sol))
        expected_b = b_bin
        assert binsparse_equal(expected_b, actual_b), (
            f"CG residual mismatch for {param.dataset.name}"
        )

    def benchmark(self, xp, data: list, meta: dict):
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
