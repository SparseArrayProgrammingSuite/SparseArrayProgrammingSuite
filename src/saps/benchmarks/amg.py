from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy, to_sparse

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
from saps.util.error_bounds import operation_error_bound, summation_error_bound


def _build_aggregates(A: scipy_sparse.csr_matrix, aggregate_size: int) -> np.ndarray:
    n = A.shape[0]
    indptr = A.indptr
    indices = A.indices
    agg = np.full(n, -1, dtype=np.int64)
    next_agg = 0
    for seed in range(n):
        if agg[seed] != -1:
            continue
        agg[seed] = next_agg
        members = 1
        for k in range(indptr[seed], indptr[seed + 1]):
            if members >= aggregate_size:
                break
            neighbour = indices[k]
            if neighbour != seed and agg[neighbour] == -1:
                agg[neighbour] = next_agg
                members += 1
        next_agg += 1
    return agg


def _tentative_prolongator(agg: np.ndarray) -> scipy_sparse.csr_matrix:
    n_fine = agg.shape[0]
    n_coarse = int(agg.max()) + 1 if n_fine else 0
    data = np.ones(n_fine, dtype=np.float64)
    rows = np.arange(n_fine, dtype=np.int64)
    cols = agg
    return scipy_sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_fine, n_coarse)
    ).tocsr()


def _build_tentative_prolongator(
    A: scipy_sparse.csr_matrix,
    *,
    aggregate_size: int,
) -> scipy_sparse.csr_matrix:
    agg = _build_aggregates(A.tocsr(), aggregate_size)
    return _tentative_prolongator(agg)


class AMGDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        pretty_name: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        rhs_index: int | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-6,
        num_sweeps: int = 2,
        jacobi_weight: float = 4.0 / 3.0,
        aggregate_size: int = 3,
        prolongation_omega: float = 4.0 / 3.0,
    ):
        name = source_name
        if rhs_index is not None:
            name = f"{source_name}_rhs{rhs_index}"
            pretty_name = f"{source_name} (Right-Hand Side {rhs_index})"
        super().__init__(
            name,
            source_name=source_name,
            pretty_name=pretty_name,
            suites=suites,
            rhs_index=rhs_index,
        )
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.num_sweeps = num_sweeps
        self.jacobi_weight = jacobi_weight
        self.aggregate_size = aggregate_size
        self.prolongation_omega = prolongation_omega

    def benchmark_meta(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "rel_tol": self.rel_tol,
            "num_sweeps": self.num_sweeps,
            "jacobi_weight": self.jacobi_weight,
            "prolongation_omega": self.prolongation_omega,
        }


class AMGSuiteSparseGenerator(Generator[AMGDataset]):
    @property
    def name(self) -> str:
        return "amg_suitesparse"

    @property
    def pretty_name(self) -> str:
        return "Smoothed Aggregation AMG SuiteSparse"

    @property
    def description(self) -> str:
        return (
            "Symmetric positive definite matrices from the SuiteSparse Matrix"
            " Collection, solved with smoothed-aggregation algebraic multigrid."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return AMGBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return AMGBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return AMGBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return AMGBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[AMGDataset]:
        # Symmetric positive definite matrices drawn from the same SuiteSparse
        # groups the PCG benchmark (which this solver is modelled on) uses.
        return [
            AMGDataset("Andrews/Andrews", max_iter=100, rel_tol=1e-06),
            AMGDataset("Bai/mhd3200b", max_iter=100, rel_tol=1e-06),
            AMGDataset("Bai/mhd4800b", max_iter=100, rel_tol=1e-06),
            AMGDataset("Bai/mhdb416", max_iter=100, rel_tol=1e-06),
            AMGDataset("Bindel/ted_B", max_iter=100, rel_tol=1e-06),
            AMGDataset("Bindel/ted_B_unscaled", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/bcsstk34", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/bcsstm39", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/crystm01", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/crystm02", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/crystm03", max_iter=100, rel_tol=1e-06),
            AMGDataset("Boeing/msc00726", max_iter=100, rel_tol=1e-06),
            AMGDataset("Botonakis/thermomech_TC", max_iter=100, rel_tol=1e-06),
            AMGDataset("Botonakis/thermomech_TK", max_iter=100, rel_tol=1e-06),
            AMGDataset("Botonakis/thermomech_dM", max_iter=100, rel_tol=1e-06),
            AMGDataset("CEMW/tmt_sym", max_iter=100, rel_tol=1e-06),
            AMGDataset("Cunningham/qa8fm", max_iter=100, rel_tol=1e-06),
            AMGDataset("FIDAP/ex3", max_iter=100, rel_tol=1e-06),
            AMGDataset("FIDAP/ex5", max_iter=100, rel_tol=1e-06),
            AMGDataset("GHS_psdef/wathen100", max_iter=100, rel_tol=1e-06),
            AMGDataset("GHS_psdef/wathen120", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk01", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk02", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk04", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk05", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk09", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk16", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk21", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstk22", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm02", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm05", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm06", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm07", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm08", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm09", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm11", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm19", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm20", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm21", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm22", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm23", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm24", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm25", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/bcsstm26", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/gr_30_30", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/lund_a", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/lund_b", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/nos3", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/nos4", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/nos5", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/nos6", max_iter=100, rel_tol=1e-06),
            AMGDataset("HB/nos7", max_iter=100, rel_tol=1e-06),
            AMGDataset("Janna/StocF-1465", max_iter=100, rel_tol=1e-06),
            AMGDataset("Lourakis/bundle1", max_iter=100, rel_tol=1e-06),
            AMGDataset("MathWorks/Muu", max_iter=100, rel_tol=1e-06),
            AMGDataset("MaxPlanck/shallow_water1", max_iter=100, rel_tol=1e-06),
            AMGDataset("MaxPlanck/shallow_water2", max_iter=100, rel_tol=1e-06),
            AMGDataset("McRae/ecology2", max_iter=100, rel_tol=1e-06),
            AMGDataset("Nasa/nasa2146", max_iter=100, rel_tol=1e-06, rhs_index=0),
            AMGDataset("Norris/fv1", max_iter=100, rel_tol=1e-06),
            AMGDataset("Norris/fv2", max_iter=100, rel_tol=1e-06),
            AMGDataset("Norris/fv3", max_iter=100, rel_tol=1e-06),
            AMGDataset("Oberwolfach/LF10", max_iter=100, rel_tol=1e-06),
            AMGDataset("Oberwolfach/LFAT5", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/bodyy4", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/bodyy5", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh1e1", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh1em1", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh1em6", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh2e1", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh2em5", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh3e1", max_iter=100, rel_tol=1e-06),
            AMGDataset("Pothen/mesh3em5", max_iter=100, rel_tol=1e-06),
            AMGDataset("TKK/plbuckle", max_iter=100, rel_tol=1e-06),
            AMGDataset("UTEP/Dubcova1", max_iter=100, rel_tol=1e-06),
            AMGDataset("UTEP/Dubcova2", max_iter=100, rel_tol=1e-06),
            AMGDataset("UTEP/Dubcova3", max_iter=100, rel_tol=1e-06),
            AMGDataset("Um/2cubes_sphere", max_iter=100, rel_tol=1e-06, rhs_index=0),
        ]

    def generate(self, dataset: AMGDataset) -> DataInstance:
        if dataset.A is not None:
            A_bin = from_scipy(scipy_sparse.coo_array(dataset.A))
            b = dataset.b
        else:
            A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(
                dataset.source_name,
                rhs_index=dataset.rhs_index,
            )
        A_csr = to_scipy(A_bin).tocsr()
        tentative = _build_tentative_prolongator(
            A_csr, aggregate_size=dataset.aggregate_size
        )
        return DataInstance(
            inputs=[
                A_bin,
                from_numpy(b),
                from_numpy(np.zeros(A_bin.shape[1])),
                from_scipy(tentative.tocoo()),
            ],
            meta=dataset.benchmark_meta(),
            ref_outputs=[from_scipy(np.abs(A_csr).tocoo())],
            ref_meta={"bnorm": float(np.linalg.norm(b))},
        )


class AMGTestGenerator(AMGSuiteSparseGenerator):
    @property
    def name(self) -> str:
        return "amg_test"

    @property
    def pretty_name(self) -> str:
        return "Smoothed Aggregation AMG Test"

    @property
    def description(self) -> str:
        return "Small inlined symmetric positive definite systems."

    @property
    def datasets(self) -> list[AMGDataset]:
        return [
            AMGDataset(
                "3x3_tridiagonal",
                pretty_name="3x3 Tridiagonal",
                suites=["test"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                b=np.array([4.0, 8.0, 16.0]),
            ),
            AMGDataset(
                "3x3_dense",
                pretty_name="3x3 Dense",
                suites=["test"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                b=np.array([13.0, -3.0, 8.0]),
            ),
            AMGDataset(
                "4x4_tridiagonal",
                pretty_name="4x4 Tridiagonal",
                suites=["test"],
                A=np.array(
                    [
                        [8.0, -1.0, 0.0, 0.0],
                        [-1.0, 8.0, -1.0, 0.0],
                        [0.0, -1.0, 8.0, -1.0],
                        [0.0, 0.0, -1.0, 8.0],
                    ]
                ),
                b=np.array([8.0, -2.0, 6.0, 15.0]),
            ),
            AMGDataset(
                "3x3_indefinite_sparse",
                pretty_name="3x3 Indefinite Sparse",
                suites=["test"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                b=np.array([40.0, 10.0, -18.0]),
            ),
            AMGDataset(
                "3x3_scaled_tridiagonal",
                pretty_name="3x3 Scaled Tridiagonal",
                suites=["test"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                b=np.array([118.0, 116.0, 118.0]),
            ),
            AMGDataset(
                "5x5_sparse",
                pretty_name="5x5 Sparse",
                suites=["test"],
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
            ),
        ]


class AMGBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "amg"

    @property
    def pretty_name(self) -> str:
        return "Algebraic Multigrid (AMG)"

    @property
    def description(self) -> str:
        return (
            "Solves sparse symmetric positive definite linear systems with"
            " conjugate gradient, preconditioned by a smoothed-aggregation"
            " algebraic-multigrid two-level V-cycle. The"
            " coarse operator is formed with the Galerkin triple product R @ A @ P,"
            " where the restriction R is short and wide, the prolongation P is tall"
            " and narrow, and A is square."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard-solvers"]

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
        return [Contributor("Aarav Joglekar", "ajoglekar32@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="PyAMG: Algebraic Multigrid Solvers in Python",
                authors=[
                    Author("Nathan Bell"),
                    Author("Luke N. Olson"),
                    Author("Jacob Schroder"),
                ],
                journal="Journal of Open Source Software",
                volume=7,
                number=72,
                pages="4142",
                year=2022,
                url="https://doi.org/10.21105/joss.04142",
                doi="10.21105/joss.04142",
            ),
            Ref(
                title="Algebraic Multigrid",
                authors=[Author("Luke N. Olson")],
                booktitle="Encyclopedia of Applied and Computational Mathematics",
                publisher="Springer",
                year=2015,
                url="https://lukeo.cs.illinois.edu/files/2015_Ol_encamg.pdf",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the initial benchmark function, but AI was used to fix bugs in the implementation. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Algebraic multigrid is one of the most effective solvers for the sparse"
            " symmetric positive definite systems that arise from discretized PDEs."
            " The two-level V-cycle forms its coarse operator with the Galerkin"
            " triple product R @ A @ P. Because the restriction R is short and wide"
            " and the prolongation P is tall and narrow while A is square, these"
            " sparse-sparse matrix multiplications exercise exactly the U * A * V"
            " SpGEMM shape that sparse autoscheduling must handle well. Keeping the"
            " operators sparse is what makes the setup and each cycle scale with the"
            " number of non-zeros rather than the square of the dimension."
        )

    @property
    def generators(self):
        return [AMGTestGenerator(), AMGSuiteSparseGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        A_bin, b_bin, _x0_bin, _tentative_bin = self._input
        try:
            A_coo = to_scipy(A_bin).tocoo()
        except TypeError:
            A_coo = scipy_sparse.coo_array(to_numpy(A_bin))
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.row, A_coo.col)),
            data=A_coo.data,
            shape=A_coo.shape,
        )
        assert self._ref_outputs is not None, "No cached absolute matrix"
        assert self._ref_meta is not None, "No cached reference norms"
        abs_A = to_sparse(self._ref_outputs[0])
        b = to_numpy(b_bin)
        x_sol = to_numpy(self._output[0])
        r = np.asarray(b - A @ x_sol).ravel()
        roundoff = summation_error_bound(
            np,
            A.shape[1] + 1,
            abs_A @ np.abs(x_sol),
            np.result_type(A.dtype, x_sol.dtype),
        )
        roundoff += operation_error_bound(np, r, r.dtype)
        residual = np.linalg.norm(r)
        bound = np.linalg.norm(roundoff)
        bound += summation_error_bound(np, r.size + 2, residual + bound, r.dtype)
        assert np.isfinite(residual) and np.isfinite(bound), (
            "Non-finite residual or error bound"
        )

        rel_tol = (self._meta or {}).get("rel_tol", 1e-6)
        bnorm = self._ref_meta["bnorm"]
        abs_tol = (self._meta or {}).get("abs_tol", 1e-20)
        limit = max(rel_tol * bnorm, abs_tol) + bound
        assert residual <= limit, (
            f"AMG residual too high for {param.dataset.name}: {residual}"
        )

    def benchmark(self, xp, meta: dict[str, Any], A, b, x0, tentative):
        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 100)
        num_sweeps = meta.get("num_sweeps", 2)
        weight = meta.get("jacobi_weight", 4.0 / 3.0)
        omega = meta.get("prolongation_omega", 4.0 / 3.0)

        # Smoothed prolongator P = (I - omega / rho * D^{-1} A) @ tentative.
        diag = xp.diagonal(A)
        diag_inv = xp.replace(1.0 / diag, xp.inf, 0)
        Dinv_A = A * xp.reshape(diag_inv, (-1, 1))
        # Cheap spectral-radius bound: ||D^{-1} A||_inf (max absolute row sum).
        rho = xp.max(xp.sum(xp.abs(Dinv_A), axis=1))
        rho = float(rho) if rho > 0 else 1.0
        scale = omega / rho
        P = tentative - scale * (Dinv_A @ tentative)

        R = xp.matrix_transpose(P)
        # Coarse operator via the Galerkin triple product A_c = R @ A @ P. R is
        # short and wide, P is tall and narrow, and A is square, so this is the
        # U * A * V SpGEMM shape.
        A_coarse = R @ A @ P

        diag_coarse = xp.diagonal(A_coarse)

        # Weighted Jacobi only converges when weight * rho(D^{-1} A) < 2, so each
        # level scales the weight by its own spectral-radius bound.
        diag_coarse_inv = xp.replace(1.0 / diag_coarse, xp.inf, 0)
        Dinv_A_coarse = A_coarse * xp.reshape(diag_coarse_inv, (-1, 1))
        rho_coarse = xp.max(xp.sum(xp.abs(Dinv_A_coarse), axis=1))
        rho_coarse = float(rho_coarse) if rho_coarse > 0 else 1.0
        fine_weight = weight / rho
        coarse_weight = weight / rho_coarse

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        tol_sq = tolerance * tolerance

        x = x0
        r = b - A @ x
        rr = xp.vecdot(r, r)[()]
        p = xp.zeros(A.shape[0], dtype=b.dtype)
        old_rz = 1.0
        it = 0
        if rr >= tol_sq:
            while it < max_iter:
                # Precondition with one V-cycle on A z = r, starting from z = 0.
                z = xp.zeros(A.shape[0], dtype=b.dtype)
                # Pre-smoothing: a few weighted-Jacobi relaxations on the fine grid.
                for _ in range(num_sweeps):
                    z = z + fine_weight * xp.replace((r - A @ z) / diag, xp.nan, 0)
                residual = r - A @ z
                coarse_b = R @ residual
                coarse_x = xp.zeros(A_coarse.shape[0], dtype=b.dtype)
                for _ in range(num_sweeps + 6):
                    coarse_x = coarse_x + coarse_weight * xp.replace(
                        (coarse_b - A_coarse @ coarse_x) / diag_coarse, xp.nan, 0
                    )
                z = z + P @ coarse_x

                for _ in range(num_sweeps):
                    z = z + fine_weight * xp.replace((r - A @ z) / diag, xp.nan, 0)

                # Conjugate gradient step, preconditioned by the V-cycle.
                rz = xp.vecdot(r, z)[()]
                p = z + (rz / old_rz) * p
                Ap = A @ p
                alpha = rz / xp.vecdot(p, Ap)[()]
                x = x + alpha * p
                r = r - alpha * Ap
                old_rz = rz

                rr = xp.vecdot(r, r)[()]
                it += 1
                if rr < tol_sq:
                    break

        return x
