from typing import Any

import numpy as np

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

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
from saps_framework.binsparse_utils import from_coo, to_coo


class GMRESDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        nnz: int | None = None,
        suites: list[str] | None = None,
        A: Any | None = None,
        b: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        ref_meta: dict[str, Any] | None = None,
    ):
        super().__init__(
            source_name,
            pretty_name=f"GMRES {source_name}",
            suites=suites,
            nnz=nnz,
        )
        self.A = A
        self.b = b
        self.x0 = x0
        self.benchmark_meta = meta or {}
        self.ref_meta = ref_meta or {}


def gmres_random_system(seed):
    import scipy.sparse

    rng = np.random.default_rng(seed)
    n = 50
    A = scipy.sparse.random(n, n, density=0.1, random_state=rng)
    A = A + scipy.sparse.eye(n) * n
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A.tocoo(), b, np.zeros(n)


class GMRESTestGenerator(Generator[GMRESDataset]):
    @property
    def name(self) -> str:
        return "gmres_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "GMRES Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined matrices and seeded systems from the GMRES pytest examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return GMRESBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return GMRESBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Uses small inlined linear systems to verify GMRES convergence."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[GMRESDataset]:
        random_42 = gmres_random_system(42)
        random_123 = gmres_random_system(123)
        return [
            GMRESDataset(
                "test_gmres_random_42",
                suites=["test", "trace"],
                A=random_42[0],
                b=random_42[1],
                x0=random_42[2],
                meta={"restart": 20, "tol": 1e-8, "max_iter": 1000},
                ref_meta={"residual_tol": 1e-5},
            ),
            GMRESDataset(
                "test_gmres_random_123",
                suites=["test", "trace"],
                A=random_123[0],
                b=random_123[1],
                x0=random_123[2],
                meta={"restart": 20, "tol": 1e-8, "max_iter": 1000},
                ref_meta={"residual_tol": 1e-5},
            ),
            GMRESDataset(
                "test_gmres_diagonal",
                suites=["test", "trace"],
                A=np.array([[2.0, 0.0], [0.0, 3.0]]),
                b=np.array([4.0, 9.0]),
                x0=np.zeros(2),
                meta={"restart": 2, "tol": 1e-8, "max_iter": 100},
                ref_meta={"residual_tol": 1e-6},
            ),
            GMRESDataset(
                "test_gmres_3x3",
                suites=["test", "trace"],
                A=np.array([[10.0, 2.0, 1.0], [1.0, 20.0, 1.0], [1.0, 2.0, 10.0]]),
                b=np.array([13.0, 22.0, 13.0]),
                x0=np.zeros(3),
                meta={"restart": 3, "tol": 1e-8, "max_iter": 100},
                ref_meta={"residual_tol": 1e-6},
            ),
            GMRESDataset(
                "test_gmres_4x4",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [4.0, -1.0, 0.0, 0.0],
                        [-1.0, 4.0, -1.0, 0.0],
                        [0.0, -1.0, 4.0, -1.0],
                        [0.0, 0.0, -1.0, 3.0],
                    ]
                ),
                b=np.array([3.0, 2.0, 2.0, 2.0]),
                x0=np.zeros(4),
                meta={"restart": 4, "tol": 1e-8, "max_iter": 100},
                ref_meta={"residual_tol": 1e-6},
            ),
        ]

    def generate(self, dataset: GMRESDataset) -> DataInstance:
        if dataset.A is None or dataset.b is None or dataset.x0 is None:
            raise ValueError("GMRES test datasets must define A, b, and x0.")
        A = dataset.A.tocoo() if hasattr(dataset.A, "tocoo") else None
        A_bin = (
            from_coo((A.row, A.col), A.data, A.shape)
            if A is not None
            else from_numpy(dataset.A)
        )
        return DataInstance(
            inputs=[
                A_bin,
                from_numpy(dataset.b),
                from_numpy(dataset.x0),
            ],
            meta=dataset.benchmark_meta,
            ref_meta=dataset.ref_meta,
        )


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
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
    def cacheable(self) -> bool:
        return False

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
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(dataset.source_name)
        x_bin = from_numpy(np.zeros(A_bin.shape[1]))
        b_bin = from_numpy(b)
        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta={})


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
        return [GMRESTestGenerator(), GMRESGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or "residual_tol" not in self._ref_meta:
            return

        A_bin, b_bin, _x0_bin = self._input
        A_coo = to_coo(A_bin)
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.indices_0, A_coo.indices_1)),
            data=A_coo.values,
            shape=A_coo.shape,
        )
        b = to_numpy(b_bin)
        x_sol = to_numpy(self._output[0])
        residual = np.linalg.norm(b - A @ x_sol)
        assert residual < self._ref_meta["residual_tol"], (
            f"GMRES residual too high for {param.dataset.name}: {residual}"
        )

    def benchmark(self, xp, data: list, meta: dict):
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
