from typing import Any

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
    suite_sparse_rhs_dataset_name,
)
from saps_framework.binsparse_utils import binsparse_equal


class JacobiDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        x: np.ndarray | None = None,
        rhs_index: int | None = None,
        max_iter: int = 1000,
        rel_tol: float = 1e-6,
    ):
        dataset_name = suite_sparse_rhs_dataset_name(source_name, rhs_index)
        super().__init__(
            dataset_name,
            source_name=source_name,
            pretty_name=f"Jacobi {source_name}",
            suites=suites,
            rhs_index=rhs_index,
        )
        self.A = A
        self.b = b
        self.x = x
        self.max_iter = max_iter
        self.rel_tol = rel_tol

    def benchmark_meta(self) -> dict[str, Any]:
        return {"max_iter": self.max_iter, "rel_tol": self.rel_tol}


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
    def datasets(self) -> list[JacobiDataset]:
        return [
            JacobiDataset(
                "test_3x3",
                suites=["test", "trace"],
                A=np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]),
                b=np.array([5.0, 8.0, 8.0]),
                x=np.zeros((3,)),
            ),
            JacobiDataset(
                "test_4x4",
                suites=["test", "trace"],
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
                suites=["test", "trace"],
                A=np.array([[20.0, 3.0, 1.0], [2.0, 15.0, 4.0], [1.0, 2.0, 18.0]]),
                b=np.array([24.0, 21.0, 21.0]),
                x=np.zeros((3,)),
            ),
        ]

    def generate(self, dataset: JacobiDataset):
        if dataset.A is None or dataset.b is None or dataset.x is None:
            raise ValueError("Jacobi test datasets must define A, b, and x.")

        return DataInstance(
            inputs=[
                from_numpy(dataset.A),
                from_numpy(dataset.b),
                from_numpy(dataset.x),
            ],
            meta=dataset.benchmark_meta(),
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
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[JacobiDataset]:
        return [
    JacobiDataset("Andrews/Andrews", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/cdde2", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/cdde4", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/cdde6", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/dw256B", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/dwb512", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bai/pde900", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bindel/ted_B_unscaled", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Boeing/bcsstm39", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Bourchtein/atmosmodd", max_iter=1000, rel_tol=1e-06, rhs_index=1),
    JacobiDataset("Bourchtein/atmosmodj", max_iter=1000, rel_tol=1e-06, rhs_index=1),
    JacobiDataset("Bourchtein/atmosmodl", max_iter=1000, rel_tol=1e-06, rhs_index=1),
    JacobiDataset("Bourchtein/atmosmodm", max_iter=1000, rel_tol=1e-06, rhs_index=1),
    JacobiDataset("Cunningham/qa8fk", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("FEMLAB/problem1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Freescale/circuit5M_dc", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("GHS_psdef/jnlbrng1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("GHS_psdef/minsurfo", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("GHS_psdef/obstclae", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Grund/poli", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("Grund/poli3", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Grund/poli4", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Grund/poli_large", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/arc130", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm02", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm05", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm06", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm08", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm09", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm11", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm19", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm20", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm21", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm22", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm23", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm24", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm25", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/bcsstm26", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_183_1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_183_3", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_183_4", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_183_6", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_541_1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_680_1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_680_2", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/fs_760_1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/gr_30_30", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/jpwh_991", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/steam3", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("HB/watt_1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Hamm/add32", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("MathWorks/tomography", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("MaxPlanck/shallow_water1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("MaxPlanck/shallow_water2", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Mulvey/finan512", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nasa/nasa2146", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("Nemeth/nemeth02", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth03", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth04", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth05", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth06", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth07", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth08", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth09", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth10", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth11", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth12", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Nemeth/nemeth13", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Norris/fv1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Norris/fv2", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Norris/torso2", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh1e1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh1em1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh1em6", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh2e1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh2em5", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh3e1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Pothen/mesh3em5", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("QLi/majorbasis", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Simon/raefsky5", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("Simon/raefsky6", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("TOKAMAK/utm1700b", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("TOKAMAK/utm3060", max_iter=1000, rel_tol=1e-06, rhs_index=0),
    JacobiDataset("VLSI/ss1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Wang/swang1", max_iter=1000, rel_tol=1e-06),
    JacobiDataset("Wang/swang2", max_iter=1000, rel_tol=1e-06),
        ]

    def generate(self, dataset: JacobiDataset):
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(
            dataset.source_name,
            rhs_index=dataset.rhs_index,
        )
        x_bin = from_numpy(np.zeros(A_bin.shape[1]))
        b_bin = from_numpy(b)

        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta=dataset.benchmark_meta())


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
            f"Jacobi residual mismatch for {param.dataset.name}"
        )

    def _norm(self, xp, v):
        return xp.sqrt(xp.sum(xp.multiply(v, v)))

    def benchmark(self, xp, data: list, meta: dict):
        A, b, x = data

        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 1000)

        tolerance = max(rel_tol * self._norm(xp, b)[()], abs_tol)
        d = xp.with_fill_value(xp.diagonal(A), 1)
        if xp.any(d == 0):
            raise ValueError("Jacobi requires nonzero diagonal entries.")

        r = b - A @ x
        it = 0

        while self._norm(xp, r)[()] >= tolerance and it < max_iter:
            x = x + r / d

            r = b - A @ x
            it += 1
        if it >= max_iter:
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
