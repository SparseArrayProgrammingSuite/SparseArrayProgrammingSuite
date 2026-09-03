from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

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
    suite_sparse_rhs_dataset_name,
)


class GMRESDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        suites: list[str] | None = None,
        A: Any | None = None,
        b: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        ref_meta: dict[str, Any] | None = None,
        rhs_index: int | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-6,
        restart: int = 50,
    ):
        dataset_name = suite_sparse_rhs_dataset_name(source_name, rhs_index)
        super().__init__(
            dataset_name,
            source_name=source_name,
            pretty_name=f"GMRES {source_name}",
            suites=suites,
            rhs_index=rhs_index,
        )
        self.A = A
        self.b = b
        self.x0 = x0
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.restart = restart
        self.ref_meta = ref_meta or {}

    def benchmark_meta(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "rel_tol": self.rel_tol,
            "restart": self.restart,
        }


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
                restart=20,
                rel_tol=1e-8,
                max_iter=1000,
                ref_meta={"residual_tol": 1e-5},
            ),
            GMRESDataset(
                "test_gmres_random_123",
                suites=["test", "trace"],
                A=random_123[0],
                b=random_123[1],
                x0=random_123[2],
                restart=20,
                rel_tol=1e-8,
                max_iter=1000,
                ref_meta={"residual_tol": 1e-5},
            ),
            GMRESDataset(
                "test_gmres_diagonal",
                suites=["test", "trace"],
                A=np.array([[2.0, 0.0], [0.0, 3.0]]),
                b=np.array([4.0, 9.0]),
                x0=np.zeros(2),
                restart=2,
                rel_tol=1e-8,
                max_iter=100,
                ref_meta={"residual_tol": 1e-6},
            ),
            GMRESDataset(
                "test_gmres_3x3",
                suites=["test", "trace"],
                A=np.array([[10.0, 2.0, 1.0], [1.0, 20.0, 1.0], [1.0, 2.0, 10.0]]),
                b=np.array([13.0, 22.0, 13.0]),
                x0=np.zeros(3),
                restart=3,
                rel_tol=1e-8,
                max_iter=100,
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
                restart=4,
                rel_tol=1e-8,
                max_iter=100,
                ref_meta={"residual_tol": 1e-6},
            ),
        ]

    def generate(self, dataset: GMRESDataset) -> DataInstance:
        if dataset.A is None or dataset.b is None or dataset.x0 is None:
            raise ValueError("GMRES test datasets must define A, b, and x0.")
        A = dataset.A.tocoo() if hasattr(dataset.A, "tocoo") else None
        A_bin = from_scipy(A) if A is not None else from_numpy(dataset.A)
        return DataInstance(
            inputs=[
                A_bin,
                from_numpy(dataset.b),
                from_numpy(dataset.x0),
            ],
            meta=dataset.benchmark_meta(),
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
            GMRESDataset("Andrews/Andrews", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/ins2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net100", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net125", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net150", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net25", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net50", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Andrianov/net75", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/bfwa62", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/bfwb398", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/bfwb62", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/bfwb782", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/ck104", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/dw256B", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/dwb512", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/pde225", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/rdb200", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bai/rdb200l", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bindel/ted_B", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bindel/ted_B_unscaled", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bomhof/circuit_1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Botonakis/thermomech_TC", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Botonakis/thermomech_dM", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Bourchtein/atmosmodd", max_iter=100, rel_tol=1e-06, rhs_index=1),
    GMRESDataset("Bourchtein/atmosmodj", max_iter=100, rel_tol=1e-06, rhs_index=1),
    GMRESDataset("Bourchtein/atmosmodl", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Bourchtein/atmosmodl", max_iter=100, rel_tol=1e-06, rhs_index=1),
    GMRESDataset("Bourchtein/atmosmodm", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Bourchtein/atmosmodm", max_iter=100, rel_tol=1e-06, rhs_index=1),
    GMRESDataset("Brunetiere/thermal", max_iter=100, rel_tol=1e-06),
    GMRESDataset("CPM/cz148", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Cunningham/m3plates", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Cunningham/qa8fk", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Cunningham/qa8fm", max_iter=100, rel_tol=1e-06),
    GMRESDataset("FEMLAB/poisson2D", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("FEMLAB/problem1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("FIDAP/ex29", max_iter=100, rel_tol=1e-06),
    GMRESDataset("FIDAP/ex37", max_iter=100, rel_tol=1e-06),
    GMRESDataset("FIDAP/ex5", max_iter=100, rel_tol=1e-06),
    GMRESDataset("FIDAP/ex7", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Freescale/circuit5M_dc", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_indef/blockqp1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_indef/boyd1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("GHS_indef/boyd2", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("GHS_indef/laser", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_indef/qpband", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_psdef/jnlbrng1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_psdef/minsurfo", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_psdef/obstclae", max_iter=100, rel_tol=1e-06),
    GMRESDataset("GHS_psdef/wathen120", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Grund/b1_ss", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Grund/meg4", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Grund/poli", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Grund/poli3", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Grund/poli4", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Grund/poli_large", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/arc130", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcspwr01", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcspwr02", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstk01", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstk02", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstk20", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm01", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm02", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm03", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm04", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm05", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm06", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm08", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm09", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm11", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm19", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm20", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm21", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/bcsstm22", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/can_144", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/can_24", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/can_61", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/can_73", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/curtis54", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/fs_541_1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/fs_760_1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/fs_760_2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/fs_760_3", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/gr_30_30", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/jpwh_991", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/lap_25", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/nos4", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/nos7", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=1),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=10),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=11),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=12),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=14),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=15),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=16),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=17),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=18),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=19),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=2),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=3),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=4),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=5),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=6),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=62),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=7),
    GMRESDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=8),
    GMRESDataset("HB/pores_1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/psmigr_3", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/steam2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/steam3", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/watt_1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("HB/watt_2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Hamrle/Hamrle1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("MKS/fp", max_iter=100, rel_tol=1e-06),
    GMRESDataset("MathWorks/Muu", max_iter=100, rel_tol=1e-06),
    GMRESDataset("MathWorks/tomography", max_iter=100, rel_tol=1e-06),
    GMRESDataset("MaxPlanck/shallow_water1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("MaxPlanck/shallow_water2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Morandini/rotor1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Mulvey/finan512", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nasa/nasa2146", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Nemeth/nemeth02", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth03", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth04", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth05", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth06", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth07", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth08", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth09", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth10", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth11", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth12", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth13", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth16", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth17", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth18", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth19", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth20", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth21", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth22", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth23", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth24", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth25", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Nemeth/nemeth26", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Norris/fv1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Norris/fv2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Norris/torso2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Oberwolfach/LF10", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Oberwolfach/LFAT5", max_iter=100, rel_tol=1e-06),
    GMRESDataset("PARSEC/Si2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh1e1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh1em1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh1em6", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh2e1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh2em5", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh3e1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/mesh3em5", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Pothen/sphere2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Precima/analytics", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Rajat/rajat13", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Rommes/bips98_1450", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Rommes/bips98_606", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Rommes/ww_36_pmec_36", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/ASIC_100k", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/ASIC_100ks", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/ASIC_320ks", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/ASIC_680k", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/ASIC_680ks", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_31", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_32", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_33", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_34", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_35", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_36", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_37", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_38", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_40", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_41", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_42", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_43", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_44", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_45", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_46", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_47", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_48", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_49", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_50", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_51", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_52", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_53", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_54", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_55", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_57", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_58", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_59", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_60", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_61", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_62", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_63", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_64", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_65", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_66", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_67", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_68", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_dcop_69", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_trans_01", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/adder_trans_02", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Sandia/mult_dcop_02", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Schenk_AFE/af_shell3", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Schenk_AFE/af_shell4", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Schenk_AFE/af_shell7", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Schenk_AFE/af_shell8", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("Simon/raefsky5", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("TOKAMAK/utm1700b", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("TOKAMAK/utm3060", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("VDOL/hangGlider_1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("VDOL/tumorAntiAngiogenesis_1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("VDOL/tumorAntiAngiogenesis_2", max_iter=100, rel_tol=1e-06, rhs_index=0),
    GMRESDataset("VLSI/ss1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Wang/swang1", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Wang/swang2", max_iter=100, rel_tol=1e-06),
    GMRESDataset("Zhao/Zhao1", max_iter=100, rel_tol=1e-06),
        ]

    def generate(self, dataset: GMRESDataset):
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(
            dataset.source_name,
            rhs_index=dataset.rhs_index,
        )
        x_bin = from_numpy(np.zeros(A_bin.shape[1]))
        b_bin = from_numpy(b)
        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta=dataset.benchmark_meta())


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
        try:
            A_coo = to_scipy(A_bin).tocoo()
        except TypeError:
            A_coo = scipy_sparse.coo_array(to_numpy(A_bin))
        A = pydata_sparse.COO(
            coords=np.stack((A_coo.row, A_coo.col)),
            data=A_coo.data,
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
        rel_tol = meta.get("rel_tol", 1e-6)
        max_iter = meta.get("max_iter", 100)

        itcount = 0
        r0 = b - A @ x0
        initial_beta = xp.linalg.norm(r0)[()]
        if initial_beta < rel_tol:
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
                if r0_norm / initial_beta < rel_tol:
                    return [x0]

                itcount += 1
                if itcount >= max_iter:
                    break

            beta = r0_norm

        xsol = x0
        return [xsol]
