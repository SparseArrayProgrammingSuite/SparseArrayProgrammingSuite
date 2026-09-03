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


class CGDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        x: np.ndarray | None = None,
        rhs_index: int | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-6,
    ):
        dataset_name = suite_sparse_rhs_dataset_name(source_name, rhs_index)
        super().__init__(
            dataset_name,
            source_name=source_name,
            pretty_name=f"CG {source_name}",
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
            meta=dataset.benchmark_meta(),
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
            CGDataset(
                "Andrews/Andrews", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/ins2", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net100", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net125", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net150", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net25", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net50", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Andrianov/net75", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset("Bai/bfwb398", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bai/bfwb62", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bai/bfwb782", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bai/dw256B", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bai/dwb512", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bai/odepb400", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Bindel/ted_B", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "Bindel/ted_B_unscaled",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGDataset(
                "Boeing/crystm01", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Boeing/crystm02", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Boeing/crystm03", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Botonakis/thermomech_TC",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGDataset(
                "Botonakis/thermomech_dM",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGDataset(
                "Bourchtein/atmosmodd",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGDataset(
                "Bourchtein/atmosmodj",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGDataset(
                "Bourchtein/atmosmodl",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGDataset(
                "Bourchtein/atmosmodm",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGDataset(
                "Brunetiere/thermal", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Cunningham/m3plates", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Cunningham/qa8fm", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "FEMLAB/poisson2D",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "FEMLAB/problem1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset("FIDAP/ex29", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("FIDAP/ex37", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("FIDAP/ex5", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("FIDAP/ex7", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "GHS_indef/blockqp1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "GHS_indef/laser", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "GHS_indef/qpband", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "GHS_psdef/jnlbrng1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "GHS_psdef/minsurfo", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "GHS_psdef/obstclae", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset("Grund/meg4", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "Grund/poli",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset("HB/bcspwr01", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcspwr02", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstk01", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstk02", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm01", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm02", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm03", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm04", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm05", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm06", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm08", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm09", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm11", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm19", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm20", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm21", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/bcsstm22", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_144", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_24", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_61", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_62", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_73", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/can_96", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/dwt_59", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/dwt_66", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/dwt_72", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/fs_541_1", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/gr_30_30", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/lap_25", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/nos4", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=10,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=11,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=12,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=14,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=15,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=16,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=17,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=18,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=19,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=2,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=3,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=4,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=5,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=6,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=62,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=7,
            ),
            CGDataset(
                "HB/orani678",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=8,
            ),
            CGDataset("HB/watt_1", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("HB/watt_2", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "Hamm/add32",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset("MKS/fp", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "MathWorks/Muu", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "MathWorks/tomography", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "MaxPlanck/shallow_water1",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGDataset(
                "MaxPlanck/shallow_water2",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGDataset(
                "Mulvey/finan512", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nasa/nasa2146",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "Nemeth/nemeth02", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth03", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth04", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth05", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth06", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth07", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth08", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth09", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth10", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth11", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth12", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth13", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth16", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth17", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth18", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth19", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth20", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth21", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth22", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth23", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth24", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth25", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Nemeth/nemeth26", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset("Norris/fv1", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset("Norris/fv2", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "Oberwolfach/LF10", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Oberwolfach/LFAT5", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset("PARSEC/Si2", suites=["standard"], max_iter=100, rel_tol=1e-06),
            CGDataset(
                "Pothen/mesh1e1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh1em1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh1em6", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh2e1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh2em5", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh3e1", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/mesh3em5", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Pothen/sphere2", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Precima/analytics", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/ASIC_100k", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/ASIC_100ks", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/ASIC_320ks", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_61", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_62", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_63", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_64", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_65", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_66", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_67", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_68", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Sandia/adder_dcop_69", suites=["standard"], max_iter=100, rel_tol=1e-06
            ),
            CGDataset(
                "Schenk_AFE/af_shell3",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "Schenk_AFE/af_shell4",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "Schenk_AFE/af_shell7",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "Schenk_AFE/af_shell8",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "VDOL/hangGlider_1",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "VDOL/tumorAntiAngiogenesis_1",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset(
                "VDOL/tumorAntiAngiogenesis_2",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGDataset("VLSI/ss1", suites=["standard"], max_iter=100, rel_tol=1e-06),
        ]

    def generate(self, dataset: CGDataset) -> DataInstance:
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(
            dataset.source_name,
            rhs_index=dataset.rhs_index,
        )
        x_bin = from_numpy(np.zeros(A_bin.shape[1]))
        b_bin = from_numpy(b)

        return DataInstance(inputs=[A_bin, b_bin, x_bin], meta=dataset.benchmark_meta())


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
        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 100)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        tol_sq = tolerance * tolerance

        r = b - A @ x
        p = r
        rr = xp.vecdot(r, r)[()]
        it = 0

        if rr >= tol_sq:
            while it < max_iter:
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
