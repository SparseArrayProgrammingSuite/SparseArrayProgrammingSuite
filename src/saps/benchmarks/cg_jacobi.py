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
from saps.downloaders.suitesparse import random_rhs_for_matrix
from saps.util.error_bounds import operation_error_bound, summation_error_bound


def _dense(array: BinsparseTensor):
    try:
        return to_numpy(array)
    except TypeError:
        return np.asarray(to_sparse(array).todense())


def _generate_cg_data(source, A=None, rhs_index=None):
    if A is not None:
        import scipy.sparse as sp

        A = sp.coo_matrix(A)
        b = random_rhs_for_matrix(A)
        A_bin = from_scipy(A)
    else:
        A_bin, b, _has_real_rhs = fetch_suitesparse_linear_system(
            source,
            rhs_index=rhs_index,
        )
    x0 = np.zeros(A_bin.shape[1])
    return (A_bin, b, x0)


class CGJacobiDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        pretty_name: str | None = None,
        A=None,
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
        rhs_index: int | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-6,
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
        self.ref_meta = ref_meta
        self.max_iter = max_iter
        self.rel_tol = rel_tol

    def benchmark_meta(self) -> dict[str, Any]:
        return {"max_iter": self.max_iter, "rel_tol": self.rel_tol}


class CGJacobiSuiteSparseGenerator(Generator[CGJacobiDataset]):
    @property
    def name(self) -> str:
        return "cg_jacobi_suitesparse"

    @property
    def pretty_name(self) -> str:
        return "Jacobi-Preconditioned Conjugate Gradients SuiteSparse"

    @property
    def description(self) -> str:
        return (
            "Data collected from SuiteSparse Matrix Collection consisting of symmetric"
            " positive definite matrices, particularly those with a low convergence"
            " criteria."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return CGJacobiBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return CGJacobiBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return CGJacobiBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return CGJacobiBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[CGJacobiDataset]:
        return [
            CGJacobiDataset(
                "Andrews/Andrews",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/ins2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net100",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net125",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net150",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net25",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net50",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Andrianov/net75",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Bai/bfwb398", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bai/bfwb62", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bai/bfwb782", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bai/dw256B", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bai/dwb512", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bai/mhd3200b",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Bai/mhd4800b",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Bai/mhdb416", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Bindel/ted_B",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Bindel/ted_B_unscaled",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/bcsstk34",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/bcsstm39",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/crystm01",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/crystm02",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/crystm03",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Boeing/msc00726",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Botonakis/FEM_3D_thermal1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Botonakis/FEM_3D_thermal2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Botonakis/thermomech_TC",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Botonakis/thermomech_dM",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Bourchtein/atmosmodd",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGJacobiDataset(
                "Bourchtein/atmosmodj",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGJacobiDataset(
                "Bourchtein/atmosmodl",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGJacobiDataset(
                "Bourchtein/atmosmodm",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGJacobiDataset(
                "Brunetiere/thermal",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Cunningham/qa8fm",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "FEMLAB/poisson2D",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "FEMLAB/problem1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "FIDAP/ex29", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "FIDAP/ex37", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "FIDAP/ex5", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "FIDAP/ex7", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Freescale/circuit5M_dc",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "GHS_psdef/jnlbrng1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "GHS_psdef/minsurfo",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "GHS_psdef/obstclae",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "GHS_psdef/wathen100",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "GHS_psdef/wathen120",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Grund/poli",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Guettel/TEM27623",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "HB/bcspwr01", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcspwr02", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstk01", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstk02", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstk04", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstk08", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstk22", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm02", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm05", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm06", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm07", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm08", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm09", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm11", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm19", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm20", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm21", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm22", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm23", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm24", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm25", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/bcsstm26", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_144", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_24", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_61", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_62", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_73", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/can_96", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/dwt_59", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/dwt_66", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/dwt_72", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/fs_541_1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/gr_30_30", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/jpwh_991", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/lap_25", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/lund_a", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/lund_b", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/nos4", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/nos6", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/nos7", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=1,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=10,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=11,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=12,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=14,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=15,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=16,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=17,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=18,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=19,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=2,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=3,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=4,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=5,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=6,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=62,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=7,
            ),
            CGJacobiDataset(
                "HB/orani678",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=8,
            ),
            CGJacobiDataset(
                "HB/watt_1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Hamm/add32",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Lourakis/bundle1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "MathWorks/Muu",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "MathWorks/tomography",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "MaxPlanck/shallow_water1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "MaxPlanck/shallow_water2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Mulvey/finan512",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nasa/nasa2146",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth02",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth03",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth04",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth05",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth06",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth07",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth08",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth09",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth10",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth11",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth12",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth13",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth16",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Nemeth/nemeth17",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Norris/fv1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Norris/fv2", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Oberwolfach/LF10",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Oberwolfach/LFAT5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "PARSEC/Si2", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGJacobiDataset(
                "Pothen/bodyy4",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh1e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh1em1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh1em6",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh2e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh2em5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh3e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/mesh3em5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Pothen/sphere2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Sandia/ASIC_100ks",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Sandia/ASIC_320ks",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Sandia/ASIC_680ks",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGJacobiDataset(
                "Schenk_AFE/af_shell3",
                suites=["standard", "trace", "train"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Schenk_AFE/af_shell4",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Schenk_AFE/af_shell7",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Schenk_AFE/af_shell8",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "Um/2cubes_sphere",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "VDOL/hangGlider_1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "VDOL/tumorAntiAngiogenesis_1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "VDOL/tumorAntiAngiogenesis_2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGJacobiDataset(
                "VLSI/ss1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
        ]

    def generate(self, dataset: CGJacobiDataset) -> DataInstance:
        A_bin, b, x0 = _generate_cg_data(
            dataset.source_name,
            dataset.A,
            rhs_index=dataset.rhs_index,
        )
        A = to_scipy(A_bin).tocoo()
        M = A.diagonal()
        M_bin = from_numpy(M)
        b_bin = from_numpy(b)
        x0_bin = from_numpy(x0)
        return DataInstance(
            inputs=[A_bin, b_bin, x0_bin, M_bin],
            meta=dataset.benchmark_meta(),
            ref_outputs=[from_scipy(np.abs(A))],
            ref_meta={**(dataset.ref_meta or {}), "bnorm": float(np.linalg.norm(b))},
        )


class CGJacobiTestGenerator(CGJacobiSuiteSparseGenerator):
    @property
    def name(self) -> str:
        return "cg_jacobi_test"

    @property
    def pretty_name(self) -> str:
        return "Jacobi-Preconditioned Conjugate Gradients Test"

    @property
    def description(self) -> str:
        return "Small inlined symmetric positive definite systems."

    @property
    def datasets(self) -> list[CGJacobiDataset]:
        return [
            CGJacobiDataset(
                "3x3_tridiagonal",
                pretty_name="3x3 Tridiagonal",
                suites=["test"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            CGJacobiDataset(
                "3x3_dense",
                pretty_name="3x3 Dense",
                suites=["test"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            CGJacobiDataset(
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
                ref_meta={"check_residual": True},
            ),
            CGJacobiDataset(
                "3x3_indefinite_sparse",
                pretty_name="3x3 Indefinite Sparse",
                suites=["test"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            CGJacobiDataset(
                "3x3_scaled_tridiagonal",
                pretty_name="3x3 Scaled Tridiagonal",
                suites=["test"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            CGJacobiDataset(
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
                ref_meta={"check_residual": True},
            ),
        ]


class CGJacobiBenchmark(Benchmark):
    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Benjamin Berol", "bberol3@gatech.edu")]

    @property
    def description(self) -> str:
        return (
            "Hand-written code modelling the algorithm structure outlined in "
            "https://www.netlib.org/templates/templates.pdf Page 13."
        )

    @property
    def motivation(self) -> str:
        return (
            '"The preconditioned conjugate gradient method is well established '
            "for solving linear systems of equations that arise from the "
            "discretization of partial differential equations. Point and block "
            'Jacobi preconditioning are both common preconditioning techniques." '
            "Sparsity enhances the functionality of both the solver and the "
            "preconditioner. Similar to normal conjugate gradient, the SpMV "
            "done once per iteration reduces complexity from O(n^2) to O(nnz). "
            "Furthermore, the sparse block Jacobi preconditioner avoids filling "
            "in all the 0s around the blocks, which prevents memory overhead "
            "and keeps the per-iteration block solve cost proportional to the "
            "block size instead of the full matrix dimension."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Block Jacobi Preconditioning of the Conjugate Gradient Method on"
                    " a Vector Processor"
                ),
                authors=[Author("M. Hegland"), Author("P. E. Saylor")],
                journal="International Journal of Computer Mathematics",
                volume=44,
                number="1-4",
                pages="71-89",
                year=1992,
                doi="10.1080/00207169208804096",
            ),
            Ref(
                title=(
                    "Templates for the Solution of Linear Systems: Building Blocks"
                    " for Iterative Methods"
                ),
                authors=[
                    Author("Richard Barrett"),
                    Author("Michael Berry"),
                    Author("Tony F. Chan"),
                    Author("James Demmel"),
                    Author("June Donato"),
                    Author("Jack Dongarra"),
                    Author("Victor Eijkhout"),
                    Author("Roldan Pozo"),
                    Author("Charles Romine"),
                    Author("Henk van der Vorst"),
                ],
                publisher="SIAM",
                year=1994,
                url="https://www.netlib.org/templates/templates.pdf",
                doi="10.1137/1.9781611971538",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        A_bin, b_bin, _x0_bin, _M_bin = self._input
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
        # A sparse framework may return x as a sparse vector.
        x_sol = _dense(self._output[0])
        r = np.asarray(b - A @ x_sol).ravel()
        # Bound the matrix-vector product and the subtraction separately.
        roundoff = summation_error_bound(
            np,
            A.shape[1] + 1,
            abs_A @ np.abs(x_sol),
            np.result_type(A.dtype, x_sol.dtype),
        )
        roundoff += operation_error_bound(np, r, r.dtype)
        residual = np.linalg.norm(r)
        bound = np.linalg.norm(roundoff)
        # The norm adds a sum of squares and a square root.
        bound += summation_error_bound(np, r.size + 2, residual + bound, r.dtype)
        assert np.isfinite(residual) and np.isfinite(bound), (
            "Non-finite residual or error bound"
        )
        if self._ref_meta and self._ref_meta.get("check_residual"):
            assert residual < 1e-6 * self._ref_meta["bnorm"] + 1e-6 + bound, (
                f"Preconditioned CG residual too high for {param.dataset.name}"
            )
            return

        # PCG returns once ||b - Ax|| drops below rel_tol * ||b||.
        rel_tol = (self._meta or {}).get("rel_tol", 1e-6)
        bnorm = self._ref_meta["bnorm"]
        abs_tol = (self._meta or {}).get("abs_tol", 1e-20)
        limit = max(rel_tol * bnorm, abs_tol) + bound
        assert residual <= limit, (
            f"Preconditioned CG residual too high for {param.dataset.name}: {residual}"
        )

    @property
    def name(self) -> str:
        return "cg_jacobi"

    @property
    def pretty_name(self) -> str:
        return "Jacobi-Preconditioned Conjugate Gradients"

    @property
    def generators(self):
        return [CGJacobiTestGenerator(), CGJacobiSuiteSparseGenerator()]

    def benchmark(self, xp, meta: dict[str, Any], A, b, x0, M):
        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 100)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        # tol_sq used to avoid having to sqrt dot products when checking tolerance
        tol_sq = tolerance * tolerance

        x = x0
        r = b - A @ x
        z = xp.replace(r / M, xp.nan, 0)
        rho = xp.vecdot(r, z)
        p = z
        it = 0
        rr = xp.vecdot(r, r)[()]

        if rr >= tol_sq:
            while it < max_iter:
                Ap = A @ p
                alpha = rho / xp.vecdot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap

                new_rr = xp.vecdot(r, r)[()]

                it += 1

                if new_rr < tol_sq:
                    break

                z = xp.replace(r / M, xp.nan, 0)
                new_rho = xp.vecdot(r, z)
                beta = new_rho / rho
                p = z + beta * p
                rho = new_rho
                rr = new_rr

        return x
