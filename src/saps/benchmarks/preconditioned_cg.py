from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

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
from saps.downloaders.suitesparse import random_rhs_for_matrix

BLOCK_JACOBI_BLOCK_SIZE = 16


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


class PreconditionedCGDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        *,
        A=None,
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
        rhs_index: int | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-6,
    ):
        dataset_name = suite_sparse_rhs_dataset_name(source_name, rhs_index)
        super().__init__(
            dataset_name,
            source_name=source_name,
            pretty_name=f"Preconditioned CG {source_name}",
            suites=suites,
            rhs_index=rhs_index,
        )
        self.A = A
        self.ref_meta = ref_meta
        self.max_iter = max_iter
        self.rel_tol = rel_tol

    def benchmark_meta(self) -> dict[str, Any]:
        return {"max_iter": self.max_iter, "rel_tol": self.rel_tol}


class BlockJacobiCGGenerator(Generator[PreconditionedCGDataset]):
    @property
    def name(self) -> str:
        return "block_jacobi_cg_inputs"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi CG SuiteSparse Data Generator"

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
        return PreconditionedCGBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return PreconditionedCGBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return PreconditionedCGBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return PreconditionedCGBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset(
                "test_A0",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A1",
                suites=["test", "trace"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A2",
                suites=["test", "trace"],
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
            PreconditionedCGDataset(
                "test_A3",
                suites=["test", "trace"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A4",
                suites=["test", "trace"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A5",
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
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset("Andrews/Andrews", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net100", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net125", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net150", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net25", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net50", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net75", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/dw256B", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/dwb512", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhd3200b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhd4800b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhdb416", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bindel/ted_B", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bindel/ted_B_unscaled", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/bcsstk34", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/bcsstm39", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm01", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm03", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/FEM_3D_thermal1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/FEM_3D_thermal2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/thermomech_TC", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/thermomech_dM", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Brunetiere/thermal", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Cunningham/qa8fm", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FEMLAB/poisson2D", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("FEMLAB/problem1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex29", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex37", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex7", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Freescale/circuit5M_dc", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/jnlbrng1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/minsurfo", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/obstclae", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/wathen100", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/wathen120", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Grund/poli", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Guettel/TEM27623", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk01", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk03", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk04", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk05", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk08", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk22", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm05", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm06", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm07", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm08", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm09", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm11", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm12", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm19", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm20", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm21", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm22", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm23", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm24", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm25", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm26", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/fs_541_1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/gr_30_30", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/lund_a", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/lund_b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos4", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos6", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos7", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Hamm/add32", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Lourakis/bundle1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MathWorks/Muu", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MathWorks/tomography", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MaxPlanck/shallow_water1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MaxPlanck/shallow_water2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Mulvey/finan512", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nasa/nasa2146", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Norris/fv1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Norris/fv2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Oberwolfach/LF10", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Oberwolfach/LFAT5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("PARSEC/Si2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/bodyy4", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1em1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1em6", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh2e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh2em5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh3e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh3em5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Sandia/ASIC_100ks", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Sandia/ASIC_320ks", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Um/2cubes_sphere", max_iter=100, rel_tol=1e-06, rhs_index=0),
        ]

    def generate(self, dataset: PreconditionedCGDataset) -> DataInstance:
        import scipy.sparse as sp

        A_bin, b, x0 = _generate_cg_data(
            dataset.source_name,
            dataset.A,
            rhs_index=dataset.rhs_index,
        )
        A_csr = to_scipy(A_bin).tocsr()
        # Create one block for every processor modelled after
        # this example: https://petsc.org/main/src/ksp/ksp/tutorials/ex7.c.html
        n = A_csr.shape[0]
        block_size = min(BLOCK_JACOBI_BLOCK_SIZE, n)
        blocks = []
        i = 0
        while i < n:
            j = min(i + block_size, n)
            A_ii = A_csr[i:j, i:j].toarray()
            L_i = np.linalg.cholesky(A_ii)
            blocks.append(L_i)
            i = j
        M = sp.block_diag(blocks).tocoo()
        M_bin = from_scipy(M)
        b_bin = from_numpy(b)
        x0_bin = from_numpy(x0)
        return DataInstance(
            inputs=[A_bin, b_bin, x0_bin, M_bin],
            meta=dataset.benchmark_meta(),
            ref_meta=dataset.ref_meta,
        )


class JacobiCGGenerator(Generator[PreconditionedCGDataset]):
    @property
    def name(self) -> str:
        return "jacobi_cg_inputs"

    @property
    def pretty_name(self) -> str:
        return "Jacobi CG SuiteSparse Data Generator"

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
        return PreconditionedCGBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return PreconditionedCGBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return PreconditionedCGBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return PreconditionedCGBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[PreconditionedCGDataset]:
        return [
            PreconditionedCGDataset(
                "test_A0",
                suites=["test", "trace"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A1",
                suites=["test", "trace"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A2",
                suites=["test", "trace"],
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
            PreconditionedCGDataset(
                "test_A3",
                suites=["test", "trace"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A4",
                suites=["test", "trace"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            PreconditionedCGDataset(
                "test_A5",
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
                ref_meta={"check_residual": True},
            ),
             PreconditionedCGDataset("Andrews/Andrews", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/ins2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net100", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net125", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net150", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net25", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net50", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Andrianov/net75", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/bfwb398", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/bfwb62", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/bfwb782", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/dw256B", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/dwb512", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhd3200b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhd4800b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bai/mhdb416", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bindel/ted_B", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bindel/ted_B_unscaled", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/bcsstk34", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/bcsstm39", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm01", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/crystm03", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Boeing/msc00726", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/FEM_3D_thermal1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/FEM_3D_thermal2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/thermomech_TC", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Botonakis/thermomech_dM", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Bourchtein/atmosmodd", max_iter=100, rel_tol=1e-06, rhs_index=1),
    PreconditionedCGDataset("Bourchtein/atmosmodj", max_iter=100, rel_tol=1e-06, rhs_index=1),
    PreconditionedCGDataset("Bourchtein/atmosmodl", max_iter=100, rel_tol=1e-06, rhs_index=1),
    PreconditionedCGDataset("Bourchtein/atmosmodm", max_iter=100, rel_tol=1e-06, rhs_index=1),
    PreconditionedCGDataset("Brunetiere/thermal", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Cunningham/qa8fm", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FEMLAB/poisson2D", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("FEMLAB/problem1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex29", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex37", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("FIDAP/ex7", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Freescale/circuit5M_dc", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/jnlbrng1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/minsurfo", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/obstclae", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/wathen100", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("GHS_psdef/wathen120", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Grund/poli", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Guettel/TEM27623", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcspwr01", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcspwr02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk01", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk04", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk08", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstk22", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm05", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm06", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm07", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm08", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm09", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm11", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm19", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm20", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm21", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm22", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm23", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm24", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm25", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/bcsstm26", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_144", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_24", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_61", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_62", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_73", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/can_96", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/dwt_59", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/dwt_66", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/dwt_72", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/fs_541_1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/gr_30_30", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/jpwh_991", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/lap_25", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/lund_a", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/lund_b", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos4", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos6", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/nos7", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=1),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=10),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=11),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=12),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=14),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=15),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=16),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=17),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=18),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=19),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=2),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=3),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=4),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=5),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=6),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=62),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=7),
    PreconditionedCGDataset("HB/orani678", max_iter=100, rel_tol=1e-06, rhs_index=8),
    PreconditionedCGDataset("HB/watt_1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Hamm/add32", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Lourakis/bundle1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MathWorks/Muu", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MathWorks/tomography", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MaxPlanck/shallow_water1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("MaxPlanck/shallow_water2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Mulvey/finan512", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nasa/nasa2146", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Nemeth/nemeth02", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth03", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth04", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth05", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth06", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth07", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth08", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth09", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth10", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth11", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth12", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth13", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth16", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Nemeth/nemeth17", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Norris/fv1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Norris/fv2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Oberwolfach/LF10", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Oberwolfach/LFAT5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("PARSEC/Si2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/bodyy4", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1em1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh1em6", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh2e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh2em5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh3e1", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/mesh3em5", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Pothen/sphere2", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Sandia/ASIC_100ks", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Sandia/ASIC_320ks", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Sandia/ASIC_680ks", max_iter=100, rel_tol=1e-06),
    PreconditionedCGDataset("Schenk_AFE/af_shell3", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Schenk_AFE/af_shell4", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Schenk_AFE/af_shell7", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Schenk_AFE/af_shell8", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("Um/2cubes_sphere", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("VDOL/hangGlider_1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("VDOL/tumorAntiAngiogenesis_1", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("VDOL/tumorAntiAngiogenesis_2", max_iter=100, rel_tol=1e-06, rhs_index=0),
    PreconditionedCGDataset("VLSI/ss1", max_iter=100, rel_tol=1e-06),

        ]

    def generate(self, dataset: PreconditionedCGDataset) -> DataInstance:
        A_bin, b, x0 = _generate_cg_data(
            dataset.source_name,
            dataset.A,
            rhs_index=dataset.rhs_index,
        )
        M = to_scipy(A_bin).diagonal()
        M_bin = from_numpy(M)
        b_bin = from_numpy(b)
        x0_bin = from_numpy(x0)
        return DataInstance(
            inputs=[A_bin, b_bin, x0_bin, M_bin],
            meta=dataset.benchmark_meta(),
            ref_meta=dataset.ref_meta,
        )


class _PreconditionedCGBase(Benchmark, ABC):
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
            ),
            Ref(
                title="",
                authors=[],
                url="https://www.netlib.org/templates/templates.pdf",
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

    @abstractmethod
    def _solve_cg(self, xp, M, r):
        raise NotImplementedError

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or not self._ref_meta.get("check_residual"):
            return

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
        b = to_numpy(b_bin)
        x_sol = to_numpy(self._output[0])
        residual = b - A @ x_sol
        assert np.linalg.norm(residual) < 1e-6 * np.linalg.norm(b) + 1e-6, (
            f"Preconditioned CG residual too high for {param.dataset.name}"
        )

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        A, b, x0, M = data
        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 100)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        # tol_sq used to avoid having to sqrt dot products when checking tolerance
        tol_sq = tolerance * tolerance

        x = x0
        r = b - A @ x
        z = self._solve_cg(xp, M, r)
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

                z = self._solve_cg(xp, M, r)
                new_rho = xp.vecdot(r, z)
                beta = new_rho / rho
                p = z + beta * p
                rho = new_rho
                rr = new_rr

        x_solution = x
        return [x_solution]


class _BlockJacobiCGMixin:
    @property
    def generators(self):
        return [BlockJacobiCGGenerator()]

    def _solve_cg(self, xp, M, r):
        y = xp.linalg.solve(M, r)
        return xp.linalg.solve(M.T, y)


class _JacobiCGMixin:
    @property
    def generators(self):
        return [JacobiCGGenerator()]

    def _solve_cg(self, xp, M, r):
        output = r / M
        if hasattr(xp, "with_fill_value"):
            return xp.with_fill_value(output, 0)
        return output


class PreconditionedCGBenchmark(_BlockJacobiCGMixin, _PreconditionedCGBase):
    @property
    def name(self) -> str:
        return "preconditioned_cg"

    @property
    def pretty_name(self) -> str:
        return "Preconditioned Conjugate Gradient (Block Jacobi)"


class JacobiPreconditionedCGBenchmark(_JacobiCGMixin, _PreconditionedCGBase):
    @property
    def name(self) -> str:
        return "jacobi_preconditioned_cg"

    @property
    def pretty_name(self) -> str:
        return "Preconditioned Conjugate Gradient (Jacobi)"
