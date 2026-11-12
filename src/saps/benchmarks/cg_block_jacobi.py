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
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.benchmarks.suitesparse import (
    SuiteSparseDataset,
    fetch_suitesparse_linear_system,
)
from saps.downloaders.suitesparse import random_rhs_for_matrix
from saps.util.error_bounds import operation_error_bound, summation_error_bound

BLOCK_JACOBI_BLOCK_SIZE = 16


def _dense(array: BinsparseTensor):
    try:
        return to_numpy(array)
    except TypeError:
        return np.asarray(to_sparse(array).todense())


def _block_jacobi_preconditioner(A_bin):
    """Lower Cholesky factors of A's diagonal blocks, as one block-diagonal matrix.

    One block for every processor, modelled after
    https://petsc.org/main/src/ksp/ksp/tutorials/ex7.c.html. Every block factors
    at once; identity pads a short last block without changing its factor.
    """
    A = to_scipy(A_bin).tocoo()
    A.sum_duplicates()
    n = A.shape[0]
    size = min(BLOCK_JACOBI_BLOCK_SIZE, n)
    count = -(-n // size)
    blocks = np.zeros((count, size, size), dtype=A.dtype)
    inside = A.row // size == A.col // size
    rows, cols = A.row[inside], A.col[inside]
    blocks[rows // size, rows % size, cols % size] = A.data[inside]
    padding = count * size - n
    blocks[-1, size - padding :, size - padding :] = np.eye(padding)
    factors = np.linalg.cholesky(blocks)
    # Blocks are stored whole, upper zeros included, as block_diag stores them.
    block, row, col = np.indices(factors.shape, dtype=np.int32).reshape(3, -1)
    row, col = block * size + row, block * size + col
    keep = (row < n) & (col < n)
    return from_scipy(
        scipy_sparse.coo_array(
            (factors.ravel()[keep], (row[keep], col[keep])), shape=(n, n)
        )
    )


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


class CGBlockJacobiDataset(SuiteSparseDataset):
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


class CGBlockJacobiSuiteSparseGenerator(Generator[CGBlockJacobiDataset]):
    @property
    def name(self) -> str:
        return "cg_block_jacobi_suitesparse"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi-Preconditioned Conjugate Gradients SuiteSparse"

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
        return CGBlockJacobiBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return CGBlockJacobiBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return CGBlockJacobiBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return CGBlockJacobiBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[CGBlockJacobiDataset]:
        return [
            CGBlockJacobiDataset(
                "Andrews/Andrews",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net100",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net125",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net150",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net25",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net50",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Andrianov/net75",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Bai/dw256B", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Bai/dwb512", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Bai/mhd3200b",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Bai/mhd4800b",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Bai/mhdb416", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Bindel/ted_B",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Bindel/ted_B_unscaled",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Boeing/bcsstk34",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Boeing/bcsstm39",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Boeing/crystm01",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Boeing/crystm02",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Boeing/crystm03",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Botonakis/FEM_3D_thermal1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Botonakis/FEM_3D_thermal2",
                suites=["standard", "trace", "train"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Botonakis/thermomech_TC",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Botonakis/thermomech_dM",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Brunetiere/thermal",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Cunningham/qa8fm",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "FEMLAB/poisson2D",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGBlockJacobiDataset(
                "FEMLAB/problem1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "FIDAP/ex29", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "FIDAP/ex37", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "FIDAP/ex5", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "FIDAP/ex7", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Freescale/circuit5M_dc",
                suites=["standard"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "GHS_psdef/jnlbrng1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "GHS_psdef/minsurfo",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "GHS_psdef/obstclae",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "GHS_psdef/wathen100",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "GHS_psdef/wathen120",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Grund/poli",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGBlockJacobiDataset(
                "Guettel/TEM27623",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk01", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk02", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk03", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk04", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk05", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk08", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstk22", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm02", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm05", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm06", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm07", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm08", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm09", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm11", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm12", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm19", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm20", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm21", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm22", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm23", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm24", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm25", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/bcsstm26", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/fs_541_1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/gr_30_30", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/lund_a", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/lund_b", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/nos1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/nos4", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/nos6", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "HB/nos7", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Hamm/add32",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGBlockJacobiDataset(
                "Lourakis/bundle1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "MathWorks/Muu",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "MathWorks/tomography",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "MaxPlanck/shallow_water1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "MaxPlanck/shallow_water2",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Mulvey/finan512",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Nasa/nasa2146",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
            CGBlockJacobiDataset(
                "Norris/fv1", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Norris/fv2", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Oberwolfach/LF10",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Oberwolfach/LFAT5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "PARSEC/Si2", suites=["standard", "trace"], max_iter=100, rel_tol=1e-06
            ),
            CGBlockJacobiDataset(
                "Pothen/bodyy4",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh1e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh1em1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh1em6",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh2e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh2em5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh3e1",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Pothen/mesh3em5",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Sandia/ASIC_100ks",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Sandia/ASIC_320ks",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
            ),
            CGBlockJacobiDataset(
                "Um/2cubes_sphere",
                suites=["standard", "trace"],
                max_iter=100,
                rel_tol=1e-06,
                rhs_index=0,
            ),
        ]

    def generate(self, dataset: CGBlockJacobiDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        # The preconditioner is costly to factor, so it is cached apart from A.
        reference = CGBlockJacobiSuiteSparseReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.inputs.extend(reference.inputs)
        return problem

    def generate_inputs(self, dataset: CGBlockJacobiDataset) -> DataInstance:
        A_bin, b, x0 = _generate_cg_data(
            dataset.source_name,
            dataset.A,
            rhs_index=dataset.rhs_index,
        )
        return DataInstance(
            inputs=[A_bin, from_numpy(b), from_numpy(x0)],
            meta=dataset.benchmark_meta(),
            ref_outputs=[from_scipy(np.abs(to_scipy(A_bin).tocsr()).tocoo())],
            ref_meta={**(dataset.ref_meta or {}), "bnorm": float(np.linalg.norm(b))},
        )


class CGBlockJacobiSuiteSparseReferenceGenerator(
    ReferenceGenerator[CGBlockJacobiDataset]
):
    @property
    def name(self) -> str:
        return "cg_block_jacobi_suitesparse_reference"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi-Preconditioned Conjugate Gradients SuiteSparse Reference"

    @property
    def description(self) -> str:
        return (
            "Block Jacobi preconditioner for Block Jacobi-Preconditioned Conjugate"
            " Gradients SuiteSparse, factored once and cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[CGBlockJacobiDataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in CGBlockJacobiSuiteSparseGenerator().datasets
        ]

    def generate(self, dataset: ReferenceDataset[CGBlockJacobiDataset]) -> DataInstance:
        A_bin = (
            CGBlockJacobiSuiteSparseGenerator()
            .generate_inputs(dataset.dataset)
            .inputs[0]
        )
        return DataInstance(inputs=[_block_jacobi_preconditioner(A_bin)], meta={})


class CGBlockJacobiSuiteSparseReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return CGBlockJacobiSuiteSparseReferenceGenerator()


class CGBlockJacobiTestGenerator(CGBlockJacobiSuiteSparseGenerator):
    @property
    def name(self) -> str:
        return "cg_block_jacobi_test"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi-Preconditioned Conjugate Gradients Test"

    @property
    def description(self) -> str:
        return "Small inlined symmetric positive definite systems."

    @property
    def datasets(self) -> list[CGBlockJacobiDataset]:
        return [
            CGBlockJacobiDataset(
                "3x3_tridiagonal",
                pretty_name="3x3 Tridiagonal",
                suites=["test"],
                A=np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
                ref_meta={"check_residual": True},
            ),
            CGBlockJacobiDataset(
                "3x3_dense",
                pretty_name="3x3 Dense",
                suites=["test"],
                A=np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
                ref_meta={"check_residual": True},
            ),
            CGBlockJacobiDataset(
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
            CGBlockJacobiDataset(
                "3x3_indefinite_sparse",
                pretty_name="3x3 Indefinite Sparse",
                suites=["test"],
                A=np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
                ref_meta={"check_residual": True},
            ),
            CGBlockJacobiDataset(
                "3x3_scaled_tridiagonal",
                pretty_name="3x3 Scaled Tridiagonal",
                suites=["test"],
                A=np.array(
                    [[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]
                ),
                ref_meta={"check_residual": True},
            ),
            CGBlockJacobiDataset(
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

    def generate(self, dataset: CGBlockJacobiDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        problem.inputs.append(_block_jacobi_preconditioner(problem.inputs[0]))
        return problem


class CGBlockJacobiBenchmark(Benchmark):
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
        return ["solvers", "iterative-fixed"]

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
        return "cg_block_jacobi"

    @property
    def pretty_name(self) -> str:
        return "Block Jacobi-Preconditioned Conjugate Gradients"

    @property
    def generators(self):
        return [CGBlockJacobiTestGenerator(), CGBlockJacobiSuiteSparseGenerator()]

    def benchmark(self, xp, meta: dict[str, Any], A, b, x0, M):
        rel_tol = meta.get("rel_tol", 1e-6)
        abs_tol = meta.get("abs_tol", 1e-20)
        max_iter = meta.get("max_iter", 100)

        tolerance = max(rel_tol * xp.sqrt(xp.vecdot(b, b))[()], abs_tol)
        # tol_sq used to avoid having to sqrt dot products when checking tolerance
        tol_sq = tolerance * tolerance

        x = x0
        r = b - A @ x
        y = xp.linalg.solve(M, r)
        z = xp.linalg.solve(M.T, y)
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

                y = xp.linalg.solve(M, r)
                z = xp.linalg.solve(M.T, y)
                new_rho = xp.vecdot(r, z)
                beta = new_rho / rho
                p = z + beta * p
                rho = new_rho
                rr = new_rr

        return x
