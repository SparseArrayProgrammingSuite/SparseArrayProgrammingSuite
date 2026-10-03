import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import fetch_suitesparse_matrix
from saps_framework.binsparse_utils import assert_coo_allclose


class MatrixVectorMultiplicationDenseDataset(Dataset):
    def __init__(
        self,
        name: str,
        dim1: int,
        dim2: int,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Dense MatVec Input {self._pretty_name}."
        self._suites = suites or ["dense", "test"]
        self.dim1 = dim1
        self.dim2 = dim2

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class MatrixVectorMultiplicationDenseGenerator(Generator):
    @property
    def name(self) -> str:
        return "matrix_vector_multiplication_dense"

    @property
    def pretty_name(self) -> str:
        return "Matrix-Vector Multiplication Dense"

    @property
    def description(self) -> str:
        return "Dense input generator for matrix-vector multiplication."

    @property
    def suites(self) -> list[str]:
        return ["dense"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kyle Deeds", "kdeeds@bu.edu")]

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
        return "Generate dense matrices for matrix-vector multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            MatrixVectorMultiplicationDenseDataset(
                "small", 10, 10, pretty_name="Small", suites=["dense", "test"]
            ),
            MatrixVectorMultiplicationDenseDataset(
                "medium", 100, 100, pretty_name="Medium", suites=["dense", "test"]
            ),
            MatrixVectorMultiplicationDenseDataset(
                "large", 1000, 1000, pretty_name="Large", suites=["dense"]
            ),
            # Non-square A, which the SuiteSparse suite does not cover.
            MatrixVectorMultiplicationDenseDataset(
                "rectangular",
                100,
                150,
                pretty_name="Rectangular",
                suites=["dense", "test"],
            ),
        ]

    def generate(self, dataset: MatrixVectorMultiplicationDenseDataset) -> DataInstance:
        gen = np.random.Generator(np.random.PCG64(42))
        A = gen.random((dataset.dim1, dataset.dim2))
        b = gen.random((dataset.dim2,))
        ref_outputs = None
        if "test" in dataset.suites:
            ref_outputs = [from_numpy(np.matmul(A, b))]
        return DataInstance(
            [from_numpy(A), from_numpy(b)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class MatrixVectorMultiplicationSuiteSparseDataset(Dataset):
    def __init__(
        self,
        name: str,
        matrix: str,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self.matrix = matrix
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = (
            description or f"Suite Sparse MatVec Input {self._pretty_name}."
        )
        self._suites = suites or ["sparse", "test"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


# The SpMV benchmark suite of Vuduc's autotuning study: 26 of its 28 matrices,
# omitting `bai` and `vavasis3`, which SuiteSparse does not carry under those
# names.
# (matrix name, SuiteSparse kind, include in the correctness test suite)
_SPMV_MATRICES: list[tuple[str, str, bool]] = [
    ("HB/gemat11", "power network problem sequence", False),
    ("Grund/bayer02", "chemical process simulation problem", False),
    ("Grund/bayer10", "chemical process simulation problem", False),
    ("HB/orani678", "economic problem", False),
    ("Zitney/rdist1", "chemical process simulation problem", False),
    ("Hamm/memplus", "circuit simulation problem", False),
    ("Wang/wang4", "semiconductor device problem", False),
    ("Brethour/coater2", "computational fluid dynamics problem", False),
    ("ATandT/onetone2", "frequency-domain circuit simulation problem", False),
    ("Mallya/lhr10", "chemical process simulation problem", False),
    ("Cote/vibrobox", "acoustics problem", False),
    ("Goodwin/goodwin", "computational fluid dynamics problem", False),
    ("Nasa/pwt", "duplicate structural problem", False),
    ("Mulvey/finan512", "economic problem", False),
    ("Boeing/crystk02", "materials problem", False),
    ("Goodwin/rim", "computational fluid dynamics problem", False),
    ("Simon/olafu", "structural problem", False),
    ("FIDAP/ex11", "computational fluid dynamics problem", False),
    ("Simon/raefsky4", "structural problem", False),
    ("Boeing/bcsstk35", "structural problem", False),
    ("Simon/raefsky3", "computational fluid dynamics problem", False),
    ("Simon/venkat01", "computational fluid dynamics problem sequence", False),
    ("Boeing/crystk03", "materials problem", False),
    ("Boeing/ct20stif", "structural problem", False),
    ("Nasa/nasasrb", "structural problem", False),
    ("Rothberg/3dtube", "computational fluid dynamics problem", False),
]


class MatrixVectorMultiplicationSuiteSparseGenerator(Generator):
    @property
    def name(self) -> str:
        return "matrix_vector_multiplication_suitesparse"

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return "Matrix-Vector Multiplication SuiteSparse"

    @property
    def description(self) -> str:
        return (
            "Sparse input generator for sparse matrix-vector multiplication,"
            " drawing real matrices from the SuiteSparse Matrix Collection"
            " across a range of application domains."
        )

    @property
    def suites(self) -> list[str]:
        return ["sparse"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kyle Deeds", "kdeeds@bu.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Finch: Sparse and Structured Tensor Programming with Control Flow"
                ),
                authors=[
                    Author("W. Ahrens"),
                    Author("T. Collin"),
                    Author("R. Patel"),
                    Author("K. Deeds"),
                    Author("C. Hong"),
                    Author("S. Amarasinghe"),
                ],
                journal="Proc. ACM Program. Lang. OOPSLA",
                year=2025,
                url="https://dl.acm.org/doi/pdf/10.1145/3720473",
            ),
            Ref(
                title=("The University of Florida Sparse Matrix Collection"),
                authors=[
                    Author("T. Davis"),
                    Author("Y. Hu"),
                ],
                journal="ACM Trans. Math. Softw.",
                year=2011,
                url="https://dl.acm.org/doi/pdf/10.1145/2049662.2049663",
            ),
            Ref(
                title="Automatic Performance Tuning of Sparse Matrix Kernels",
                authors=[Author("R. Vuduc")],
                institution="University of California, Berkeley",
                year=2003,
                url="https://bebop.cs.berkeley.edu/pubs/vuduc2003-dissertation.pdf",
            ),
            Ref(
                title="Evaluation Criteria for Sparse Matrix Storage Formats",
                authors=[Author("D. Langr"), Author("P. Tvrdik")],
                journal="IEEE Trans. Parallel Distrib. Syst.",
                volume=27,
                number=2,
                pages="428-440",
                year=2016,
                doi="10.1109/TPDS.2015.2401575",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was not used to write the benchmark function. "
            "Generative AI might be used for dataset collecting and parsing. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Generate real sparse matrices for matrix-vector multiplication, "
            "using the benchmark suite of Vuduc's SpMV autotuning study: 26 of "
            "its 28 matrices, spanning fluid dynamics, structural, materials, "
            "chemical process, economic, circuit and semiconductor problems. "
            "`bai` and `vavasis3` are omitted because SuiteSparse does not "
            "carry them under those names."
        )

    @property
    def datasets(self) -> list[Dataset]:
        return [
            MatrixVectorMultiplicationSuiteSparseDataset(
                name,
                name,
                suites=["sparse", "test"] if in_test_suite else ["sparse"],
                description=(
                    f"SuiteSparse matrix {name}, included to represent the "
                    f"'{kind}' domain in the matrix-vector suite."
                ),
            )
            for name, kind, in_test_suite in _SPMV_MATRICES
        ]

    def generate(
        self, dataset: MatrixVectorMultiplicationSuiteSparseDataset
    ) -> DataInstance:
        raw = fetch_suitesparse_matrix(dataset.matrix)
        A_bin = raw.inputs[0]
        A_coo = to_scipy(A_bin).tocoo()

        gen = np.random.Generator(np.random.PCG64(42))
        b = gen.random((A_coo.shape[1],))

        ref_outputs = None
        if "test" in dataset.suites:
            output = A_coo @ b
            ref_outputs = [from_numpy(output)]

        return DataInstance(
            [A_bin, from_numpy(b)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Densities (fraction of nonzeros) for the uniform random sparse generator,
# spanning very sparse to moderately dense.
UNIFORM_SPARSE_DENSITIES = [0.00001, 0.0001, 0.001, 0.01, 0.1]


class MatrixVectorMultiplicationUniformRandomDataset(Dataset):
    def __init__(
        self,
        name: str,
        dim: int,
        density: float,
        seed: int = 0,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or (
            f"Uniform random sparse matmul input {self._pretty_name}."
        )
        self._suites = suites or ["sparse", "test"]
        self.dim = dim
        self.density = density
        self.seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class MatrixVectorMultiplicationUniformRandomGenerator(Generator):
    @property
    def name(self) -> str:
        return "matrix_vector_multiplication_uniform_random"

    @property
    def pretty_name(self) -> str:
        return "Matrix-Vector Multiplication Uniform Random"

    @property
    def description(self) -> str:
        return (
            "Generates a pair of uniform random sparse matrices for sparse "
            "general matrix-matrix multiplication (SpGEMM) at a range of densities."
        )

    @property
    def suites(self) -> list[str]:
        return ["sparse"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kyle Deeds", "kdeeds@bu.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was not used to write the benchmark function. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Generate uniform random sparse matrices for matrix-vector "
            "multiplication over a range of densities. The density values are "
            "just a synthetic sweep."
        )

    @property
    def datasets(self) -> list[Dataset]:
        return [
            MatrixVectorMultiplicationUniformRandomDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"density_{density:.0e}",
                pretty_name=f"Density {density:.0e}",
                dim=5000,
                density=density,
                suites=["sparse", "test"],
            )
            for density in UNIFORM_SPARSE_DENSITIES
        ]

    def generate(
        self, dataset: MatrixVectorMultiplicationUniformRandomDataset
    ) -> DataInstance:
        import scipy.sparse as sps

        rng = np.random.default_rng(dataset.seed)
        A = sps.random_array(
            (dataset.dim, dataset.dim),
            density=dataset.density,
            format="coo",
            rng=rng,
        )
        gen = np.random.Generator(np.random.PCG64(42))
        b = gen.random((dataset.dim,))
        ref_outputs = None
        if "test" in dataset.suites:
            output = A @ b
            ref_outputs = [from_numpy(output)]
        return DataInstance(
            [
                from_scipy(A),
                from_numpy(b),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class MatrixVectorMultiplicationBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "matrix_vector_multiplication"

    @property
    def pretty_name(self) -> str:
        return "Matrix-Vector Multiplication"

    @property
    def motivation(self) -> str:
        return (
            "Matrix-vector multiplication is the key operator in linear algebra"
            "and it is widely used in almost every sparse array application. "
        )

    @property
    def description(self) -> str:
        return "The multiplication of a matrix and a vector.C_i = \\sum_j A_ij B_j"

    @property
    def suites(self) -> list[str]:
        return ["microkernel"]

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705</concept_id>
<concept_desc>Mathematics of computing~Mathematical software</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003705.10011686</concept_id>
<concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kyle Deeds", "kdeeds@bu.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Finch: Sparse and Structured Tensor Programming with Control Flow"
                ),
                authors=[
                    Author("W. Ahrens"),
                    Author("T. Collin"),
                    Author("R. Patel"),
                    Author("K. Deeds"),
                    Author("C. Hong"),
                    Author("S. Amarasinghe"),
                ],
                journal="Proc. ACM Program. Lang. OOPSLA",
                year=2025,
                url="https://dl.acm.org/doi/pdf/10.1145/3720473",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was not used to write the benchmark function. "
            "This statement was written manually."
        )

    @property
    def generators(self) -> list[Generator]:
        return [
            MatrixVectorMultiplicationDenseGenerator(),
            MatrixVectorMultiplicationSuiteSparseGenerator(),
            MatrixVectorMultiplicationUniformRandomGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A = data[0]
        b = data[1]
        return [xp.matmul(A, b)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert_coo_allclose(self._ref_outputs[0], self._output[0])
