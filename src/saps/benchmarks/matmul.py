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


class DenseMatmulDataset(Dataset):
    def __init__(
        self,
        name: str,
        dim1: int,
        dim2: int,
        dim3: int,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Dense Matmul Input {self._pretty_name}."
        self._suites = suites or ["dense", "test"]
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

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


class DenseMatmulGenerator(Generator):
    @property
    def name(self) -> str:
        return "dense_matmul_generator"

    @property
    def pretty_name(self) -> str:
        return "Dense Matmul Generator"

    @property
    def description(self) -> str:
        return "Dense input generator for matrix multiplication."

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
        return "Generate dense matrices for matrix multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            DenseMatmulDataset("small", 10, 10, 10, suites=["dense", "test"]),
            DenseMatmulDataset("medium", 100, 100, 100, suites=["dense", "test"]),
            DenseMatmulDataset("large", 1000, 1000, 1000, suites=["dense"]),
        ]

    def generate(self, dataset: DenseMatmulDataset) -> DataInstance:
        gen = np.random.Generator(np.random.PCG64(42))
        A = gen.random((dataset.dim1, dataset.dim2))
        B = gen.random((dataset.dim2, dataset.dim3))
        ref_outputs = None
        if "test" in dataset.suites:
            ref_outputs = [from_numpy(np.matmul(A, B))]
        return DataInstance(
            [from_numpy(A), from_numpy(B)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class SuiteSparseMatmulDataset(Dataset):
    def __init__(
        self,
        name: str,
        matrix_1: str,
        matrix_2: str,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self.matrix_1 = matrix_1
        self.matrix_2 = matrix_2
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = (
            description or f"Suite Sparse Matmul Input {self._pretty_name}."
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


# Buluc and Gilbert motivate SpGEMM by two application classes: graph
# algorithms, where A*A gives two-hop neighbourhoods (triangle counting,
# multi-source BFS), and algebraic multigrid, where SpGEMM forms the Galerkin
# coarse-grid operator. Both benchmark A*A, so the suite covers both classes.
# Only small matrices join the test suite: it runs the NumPy framework, which
# densifies, making cost scale with n**3 rather than with nnz.
# (matrix name, application class, include in the correctness test suite)
_MATMUL_MATRICES: list[tuple[str, str, bool]] = [
    ("email", "graph algorithms", True),
    ("email-Eu-core", "graph algorithms", True),
    ("ca-GrQc", "graph algorithms", True),
    ("bcsstk09", "algebraic multigrid", True),
    ("Chebyshev3", "algebraic multigrid", True),
    ("CollegeMsg", "graph algorithms", False),
    ("wiki-vote", "graph algorithms", False),
    ("ca-HepPh", "graph algorithms", False),
    ("Muu", "algebraic multigrid", False),
    ("fv2", "algebraic multigrid", False),
    ("Dubcova1", "algebraic multigrid", False),
]


class SuiteSparseMatmulGenerator(Generator):
    @property
    def name(self) -> str:
        return "suitesparse_matmul_generator"

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return "Suite Sparse Matmul Generator"

    @property
    def description(self) -> str:
        return (
            "Sparse input generator for sparse matrix-matrix multiplication,"
            " drawing real matrices from the SuiteSparse Matrix Collection"
            " across the application classes that motivate SpGEMM."
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
                title=(
                    "Parallel Sparse Matrix-Matrix Multiplication and Indexing: "
                    "Implementation and Experiments"
                ),
                authors=[Author("A. Buluç"), Author("J. R. Gilbert")],
                journal="SIAM Journal on Scientific Computing",
                volume=34,
                number=4,
                pages="170-191",
                year=2012,
                doi="10.1137/110848244",
            ),
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
            "Generate real sparse matrices for squaring (A*A), the operation "
            "both classes of SpGEMM application reduce to: graph algorithms, "
            "where A*A gives two-hop neighbourhoods, and algebraic multigrid, "
            "where SpGEMM forms the coarse-grid operator."
        )

    @property
    def datasets(self) -> list[Dataset]:
        return [
            SuiteSparseMatmulDataset(
                name,
                name,
                name,
                suites=["sparse", "test"] if in_test_suite else ["sparse"],
                description=(
                    f"SuiteSparse matrix {name}, squared to represent "
                    f"{application} in the matrix-matrix suite."
                ),
            )
            for name, application, in_test_suite in _MATMUL_MATRICES
        ]

    def generate(self, dataset: SuiteSparseMatmulDataset) -> DataInstance:
        A_bin = fetch_suitesparse_matrix(dataset.matrix_1).inputs[0]
        B_bin = fetch_suitesparse_matrix(dataset.matrix_2).inputs[0]
        A_coo = to_scipy(A_bin).tocoo()
        B_coo = to_scipy(B_bin).tocoo()

        ref_outputs = None
        if "test" in dataset.suites:
            output_coo = (A_coo @ B_coo).tocoo()
            ref_outputs = [from_scipy(output_coo)]

        return DataInstance(
            [A_bin, B_bin],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Densities (fraction of nonzeros) for the uniform random sparse generator,
# spanning very sparse to moderately dense.
UNIFORM_SPARSE_DENSITIES = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# Above this density the product is effectively fully dense, and building the
# reference output overruns the test suite's per-benchmark timeout.
TEST_SUITE_MAX_DENSITY = 0.01


class UniformRandomMatmulDataset(Dataset):
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


class UniformRandomMatmulGenerator(Generator):
    @property
    def name(self) -> str:
        return "uniform_random_matmul_generator"

    @property
    def pretty_name(self) -> str:
        return "Uniform Random Sparse Matmul Generator"

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
        return [
            Ref(
                title=(
                    "Parallel Sparse Matrix-Matrix Multiplication and Indexing: "
                    "Implementation and Experiments"
                ),
                authors=[
                    Author("A. Buluç"),
                    Author("J. R. Gilbert"),
                ],
                journal="SIAM Journal on Scientific Computing",
                volume=34,
                number=4,
                pages="170-191",
                year=2012,
                doi="10.1137/110848244",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was not used to write the benchmark function itself. "
            "Generative AI might be used for dataset collecting and parsing. "
            "This statement was written manually."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[Dataset]:
        return [
            UniformRandomMatmulDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"uniform-{density:.0e}",
                dim=5000,
                density=density,
                suites=["sparse", "test"]
                if density <= TEST_SUITE_MAX_DENSITY
                else ["sparse"],
            )
            for density in UNIFORM_SPARSE_DENSITIES
        ]

    def generate(self, dataset: UniformRandomMatmulDataset) -> DataInstance:
        import scipy.sparse as sps

        rng = np.random.default_rng(dataset.seed)
        A = sps.random_array(
            (dataset.dim, dataset.dim),
            density=dataset.density,
            format="coo",
            rng=rng,
        )
        B = sps.random_array(
            (dataset.dim, dataset.dim),
            density=dataset.density,
            format="coo",
            rng=rng,
        )
        ref_outputs = None
        if "test" in dataset.suites:
            output_coo = (A @ B).tocoo()
            ref_outputs = [from_scipy(output_coo)]
        return DataInstance(
            [from_scipy(A), from_scipy(B)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class MatrixMultiplicationBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "matrix_multiplication"

    @property
    def pretty_name(self) -> str:
        return "Matrix Multiplication"

    @property
    def motivation(self) -> str:
        return (
            "Matrix multiplication is the key operator in linear algebra"
            "and it is widely used in almost every sparse array application. "
        )

    @property
    def description(self) -> str:
        return "The multiplication of two matrices.C_ik = \\sum_k A_ij B_jk"

    @property
    def suites(self) -> list[str]:
        return ["micro-benchmark"]

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
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator]:
        return [
            DenseMatmulGenerator(),
            SuiteSparseMatmulGenerator(),
            UniformRandomMatmulGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A = data[0]
        B = data[1]
        return [xp.matmul(A, B)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert_coo_allclose(self._ref_outputs[0], self._output[0])
