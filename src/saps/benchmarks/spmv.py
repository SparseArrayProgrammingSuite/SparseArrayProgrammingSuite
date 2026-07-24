import numpy as np

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
from saps_framework import BinsparseFormat


class DenseMatVecDataset(Dataset):
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


class DenseMatVecGenerator(Generator):
    @property
    def name(self) -> str:
        return "dense_matmul_generator"

    @property
    def pretty_name(self) -> str:
        return "Dense MatVec Generator"

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
            DenseMatVecDataset("small", 10, 10, suites=["dense", "test"]),
            DenseMatVecDataset("medium", 100, 100, suites=["dense", "test"]),
            DenseMatVecDataset("large", 1000, 1000, suites=["dense"]),
        ]

    def generate(self, dataset: DenseMatVecDataset) -> DataInstance:
        gen = np.random.Generator(np.random.PCG64(42))
        A = gen.random((dataset.dim1, dataset.dim2))
        b = gen.random((dataset.dim2,))
        ref_outputs = None
        if "test" in dataset.suites:
            ref_outputs = [BinsparseFormat.from_numpy(np.matmul(A, b))]
        return DataInstance(
            [BinsparseFormat.from_numpy(A), BinsparseFormat.from_numpy(b)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class SuiteSparseMatVecDataset(Dataset):
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


class SuiteSparseMatVecGenerator(Generator):
    @property
    def name(self) -> str:
        return "suitesparse_matmul_generator"

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return "Suite Sparse MatVec Generator"

    @property
    def description(self) -> str:
        return (
            "Sparse input generator for matrix multiplication"
            " based on the suite sparse matrix collection."
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
                    "Finch: Sparse and Structured Array Programming with Control Flow"
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
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was not used to write the benchmark function. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse matrices for matrix multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            SuiteSparseMatVecDataset(
                "email-Eu-core", "email-Eu-core", suites=["sparse", "test"]
            ),
            SuiteSparseMatVecDataset(
                "CollegeMsg", "CollegeMsg", suites=["sparse", "test"]
            ),
            SuiteSparseMatVecDataset("wiki-vote", "wiki-vote", suites=["sparse"]),
        ]

    def generate(self, dataset: SuiteSparseMatVecDataset) -> DataInstance:
        raw = fetch_suitesparse_matrix(dataset.matrix)
        A_bin = raw.inputs[0]
        A_coo = A_bin.to_scipy_coo()

        gen = np.random.Generator(np.random.PCG64(42))
        b = gen.random((A_coo.shape[1],))

        ref_outputs = None
        if "test" in dataset.suites:
            output = A_coo @ b
            ref_outputs = [BinsparseFormat.from_numpy(output)]

        return DataInstance(
            [A_bin, BinsparseFormat.from_numpy(b)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Densities (fraction of nonzeros) for the uniform random sparse generator,
# spanning very sparse to moderately dense.
UNIFORM_SPARSE_DENSITIES = [0.00001, 0.0001, 0.001, 0.01, 0.1]


class UniformRandomMatVecDataset(Dataset):
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


class UniformRandomMatVecGenerator(Generator):
    @property
    def name(self) -> str:
        return "uniform_random_matmul_generator"

    @property
    def pretty_name(self) -> str:
        return "Uniform Random Sparse MatVec Generator"

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
            "Generative AI was not used to write the benchmark function. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[Dataset]:
        return [
            UniformRandomMatVecDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"uniform-{density:.0e}",
                dim=5000,
                density=density,
                suites=["sparse", "test"],
            )
            for density in UNIFORM_SPARSE_DENSITIES
        ]

    def generate(self, dataset: UniformRandomMatVecDataset) -> DataInstance:
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
            ref_outputs = [BinsparseFormat.from_numpy(output)]
        return DataInstance(
            [
                BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape),
                BinsparseFormat.from_numpy(b),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class MatrixVectorBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "matrix_vector_multiplication"

    @property
    def pretty_name(self) -> str:
        return "Matrix Vector Multiplication"

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
        return ["micro-benchmark"]

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
                    "Finch: Sparse and Structured Array Programming with Control Flow"
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
            "Generative AI was not used to write the benchmark function."
            "This statement was written manually."
        )

    @property
    def generators(self) -> list[Generator]:
        return [
            DenseMatVecGenerator(),
            SuiteSparseMatVecGenerator(),
            UniformRandomMatVecGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A = data[0]
        b = data[1]
        return [xp.matmul(A, b)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        ref_coo = BinsparseFormat.to_coo(self._ref_outputs[0])
        out_coo = BinsparseFormat.to_coo(self._output[0])
        assert ref_coo.data["shape"] == out_coo.data["shape"]
        assert np.allclose(ref_coo.data["values"], out_coo.data["values"])
