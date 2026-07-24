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


class DenseElementwiseDataset(Dataset):
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
        self._description = (
            description or f"Dense Elementwise Input {self._pretty_name}."
        )
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


class DenseElementwiseGenerator(Generator):
    @property
    def name(self) -> str:
        return "dense_elementwise_generator"

    @property
    def pretty_name(self) -> str:
        return "Dense Elementwise Generator"

    @property
    def description(self) -> str:
        return "Dense input generator for elementwise multiplication."

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
        return "Generate dense matrices for elementwise multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            DenseElementwiseDataset("small", 10, 10, suites=["dense", "test"]),
            DenseElementwiseDataset("medium", 100, 100, suites=["dense", "test"]),
            DenseElementwiseDataset("large", 1000, 1000, suites=["dense"]),
        ]

    def generate(self, dataset: DenseElementwiseDataset) -> DataInstance:
        gen = np.random.Generator(np.random.PCG64(42))
        A = gen.random((dataset.dim1, dataset.dim2))
        B = gen.random((dataset.dim1, dataset.dim2))
        ref_outputs = None
        if "test" in dataset.suites:
            ref_outputs = [BinsparseFormat.from_numpy(np.multiply(A, B))]
        return DataInstance(
            [BinsparseFormat.from_numpy(A), BinsparseFormat.from_numpy(B)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


def _add_sparse_noise(coo, rng: np.random.Generator, noise_fraction: float = 0.05):
    """Return a copy of *coo* with ~noise_fraction of its nonzeros replaced by
    new, randomly placed nonzeros, so nnz stays roughly unchanged while the
    sparsity pattern shifts. New values are resampled (with replacement) from
    the existing nonzero values, so the value distribution/dtype is preserved.
    """
    import scipy.sparse as sps

    n = coo.data.shape[0]
    k = int(n * noise_fraction)
    if k == 0:
        return coo

    keep = rng.choice(n, size=n - k, replace=False)
    rows = coo.row[keep]
    cols = coo.col[keep]
    values = coo.data[keep]

    new_rows = rng.integers(0, coo.shape[0], size=k)
    new_cols = rng.integers(0, coo.shape[1], size=k)
    new_values = rng.choice(coo.data, size=k)

    rows = np.concatenate([rows, new_rows])
    cols = np.concatenate([cols, new_cols])
    values = np.concatenate([values, new_values])
    return sps.coo_matrix((values, (rows, cols)), shape=coo.shape)


class SuiteSparseElementwiseDataset(Dataset):
    def __init__(
        self,
        name: str,
        matrix: str,
        noise_fraction: float = 0.05,
        seed: int = 0,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self.matrix = matrix
        self.noise_fraction = noise_fraction
        self.seed = seed
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = (
            description or f"Suite Sparse Elementwise Input {self._pretty_name}."
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


class SuiteSparseElementwiseGenerator(Generator):
    @property
    def name(self) -> str:
        return "suitesparse_elementwise_generator"

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return "Suite Sparse Elementwise Generator"

    @property
    def description(self) -> str:
        return (
            "Sparse input generator for elementwise multiplication"
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
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse matrices for elementwise multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            SuiteSparseElementwiseDataset(
                "email-Eu-core", "email-Eu-core", suites=["sparse", "test"]
            ),
            SuiteSparseElementwiseDataset(
                "CollegeMsg", "CollegeMsg", suites=["sparse", "test"]
            ),
            SuiteSparseElementwiseDataset("wiki-vote", "wiki-vote", suites=["sparse"]),
        ]

    def generate(self, dataset: SuiteSparseElementwiseDataset) -> DataInstance:
        base_coo = fetch_suitesparse_matrix(dataset.matrix).inputs[0].to_scipy_coo()

        # A and B are independent random perturbations of the same real matrix,
        # so they have distinct (but similarly structured) sparsity patterns
        # rather than being identical, while nnz stays roughly unchanged.
        rng = np.random.default_rng(dataset.seed)
        A_coo = _add_sparse_noise(base_coo, rng, dataset.noise_fraction)
        B_coo = _add_sparse_noise(base_coo, rng, dataset.noise_fraction)

        ref_outputs = None
        if "test" in dataset.suites:
            output_coo = A_coo.multiply(B_coo).tocoo()
            ref_outputs = [
                BinsparseFormat.from_coo(
                    (output_coo.row, output_coo.col), output_coo.data, output_coo.shape
                )
            ]

        return DataInstance(
            [
                BinsparseFormat.from_coo(
                    (A_coo.row, A_coo.col), A_coo.data, A_coo.shape
                ),
                BinsparseFormat.from_coo(
                    (B_coo.row, B_coo.col), B_coo.data, B_coo.shape
                ),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Densities (fraction of nonzeros) for the uniform random sparse generator,
# spanning very sparse to moderately dense.
UNIFORM_SPARSE_DENSITIES = [0.00001, 0.0001, 0.001, 0.01, 0.1]


class UniformRandomElementwiseDataset(Dataset):
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
            f"Uniform random sparse elementwise input {self._pretty_name}."
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


class UniformRandomElementwiseGenerator(Generator):
    @property
    def name(self) -> str:
        return "uniform_random_elementwise_generator"

    @property
    def pretty_name(self) -> str:
        return "Uniform Random Sparse Elementwise Generator"

    @property
    def description(self) -> str:
        return (
            "Generates a pair of uniform random sparse matrices for elementwise "
            "multiplication at a range of densities."
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
            "Generative AI was not used to write the benchmark function itself."
            "This statement was written manually."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[Dataset]:
        return [
            UniformRandomElementwiseDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"uniform-{density:.0e}",
                dim=5000,
                density=density,
                suites=["sparse", "test"],
            )
            for density in UNIFORM_SPARSE_DENSITIES
        ]

    def generate(self, dataset: UniformRandomElementwiseDataset) -> DataInstance:
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
            output_coo = A.multiply(B).tocoo()
            ref_outputs = [
                BinsparseFormat.from_coo(
                    (output_coo.row, output_coo.col),
                    output_coo.data,
                    output_coo.shape,
                )
            ]
        return DataInstance(
            [
                BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape),
                BinsparseFormat.from_coo((B.row, B.col), B.data, B.shape),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class ElementwiseBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "elementwise_multiplication"

    @property
    def pretty_name(self) -> str:
        return "Elementwise Multiplication"

    @property
    def motivation(self) -> str:
        return (
            "Elementwise multiplication is a fundamental operator in array "
            "programming and it is widely used in almost every sparse array "
            "application. "
        )

    @property
    def description(self) -> str:
        return "The elementwise multiplication of two matrices.C_ij = A_ij * B_ij"

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. This statement was written by hand."
        )

    @property
    def generators(self) -> list[Generator]:
        return [
            DenseElementwiseGenerator(),
            SuiteSparseElementwiseGenerator(),
            UniformRandomElementwiseGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A, B = data.inputs[0], data.inputs[1]
        return xp.multiply(A, B)

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
