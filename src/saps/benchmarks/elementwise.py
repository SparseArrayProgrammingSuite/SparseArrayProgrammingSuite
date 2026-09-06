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
            ref_outputs = [from_numpy(np.multiply(A, B))]
        return DataInstance(
            [from_numpy(A), from_numpy(B)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# PASTA varies whether element-wise operands share a nonzero pattern, since the
# output size is only predictable when they do. Overlap is that parameter here.
_ELEMENTWISE_OVERLAPS = [1.0, 0.75, 0.5, 0.25]


def _matrix_with_overlap(coo, rng: np.random.Generator, overlap: float):
    """Return a matrix with the same shape and nonzero count as *coo* that
    shares *overlap* of its nonzero positions, the rest placed at random.
    Values are resampled from *coo*, preserving the value distribution/dtype.
    """
    import scipy.sparse as sps

    n = coo.data.shape[0]
    shared = int(round(n * overlap))
    keep = rng.choice(n, size=shared, replace=False)
    rows = [coo.row[keep]]
    cols = [coo.col[keep]]
    if n - shared:
        rows.append(rng.integers(0, coo.shape[0], size=n - shared))
        cols.append(rng.integers(0, coo.shape[1], size=n - shared))
    return sps.coo_matrix(
        (rng.choice(coo.data, size=n), (np.concatenate(rows), np.concatenate(cols))),
        shape=coo.shape,
    )


# Real matrices for the element-wise suite, each paired with a copy at every
# overlap in _ELEMENTWISE_OVERLAPS.
# (matrix name, include in the correctness test suite)
_ELEMENTWISE_MATRICES: list[tuple[str, bool]] = [
    ("email-Eu-core", True),
    ("ca-GrQc", True),
    ("wiki-vote", False),
]


class SuiteSparseElementwiseDataset(Dataset):
    def __init__(
        self,
        name: str,
        matrix: str,
        overlap: float = 1.0,
        seed: int = 0,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self.matrix = matrix
        self.overlap = overlap
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
            Ref(
                title="PASTA: A Parallel Sparse Tensor Algorithm Benchmark Suite",
                authors=[
                    Author("J. Li"),
                    Author("Y. Ma"),
                    Author("X. Wu"),
                    Author("A. Li"),
                    Author("K. Barker"),
                ],
                journal="CCF Transactions on High Performance Computing",
                volume=1,
                number=2,
                pages="111-130",
                year=2019,
                doi="10.1007/s42514-019-00012-w",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. "
            "Generative AI was used to debug code. Generative AI might be used "
            "for dataset collecting and parsing. This statement was written by "
            "hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Generate real sparse matrices for elementwise multiplication, each "
            "paired with a matrix sharing a controlled fraction of its nonzero "
            "positions. PASTA identifies that overlap as what makes the output "
            "size unpredictable; the matrices themselves are not from PASTA, "
            "whose datasets are sparse tensors."
        )

    @property
    def datasets(self) -> list[Dataset]:
        return [
            SuiteSparseElementwiseDataset(
                f"{matrix}-overlap-{int(overlap * 100)}",
                matrix,
                overlap=overlap,
                suites=["sparse", "test"]
                if in_test_suite and overlap in (1.0, 0.5)
                else ["sparse"],
                description=(
                    f"SuiteSparse matrix {matrix} multiplied elementwise by a "
                    f"matrix sharing {int(overlap * 100)}% of its nonzero "
                    f"positions."
                ),
            )
            for matrix, in_test_suite in _ELEMENTWISE_MATRICES
            for overlap in _ELEMENTWISE_OVERLAPS
        ]

    def generate(self, dataset: SuiteSparseElementwiseDataset) -> DataInstance:
        base_coo = to_scipy(fetch_suitesparse_matrix(dataset.matrix).inputs[0]).tocoo()

        rng = np.random.default_rng(dataset.seed)
        A_coo = base_coo
        B_coo = _matrix_with_overlap(base_coo, rng, dataset.overlap)

        ref_outputs = None
        if "test" in dataset.suites:
            output_coo = A_coo.multiply(B_coo).tocoo()
            ref_outputs = [from_scipy(output_coo)]

        return DataInstance(
            [
                from_scipy(A_coo),
                from_scipy(B_coo),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Fixed occupancy for the random element-wise inputs, so that overlap is the
# only variable. At this dimension it gives 50 nonzeros per row.
_ELEMENTWISE_UNIFORM_DENSITY = 0.01


class UniformRandomElementwiseDataset(Dataset):
    def __init__(
        self,
        name: str,
        dim: int,
        density: float,
        overlap: float = 1.0,
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
        self.overlap = overlap
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
        return [
            Ref(
                title="PASTA: A Parallel Sparse Tensor Algorithm Benchmark Suite",
                authors=[
                    Author("J. Li"),
                    Author("Y. Ma"),
                    Author("X. Wu"),
                    Author("A. Li"),
                    Author("K. Barker"),
                ],
                journal="CCF Transactions on High Performance Computing",
                volume=1,
                number=2,
                pages="111-130",
                year=2019,
                doi="10.1007/s42514-019-00012-w",
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
        return (
            "Generate uniform random element-wise inputs whose nonzero patterns "
            "overlap by a controlled fraction, spanning PASTA's two cases: "
            "operands sharing a pattern, where the output size is known in "
            "advance, and operands whose patterns differ, where it is not. The "
            "overlap parameterisation follows PASTA; the inputs are synthetic."
        )

    @property
    def datasets(self) -> list[Dataset]:
        return [
            UniformRandomElementwiseDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"uniform-overlap-{int(overlap * 100)}",
                dim=5000,
                density=_ELEMENTWISE_UNIFORM_DENSITY,
                overlap=overlap,
                suites=["sparse", "test"],
            )
            for overlap in _ELEMENTWISE_OVERLAPS
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
        B = _matrix_with_overlap(A, rng, dataset.overlap)
        ref_outputs = None
        if "test" in dataset.suites:
            output_coo = A.multiply(B).tocoo()
            ref_outputs = [from_scipy(output_coo)]
        return DataInstance(
            [
                from_scipy(A),
                from_scipy(B),
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
        return "The elementwise multiplication of two matrices. C_ij = A_ij * B_ij"

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
</ccs2012>
"""

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
        A, B = data[0], data[1]
        return [xp.multiply(A, B)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert_coo_allclose(self._ref_outputs[0], self._output[0])
