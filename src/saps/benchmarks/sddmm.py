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


class SDDMMSuiteSparseDataset(Dataset):
    def __init__(
        self,
        name: str,
        middle_dim: int,
        matrix_name: str,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"SDDMM Input {self._pretty_name}."
        self._suites = suites or ["sparse", "test"]
        self.middle_dim = middle_dim
        self.matrix_name = matrix_name

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


# Bharadwaj et al. run SDDMM on real-world matrices at an embedding width of
# r = 128, the value used here.
SDDMM_EMBEDDING_WIDTH = 128

# SDDMM's two applications are collaborative filtering and graph neural networks
# with self-attention. The graphs below cover the latter; the Erdos-Renyi
# generator below covers the synthetic case the same paper uses. Sizes stay
# modest because the benchmark forms the dense A @ B product, which costs n**2.
# (matrix name, include in the correctness test suite)
_SDDMM_GRAPHS: list[tuple[str, bool]] = [
    ("email-Eu-core", True),
    ("email", True),
    ("ca-GrQc", True),
    ("wiki-vote", False),
    ("ca-HepPh", False),
]


class SDDMMSuiteSparseGenerator(Generator):
    @property
    def name(self) -> str:
        return "suitesparse_sddmm_generator"

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def pretty_name(self) -> str:
        return "SuiteSparse SDDMM Generator"

    @property
    def description(self) -> str:
        return (
            "Input generator for SDDMM, sampling a dense product at the nonzero"
            " pattern of a real graph from the SuiteSparse Matrix Collection."
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
                title="Distributed-Memory Sparse Kernels for Machine Learning",
                authors=[
                    Author("V. Bharadwaj"),
                    Author("A. Buluç"),
                    Author("J. Demmel"),
                ],
                conference=(
                    "IEEE International Parallel and Distributed Processing "
                    "Symposium (IPDPS)"
                ),
                year=2022,
                url="https://arxiv.org/abs/2203.07673",
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
        return "Generate matrices for sampled dense-dense matrix multiplication."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            SDDMMSuiteSparseDataset(
                f"{matrix}-{SDDMM_EMBEDDING_WIDTH}",
                SDDMM_EMBEDDING_WIDTH,
                matrix,
                suites=["sparse", "test"] if in_test_suite else ["sparse"],
                description=(
                    f"SDDMM over the nonzero pattern of SuiteSparse graph "
                    f"{matrix} with embedding width {SDDMM_EMBEDDING_WIDTH}."
                ),
            )
            for matrix, in_test_suite in _SDDMM_GRAPHS
        ]

    def generate(self, dataset: SDDMMSuiteSparseDataset) -> DataInstance:
        S_bin = fetch_suitesparse_matrix(dataset.matrix_name).inputs[0]
        sample_matrix = to_scipy(S_bin).tocoo()

        gen = np.random.Generator(np.random.PCG64(42))
        A = gen.random((sample_matrix.shape[0], dataset.middle_dim))
        B = gen.random((dataset.middle_dim, sample_matrix.shape[1]))
        ref_outputs = None
        if "test" in dataset.suites:
            ref = sample_matrix.multiply(np.matmul(A, B)).tocoo()
            ref_outputs = [from_scipy(ref)]
        return DataInstance(
            [S_bin, from_numpy(A), from_numpy(B)],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


# Nonzeros per row of the sampling matrix, the sweep Bharadwaj et al. use for
# their Erdos-Renyi experiments. Parameterising by row occupancy rather than by
# density keeps every row sampled: at this dimension a density of 1e-5 would
# leave 95% of rows empty, so SDDMM would sample almost nothing.
_SDDMM_NONZEROS_PER_ROW = [21, 43, 65, 85, 107, 128, 149]


class UniformRandomSDDMMDataset(Dataset):
    def __init__(
        self,
        name: str,
        dim: int,
        middle_dim: int,
        density: float,
        seed: int = 0,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or (
            f"Uniform random SDDMM input {self._pretty_name}."
        )
        self._suites = suites or ["sparse", "test"]
        self.dim = dim
        self.middle_dim = middle_dim
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


class UniformRandomSDDMMGenerator(Generator):
    @property
    def name(self) -> str:
        return "uniform_random_sddmm_generator"

    @property
    def pretty_name(self) -> str:
        return "Uniform Random SDDMM Generator"

    @property
    def description(self) -> str:
        return (
            "Generates SDDMM inputs with a uniform random sparse sampling matrix "
            "at a range of densities."
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
                title="Distributed-Memory Sparse Kernels for Machine Learning",
                authors=[
                    Author("V. Bharadwaj"),
                    Author("A. Buluç"),
                    Author("J. Demmel"),
                ],
                conference=(
                    "IEEE International Parallel and Distributed Processing "
                    "Symposium (IPDPS)"
                ),
                year=2022,
                url="https://arxiv.org/abs/2203.07673",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function. "
            "Generative AI might be used for dataset collecting and parsing. "
            "This statement was written manually."
        )

    @property
    def motivation(self) -> str:
        return (
            "Generate Erdos-Renyi sampling matrices for SDDMM, the synthetic "
            "workload Bharadwaj et al. use, swept over their range of nonzeros "
            "per row so that row occupancy rather than matrix size varies."
        )

    @property
    def datasets(self) -> list[Dataset]:
        # Bounded by the dense A @ B product the benchmark forms, which costs
        # dim**2 regardless of how sparse the sampling matrix is.
        dim = 5000
        return [
            UniformRandomSDDMMDataset(
                # No dots in the name: the framework parses params as
                # "generator.dataset" by splitting on ".".
                f"uniform-{nonzeros_per_row}-per-row",
                dim=dim,
                middle_dim=SDDMM_EMBEDDING_WIDTH,
                density=nonzeros_per_row / dim,
                suites=["sparse", "test"]
                if nonzeros_per_row in (21, 85, 149)
                else ["sparse"],
            )
            for nonzeros_per_row in _SDDMM_NONZEROS_PER_ROW
        ]

    def generate(self, dataset: UniformRandomSDDMMDataset) -> DataInstance:
        import scipy.sparse as sps

        rng = np.random.default_rng(dataset.seed)
        S = sps.random_array(
            (dataset.dim, dataset.dim),
            density=dataset.density,
            format="coo",
            rng=rng,
        )
        A = rng.random((dataset.dim, dataset.middle_dim))
        B = rng.random((dataset.middle_dim, dataset.dim))
        ref_outputs = None
        if "test" in dataset.suites:
            ref = S.multiply(np.matmul(A, B)).tocoo()
            ref_outputs = [from_scipy(ref)]
        return DataInstance(
            [
                from_scipy(S),
                from_numpy(A),
                from_numpy(B),
            ],
            meta={"dataset": dataset.name},
            ref_outputs=ref_outputs,
        )


class SDDMMBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "sddmm"

    @property
    def pretty_name(self) -> str:
        return "Sampled Dense-Dense Matrix Multiplication"

    @property
    def motivation(self) -> str:
        return (
            "Sampled matrix multiplication is a performant core primitive"
            "for machine learning algorithms like ALS, Sparse Factor Analysis,"
            " and Graph Neural Networks."
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
                    "Sampled Dense Matrix Multiplication for "
                    "High-Performance Machine Learning"
                ),
                authors=[
                    Author("I. Nisa"),
                    Author("A. Sukumaran-Rajam"),
                    Author("S. Kurt"),
                    Author("C. Hong"),
                    Author("P. Sadayappan"),
                ],
                journal="IEEE HiPC",
                year=2018,
                url="https://ieeexplore.ieee.org/iel7/8632556/8638028/08638042.pdf",
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
        return [SDDMMSuiteSparseGenerator(), UniformRandomSDDMMGenerator()]

    def benchmark(self, xp, data: list, meta: dict):
        S = data[0]
        A = data[1]
        B = data[2]
        return [xp.multiply(S, A @ B)]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert_coo_allclose(self._ref_outputs[0], self._output[0])
