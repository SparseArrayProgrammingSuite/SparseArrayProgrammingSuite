from typing import Any

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
from saps_framework import BinsparseFormat


class HOSVDDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str,
        description: str,
        suites: list[str],
        shape: tuple[int, int, int],
        ranks: tuple[int, int, int],
        seed: int = 42,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.shape = shape
        self.ranks = ranks
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

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["shape"] = self.shape
        data["ranks"] = self.ranks
        data["seed"] = self.seed
        return data


def _reconstruct_tensor(core, factors):
    num_modes = len(factors)
    core_idx = "".join(chr(65 + m) for m in range(num_modes))
    result_idx = "".join(chr(97 + m) for m in range(num_modes))
    terms = [core_idx]
    operands = [core]
    for mode, factor in enumerate(factors):
        terms.append(f"{result_idx[mode]}{core_idx[mode]}")
        operands.append(factor)
    return np.einsum(f"{','.join(terms)}->{result_idx}", *operands)


class HOSVDDenseGenerator(Generator[HOSVDDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "hosvd_dense_inputs"

    @property
    def pretty_name(self) -> str:
        return "Dense Low-Rank HOSVD Input Generator"

    @property
    def description(self) -> str:
        return "Generates a dense low-rank 3D tensor using random factor matrices."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return HOSVDBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return HOSVDBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return (
            "The data for this benchmark was created by randomly generating "
            "factor matrices that were both sparse and dense. These factor "
            "matrices were used to construct a factorizable matrix."
        )

    @property
    def datasets(self) -> list[HOSVDDataset]:
        return [
            HOSVDDataset(
                "small_dense_hosvd",
                "Small Dense HOSVD Tensor",
                "Dense low-rank 3D tensor using random factor matrices.",
                ["test", "trace"],
                (10, 10, 10),
                (3, 3, 3),
            )
        ]

    def generate(self, dataset: HOSVDDataset):
        dim1, dim2, dim3 = dataset.shape
        ranks = dataset.ranks
        rng = np.random.default_rng(dataset.seed)

        G = rng.random(ranks).astype(np.float64)
        A = rng.random((dim1, ranks[0])).astype(np.float64)
        B = rng.random((dim2, ranks[1])).astype(np.float64)
        C = rng.random((dim3, ranks[2])).astype(np.float64)

        X_dense = np.einsum("pqr,ip,jq,kr->ijk", G, A, B, C)

        X_bin = BinsparseFormat.from_numpy(X_dense)

        ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))
        return DataInstance(
            inputs=[X_bin, ranks_bin],
            meta={"max_iter": 50, "tolerance": 1e-8},
            ref_meta={"check_reconstruction": True},
        )


class HOSVDSparseGenerator(Generator[HOSVDDataset]):
    @property
    def name(self) -> str:
        return "hosvd_sparse_inputs"

    @property
    def pretty_name(self) -> str:
        return "Sparse Low-Rank HOSVD Input Generator"

    @property
    def description(self) -> str:
        return "Generates a sparse low-rank 3D tensor using random factor matrices."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return HOSVDBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return HOSVDBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return (
            "The data for this benchmark was created by randomly generating "
            "factor matrices that were both sparse and dense. These factor "
            "matrices were used to construct a factorizable matrix."
        )

    @property
    def datasets(self) -> list[HOSVDDataset]:
        return [
            HOSVDDataset(
                "small_sparse",
                "Small sparse HOSVD Tensor",
                "sparse_small Small Sparse HOSVD Tensor Sparse low-rank 3D tensor"
                " using random factor matrices.",
                [],
                (20, 20, 20),
                (3, 3, 3),
            )
        ]

    def generate(self, dataset: HOSVDDataset):
        dim1, dim2, dim3 = dataset.shape
        ranks = dataset.ranks
        rng = np.random.default_rng(dataset.seed)

        def get_sparse_factor(rows, cols, density=0.2):
            nnz = int(rows * cols * density)
            if nnz < 1:
                nnz = 1
            indices = rng.choice(rows * cols, nnz, replace=False)
            mat = np.zeros(rows * cols)
            mat[indices] = rng.random(nnz)
            return mat.reshape((rows, cols)).astype(np.float64)

        G = get_sparse_factor(ranks[0], ranks[1] * ranks[2], density=0.5).reshape(ranks)
        A = get_sparse_factor(dim1, ranks[0], density=0.2)
        B = get_sparse_factor(dim2, ranks[1], density=0.2)
        C = get_sparse_factor(dim3, ranks[2], density=0.2)

        X_dense = np.einsum("pqr,ip,jq,kr->ijk", G, A, B, C)

        indices = np.nonzero(X_dense)
        values = X_dense[indices]

        X_bin = BinsparseFormat.from_coo(indices, values, (dim1, dim2, dim3))

        ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))
        return DataInstance(
            inputs=[X_bin, ranks_bin],
            meta={"max_iter": 50, "tolerance": 1e-8},
        )


class HOSVDBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "hosvd"

    @property
    def pretty_name(self) -> str:
        return "High-Order SVD (Tucker Decomposition)"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def description(self) -> str:
        return (
            "This code implements the Tucker Decomposition or HOSVD algorithm for"
            " decomposing high-order tensors into a core-tensor that can be projected"
            " onto factor matrices along each mode. A typical 3D tensor will have 3"
            " modes (row, column, and frontal) and thus 3 factor matrices. The"
            " algorithm starts by finding the initial factor matrices by performing SVD"
            " on matrix unfoldings along each mode. Certain columns in these factor"
            " matrices are selected based on the ranks parameter. Then, the algorithm"
            " iteratively updates each factor matrix by projecting the original tensor"
            " onto other factor matrices. The iteration continues until max iterations"
            " is reached or the change in factor matrices becomes insignificant. The"
            " resulting factor matrices and the core tensor are returned by the"
            " benchmark function."
        )

    @property
    def motivation(self) -> str:
        return (
            "Tensor decomposition are essential for efficiently analyzing"
            " multi-dimensional data and can cut out noise during preprocessing. Tensor"
            " decomposition has applications in signal processing, computer vision,"
            " numerical linear algebra, and many other fields. HOSVD or Tucker"
            " Decomposition is one of the most widely used methods for tensor"
            " decomposition on high-level tensors, which are tensors with 3 or more"
            " dimensions."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Tensor Decompositions and Applications",
                authors=[Author("Tamara G. Kolda"), Author("Brett W. Bader")],
                journal="SIAM Review",
                publisher="Society for Industrial & Applied Mathematics (SIAM)",
                volume="51",
                number="3",
                pages="455-500",
                year=2009,
                url="https://doi.org/10.1137/07070111x",
                doi="10.1137/07070111x",
            ),
            Ref(
                title=(
                    "Harnessing Tensor Decomposition for High-Dimensional "
                    "Machine Learning"
                ),
                authors=[
                    Author("Evgeni Rustik"),
                    Author("Emiliya Viktoriia"),
                    Author("Aliona Tatyana"),
                ],
                publisher="Institute of Electrical and Electronics Engineers (IEEE)",
                year=2025,
                url="https://doi.org/10.36227/techrxiv.174417403.38431928/v1",
                doi="10.36227/techrxiv.174417403.38431928/v1",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function. Generative"
            " AI might have been used to construct tests. This statement is written by"
            " hand."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
         return """
        <ccs2012>
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
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010147.10010257.10010293.10010309</concept_id>
<concept_desc>Computing methodologies~Factorization methods</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def generators(self):
        return [HOSVDDenseGenerator(), HOSVDSparseGenerator()]

    def benchmark(self, xp, data: list, meta: dict):
        X, ranks = data
        max_iter = meta.get("max_iter", 50)
        tolerance = meta.get("tolerance", 1e-8)

        dimensions = X.shape
        num_modes = len(dimensions)

        # initial HOSVD by performing SVD on matrix unfoldings along each mode
        initial_factors: list[Any] = [None] * num_modes
        for mode in range(num_modes):
            perm = [mode] + list(range(mode)) + list(range(mode + 1, num_modes))
            unfold = xp.reshape(xp.transpose(X, perm), (dimensions[mode], -1))

            U, S, Vt = xp.linalg.svd(unfold, full_matrices=False)
            initial_factors[mode] = U[:, : ranks[mode]]

        # iteration to update each factor matrix by projecting the original
        # tensor onto other factor matrices
        for _iteration in range(max_iter):
            prev_factors = initial_factors[:]
            for mode in range(num_modes):
                initial_factors[mode] = initial_factors[mode]

                if mode == 0:
                    update = xp.einsum(
                        "Y[i, r1, r2] += X[i, j, k] * B[j, r1] * C[k, r2]",
                        X=X,
                        B=initial_factors[1],
                        C=initial_factors[2],
                    )
                elif mode == 1:
                    update = xp.einsum(
                        "Y[r0, j, r2] += X[i, j, k] * A[i, r0] * C[k, r2]",
                        X=X,
                        A=initial_factors[0],
                        C=initial_factors[2],
                    )
                elif mode == 2:
                    update = xp.einsum(
                        "Y[r0, r1, k] += X[i, j, k] * A[i, r0] * B[j, r1]",
                        X=X,
                        A=initial_factors[0],
                        B=initial_factors[1],
                    )

                perm = [mode] + list(range(mode)) + list(range(mode + 1, num_modes))
                unfold_update = xp.reshape(
                    xp.transpose(update, perm), (dimensions[mode], -1)
                )

                U, S, Vt = xp.linalg.svd(unfold_update, full_matrices=False)
                initial_factors[mode] = U[:, : ranks[mode]]

            # stop iterations when solutions stop changing significantly
            change = (
                xp.linalg.norm(initial_factors[0] - prev_factors[0])
                + xp.linalg.norm(initial_factors[1] - prev_factors[1])
                + xp.linalg.norm(initial_factors[2] - prev_factors[2])
            )
            if change[()] < tolerance:
                break

        core_tensor = xp.einsum(
            "G[p, q, r] += X[i, j, k] * A[i, p] * B[j, q] * C[k, r]",
            X=X,
            A=initial_factors[0],
            B=initial_factors[1],
            C=initial_factors[2],
        )
        return [
            core_tensor,
            initial_factors[0],
            initial_factors[1],
            initial_factors[2],
        ]

    def check(self, param):
        super().check(param)
        if not self._ref_meta or not self._ref_meta.get("check_reconstruction"):
            return
        X = self._input[0].data["values"].reshape(self._input[0].data["shape"])
        core = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        factors = [
            output.data["values"].reshape(output.data["shape"])
            for output in self._output[1:]
        ]
        X_rec = _reconstruct_tensor(core, factors)
        error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert error < 1e-5
