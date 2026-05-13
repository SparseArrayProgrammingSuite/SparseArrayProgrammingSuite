import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


class CP4FactorizeableDataset(Dataset):
    def __init__(self, name, pretty_name, tags, shape, rank):
        self._name = name
        self._pretty_name = pretty_name
        self._tags = tags
        self.shape = shape
        self.rank = rank

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return f"rank = {self.rank}, shape = {self.shape}"

    @property
    def tags(self) -> list[str]:
        return self._tags


class CP4FactorizeableGenerator(Generator):
    @property
    def name(self):
        return "cp_factorizable"

    @property
    def pretty_name(self):
        return "Factorizable Tensor for CP Decomposition"

    @property
    def description(self):
        return """
            Generating a small factorizable tensor by creating random factor matrices
            and reconstructing the tensor from them (tensor should decompose easily
            with low reconstruction error).
            """

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI was used to debug some parts of the code. This statement"
            " was written by hand."
        )

    @property
    def motivation(self):
        return ""

    @property
    def references(self):
        return []

    @property
    def tags(self):
        return ["tensor-factorization", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Grace Wang", "gwang426@gatech.edu"),
            Contributor("Kseniia Suleimanova", "kseniiasuleimanova@gmail.com"),
        ]

    @property
    def datasets(self):
        return [
            CP4FactorizeableDataset(
                name="cp_factorizeable_small",
                pretty_name="Small Factorizeable CP Tensor",
                tags=[
                    "small",
                ],
                shape=(20, 20, 20, 20),
                rank=4,
            ),
        ]

    def generate(self, dataset: CP4FactorizeableDataset):
        dim1, dim2, dim3, dim4 = dataset.shape
        rank = dataset.rank

        rng = np.random.default_rng(42)

        # Create simple factor matrices for easier decomposition
        A = rng.random((dim1, rank)).astype(np.float32)
        B = rng.random((dim2, rank)).astype(np.float32)
        C = rng.random((dim3, rank)).astype(np.float32)
        D = rng.random((dim4, rank)).astype(np.float32)

        A = A / np.linalg.norm(A, axis=0, keepdims=True)
        B = B / np.linalg.norm(B, axis=0, keepdims=True)
        C = C / np.linalg.norm(C, axis=0, keepdims=True)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        lambdas = 100 * np.pow(2.0, -np.arange(rank))
        A = A * lambdas

        X = np.einsum("ir,jr,kr,lr->ijkl", A, B, C, D)

        X = BinsparseFormat.from_numpy(X)
        max_iter = 100

        return (X, rank, max_iter)


class CP4_ALS(Benchmark):
    @property
    def name(self):
        return "cp4_als"

    @property
    def pretty_name(self):
        return (
            "CANDECOMP/PARAFAC (CP) Decomposition of order 4"
            " via Alternating Least Squares (ALS)"
        )

    @property
    def description(self):
        return (
            "Computes the CP decomposition using Alternating Least Squares (ALS). "
            "Factorizes a 4-order tensor X into factor matrices A, B, C such that: "
            "$X \approx \\sum_{r=1}^{R} \\lambda_r \\cdot a_r \\circ b_r \\circ c_r"
            " \\circ d_r$ "
            "where $\\circ$ denotes the outeer product, R is the rank, and $\\lambda$ "
            "are the weights."
            "Handwritten code based on the standard CP-ALS algorithm from Kolda and"
            " Bader (2009). "
        )

    @property
    def tags(self):
        return ["tensor-factorization", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Grace Wang", "gwang426@gatech.edu"),
            Contributor("Kseniia Suleimanova", "kseniiasuleimanova@gmail.com"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title="Tensor Decompositions and Applications",
                authors=[Author("T. G. Kolda"), Author("B. W. Bader")],
                journal="SIAM Review",
                volume=51,
                number=3,
                pages="455-500",
                year=2009,
                doi="10.1137/07070111X",
            ),
            Ref(
                title="CS 18.335 Final Project: CP Decomposition",
                authors=[Author("Willow Ahrens"), Author("Alvin Shi")],
                url="https://github.com/willow-ahrens/18.335FinalProject/blob/submit/TFGDCANDECOMP.py",
                year=2025,
            ),
            Ref(
                title="Tensorly",
                authors=[
                    Author("Tensorly Contributors"),
                ],
                url="https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_cp.py",
                year=2025,
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI was used to debug some parts of the code. This statement"
            " was written by hand."
        )

    @property
    def motivation(self):
        return (
            "The Alternating Least Squares (ALS) algorithm for CANDECOMP/PARAFAC (CP)"
            " plays a critical role in tensor decomposition, which has applications in"
            " various fields such as signal processing, pscyhometrics, neuroscience,"
            " and graph analysis. Within the ALS algorithm, the Matricized-Tensor Times"
            " Khatri-Rao Product (MTTKRP) operation is a computationally intensive step"
            " that often dominates the overall runtime. Efficiently implementing MTTKRP"
            " is crucial for the performance of the ALS algorithm. The input tensor is"
            " sparse, and the ALS algorithm takes advantage of this sparsity through"
            " its MTTKRP kernel, which process only the non-zero elements of the"
            " tensor. For sparse tensors with nnz << I * J * K, the complexity reduces"
            " from O(I * J * K * R) to O(nnz * R), where nnz is the number of non-zero"
            " elements in the tensor and R is the decomposition rank. This makes it"
            " practical to work with large-scale applications."
        )

    @property
    def generators(self):
        return [
            CP4FactorizeableGenerator(),
        ]

    """
    benchmark(X_bench, rank, max_iter)

    Args:
    ----
    xp: The array API module to use
    X_bench: The input 4th-order sparse tensor in binsparse format
    rank: Number of components (rank) for the decomposition
    max_iter: Maximum number of ALS iterations

    Returns:
    -------
    Tuple of (A_bench, B_bench, C_bench, lambda_bench) in binsparse format where:
    - A, B, and C are the normalized factor matrices
    - lambda are the component weights
    """

    def benchmark(self, data, meta):
        (X,) = data
        rank = meta["rank"]
        max_iter = meta["max_iter"]
        dim1, dim2, dim3, dim4 = X.shape
        dtype = X.dtype

        A = xp.from_binsparse(
            BinsparseFormat.from_numpy(
                np.random.default_rng(0).random((dim1, rank)).astype(dtype)
            )
        )
        B = xp.from_binsparse(
            BinsparseFormat.from_numpy(
                np.random.default_rng(0).random((dim2, rank)).astype(dtype)
            )
        )
        C = xp.from_binsparse(
            BinsparseFormat.from_numpy(
                np.random.default_rng(0).random((dim3, rank)).astype(dtype)
            )
        )
        D = xp.from_binsparse(
            BinsparseFormat.from_numpy(
                np.random.default_rng(0).random((dim4, rank)).astype(dtype)
            )
        )

        for _iteration in range(max_iter):
            (A, B, C, D) = (A, B, C, D)
            # Update A
            mttkrp_result = xp.einsum(
                "mttkrp_result[i, r] += X[i, j, k, l] * B[j, r] * C[k, r] * D[l, r]",
                X=X,
                B=B,
                C=C,
                D=D,
            )
            DtD = xp.einsum("DtD[r, s] += D[l, r] * D[l, s]", D=D)
            CtC = xp.einsum("CtC[r, s] += C[k, r] * C[k, s]", C=C)
            BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B=B)

            G = xp.multiply(xp.multiply(DtD, CtC), BtB)
            # G = G + xp.eye(rank, dtype=dtype) * epsilon2
            G_pinv = xp.linalg.pinv(G)
            A = xp.matmul(mttkrp_result, G_pinv)

            # Update B
            mttkrp_result = xp.einsum(
                "mttkrp_result[j, r] += X[i, j, k, l] * A[i, r] * C[k, r] * D[l, r]",
                X=X,
                A=A,
                C=C,
                D=D,
            )
            AtA = xp.einsum("AtA[r, s] += A[i, r] * A[i, s]", A=A)
            G = xp.multiply(xp.multiply(DtD, CtC), AtA)
            # G = G + xp.eye(rank, dtype=dtype) * epsilon2
            G_pinv = xp.linalg.pinv(G)
            B = xp.matmul(mttkrp_result, G_pinv)

            # Update C
            mttkrp_result = xp.einsum(
                "mttkrp_result[k, r] += X[i, j, k, l] * A[i, r] * B[j, r] * D[l, r]",
                X=X,
                A=A,
                B=B,
                D=D,
            )
            BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B=B)
            G = xp.multiply(xp.multiply(DtD, BtB), AtA)
            # G = G + xp.eye(rank, dtype=dtype) * epsilon2
            G_pinv = xp.linalg.pinv(G)
            C = xp.matmul(mttkrp_result, G_pinv)

            # Update D
            mttkrp_result = xp.einsum(
                "mttkrp_result[l, r] += X[i, j, k, l] * A[i, r] * B[j, r] * C[k, r]",
                X=X,
                A=A,
                B=B,
                C=C,
            )
            CtC = xp.einsum("CtC[r, s] += C[k, r] * C[k, s]", C=C)
            G = xp.multiply(xp.multiply(CtC, BtB), AtA)
            # G = G + xp.eye(rank, dtype=dtype) * epsilon2
            G_pinv = xp.linalg.pinv(G)
            D = xp.matmul(mttkrp_result, G_pinv)

        (A, B, C, D) = (A, B, C, D)

        # Normalizing factors
        A_norms_sq = xp.einsum("norms[r] += A[i, r] * A[i, r]", A=A)
        B_norms_sq = xp.einsum("norms[r] += B[j, r] * B[j, r]", B=B)
        C_norms_sq = xp.einsum("norms[r] += C[k, r] *C[k, r]", C=C)
        D_norms_sq = xp.einsum("norms[r] += D[l, r] *D[l, r]", D=D)

        A_norms = xp.sqrt(A_norms_sq)
        B_norms = xp.sqrt(B_norms_sq)
        C_norms = xp.sqrt(C_norms_sq)
        D_norms = xp.sqrt(D_norms_sq)

        # Computing lambda
        lambda_vals = xp.multiply(
            xp.multiply(xp.multiply(A_norms, B_norms), C_norms), D_norms
        )

        A_norms_2d = xp.expand_dims(A_norms, 0)
        B_norms_2d = xp.expand_dims(B_norms, 0)
        C_norms_2d = xp.expand_dims(C_norms, 0)
        D_norms_2d = xp.expand_dims(D_norms, 0)

        # Case: avoiding division by zero
        eps = 1e-10
        A_norms_safe = xp.maximum(A_norms_2d, eps)
        B_norms_safe = xp.maximum(B_norms_2d, eps)
        C_norms_safe = xp.maximum(C_norms_2d, eps)
        D_norms_safe = xp.maximum(D_norms_2d, eps)

        A = xp.divide(A, A_norms_safe)
        B = xp.divide(B, B_norms_safe)
        C = xp.divide(C, C_norms_safe)
        D = xp.divide(D, D_norms_safe)

        # Now compute everything at once

        (A, B, C, D, lambda_vals) = (A, B, C, D, lambda_vals)

        # Convert to binsparse format
        A_bench_out = xp.to_binsparse(A)
        B_bench_out = xp.to_binsparse(B)
        C_bench_out = xp.to_binsparse(C)
        D_bench_out = xp.to_binsparse(D)
        lambda_bench_out = xp.to_binsparse(lambda_vals)

        return (A_bench_out, B_bench_out, C_bench_out, D_bench_out, lambda_bench_out)
