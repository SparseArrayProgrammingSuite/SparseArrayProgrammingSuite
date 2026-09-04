import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.frostt import fetch_frostt_tensor


class CP4FactorizeableDataset(Dataset):
    def __init__(self, name, pretty_name, suites, shape, rank, max_iter=100):
        self._name = name
        self._pretty_name = pretty_name
        self._suites = suites
        self.shape = shape
        self.rank = rank
        self.max_iter = max_iter

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
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


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
    def suites(self):
        return []

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
                name="cp_factorizeable_tiny",
                pretty_name="Tiny Factorizeable CP Tensor",
                suites=["test", "trace"],
                shape=(5, 5, 5, 5),
                rank=1,
                max_iter=20,
            ),
            CP4FactorizeableDataset(
                name="cp_factorizeable_small",
                pretty_name="Small Factorizeable CP Tensor",
                suites=[],
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
        dtype = X.dtype
        initial_A = from_numpy(
            np.random.default_rng(0).random((dim1, rank)).astype(dtype)
        )
        initial_B = from_numpy(
            np.random.default_rng(0).random((dim2, rank)).astype(dtype)
        )
        initial_C = from_numpy(
            np.random.default_rng(0).random((dim3, rank)).astype(dtype)
        )
        initial_D = from_numpy(
            np.random.default_rng(0).random((dim4, rank)).astype(dtype)
        )

        X = from_numpy(X)
        max_iter = dataset.max_iter

        return DataInstance(
            inputs=[X, initial_A, initial_B, initial_C, initial_D],
            meta={"rank": rank, "max_iter": max_iter},
            ref_meta={"check_reconstruction": True, "rel_error_tol": 0.1},
        )


class CP4FrosttDataset(Dataset):
    def __init__(self, name, pretty_name, tensor_name, rank, max_iter=5, suites=None):
        self._name = name
        self._pretty_name = pretty_name
        self.tensor_name = tensor_name
        self.rank = rank
        self.max_iter = max_iter
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return f"FROSTT tensor {self.tensor_name}, rank = {self.rank}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class CP4FrosttGenerator(Generator[CP4FrosttDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self):
        return "cp4_frostt_inputs"

    @property
    def pretty_name(self):
        return "FROSTT Sparse Tensor Generator for CP4-ALS"

    @property
    def description(self):
        return (
            "Real 4th-order sparse tensors downloaded from FROSTT (frostt.io),"
            " factorized directly. No dense reconstruction check is performed since"
            " these tensors are stored in genuinely sparse (COO) form."
        )

    @property
    def suites(self):
        return ["standard"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return []

    @property
    def references(self):
        return [
            Ref(
                title=(
                    "FROSTT: The Formidable Repository of Open Sparse Tensors and Tools"
                ),
                authors=[
                    Author("Shaden Smith"),
                    Author("Jee W. Choi"),
                    Author("Jiajia Li"),
                    Author("Richard Vuduc"),
                    Author("Jongsoo Park"),
                    Author("Xing Liu"),
                    Author("George Karypis"),
                ],
                url="http://frostt.io/",
                year=2017,
            )
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the CP-ALS algorithm itself, which"
            " predates this generator. This generator and its FROSTT data-fetching"
            " were written by a generative AI assistant (Claude) at the user's"
            " direction."
        )

    @property
    def motivation(self):
        return (
            "Real sparse tensors from FROSTT exercise CP-ALS's MTTKRP kernel against"
            " genuinely irregular sparsity patterns, unlike the synthetic"
            " low-rank-by-construction tensors generated elsewhere in this file."
        )

    @property
    def datasets(self):
        return [
            CP4FrosttDataset(
                name=f"cp4_frostt_{tensor_name}",
                pretty_name=f"CP4 FROSTT {tensor_name}",
                tensor_name=tensor_name,
                rank=rank,
                max_iter=max_iter,
                suites=suites,
            )
            for tensor_name, rank, max_iter, suites in [
                ("toy", 2, 5, []),
                ("nips", 10, 5, []),
                ("uber_pickups", 10, 5, []),
                ("chicago_crime_comm", 10, 5, []),
                ("enron", 10, 5, []),
                ("flickr_4d", 10, 5, []),
                ("delicious_4d", 10, 5, []),
            ]
        ]

    def generate(self, dataset: CP4FrosttDataset):
        raw = fetch_frostt_tensor(dataset.tensor_name)
        X = raw.inputs[0]
        rank = dataset.rank
        dim1, dim2, dim3, dim4 = raw.meta["shape"]
        dtype = to_numpy(X).dtype

        rng = np.random.default_rng(0)
        initial_A = from_numpy(rng.random((dim1, rank)).astype(dtype))
        initial_B = from_numpy(rng.random((dim2, rank)).astype(dtype))
        initial_C = from_numpy(rng.random((dim3, rank)).astype(dtype))
        initial_D = from_numpy(rng.random((dim4, rank)).astype(dtype))

        return DataInstance(
            inputs=[X, initial_A, initial_B, initial_C, initial_D],
            meta={"rank": rank, "max_iter": dataset.max_iter},
        )


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
    def suites(self):
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
            CP4FrosttGenerator(),
        ]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta or not self._ref_meta.get("check_reconstruction"):
            return

        X = to_numpy(self._input[0])
        A = to_numpy(self._output[0])
        B = to_numpy(self._output[1])
        C = to_numpy(self._output[2])
        D = to_numpy(self._output[3])
        lambda_vals = to_numpy(self._output[4])
        dim1, dim2, dim3, dim4 = X.shape
        rank = self._meta["rank"]

        assert A.shape == (dim1, rank)
        assert B.shape == (dim2, rank)
        assert C.shape == (dim3, rank)
        assert D.shape == (dim4, rank)
        assert lambda_vals.shape == (rank,)

        Y = np.einsum("r,ir,jr,kr,lr->ijkl", lambda_vals, A, B, C, D)
        rel_error = np.linalg.norm(Y - X) / np.linalg.norm(X)
        assert rel_error < self._ref_meta["rel_error_tol"], (
            f"CP4 reconstruction error too high: {rel_error:.6f}"
        )

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

    def benchmark(self, xp, data, meta):
        X, A, B, C, D = data
        max_iter = meta["max_iter"]

        for _iteration in range(max_iter):
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

        return [A, B, C, D, lambda_vals]
