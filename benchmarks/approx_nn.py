import numpy as np
import saps
from saps.benchmark import Dataset, Generator, Benchmark, Contributor, Author, Ref

xp = saps.xp

"""
Name: Random Numerical Linear Algenra
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to create python tests that are for RLA methods.
Specifically, I will first show the application of the JL Lemma for NN.
My goal is to write benchmarks on applications of RNLA,
for graph algorithms, PDEs, and Scientific Machine Learning

https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615

Murray, R., Demmel, J., Mahoney, M. W., Erichson, N. B.,
Melnichenko, M., Malik, O. A., ... & Dongarra, J. (2023).
Randomized numerical linear algebra: A perspective on the field with an eye to software.
arXiv preprint arXiv:2302.11474.
Role of sparsity (How sparsity is used in the problem):
The inputs to the matrix multiply are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""

class JLApproxNNDataset(Dataset):
    def __init__(self, name, pretty_name, description, tags, n_samples, n_features, n_queries, k, eps, seed):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_queries = n_queries
        self.k = k
        self.eps = eps
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
    def tags(self) -> list[str]:
        return self._tags


class JLApproxNNGenerator(Generator):
    @property
    def name(self) -> str:
        return "jl_projection_inputs"

    @property
    def pretty_name(self) -> str:
        return "JL Projection Input Generator"

    @property
    def description(self) -> str:
        return "Generates random data/query matrices and sparse random projection matrices for JL approximate nearest-neighbor."

    @property
    def tags(self) -> list[str]:
        return ["rnla", "projection", "approximate-nearest-neighbor", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Random projection implementation reference",
                authors=[Author("scikit-learn contributors")],
                url="https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "No generative AI was used to construct the benchmark function itself. Generative AI might have been used to construct tests."

    @property
    def motivation(self) -> str:
        return "The purpose of this is to create python tests that are for RLA methods. Specifically, I will first show the application of the JL Lemma for NN."

    @property
    def datasets(self) -> list[Dataset]:
        return [
            JLApproxNNDataset(
                name="small",
                pretty_name="Small JL ANN",
                description="Small random dense data and query matrices with sparse random projection.",
                tags=["small", "rnla", "sparse"],
                n_samples=256,
                n_features=128,
                n_queries=32,
                k=5,
                eps=0.1,
                seed=40,
            ),
            JLApproxNNDataset(
                name="medium",
                pretty_name="Medium JL ANN",
                description="Medium random dense data and query matrices with sparse random projection.",
                tags=["medium", "rnla", "sparse"],
                n_samples=1024,
                n_features=256,
                n_queries=64,
                k=5,
                eps=0.1,
                seed=41,
            ),
            JLApproxNNDataset(
                name="large",
                pretty_name="Large JL ANN",
                description="Large random dense data and query matrices with sparse random projection.",
                tags=["large", "rnla", "sparse"],
                n_samples=4096,
                n_features=512,
                n_queries=128,
                k=5,
                eps=0.1,
                seed=42,
            ),
        ]

    def generate(self, dataset: JLApproxNNDataset):
        rng = np.random.default_rng(dataset.seed)
        data = rng.standard_normal((dataset.n_samples, dataset.n_features))
        query = rng.standard_normal((dataset.n_queries, dataset.n_features))
        eps = dataset.eps
        seed = dataset.seed
        n_samples, n_features = data.shape
        #  Johnson Lindenstrauss Theorem Lemmna.
        # The eps represents the disortion of distance by epsilon,
        # between the the original space and the reduced subspace
        target_dim = np.ceil(np.log(n_samples) / (eps * eps)).astype(int)

        rng = np.random.default_rng(seed)
        # return rng.standard_normal((n_features, np.round(target_dim).astype(int)))

        s = np.sqrt(n_features)  # s = 1/density
        density = 1.0 / s  # probability of a nonzero entry = density.
        density_half = density / 2.0  # probability for + or -
        scale = np.sqrt(s / target_dim)  # scale = sqrt(s / n_components)

        U_Neg = sp.sparse.random(
            n_features,
            target_dim,
            density_half,
            data_rvs=lambda k: np.full(
                k, -scale, dtype=float
            ),  # specified dtype to see of that made a difference
            random_state=rng,
        )
        U_Pos = sp.sparse.random(
            n_features,
            target_dim,
            density_half,
            data_rvs=lambda k: np.full(
                k, scale, dtype=float
            ),  # specified dtype to see of that made a difference
            random_state=rng,
        )
        projection_matrix = (U_Neg + U_Pos).toarray()

        meta = {"k": dataset.k, "eps": dataset.eps}
        return meta, [data, query, projection_matrix]


class JLApproxNearestNeighbor(Benchmark):
    @property
    def tag(self):
        return "jl_approx_nn"

    @property
    def name(self):
        return "Random Numerical Linear Algenra"

    @property
    def pretty_name(self):
        return "Johnson-Lindenstrauss Approximate Nearest Neighbor"

    @property
    def description(self):
        return (
            "Benchmarks Johnson-Lindenstrauss projection followed by k-nearest-neighbor "
            "ranking in projected space."
        )

    @property
    def tags(self):
        return ["rnla", "approximate-nearest-neighbor", "projection", "sparse"]

    @property
    def authors(self):
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self):
        return [
            Ref(
                title="Random projection implementation reference",
                authors=[Author("scikit-learn contributors")],
                url="https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615",
            ),
            Ref(
                title="Randomized numerical linear algebra: A perspective on the field with an eye to software",
                authors=[
                    Author("Murray, R."),
                    Author("Demmel, J."),
                    Author("Mahoney, M. W."),
                    Author("Erichson, N. B."),
                    Author("Melnichenko, M."),
                    Author("Malik, O. A."),
                    Author("Dongarra, J."),
                ],
                year=2023,
                url="https://arxiv.org/abs/2302.11474",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self):
        return (
            "The purpose of this is to create python tests that are for RLA methods. "
            "Specifically, I will first show the application of the JL Lemma for NN. "
            "My goal is to write benchmarks on applications of RNLA, for graph "
            "algorithms, PDEs, and Scientific Machine Learning"
        )

    @property
    def generators(self):
        return [JLApproxNNGenerator()]

    def benchmark(self, data, meta):
        data_bench, query_bench, projection_matrix = data
        k = meta["k"]
        eps = meta["eps"]
        data = data_bench
        query = query_bench
        P = projection_matrix

        n_samples, n_features = data.shape
        #  Johnson Lindenstrauss Theorem Lemmna.
        # The eps represents the disortion of distance by epsilon,
        # between the the original space and the reduced subspace
        target_dim = np.log(n_samples) / (eps * eps)
        if target_dim > n_features:
            target_dim = n_features

        # Project to lower subspace
        projected_data = xp.matmul(data, P)
        projected_query = xp.matmul(query, P)

        # -----K Nearest Neighbour from here on out--------

        # Euclidean distances
        diff = xp.einsum(
            "X[i, j, k] = Q[i, k] - D[j, k]", Q=projected_query, D=projected_data
        )
        distances = xp.sqrt(xp.sum(diff**2, axis=-1))

        # Get nearest k neighbors.
        sorted_indices = xp.argsort(distances)

        # Get nearest indices and associated distances.
        nearest_indices = xp.take(sorted_indices, xp.arange(k), axis=1)
        nearest_distances = xp.take(xp.sort(distances), xp.arange(k), axis=1)

        nearest_indices = xp.to_binsparse(nearest_indices)
        nearest_distances = xp.to_binsparse(nearest_distances)
        return [nearest_indices, nearest_distances]
