import logging

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


class JLApproxNNDataset(Dataset):
    def __init__(
        self,
        name,
        pretty_name,
        description,
        suites,
        n_samples,
        n_features,
        n_queries,
        k,
        eps,
        seed,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
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
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class JLApproxNNGenerator(Generator[JLApproxNNDataset]):
    @property
    def name(self) -> str:
        return "jl_projection_inputs"

    @property
    def pretty_name(self) -> str:
        return "JL Projection Input Generator"

    @property
    def description(self) -> str:
        return (
            "Generates uniformly random data/query matrices and sparse random"
            " projection matrices for JL approximate nearest-neighbor."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu"),
            Contributor("Willow Ahrens", "ahrens@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Randomized Numerical Linear Algebra : "
                    "A Perspective on the Field With an Eye to Software"
                ),
                authors=[
                    Author("Riley Murray"),
                    Author("James Demmel"),
                    Author("Michael W. Mahoney"),
                    Author("N. Benjamin Erichson"),
                    Author("Maksim Melnichenko"),
                    Author("Osman Asif Malik"),
                    Author("Laura Grigori"),
                    Author("Piotr Luszczek"),
                    Author("Michał Dereziński"),
                    Author("Miles E. Lopes"),
                    Author("Tianyu Liang"),
                    Author("Hengrui Luo"),
                    Author("Jack Dongarra"),
                ],
                year=2023,
                url="https://arxiv.org/abs/2302.11474",
            ),
            Ref(
                title="Random projection implementation reference",
                authors=[Author("scikit-learn contributors")],
                url="https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function "
            "itself. Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "Sparse Johnson-Lindenstrauss projection is a fundamental primitive "
            "in randomized numerical linear algebra, and is used in many "
            "applications such as approximate nearest neighbor search."
        )

    @property
    def datasets(self) -> list[JLApproxNNDataset]:
        return [
            JLApproxNNDataset(
                name="small",
                pretty_name="Small JL ANN",
                description=(
                    "Small random dense data and query matrices with sparse random"
                    " projection."
                ),
                suites=[],
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
                description=(
                    "Medium random dense data and query matrices with sparse random"
                    " projection."
                ),
                suites=[],
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
                description=(
                    "Large random dense data and query matrices with sparse random"
                    " projection."
                ),
                suites=[],
                n_samples=4096,
                n_features=512,
                n_queries=128,
                k=5,
                eps=0.1,
                seed=42,
            ),
        ]

    def generate(self, dataset: JLApproxNNDataset):
        import scipy as sp

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
        projection_matrix = (U_Neg + U_Pos).tocoo()

        meta = {"k": dataset.k, "eps": dataset.eps}
        P = BinsparseFormat.from_coo(
            (
                projection_matrix.row,
                projection_matrix.col,
            ),
            projection_matrix.data,
            projection_matrix.shape,
        )

        return [
            BinsparseFormat.from_numpy(data),
            BinsparseFormat.from_numpy(query),
            P,
        ], meta


class JLApproxNearestNeighbor(Benchmark):
    @property
    def name(self):
        return "jl_approx_nn"

    @property
    def pretty_name(self):
        return "Johnson-Lindenstrauss Approximate Nearest Neighbor"

    @property
    def description(self):
        return (
            "Benchmarks Johnson-Lindenstrauss projection followed by"
            " k-nearest-neighbor ranking in projected space."
        )

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return """
        <ccs2012>
          <concept>
            <concept_desc>Computing methodologies~Machine learning algorithms</concept_desc>
          </concept>
          <concept>
            <concept_desc>Mathematics of computing~Dimensionality reduction</concept_desc>
          </concept>
          <concept>
            <concept_desc>Theory of computation~Nearest neighbor algorithms</concept_desc>
          </concept>
        </ccs2012>
        """

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
                title=(
                    "Randomized Numerical Linear Algebra : "
                    "A Perspective on the Field With an Eye to Software"
                ),
                authors=[
                    Author("Riley Murray"),
                    Author("James Demmel"),
                    Author("Michael W. Mahoney"),
                    Author("N. Benjamin Erichson"),
                    Author("Maksim Melnichenko"),
                    Author("Osman Asif Malik"),
                    Author("Laura Grigori"),
                    Author("Piotr Luszczek"),
                    Author("Michał Dereziński"),
                    Author("Miles E. Lopes"),
                    Author("Tianyu Liang"),
                    Author("Hengrui Luo"),
                    Author("Jack Dongarra"),
                ],
                journal="Arxiv",
                volume="arXiv:2302.11474",
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
        data, query, P = data
        k = meta["k"]
        eps = meta["eps"]

        n_samples, n_features = data.shape
        logging.info(
            f"Data shape: {data.shape}, Query shape: {query.shape}, "
            f"Projection shape: {P.shape}"
        )
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

        return [nearest_indices, nearest_distances]
