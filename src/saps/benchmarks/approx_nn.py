import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.netflixprize import fetch_netflixprize_matrix
from saps.benchmarks.openml import (
    OpenMLDatasetGenerator,
    fetch_openml_train_test_features,
)


class SimHashApproxNNRandomDataset(Dataset):
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
        max_tables,
        max_projections,
        candidate_target,
        target_probability,
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
        self.max_tables = max_tables
        self.max_projections = max_projections
        self.candidate_target = candidate_target
        self.target_probability = target_probability

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


# A single random hyperplane bit collides for two UNRELATED points with
# probability exactly 0.5 (Charikar 2002: P[sign match] = 1 - theta/pi, at
# theta = pi/2), regardless of dimension or dataset -- no calibration
# needed. That is what bounds candidate-set size, so tuning targets it
# directly instead of a near-neighbor similarity guess: a guess weak enough
# to need no data (e.g. 1/sqrt(n_features)) is too close to 0.5 itself to
# ever buy both a small candidate set and a real recall guarantee at once,
# since the two nearly-identical per-bit probabilities need exponentially
# more tables to tell apart as n_projections grows.
_RANDOM_COLLISION_PROBABILITY = 0.5

# Reference near-neighbor cosine similarity, used only to report an
# estimated retrieval probability alongside the chosen (n_projections,
# n_tables) -- not to choose them. 1/sqrt(n_features) is the typical cosine
# similarity between two independent random vectors in that many dimensions
# (its standard deviation around 0), i.e. one standard deviation of
# "signal" above pure randomness: a deliberately conservative estimate,
# since real near neighbors are usually far more similar than that.
_REFERENCE_SIMILARITY_SCALE = 1.0


def _collision_probability(cosine_similarity: float) -> float:
    """SimHash collision probability for two vectors at the given cosine
    similarity: P[sign(r @ u) == sign(r @ v)] = 1 - theta / pi, where theta
    is the angle between them (Charikar 2002, Theorem 1)."""
    cosine_similarity = min(1.0, max(-1.0, cosine_similarity))
    return 1.0 - math.acos(cosine_similarity) / math.pi


def _tune_lsh(dataset, n_features: int, n_samples: int):
    """Pick n_projections so that, spread over the full max_tables budget,
    the expected number of accidental (unrelated-point) matches lands near
    candidate_target -- selectivity is fundamentally about how many points
    there are to accidentally collide with, not just n_features, so
    n_samples has to enter the picture somewhere. Always spends the whole
    table budget: more tables only ever help recall for a fixed
    candidate-set size (see the module docstring for why recall isn't the
    optimization target here). estimated_retrieval_probability is reported
    for reference against a conservative near-neighbor similarity guess,
    not solved for; real recall on structured data is typically much
    better than that guess suggests."""
    n_tables = dataset.max_tables
    if dataset.candidate_target > 0:
        expected_false_matches = max(n_samples * n_tables, 1) / dataset.candidate_target
        n_projections = math.ceil(
            math.log(max(expected_false_matches, 1.0), 1.0 / _RANDOM_COLLISION_PROBABILITY)
        )
    else:
        n_projections = dataset.max_projections
    n_projections = min(dataset.max_projections, max(1, n_projections))

    reference_similarity = min(
        1.0, _REFERENCE_SIMILARITY_SCALE / math.sqrt(max(n_features, 1))
    )
    p = _collision_probability(reference_similarity)
    probability = 1 - (1 - p**n_projections) ** n_tables
    return n_projections, n_tables, probability


class SimHashApproxNNGeneratorMixin(ABC):
    projection_kind: str
    projection_description: str
    tuning_description = (
        "Generation picks n_projections so that, using the full "
        "max_tables budget, the expected number of accidental matches "
        "between unrelated points lands near candidate_target (see "
        "_tune_lsh) -- a random hyperplane bit collides for two unrelated "
        "points with probability exactly 0.5 regardless of the data, so "
        "this needs no pass over it, only its shape. It always spends the "
        "whole table budget, since more tables can only help recall for a "
        "fixed candidate-set size; estimated_retrieval_probability is "
        "reported against a conservative near-neighbor similarity guess "
        "for reference, not solved for. At benchmark time, each round "
        "requires exact agreement on a fixed-length prefix of the "
        "n_projections signs, shortening that prefix by one sign a round "
        "until candidate_target points are found."
    )

    @property
    def projection_references(self) -> list[Ref]:
        return [
            Ref(
                title="Similarity Estimation Techniques from Rounding Algorithms",
                authors=[Author("Moses S. Charikar")],
                year=2002,
                doi="10.1145/509907.509965",
                url="https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf",
            ),
        ]

    @abstractmethod
    def projection(self, n_features: int, target_dim: int, seed: int):
        """Return the scalar projections defining each hash hyperplane."""

    def _instance(self, dataset, data, query, source_meta=None):
        from scipy.sparse import issparse

        if dataset.max_tables < 1 or dataset.max_projections < 1:
            raise ValueError("max_tables and max_projections must be positive")
        if not 1 <= dataset.k <= data.shape[0] or query.shape[0] == 0:
            raise ValueError("Require nonempty queries and 1 <= k <= data rows")
        if not 0 < dataset.target_probability <= 1:
            raise ValueError("target_probability must be in (0, 1]")

        n_features = data.shape[1]
        # How many hyperplanes and tables to use follows from the data's
        # shape alone (n_features, n_samples), not its values; it needs no
        # data pass (see _tune_lsh).
        n_projections, n_tables, probability = _tune_lsh(
            dataset, n_features, data.shape[0]
        )

        projection = self.projection(
            n_features, n_tables * n_projections, dataset.seed
        )
        try:
            projection = to_numpy(projection)
        except TypeError:
            projection = to_scipy(projection).tocsr()

        def as_binsparse(array):
            return from_scipy(array) if issparse(array) else from_numpy(array)

        return DataInstance(
            inputs=[as_binsparse(x) for x in (data, query, projection)],
            meta={
                **_lsh_meta(dataset),
                "n_projections": n_projections,
                "n_tables": n_tables,
                "estimated_retrieval_probability": probability,
                "projection_kind": self.projection_kind,
                **(source_meta or {}),
            },
        )


class _DenseProjectionMixin(SimHashApproxNNGeneratorMixin):
    projection_kind = "dense"
    projection_description = (
        "Projection coefficients are independent N(0, 1) variables, so each "
        "hash hyperplane's normal is a uniformly random direction. For any "
        "fixed pair of vectors u, v with angle theta between them, "
        "P[sign(r @ u) == sign(r @ v)] = 1 - theta / pi over the random "
        "direction r (Charikar 2002, Theorem 1), independent of u and v's "
        "norms -- unlike E2LSH's Euclidean bucketing, no width calibration "
        "against the data's scale is needed."
    )

    def projection(self, n_features: int, target_dim: int, seed: int):
        # Keep Gaussian hash directions independent of the synthetic data stream.
        rng = np.random.default_rng([seed, 1])
        return from_numpy(rng.standard_normal((n_features, target_dim)))


class _SparseProjectionMixin(SimHashApproxNNGeneratorMixin):
    projection_kind = "sparse"
    projection_description = (
        "Projection coefficients are r_i = z_i * g_i, with independent "
        "z_i ~ Bernoulli(a), g_i ~ N(0, 1), and a = 1/sqrt(d) for d features "
        "-- a standard database-friendly substitute for dense Gaussian "
        "directions (Achlioptas 2003). Charikar's "
        "P[sign(r @ u) == sign(r @ v)] = 1 - theta / pi is proven for "
        "spherically symmetric r such as Gaussian; for this sparse, "
        "non-spherical r it is only an empirical approximation, not a "
        "proven guarantee."
    )

    @property
    def projection_references(self) -> list[Ref]:
        return [
            *super().projection_references,
            Ref(
                title=(
                    "Database-Friendly Random Projections: "
                    "Johnson-Lindenstrauss with Binary Coins"
                ),
                authors=[Author("Dimitris Achlioptas")],
                journal="Journal of Computer and System Sciences",
                year=2003,
                doi="10.1016/S0022-0000(03)00025-4",
            ),
        ]

    def projection(self, n_features: int, target_dim: int, seed: int):
        return _rla_projection(n_features, target_dim, seed)


class _SimHashApproxNNRandomGeneratorMixin(SimHashApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"simhash_projection_inputs_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"SimHash Projection Input Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Generates Gaussian random data/query matrices with "
            f"{self.projection_kind} projections for approximate nearest-neighbor. "
            f"{self.projection_description} {self.tuning_description}"
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
            *self.projection_references,
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
            "The benchmark algorithm was supplied by its human authors, who "
            "directed switching it from E2LSH bucketing to SimHash. "
            "Generative AI implemented the SimHash redesign and assisted "
            "with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "Random hyperplane hashing (SimHash) is a fundamental primitive "
            "in locality-sensitive hashing and randomized numerical linear "
            "algebra, used in many applications such as approximate nearest "
            "neighbor search and near-duplicate detection."
        )

    @property
    def datasets(self) -> list[SimHashApproxNNRandomDataset]:
        return [
            SimHashApproxNNRandomDataset(
                name="small",
                pretty_name="Small SimHash ANN",
                description=(
                    "Small random dense data and query matrices with random projection."
                ),
                suites=[],
                n_samples=256,
                n_features=128,
                n_queries=32,
                k=5,
                eps=0.1,
                seed=40,
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            ),
            SimHashApproxNNRandomDataset(
                name="medium",
                pretty_name="Medium SimHash ANN",
                description=(
                    "Medium random dense data and query matrices with random"
                    " projection."
                ),
                suites=[],
                n_samples=1024,
                n_features=256,
                n_queries=64,
                k=5,
                eps=0.1,
                seed=41,
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            ),
            SimHashApproxNNRandomDataset(
                name="large",
                pretty_name="Large SimHash ANN",
                description=(
                    "Large random dense data and query matrices with random projection."
                ),
                suites=[],
                n_samples=4096,
                n_features=512,
                n_queries=128,
                k=5,
                eps=0.1,
                seed=42,
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            ),
        ]

    def generate(self, dataset: SimHashApproxNNRandomDataset):
        rng = np.random.default_rng(dataset.seed)
        data = rng.standard_normal((dataset.n_samples, dataset.n_features))
        query = rng.standard_normal((dataset.n_queries, dataset.n_features))
        return self._instance(dataset, data, query)


class _SimHashApproxNNTestGeneratorMixin(_SimHashApproxNNRandomGeneratorMixin):
    @property
    def name(self) -> str:
        return f"simhash_projection_test_inputs_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"SimHash Projection Test Input Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Small SimHash approximate nearest-neighbor example. "
            f"{self.projection_description} {self.tuning_description}"
        )

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def motivation(self) -> str:
        return "Provide a small SimHash ANN example for benchmark correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[SimHashApproxNNRandomDataset]:
        return [
            SimHashApproxNNRandomDataset(
                name="test_simhash_preserves_similarity",
                pretty_name="test SimHash ANN",
                description=(
                    "Test dense data and query matrices with random projection."
                ),
                suites=["test", "trace"],
                n_samples=20,
                n_features=10,
                n_queries=4,
                k=3,
                eps=0.01,
                seed=42,
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            )
        ]

    def generate(self, dataset: SimHashApproxNNRandomDataset):
        problem = super().generate(dataset)
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_meta={"check": "simhash_preserves_cosine_similarity"},
        )


class SimHashApproxNNDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        k: int,
        eps: float,
        seed: int,
        suites: list[str],
        max_tables: int,
        max_projections: int,
        candidate_target: int,
        target_probability: float,
    ):
        self._source_name = source_name
        self.k = k
        self.eps = eps
        self.seed = seed
        self._suites = suites
        self.max_tables = max_tables
        self.max_projections = max_projections
        self.candidate_target = candidate_target
        self.target_probability = target_probability

    @property
    def name(self) -> str:
        return self._source_name

    @property
    def pretty_name(self) -> str:
        return f"SimHash ANN {self._source_name}"

    @property
    def description(self) -> str:
        return f"SimHash approximate nearest-neighbor on {self._source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


def _lsh_meta(dataset):
    return {
        "k": dataset.k,
        "eps": dataset.eps,
        "max_tables": dataset.max_tables,
        "max_projections": dataset.max_projections,
        "target_probability": dataset.target_probability,
        "candidate_target": dataset.candidate_target,
    }


def _rla_projection(n_features: int, target_dim: int, seed: int):
    import scipy as sp

    rng = np.random.default_rng(seed)
    size = n_features * target_dim
    # Binomial count plus a uniform support gives independent Bernoulli entries.
    nnz = rng.binomial(size, 1.0 / np.sqrt(n_features))
    projection = sp.sparse.random(
        n_features,
        target_dim,
        density=nnz / size,
        data_rvs=rng.standard_normal,
        random_state=rng,
    )
    return from_scipy(projection)


class _SimHashApproxNNOpenMLGeneratorMixin(SimHashApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"simhash_approx_nn_openml_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"SimHash ANN OpenML Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Loads OpenML image datasets for SimHash approximate nearest-neighbor. "
            f"{self.projection_description} {self.tuning_description}"
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            *self.projection_references,
            Ref(
                title="Gradient-Based Learning Applied to Document Recognition",
                authors=[
                    Author("Yann LeCun"),
                    Author("Léon Bottou"),
                    Author("Yoshua Bengio"),
                    Author("Patrick Haffner"),
                ],
                journal="Proceedings of the IEEE",
                year=1998,
                url="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf",
            ),
            Ref(
                title="Learning Multiple Layers of Features from Tiny Images",
                authors=[Author("Alex Krizhevsky")],
                year=2009,
                url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "The benchmark algorithm was supplied by its human authors, who "
            "directed switching it from E2LSH bucketing to SimHash. "
            "Generative AI implemented the SimHash redesign and assisted "
            "with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "MNIST and CIFAR-10 provide dense image feature matrices from OpenML "
            "for approximate nearest-neighbor search."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[SimHashApproxNNDataset]:
        return [
            SimHashApproxNNDataset(
                dataset.name,
                k=5,
                eps=0.3,
                seed=50 if dataset.name == "mnist" else 0,
                suites=["standard"],
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            )
            for dataset in OpenMLDatasetGenerator().datasets
        ]

    def generate(self, dataset: SimHashApproxNNDataset) -> DataInstance:
        train, test, source_meta = fetch_openml_train_test_features(dataset.name)

        n_features = train.shape[1]
        return self._instance(
            dataset,
            train,
            test,
            source_meta={
                "split": "openml_task",
                "openml_task_id": source_meta["task_id"],
                "openml_task_repeat": source_meta["repeat"],
                "openml_task_fold": source_meta["fold"],
                "openml_task_sample": source_meta["sample"],
                "num_train": int(train.shape[0]),
                "num_query": int(test.shape[0]),
                "num_features": int(n_features),
                "openml_data_id": source_meta["data_id"],
                "openml_name": source_meta["openml_name"],
                "openml_version": source_meta["version"],
                "source_num_rows": source_meta["num_rows"],
                "source_num_features": source_meta["num_features"],
            },
        )


class _SimHashApproxNNNetflixGeneratorMixin(SimHashApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"simhash_approx_nn_netflix_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"SimHash ANN Netflix Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Loads Netflix Prize ratings for SimHash approximate nearest-neighbor. "
            f"{self.projection_description} {self.tuning_description}"
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            *self.projection_references,
            Ref(
                title="Use of KNN for the Netflix Prize",
                authors=[Author("Vini Hong"), Author("Anastasios Tsamis")],
                institution="Stanford CS229",
                url="https://cs229.stanford.edu/proj2008/HongTsamis-UseOfKNNForTheNetflixPrize.pdf",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "The benchmark algorithm was supplied by its human authors, who "
            "directed switching it from E2LSH bucketing to SimHash. "
            "Generative AI implemented the SimHash redesign and assisted "
            "with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "The Netflix Prize dataset provides a ~480K users × 17,770 movies sparse "
            f"ratings matrix, tested with {self.projection_kind} projections."
        )

    @property
    def datasets(self) -> list[SimHashApproxNNDataset]:
        return [
            SimHashApproxNNDataset(
                "netflix",
                k=5,
                eps=0.3,
                seed=0,
                suites=["standard"],
                max_tables=64,
                max_projections=16,
                candidate_target=100,
                target_probability=0.9,
            )
        ]

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: SimHashApproxNNDataset) -> DataInstance:
        data, source_meta = fetch_netflixprize_matrix()

        return self._instance(
            dataset,
            data,
            data,
            source_meta={
                "num_train": int(data.shape[0]),
                "num_query": int(data.shape[0]),
                "num_features": int(data.shape[1]),
                "source_num_users": source_meta["num_users"],
                "source_num_movies": source_meta["num_movies"],
                "source_num_ratings": source_meta["num_ratings"],
            },
        )


class SimHashApproxNNDenseTestGenerator(
    _DenseProjectionMixin,
    _SimHashApproxNNTestGeneratorMixin,
    Generator[SimHashApproxNNRandomDataset],
):
    pass


class SimHashApproxNNSparseTestGenerator(
    _SparseProjectionMixin,
    _SimHashApproxNNTestGeneratorMixin,
    Generator[SimHashApproxNNRandomDataset],
):
    pass


class SimHashApproxNNDenseGenerator(
    _DenseProjectionMixin,
    _SimHashApproxNNRandomGeneratorMixin,
    Generator[SimHashApproxNNRandomDataset],
):
    pass


class SimHashApproxNNSparseGenerator(
    _SparseProjectionMixin,
    _SimHashApproxNNRandomGeneratorMixin,
    Generator[SimHashApproxNNRandomDataset],
):
    pass


class SimHashApproxNNDenseOpenMLGenerator(
    _DenseProjectionMixin,
    _SimHashApproxNNOpenMLGeneratorMixin,
    Generator[SimHashApproxNNDataset],
):
    pass


class SimHashApproxNNSparseOpenMLGenerator(
    _SparseProjectionMixin,
    _SimHashApproxNNOpenMLGeneratorMixin,
    Generator[SimHashApproxNNDataset],
):
    pass


class SimHashApproxNNDenseNetflixGenerator(
    _DenseProjectionMixin,
    _SimHashApproxNNNetflixGeneratorMixin,
    Generator[SimHashApproxNNDataset],
):
    pass


class SimHashApproxNNSparseNetflixGenerator(
    _SparseProjectionMixin,
    _SimHashApproxNNNetflixGeneratorMixin,
    Generator[SimHashApproxNNDataset],
):
    pass


class SimHashApproxNearestNeighbor(Benchmark):
    @property
    def name(self):
        return "simhash_approx_nn"

    @property
    def pretty_name(self):
        return "SimHash Approximate Nearest Neighbor"

    @property
    def description(self):
        return (
            "Searches progressively larger Hamming-radius buckets of SimHash"
            " signatures across LSH tables, then ranks candidates by cosine"
            " distance in the original space."
        )

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return (
            """
<ccs2012>
<concept>
<concept_id>10002951.10003317.10003347.10003356</concept_id>
<concept_desc>Information systems~Clustering and classification</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002951.10003317.10003347.10003350</concept_id>
<concept_desc>Information systems~Recommender systems</concept_desc>
<concept_significance>300</concept_significance>
</concept>
<concept>
<concept_id>10002951.10003317</concept_id>
<concept_desc>Information systems~Information retrieval</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_desc>Computing methodologies~
Machine learning algorithms</concept_desc>
</concept>
<concept>
<concept_id>10010147.10010257.10010258.10010260.10003697</concept_id>
<concept_desc>Computing methodologies~Cluster analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010147.10010257.10010258.10010260.10010271</concept_id>
<concept_desc>Computing methodologies~"""
            "Dimensionality reduction and manifold learning"
            """</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_desc>Mathematics of computing~
Dimensionality reduction</concept_desc>
</concept>
<concept>
<concept_desc>Theory of computation~
Nearest neighbor algorithms</concept_desc>
</concept>
</ccs2012>
"""
        )

    @property
    def authors(self):
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self):
        return [
            Ref(
                title="Similarity Estimation Techniques from Rounding Algorithms",
                authors=[Author("Moses S. Charikar")],
                year=2002,
                doi="10.1145/509907.509965",
                url="https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf",
            ),
            Ref(
                title="LSH Forest: Self-Tuning Indexes for Similarity Search",
                authors=[
                    Author("Mayank Bawa"),
                    Author("Tyson Condie"),
                    Author("Prasanna Ganesan"),
                ],
                year=2005,
                url="https://www.cs.princeton.edu/courses/archive/spring06/cos592/bib/LSHForest-bawa05.pdf",
            ),
            Ref(
                title=(
                    "PUFFINN: Parameterless and Universally Fast "
                    "FInding of Nearest Neighbors"
                ),
                authors=[
                    Author("Martin Aumüller"),
                    Author("Tobias Christiani"),
                    Author("Rasmus Pagh"),
                    Author("Michael Vesterli"),
                ],
                year=2019,
                url="https://arxiv.org/abs/1906.12211",
            ),
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
            "The benchmark algorithm was supplied by its human authors, who "
            "directed switching it from E2LSH bucketing to SimHash -- "
            "encoding hyperplane signs as +/-1 so LSH agreement is a matmul "
            "contraction rather than a broadcast equality compare. "
            "Generative AI implemented the SimHash redesign and assisted "
            "with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self):
        return (
            "The purpose of this is to create python tests that are for RLA methods. "
            "Specifically, I will first show the application of random hyperplane "
            "hashing (SimHash) for NN. My goal is to write benchmarks on "
            "applications of RNLA, for graph algorithms, PDEs, and Scientific "
            "Machine Learning"
        )

    @property
    def generators(self):
        return [
            SimHashApproxNNDenseTestGenerator(),
            SimHashApproxNNSparseTestGenerator(),
            SimHashApproxNNDenseGenerator(),
            SimHashApproxNNSparseGenerator(),
            SimHashApproxNNDenseOpenMLGenerator(),
            SimHashApproxNNSparseOpenMLGenerator(),
            SimHashApproxNNDenseNetflixGenerator(),
            SimHashApproxNNSparseNetflixGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        data, query, P = data
        k = meta["k"]
        n_projections = meta["n_projections"]
        n_tables = meta["n_tables"]
        n_samples, n_features = data.shape
        n_queries = query.shape[0]
        if not 1 <= k <= n_samples:
            raise ValueError("k must be between 1 and the number of data points")
        if n_tables < 1 or n_projections < 1:
            raise ValueError("n_tables and n_projections must be positive")
        if query.shape[1] != n_features or P.shape != (
            n_features,
            n_tables * n_projections,
        ):
            raise ValueError(
                "Expected query[:, features] and "
                "projection[features, n_tables * n_projections]"
            )
        candidate_target = min(n_samples, max(k, meta["candidate_target"]))
        logging.info(
            f"Data shape: {data.shape}, Query shape: {query.shape}, "
            f"Projection shape: {P.shape}, Tables: {n_tables}, "
            f"Projections per table: {n_projections}"
        )

        projected_data = xp.reshape(
            xp.matmul(data, P), (n_samples, n_tables, n_projections)
        )
        projected_query = xp.reshape(
            xp.matmul(query, P), (n_queries, n_tables, n_projections)
        )

        # SimHash bits as +/-1 (not 0/1): a sum of matching-vs-differing
        # signs is a matmul contraction, not a broadcast equality compare.
        # Computed via xp.matmul below rather than the generic einsum DSL:
        # the DSL's reduction broadcasts every operand out to the full
        # (q, n, n_tables, n_projections) shape before reducing, which is
        # intractable at realistic sizes, and numpy-style sum reductions
        # silently upcast int8 to int64 even when they don't. matmul does
        # neither -- every backend lowers it to an actual contraction and
        # preserves the input dtype.
        # Sign never fixes at 0 (unlike the projection matmul above, which
        # is linear), so a sparse backend can't represent it with a zero
        # fill value -- and its own matmul requires one. table_data/query
        # are only (n or q) x n_tables x n_projections though, tiny next to
        # the (q, n) results below, so densifying them here is free.
        table_data = xp.from_binsparse(
            xp.to_binsparse(xp.astype(xp.where(projected_data >= 0, 1, -1), xp.int8))
        )
        table_query = xp.from_binsparse(
            xp.to_binsparse(xp.astype(xp.where(projected_query >= 0, 1, -1), xp.int8))
        )

        candidates = xp.zeros((n_queries, n_samples), dtype=xp.bool)
        # Each round requires EXACT agreement on a fixed-length prefix of
        # the n_projections signs -- not a Hamming-radius count -- and
        # shortens that prefix by one sign a round, starting from the full
        # n_projections (the finest, smallest-candidate-set partition).
        # A sum of m terms, each +/-1, can only equal m if every one of
        # them agrees, so checking the prefix sum against its own maximum
        # is exactly an exact-match test. required_projections == 0
        # accepts everyone, so this always terminates.
        required_projections = n_projections
        while True:
            active = xp.sum(candidates, axis=1) < candidate_target
            if not xp.any(active):
                break
            if required_projections <= 0:
                candidates = candidates | active[:, None]
            else:
                m = required_projections
                # Batched matmul over the table axis, on just the first m
                # signs -- (t,q,m) @ (t,m,n) -> (t,q,n), moved to (q,n,t).
                # One tensor contraction; n_tables is never iterated over
                # in Python, just carried as a batch axis the backend's
                # own matmul handles.
                prefix_agreement = xp.permute_dims(
                    xp.matmul(
                        xp.permute_dims(table_query[:, :, :m], (1, 0, 2)),
                        xp.permute_dims(table_data[:, :, :m], (1, 2, 0)),
                    ),
                    (1, 2, 0),
                )
                # OR across tables via a reduction over the trailing axis,
                # not a Python loop over n_tables.
                matches = xp.any(prefix_agreement == m, axis=-1)
                candidates = candidates | (matches & active[:, None])
            required_projections -= 1

        # Rank candidates by cosine distance, what SimHash actually targets.
        # A full (q, n) matmul over every pair, same reasoning as agreement
        # above: no feature axis survives into the output, so there's no
        # broadcast to avoid by masking before the contraction.
        dot = xp.matmul(query, xp.permute_dims(data, (1, 0)))
        query_norm = xp.sqrt(xp.sum(query**2, axis=-1))
        data_norm = xp.sqrt(xp.sum(data**2, axis=-1))
        # Guard zero-norm rows so they compare as maximally dissimilar
        # instead of nan.
        denom = xp.maximum(query_norm[:, None] * data_norm[None, :], 1e-10)
        distances = 1 - dot / denom
        # Masked zeros are not zero-distance neighbors.
        distances = xp.where(candidates, distances, xp.inf)

        sorted_indices = xp.argsort(distances, axis=1)
        nearest_indices = xp.take(sorted_indices, xp.arange(k), axis=1)
        nearest_distances = xp.take_along_axis(distances, nearest_indices, axis=1)
        return [nearest_indices, nearest_distances]

    def check(self, param):
        for item in self._output:
            assert isinstance(
                item, BinsparseTensor
            ), "Output must be in binsparse format"
        if not self._ref_meta:
            return

        from scipy.spatial.distance import cdist

        data = to_numpy(self._input[0])
        query = to_numpy(self._input[1])
        nearest_ind = to_numpy(self._output[0])

        orig_distances = cdist(query, data, metric="cosine")

        true_nearest = np.min(orig_distances, axis=1)
        approx_nearest = orig_distances[
            np.arange(param.dataset.n_queries), nearest_ind[:, 0].astype(int)
        ]
        assert np.all(approx_nearest <= (1 + param.dataset.eps) * true_nearest)
