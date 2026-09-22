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


class JLApproxNNRandomDataset(Dataset):
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


# E2LSH manual, Sec. 3.3.2 (PDF page 12): recommended bucket width for a
# Gaussian projection at reference radius R = 1. Every projection kind scales
# its own reference width (see `projection`) so that this same ratio applies
# after normalizing by that kind's projected standard deviation.
_REFERENCE_WIDTH_RATIO = 4.0

# Bits used to mix each table's projection bins into one scalar code
# (benchmark() compares codes directly, so this only affects the odds of an
# accidental collision between unrelated bins, not correctness or memory).
_HASH_BITS = 31


def _collision_probability(width_ratio: float) -> float:
    """E2LSH probability that a single Gaussian-projection hash of bucket
    width w collides for two points at distance r, as a function of
    width_ratio = w / r (Datar et al. 2004)."""
    if width_ratio <= 0:
        return 0.0
    if not np.isfinite(width_ratio):
        return 1.0
    from scipy.stats import norm

    return float(
        1
        - 2 * norm.cdf(-width_ratio)
        - (2 / (width_ratio * math.sqrt(2 * math.pi)))
        * (1 - math.exp(-(width_ratio**2) / 2))
    )


def _tables_needed(hit_probability: float, target_probability: float) -> int:
    """Tables t such that 1 - (1 - hit_probability) ** t >= target_probability."""
    if hit_probability >= 1:
        return 1
    return max(
        1,
        math.ceil(math.log1p(-target_probability) / math.log1p(-hit_probability)),
    )


def _tune_lsh(dataset):
    """Pick the cheapest (n_projections, n_tables) whose analytic retrieval
    probability reaches the target, widening the reference ratio if even the
    largest allowed counts fall short. Depends only on dataset config, not
    on any data, so the actual bucket width is just `widen` away from being
    known: bucket_width = reference_width(kind) * radius * widen."""
    widen = 1.0
    while True:
        p = _collision_probability(_REFERENCE_WIDTH_RATIO * widen)
        options = []
        for n_projections in range(1, dataset.max_projections + 1):
            n_tables = _tables_needed(p**n_projections, dataset.target_probability)
            if n_tables <= dataset.max_tables:
                options.append((n_projections * n_tables, n_projections, n_tables))
        if options:
            _, n_projections, n_tables = min(options)
            probability = 1 - (1 - p**n_projections) ** n_tables
            return n_projections, n_tables, widen, probability
        widen *= 2


class JLApproxNNGeneratorMixin(ABC):
    projection_kind: str
    projection_description: str
    tuning_description = (
        "Using the E2LSH collision-probability formula, generation first picks "
        "the fewest total table x projection hashes whose analytic retrieval "
        "probability reaches the target, independent of the data, widening "
        "the reference ratio if even the largest allowed table and projection "
        "counts fall short. It then samples up to 8 query rows and max(256, "
        "k) data rows with a fixed seed and measures the median distance "
        "from each sampled query to its k-th nearest sampled data point, "
        "purely to scale the resulting bucket width to the data. This is a "
        "calibration estimate from a small sample, not full-dataset recall."
    )

    @property
    def projection_references(self) -> list[Ref]:
        return [
            Ref(
                title="E2LSH 0.1 User Manual",
                authors=[Author("Alexandr Andoni"), Author("Piotr Indyk")],
                year=2005,
                url="https://www.mit.edu/~andoni/LSH/manual.pdf#page=12",
            ),
        ]

    @abstractmethod
    def projection(self, n_features: int, target_dim: int, seed: int):
        """Return the scalar projections and their unit-radius bucket width."""

    def _instance(self, dataset, data, query, source_meta=None):
        from scipy.sparse import issparse
        from scipy.spatial.distance import cdist

        if dataset.max_tables < 1 or dataset.max_projections < 1:
            raise ValueError("max_tables and max_projections must be positive")
        if not 1 <= dataset.k <= data.shape[0] or query.shape[0] == 0:
            raise ValueError("Require nonempty queries and 1 <= k <= data rows")
        if not 0 < dataset.target_probability <= 1:
            raise ValueError("target_probability must be in (0, 1]")

        # How many hashes to use, and how far to widen the reference ratio,
        # follows from the dataset config alone; it needs no data.
        n_projections, n_tables, widen, probability = _tune_lsh(dataset)

        projection, reference_width = self.projection(
            data.shape[1], n_tables * n_projections, dataset.seed
        )
        try:
            projection = to_numpy(projection)
        except TypeError:
            projection = to_scipy(projection).tocsr()
        rng = np.random.default_rng([dataset.seed, 2])
        offsets = rng.uniform(np.finfo(float).eps, 1.0, (n_tables, n_projections))
        strides = rng.integers(1, 2**_HASH_BITS, size=n_projections)

        # Estimate a typical k-th-nearest-neighbor distance from a small
        # sample, purely to scale the bucket width to the data; it has no
        # other use.
        rng = np.random.default_rng([dataset.seed, 3])
        data = data.tocsr() if issparse(data) else data
        query = query.tocsr() if issparse(query) else query
        sample_data = data[
            rng.choice(data.shape[0], min(data.shape[0], max(256, dataset.k)), False)
        ]
        sample_query = query[rng.choice(query.shape[0], min(query.shape[0], 8), False)]
        sample_data = sample_data.toarray() if issparse(sample_data) else sample_data
        sample_query = (
            sample_query.toarray() if issparse(sample_query) else sample_query
        )
        radii = np.sort(cdist(sample_query, sample_data), axis=1)[:, dataset.k - 1]
        positive = radii[radii > 0]
        radius = float(np.median(positive)) if positive.size else 1.0

        bucket_width_scale = radius * widen
        bucket_width = reference_width * bucket_width_scale

        def as_binsparse(array):
            return from_scipy(array) if issparse(array) else from_numpy(array)

        return DataInstance(
            inputs=[
                as_binsparse(x) for x in (data, query, projection, offsets, strides)
            ],
            meta={
                **_lsh_meta(dataset),
                "bucket_width": bucket_width,
                "bucket_width_scale": bucket_width_scale,
                "n_projections": n_projections,
                "n_tables": n_tables,
                "estimated_retrieval_probability": probability,
                "calibration_radius": radius,
                "calibration_queries": sample_query.shape[0],
                "calibration_samples": sample_data.shape[0],
                "projection_kind": self.projection_kind,
                **(source_meta or {}),
            },
        )


class _DenseProjectionMixin(JLApproxNNGeneratorMixin):
    projection_kind = "dense"
    projection_description = (
        "Projection coefficients are independent N(0, 1) variables. For a fixed "
        "difference vector v, Var(r @ v) = ||v||_2^2 over random projections r, "
        "so the reference bucket width of 4 (E2LSH manual Sec. 3.3.2, final "
        "paragraph before Sec. 3.4, PDF page 12, printed page 11) applies "
        "directly at reference radius R = 1."
    )

    def projection(self, n_features: int, target_dim: int, seed: int):
        # Keep Gaussian hash directions independent of the synthetic data stream.
        rng = np.random.default_rng([seed, 1])
        return from_numpy(rng.standard_normal((n_features, target_dim))), 4.0


class _SparseProjectionMixin(JLApproxNNGeneratorMixin):
    projection_kind = "sparse"
    projection_description = (
        "Projection coefficients are r_i = z_i * g_i, with independent "
        "z_i ~ Bernoulli(a), g_i ~ N(0, 1), and a = 1/sqrt(d) for d features "
        "(Hyvonen et al., Algorithm 1, lines 8-12; Sec. II-B). Since E[r_i] = 0 "
        "and Var(r_i) = E[z_i^2] * E[g_i^2] = a, independence gives "
        "Var(r @ v) = a * sum_i(v_i^2) = a * ||v||_2^2 for a fixed difference "
        "vector v. Dense E2LSH projections have variance ||v||_2^2, so the "
        "ratio of standard deviations is sqrt(a) = d**(-1/4). We scale its "
        "width by this ratio: w = 4 * d**(-1/4), using reference radius "
        "R = 1 and the E2LSH manual's Sec. 3.3.2 recommendation (PDF page 12, "
        "printed page 11). This matches Var((r @ v) / w) between sparse and "
        "dense projections. It is our derived scale adjustment, not a width "
        "prescribed by the MRPT paper, which uses median splits. Matching "
        "variance alone does not guarantee the same projection distribution or "
        "collision probabilities."
    )

    @property
    def projection_references(self) -> list[Ref]:
        return [
            *super().projection_references,
            Ref(
                title=(
                    "Fast Nearest Neighbor Search through "
                    "Sparse Random Projections and Voting"
                ),
                authors=[
                    Author("Ville Hyvönen"),
                    Author("Teemu Pitkänen"),
                    Author("Sotiris Tasoulis"),
                    Author("Elias Jääsaari"),
                    Author("Risto Tuomainen"),
                    Author("Liang Wang"),
                    Author("Jukka Corander"),
                    Author("Teemu Roos"),
                ],
                year=2016,
                doi="10.1109/BigData.2016.7840682",
                url="https://helda.helsinki.fi/server/api/core/bitstreams/7c057e23-8530-4848-afb1-919fba55c706/content",
            ),
        ]

    def projection(self, n_features: int, target_dim: int, seed: int):
        return _rla_projection(n_features, target_dim, seed), 4.0 * n_features**-0.25


class _JLApproxNNRandomGeneratorMixin(JLApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"jl_projection_inputs_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"JL Projection Input Generator ({self.projection_kind})"

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
            "The benchmark algorithm was supplied by its human authors. Generative AI "
            "assisted with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "Sparse Johnson-Lindenstrauss projection is a fundamental primitive "
            "in randomized numerical linear algebra, and is used in many "
            "applications such as approximate nearest neighbor search."
        )

    @property
    def datasets(self) -> list[JLApproxNNRandomDataset]:
        return [
            JLApproxNNRandomDataset(
                name="small",
                pretty_name="Small JL ANN",
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
            JLApproxNNRandomDataset(
                name="medium",
                pretty_name="Medium JL ANN",
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
            JLApproxNNRandomDataset(
                name="large",
                pretty_name="Large JL ANN",
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

    def generate(self, dataset: JLApproxNNRandomDataset):
        rng = np.random.default_rng(dataset.seed)
        data = rng.standard_normal((dataset.n_samples, dataset.n_features))
        query = rng.standard_normal((dataset.n_queries, dataset.n_features))
        return self._instance(dataset, data, query)


class _JLApproxNNTestGeneratorMixin(_JLApproxNNRandomGeneratorMixin):
    @property
    def name(self) -> str:
        return f"jl_projection_test_inputs_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"JL Projection Test Input Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Small JL approximate nearest-neighbor example. "
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
        return "Provide a small JL ANN example for benchmark correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[JLApproxNNRandomDataset]:
        return [
            JLApproxNNRandomDataset(
                name="test_jl_preserves_distance",
                pretty_name="test JL ANN",
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

    def generate(self, dataset: JLApproxNNRandomDataset):
        problem = super().generate(dataset)
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_meta={"check": "jl_preserves_distance"},
        )


class JLApproxNNDataset(Dataset):
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
        return f"JL ANN {self._source_name}"

    @property
    def description(self) -> str:
        return f"JL approximate nearest-neighbor on {self._source_name}."

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


class _JLApproxNNOpenMLGeneratorMixin(JLApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"jl_approx_nn_openml_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"JL ANN OpenML Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Loads OpenML image datasets for JL approximate nearest-neighbor. "
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
            "The benchmark algorithm was supplied by its human authors. Generative AI "
            "assisted with debugging array dimensions, generator refactoring, "
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
    def datasets(self) -> list[JLApproxNNDataset]:
        return [
            JLApproxNNDataset(
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

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
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


class _JLApproxNNNetflixGeneratorMixin(JLApproxNNGeneratorMixin):
    @property
    def name(self) -> str:
        return f"jl_approx_nn_netflix_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"JL ANN Netflix Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return (
            "Loads Netflix Prize ratings for JL approximate nearest-neighbor. "
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
            "The benchmark algorithm was supplied by its human authors. Generative AI "
            "assisted with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "The Netflix Prize dataset provides a ~480K users × 17,770 movies sparse "
            f"ratings matrix, tested with {self.projection_kind} projections."
        )

    @property
    def datasets(self) -> list[JLApproxNNDataset]:
        return [
            JLApproxNNDataset(
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

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
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


class JLApproxNNDenseTestGenerator(
    _DenseProjectionMixin,
    _JLApproxNNTestGeneratorMixin,
    Generator[JLApproxNNRandomDataset],
):
    pass


class JLApproxNNSparseTestGenerator(
    _SparseProjectionMixin,
    _JLApproxNNTestGeneratorMixin,
    Generator[JLApproxNNRandomDataset],
):
    pass


class JLApproxNNDenseGenerator(
    _DenseProjectionMixin,
    _JLApproxNNRandomGeneratorMixin,
    Generator[JLApproxNNRandomDataset],
):
    pass


class JLApproxNNSparseGenerator(
    _SparseProjectionMixin,
    _JLApproxNNRandomGeneratorMixin,
    Generator[JLApproxNNRandomDataset],
):
    pass


class JLApproxNNDenseOpenMLGenerator(
    _DenseProjectionMixin, _JLApproxNNOpenMLGeneratorMixin, Generator[JLApproxNNDataset]
):
    pass


class JLApproxNNSparseOpenMLGenerator(
    _SparseProjectionMixin,
    _JLApproxNNOpenMLGeneratorMixin,
    Generator[JLApproxNNDataset],
):
    pass


class JLApproxNNDenseNetflixGenerator(
    _DenseProjectionMixin,
    _JLApproxNNNetflixGeneratorMixin,
    Generator[JLApproxNNDataset],
):
    pass


class JLApproxNNSparseNetflixGenerator(
    _SparseProjectionMixin,
    _JLApproxNNNetflixGeneratorMixin,
    Generator[JLApproxNNDataset],
):
    pass


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
            "Searches progressively wider Euclidean buckets across LSH tables,"
            " then ranks candidates by Euclidean distance in the original space."
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
            "The benchmark algorithm was supplied by its human authors. Generative AI "
            "assisted with debugging array dimensions, generator refactoring, "
            "parameter calibration, and tests."
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
        return [
            JLApproxNNDenseTestGenerator(),
            JLApproxNNSparseTestGenerator(),
            JLApproxNNDenseGenerator(),
            JLApproxNNSparseGenerator(),
            JLApproxNNDenseOpenMLGenerator(),
            JLApproxNNSparseOpenMLGenerator(),
            JLApproxNNDenseNetflixGenerator(),
            JLApproxNNSparseNetflixGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        data, query, P, offsets, strides = data
        k = meta["k"]
        width = meta["bucket_width"]
        hash_bits = _HASH_BITS
        n_projections = meta["n_projections"]
        n_tables = meta["n_tables"]
        n_samples, n_features = data.shape
        n_queries = query.shape[0]
        if not 1 <= k <= n_samples:
            raise ValueError("k must be between 1 and the number of data points")
        if not 1 <= hash_bits <= 31 or n_tables < 1 or n_projections < 1:
            raise ValueError(
                "Use 1 to 31 hash bits and positive table/projection counts"
            )
        if not np.isfinite(width) or width <= 0:
            raise ValueError("bucket_width must be finite and positive")
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
            f"Projections per table: {n_projections}, Bits: {hash_bits}"
        )

        projected_data = xp.reshape(
            xp.matmul(data, P), (n_samples, n_tables, n_projections)
        )
        projected_query = xp.reshape(
            xp.matmul(query, P), (n_queries, n_tables, n_projections)
        )

        candidates = xp.zeros((n_queries, n_samples), dtype=xp.bool)
        n_codes = 2**hash_bits
        modulus = xp.asarray(n_codes, dtype=xp.int64)
        # Widen bins from the projection's width, freezing queries at the target.
        # Positive fractional offsets eventually put all finite projections in 0.
        while True:
            active = xp.sum(candidates, axis=1) < candidate_target
            if not xp.any(active):
                break
            table_data = xp.astype(
                xp.floor(projected_data / width + offsets) % n_codes, xp.int64
            )
            table_query = xp.astype(
                xp.floor(projected_query / width + offsets) % n_codes, xp.int64
            )
            # Mix the signed bins into one uint32 code. Reduce each product
            # before summing to avoid overflowing int64 at 31 bits.
            # Codes live in [1, n_codes] (not [0, n_codes)) so that 0 is free
            # to mean "no query": zeroing an already-satisfied query's row
            # below then can't spuriously collide with any real data code.
            table_data = xp.astype(
                xp.einsum(
                    "H[n,t] += (B[n,t,h] * S[h]) % M[]",
                    B=table_data,
                    S=strides,
                    M=modulus,
                )
                % n_codes
                + 1,
                xp.uint32,
            )
            table_query = xp.astype(
                xp.einsum(
                    "H[q,t] += (B[q,t,h] * S[h]) % M[]",
                    B=table_query,
                    S=strides,
                    M=modulus,
                )
                % n_codes
                + 1,
                xp.uint32,
            )
            # Zero out already-satisfied queries so they carry no explicit
            # entries into the comparison below, instead of comparing
            # everyone and masking after.
            table_query = table_query * xp.astype(active, table_query.dtype)[:, None]
            # A query and a sample collide iff their mixed codes match in any
            # table; no need to build a one-hot code axis for it.
            matches = xp.einsum(
                "M[q,n] or= (Q[q,t] == D[n,t])", Q=table_query, D=table_data
            )
            candidates = candidates | matches
            width *= 2

        # Materialize the query/sample candidate mask before introducing the
        # feature axis. Mask each operand before subtracting so sparse backends
        # can retain zeros for non-candidates throughout the distance calculation.
        diff = xp.einsum(
            "X[q,n,f] = C[q,n] * Q[q,f] - C[q,n] * D[n,f]",
            C=candidates,
            Q=query,
            D=data,
        )
        distances = xp.sqrt(xp.sum(diff**2, axis=-1))
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

        data = to_numpy(self._input[0])
        query = to_numpy(self._input[1])
        nearest_ind = to_numpy(self._output[0])

        diff = np.expand_dims(query, axis=1) - np.expand_dims(data, axis=0)
        orig_distances = np.sqrt(np.sum(diff**2, axis=-1))

        true_nearest = np.min(orig_distances, axis=1)
        approx_nearest = orig_distances[
            np.arange(param.dataset.n_queries), nearest_ind[:, 0].astype(int)
        ]
        assert np.all(approx_nearest <= (1 + param.dataset.eps) * true_nearest)
