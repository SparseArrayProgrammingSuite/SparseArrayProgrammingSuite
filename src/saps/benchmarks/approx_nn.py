import logging
from abc import ABC, abstractmethod

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy

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
        hash_bits=31,
        n_tables=100,
        candidate_target=100,
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
        self.hash_bits = hash_bits
        self.n_tables = n_tables
        self.candidate_target = candidate_target

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


class JLApproxNNGeneratorMixin(ABC):
    projection_kind: str

    @abstractmethod
    def projection(self, n_features: int, target_dim: int, seed: int):
        """Generate the scalar projections combined in each LSH table."""

    def _instance(self, dataset, data, query, source_meta=None):
        projection = self.projection(
            data.shape[1], dataset.hash_bits * dataset.n_tables, dataset.seed
        )
        rng = np.random.default_rng([dataset.seed, 2])
        offsets = rng.uniform(
            np.finfo(float).eps, 1.0, size=(dataset.n_tables, dataset.hash_bits)
        )
        strides = rng.integers(1, 2**dataset.hash_bits, size=dataset.hash_bits)
        return DataInstance(
            inputs=[data, query, projection, from_numpy(offsets), from_numpy(strides)],
            meta={
                **_lsh_meta(dataset),
                "projection_kind": self.projection_kind,
                **(source_meta or {}),
            },
        )


class _DenseProjectionMixin(JLApproxNNGeneratorMixin):
    projection_kind = "dense"

    def projection(self, n_features: int, target_dim: int, seed: int):
        # Keep Gaussian hash directions independent of the synthetic data stream.
        rng = np.random.default_rng([seed, 1])
        return from_numpy(rng.standard_normal((n_features, target_dim)))


class _SparseProjectionMixin(JLApproxNNGeneratorMixin):
    projection_kind = "sparse"

    def projection(self, n_features: int, target_dim: int, seed: int):
        return _rla_projection(n_features, target_dim, seed)


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
            f"{self.projection_kind} projections for approximate nearest-neighbor."
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
            "The benchmark algorithm was supplied by its human authors. Generative AI "
            "assisted with debugging array dimensions, generator refactoring, "
            "and tests."
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
            ),
        ]

    def generate(self, dataset: JLApproxNNRandomDataset):
        rng = np.random.default_rng(dataset.seed)
        data = rng.standard_normal((dataset.n_samples, dataset.n_features))
        query = rng.standard_normal((dataset.n_queries, dataset.n_features))
        return self._instance(dataset, from_numpy(data), from_numpy(query))


class _JLApproxNNTestGeneratorMixin(_JLApproxNNRandomGeneratorMixin):
    @property
    def name(self) -> str:
        return f"jl_projection_test_inputs_{self.projection_kind}"

    @property
    def pretty_name(self) -> str:
        return f"JL Projection Test Input Generator ({self.projection_kind})"

    @property
    def description(self) -> str:
        return "Small JL approximate nearest-neighbor example."

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
                # Keep the raw hash axis small enough for dense-framework tests.
                hash_bits=8,
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
        seed: int = 0,
        suites: list[str] | None = None,
        hash_bits: int = 31,
        n_tables: int = 100,
        candidate_target: int = 100,
    ):
        self._source_name = source_name
        self.k = k
        self.eps = eps
        self.seed = seed
        self._suites = suites or []
        self.hash_bits = hash_bits
        self.n_tables = n_tables
        self.candidate_target = candidate_target

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
        "hash_bits": dataset.hash_bits,
        "n_tables": dataset.n_tables,
        "candidate_target": dataset.candidate_target,
    }


def _rla_projection(n_features: int, target_dim: int, seed: int):
    import scipy as sp

    # Hyvonen et al. (2016), Section II-B: N(0, 1) with probability 1/sqrt(d),
    # otherwise zero. https://doi.org/10.1109/BigData.2016.7840682
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
        return "Loads OpenML image datasets for JL approximate nearest-neighbor."

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
            "and tests."
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
            )
            for dataset in OpenMLDatasetGenerator().datasets
        ]

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
        train, test, source_meta = fetch_openml_train_test_features(dataset.name)

        n_features = train.shape[1]
        return self._instance(
            dataset,
            from_numpy(train),
            from_numpy(test),
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
        return "Loads Netflix Prize ratings for JL approximate nearest-neighbor."

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
            "and tests."
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
            )
        ]

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
        data, source_meta = fetch_netflixprize_matrix()

        train_coo = data.tocoo()
        test_coo = data.tocoo()

        return self._instance(
            dataset,
            from_scipy(train_coo),
            from_scipy(test_coo),
            source_meta={
                "num_train": int(train_coo.shape[0]),
                "num_query": int(test_coo.shape[0]),
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
            "and tests."
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
        width = meta["eps"]
        hash_bits = meta.get("hash_bits", 31)
        n_tables = meta.get("n_tables", 100)
        n_samples, n_features = data.shape
        n_queries = query.shape[0]
        if not 1 <= k <= n_samples:
            raise ValueError("k must be between 1 and the number of data points")
        if not 1 <= hash_bits <= 31 or n_tables < 1:
            raise ValueError("Use 1 to 31 hash bits and at least one table")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("eps must be a finite, positive initial bucket width")
        if query.shape[1] != n_features or P.shape != (
            n_features,
            n_tables * hash_bits,
        ):
            raise ValueError(
                "Expected query[:, features] and "
                "projection[features, n_tables * hash_bits]"
            )
        candidate_target = min(n_samples, max(k, meta.get("candidate_target", 100)))
        logging.info(
            f"Data shape: {data.shape}, Query shape: {query.shape}, "
            f"Projection shape: {P.shape}, Tables: {n_tables}, Bits: {hash_bits}"
        )

        projected_data = xp.reshape(
            xp.matmul(data, P), (n_samples, n_tables, hash_bits)
        )
        projected_query = xp.reshape(
            xp.matmul(query, P), (n_queries, n_tables, hash_bits)
        )

        candidates = xp.zeros((n_queries, n_samples), dtype=xp.bool)
        sample_indices = xp.arange(n_samples, dtype=xp.uint64)
        query_indices = xp.arange(n_queries, dtype=xp.uint64)
        n_codes = 2**hash_bits
        modulus = xp.asarray(n_codes, dtype=xp.int64)
        # Widen bins from eps, freezing each query when it reaches the target.
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
            # Mix the signed bins into one uint32 scatter index. Reduce each
            # product before summing to avoid overflowing int64 at 31 bits.
            table_data = xp.astype(
                xp.einsum(
                    "H[n,t] += (B[n,t,h] * S[h]) % M[]",
                    B=table_data,
                    S=strides,
                    M=modulus,
                )
                % n_codes,
                xp.uint32,
            )
            table_query = xp.astype(
                xp.einsum(
                    "H[q,t] += (B[q,t,h] * S[h]) % M[]",
                    B=table_query,
                    S=strides,
                    M=modulus,
                )
                % n_codes,
                xp.uint32,
            )
            for table in range(n_tables):
                # if frameworks were better, we could write:
                # matches = xp.einsum(
                #    "M[q,n] or= A[q] & (Q[q,t] == D[n,t])",
                #    A=active,
                #    Q=table_query,
                #    D=table_data,
                # )
                key_data = xp.zeros((n_samples, n_codes), dtype=xp.bool)
                key_query = xp.zeros((n_queries, n_codes), dtype=xp.bool)
                key_data[sample_indices, table_data[:, table]] = True
                key_query[query_indices, table_query[:, table]] = active
                matches = xp.einsum(
                    "M[q,n] or= Q[q,h] & D[n,h]", Q=key_query, D=key_data
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
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
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
