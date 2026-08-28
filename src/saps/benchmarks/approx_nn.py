import logging

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


class JLApproxNNTestGenerator(Generator[JLApproxNNRandomDataset]):
    @property
    def name(self) -> str:
        return "jl_projection_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "JL Projection Test Input Generator"

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
    def authors(self) -> list[Contributor]:
        return JLApproxNNGenerator().authors

    @property
    def references(self) -> list[Ref]:
        return JLApproxNNGenerator().references

    @property
    def ai_disclosure(self) -> str:
        return JLApproxNNGenerator().ai_disclosure

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
                    "test dense data and query matrices with sparse random projection."
                ),
                suites=["test", "trace"],
                n_samples=20,
                n_features=10,
                n_queries=4,
                k=3,
                eps=0.01,
                seed=42,
            )
        ]

    def generate(self, dataset: JLApproxNNRandomDataset):
        problem = JLApproxNNGenerator().generate(dataset)
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_meta={"check": "jl_preserves_distance"},
        )


class JLApproxNNGenerator(Generator[JLApproxNNRandomDataset]):
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
    def datasets(self) -> list[JLApproxNNRandomDataset]:
        return [
            JLApproxNNRandomDataset(
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
            JLApproxNNRandomDataset(
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
            JLApproxNNRandomDataset(
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

    def generate(self, dataset: JLApproxNNRandomDataset):
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

        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(data),
                BinsparseFormat.from_numpy(query),
                P,
            ],
            meta=meta,
        )


class JLApproxNNDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        k: int,
        eps: float,
        seed: int = 0,
        suites: list[str] | None = None,
    ):
        self._source_name = source_name
        self.k = k
        self.eps = eps
        self.seed = seed
        self._suites = suites or []

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


def _rla_projection(n_features: int, n_samples: int, eps: float, seed: int):
    import scipy as sp

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
    coo = (U_Neg + U_Pos).tocoo()
    return BinsparseFormat.from_coo((coo.row, coo.col), coo.data, coo.shape)


class JLApproxNNMNISTGenerator(Generator[JLApproxNNDataset]):
    @property
    def name(self) -> str:
        return "jl_approx_nn_mnist"

    @property
    def pretty_name(self) -> str:
        return "JL ANN MNIST Generator"

    @property
    def description(self) -> str:
        return "Loads MNIST images for JL approximate nearest-neighbor."

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
                title=(
                    "MNIST Dataset Classification Utilizing k-NN Classifier"
                    " with Modified Sliding-window Metric"
                ),
                authors=[Author("Aditi Grover"), Author("Bahram Toghi")],
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
            "MNIST provides 70,000 28×28 grayscale images of handwritten digits "
            "flattened to 784-dimensional vectors, with a 60K/10K train/test split."
        )

    @property
    def datasets(self) -> list[JLApproxNNDataset]:
        return [JLApproxNNDataset("mnist", k=5, eps=0.3, seed=50)]

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        data = mnist.data.astype(np.float32) / 255.0
        rng = np.random.default_rng(dataset.seed)
        train = data[rng.choice(60000, size=2000, replace=False)]
        test = data[60000:][rng.choice(10000, size=200, replace=False)]

        n_samples, n_features = train.shape
        projection = _rla_projection(n_features, n_samples, dataset.eps, dataset.seed)

        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(train),
                BinsparseFormat.from_numpy(test),
                projection,
            ],
            meta={"k": dataset.k, "eps": dataset.eps},
        )


class JLApproxNNCIFAR10Generator(Generator[JLApproxNNDataset]):
    @property
    def name(self) -> str:
        return "jl_approx_nn_cifar10"

    @property
    def pretty_name(self) -> str:
        return "JL ANN CIFAR-10 Generator"

    @property
    def description(self) -> str:
        return "Loads CIFAR-10 images for JL approximate nearest-neighbor."

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
                title="Learning Multiple Layers of Features from Tiny Images",
                authors=[Author("Alex Krizhevsky")],
                year=2009,
                url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
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
            "CIFAR-10 provides 60,000 32×32 color images across 10 classes, each "
            "flattened to a 3,072-dimensional vector. Its higher dimensionality and "
            "dense structure contrasts with the Netflix and MNIST cases."
        )

    @property
    def datasets(self) -> list[JLApproxNNDataset]:
        return [JLApproxNNDataset("cifar10", k=5, eps=0.3, seed=0)]

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
        import os
        import pickle
        import tarfile
        import urllib.request
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sparseappbench", "cifar10")
        extracted_dir = os.path.join(cache_dir, "cifar-10-batches-py")

        if not os.path.exists(extracted_dir):
            os.makedirs(cache_dir, exist_ok=True)
            tar_path = os.path.join(cache_dir, "cifar-10-python.tar.gz")
            urllib.request.urlretrieve(
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path
            )
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(cache_dir)

        train_arrays = []
        for i in range(1, 6):
            with open(os.path.join(extracted_dir, f"data_batch_{i}"), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            train_arrays.append(batch[b"data"])
        all_train = np.concatenate(train_arrays, axis=0).astype(np.float32) / 255.0

        with open(os.path.join(extracted_dir, "test_batch"), "rb") as f:
            all_test = pickle.load(f, encoding="bytes")[b"data"].astype(np.float32) / 255.0

        rng = np.random.default_rng(dataset.seed)
        train = all_train[rng.choice(len(all_train), size=2000, replace=False)]
        test = all_test[rng.choice(len(all_test), size=200, replace=False)]

        n_samples, n_features = train.shape
        projection = _rla_projection(n_features, n_samples, dataset.eps, dataset.seed)

        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(train),
                BinsparseFormat.from_numpy(test),
                projection,
            ],
            meta={"k": dataset.k, "eps": dataset.eps},
        )


class JLApproxNNNetflixGenerator(Generator[JLApproxNNDataset]):
    @property
    def name(self) -> str:
        return "jl_approx_nn_netflix"

    @property
    def pretty_name(self) -> str:
        return "JL ANN Netflix Generator"

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
            "No generative AI was used to construct the benchmark function "
            "itself. Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self) -> str:
        return (
            "The Netflix Prize dataset provides a ~480K users × 17,770 movies sparse "
            "ratings matrix. Uses sparse JL projection due to the dataset's sparsity."
        )

    @property
    def datasets(self) -> list[JLApproxNNDataset]:
        return [JLApproxNNDataset("netflix", k=5, eps=0.3, seed=0)]

    def generate(self, dataset: JLApproxNNDataset) -> DataInstance:
        import os
        import kagglehub
        import scipy.sparse
        cache_path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")

        row_list, col_list, val_list = [], [], []
        user_map = {}
        current_movie = 0
        data_files = sorted(
            f for f in os.listdir(cache_path)
            if f.startswith("combined_data") and f.endswith(".txt")
        )
        for fname in data_files:
            with open(os.path.join(cache_path, fname), "r") as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(":"):
                        current_movie = int(line[:-1]) - 1
                    else:
                        uid_str, rating_str, _ = line.split(",", 2)
                        uid = int(uid_str)
                        if uid not in user_map:
                            user_map[uid] = len(user_map)
                        row_list.append(user_map[uid])
                        col_list.append(current_movie)
                        val_list.append(float(rating_str))

        n_users = len(user_map)
        n_movies = 17770
        data = scipy.sparse.csr_matrix(
            (np.array(val_list, dtype=np.float32),
             (np.array(row_list, dtype=np.int32), np.array(col_list, dtype=np.int32))),
            shape=(n_users, n_movies),
        )

        rng = np.random.default_rng(dataset.seed)
        perm = rng.permutation(n_users)
        train_coo = data[perm[:2000]].tocoo()
        test_coo = data[perm[2000:2200]].tocoo()

        projection = _rla_projection(n_movies, 2000, dataset.eps, dataset.seed)

        return DataInstance(
            inputs=[
                BinsparseFormat.from_coo((train_coo.row, train_coo.col), train_coo.data, train_coo.shape),
                BinsparseFormat.from_coo((test_coo.row, test_coo.col), test_coo.data, test_coo.shape),
                projection,
            ],
            meta={"k": dataset.k, "eps": dataset.eps},
        )


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
        return [
            JLApproxNNTestGenerator(),
            JLApproxNNGenerator(),
            JLApproxNNMNISTGenerator(),
            JLApproxNNCIFAR10Generator(),
            JLApproxNNNetflixGenerator(),
        ]

    def benchmark(self, xp, data, meta):
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if not self._ref_meta:
            return

        data = self._input[0].data["values"].reshape(self._input[0].data["shape"])
        query = self._input[1].data["values"].reshape(self._input[1].data["shape"])
        nearest_ind = (
            self._output[0].data["values"].reshape(self._output[0].data["shape"])
        )

        diff = np.expand_dims(query, axis=1) - np.expand_dims(data, axis=0)
        orig_distances = np.sqrt(np.sum(diff**2, axis=-1))

        true_nearest = np.min(orig_distances, axis=1)
        approx_nearest = orig_distances[
            np.arange(param.dataset.n_queries), nearest_ind[:, 0].astype(int)
        ]
        assert np.all(approx_nearest <= (1 + param.dataset.eps) * true_nearest)
