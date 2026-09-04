import math
from typing import Any

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


class RPKMeansRandomDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        points,
        k: int,
        eps: float,
        c=1,
        max_iter=100,
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
    ):
        self._suites = suites or []
        self.source_name = source_name
        self.points = points
        self.k = k
        self.eps = eps
        self.c = c
        self.max_iter = max_iter
        self.ref_meta = ref_meta

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"RP k-means {self.source_name}"

    @property
    def description(self) -> str:
        return "Manual test points for random-projection k-means clustering."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["k"] = self.k
        data["eps"] = self.eps
        data["c"] = self.c
        data["max_iter"] = self.max_iter
        return data


class RPKMeansGenerator(Generator[RPKMeansRandomDataset]):
    @property
    def name(self) -> str:
        return "rp_kmeans_inputs"

    @property
    def pretty_name(self) -> str:
        return "Random Projection k-means Data Generator"

    @property
    def description(self) -> str:
        return "Test points for this benchmark were created manually."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return RPKMeansBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return RPKMeansBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return RPKMeansBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return RPKMeansBenchmark().motivation

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[RPKMeansRandomDataset]:
        return [
            RPKMeansRandomDataset(
                "three_clusters",
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, -0.1, 0.0],
                        [5.0, 5.0, 5.0],
                        [5.1, 5.0, 5.2],
                        [-10.0, -10.1, -10.2],
                        [-9.9, -9.8, -9.7],
                    ],
                    dtype=np.float32,
                ),
                k=3,
                eps=0.3,
                c=0.5,
                max_iter=5,
                suites=["test", "trace"],
                ref_meta={
                    "same": [(0, 1), (2, 3), (4, 5)],
                    "cluster_count": 3,
                },
            ),
            RPKMeansRandomDataset(
                "two_clusters",
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, -0.1, 0.0],
                        [-0.2, 0.0, 0.1],
                        [0.3, 0.2, 0.1],
                        [-20.0, -40.0, -60.0],
                    ],
                    dtype=np.float32,
                ),
                k=2,
                eps=0.2,
                c=1,
                max_iter=5,
                suites=["test", "trace"],
                ref_meta={
                    "same": [(0, 1), (1, 2), (2, 3)],
                    "different": [(0, 4)],
                },
            ),
        ]

    def generate(self, dataset: RPKMeansRandomDataset) -> DataInstance:
        A_bin = BinsparseFormat.from_numpy(dataset.points)
        _, d = dataset.points.shape
        t = int(dataset.c * math.ceil(dataset.k / dataset.eps**2))
        value = 1 / (t**0.5)
        rng = np.random.default_rng(0)
        R = np.where(rng.random((d, t)) < 0.5, value, -value).astype(
            dataset.points.dtype
        )
        R_bin = from_numpy(R)
        return DataInstance(
            inputs=[A_bin, R_bin],
            meta={
                "k": dataset.k,
                "eps": dataset.eps,
                "c": dataset.c,
                "max_iter": dataset.max_iter,
            },
            ref_meta=dataset.ref_meta,
        )


class RPKMeansDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        k: int,
        eps: float,
        c=1,
        max_iter=100,
        suites: list[str] | None = None,
    ):
        self._source_name = source_name
        self.k = k
        self.eps = eps
        self.c = c
        self.max_iter = max_iter
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._source_name

    @property
    def pretty_name(self) -> str:
        return f"RP k-means {self._source_name}"

    @property
    def description(self) -> str:
        return f"RP k-means clustering on {self._source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["k"] = self.k
        data["eps"] = self.eps
        data["c"] = self.c
        data["max_iter"] = self.max_iter
        return data


class RPKMeansMNISTGenerator(Generator[RPKMeansDataset]):
    @property
    def name(self) -> str:
        return "rp_kmeans_mnist"

    @property
    def pretty_name(self) -> str:
        return "RP k-means MNIST Generator"

    @property
    def description(self) -> str:
        return "Loads MNIST training images for RP k-means clustering."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kevin Wang", "kwang656@gatech.edu")]

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
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "MNIST provides 70,000 28×28 grayscale images of handwritten digits "
            "flattened to 784-dimensional vectors, with a 60K/10K train/test split. "
            "k = 10 aligns with the ten digit classes."
        )

    @property
    def datasets(self) -> list[RPKMeansDataset]:
        return [
            RPKMeansDataset("mnist", k=10, eps=0.3),
        ]

    def generate(self, dataset: RPKMeansDataset) -> DataInstance:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        training = X[:60000]

        n, d = training.shape
        t = int(dataset.c * math.ceil(dataset.k / dataset.eps**2))
        value = 1 / (t**0.5)
        rng = np.random.default_rng(0)
        R = np.where(rng.random((d, t)) < 0.5, value, -value).astype(np.float32)

        return DataInstance(
            inputs=[BinsparseFormat.from_numpy(training), BinsparseFormat.from_numpy(R)],
            meta={
                "k": dataset.k,
                "eps": dataset.eps,
                "c": dataset.c,
                "max_iter": dataset.max_iter,
            },
        )


class RPKMeansCIFAR10Generator(Generator[RPKMeansDataset]):
    @property
    def name(self) -> str:
        return "rp_kmeans_cifar10"

    @property
    def pretty_name(self) -> str:
        return "RP k-means CIFAR-10 Generator"

    @property
    def description(self) -> str:
        return "Loads CIFAR-10 training images for RP k-means clustering."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Maksim Krylykov", "mkrylykov3@gatech.edu")]

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
        return "No generative AI was used to implement benchmark functions."

    @property
    def motivation(self) -> str:
        return (
            "CIFAR-10 provides 60,000 32×32 color images across 10 classes, each "
            "flattened to a 3,072-dimensional vector. Its higher dimensionality and "
            "dense structure contrasts with the Netflix and MNIST cases."
        )

    @property
    def datasets(self) -> list[RPKMeansDataset]:
        return [
            RPKMeansDataset("cifar10", k=10, eps=0.3),
        ]

    def generate(self, dataset: RPKMeansDataset) -> DataInstance:
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
        training = np.concatenate(train_arrays, axis=0).astype(np.float32) / 255.0

        n, d = training.shape
        t = int(dataset.c * math.ceil(dataset.k / dataset.eps**2))
        value = 1 / (t**0.5)
        rng = np.random.default_rng(0)
        R = np.where(rng.random((d, t)) < 0.5, value, -value).astype(np.float32)

        return DataInstance(
            inputs=[BinsparseFormat.from_numpy(training), BinsparseFormat.from_numpy(R)],
            meta={
                "k": dataset.k,
                "eps": dataset.eps,
                "c": dataset.c,
                "max_iter": dataset.max_iter,
            },
        )


class RPKMeansNetflixGenerator(Generator[RPKMeansDataset]):
    @property
    def name(self) -> str:
        return "rp_kmeans_netflix"

    @property
    def pretty_name(self) -> str:
        return "RP k-means Netflix Generator"

    @property
    def description(self) -> str:
        return "Loads Netflix Prize ratings for RP k-means clustering."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Maksim Krylykov", "mkrylykov3@gatech.edu")]

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
        return "No generative AI was used to implement benchmark functions."

    @property
    def motivation(self) -> str:
        return (
            "The Netflix Prize dataset provides a ~480K users × 17,770 movies sparse "
            "ratings matrix."
        )

    @property
    def datasets(self) -> list[RPKMeansDataset]:
        return [
            RPKMeansDataset("netflix", k=10, eps=0.3),
        ]

    def generate(self, dataset: RPKMeansDataset) -> DataInstance:
        import os
        import kagglehub
        import scipy.sparse
        cache_path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")

        row_list = []
        col_list = []
        val_list = []
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

        rng = np.random.default_rng(0)
        train_coo = data[rng.permutation(n_users)[:5000]].tocoo()
        training = BinsparseFormat.from_coo(
            (train_coo.row, train_coo.col), train_coo.data, train_coo.shape
        )

        d = n_movies
        t = int(dataset.c * math.ceil(dataset.k / dataset.eps**2))
        value = 1 / (t**0.5)
        rng = np.random.default_rng(0)
        R = np.where(rng.random((d, t)) < 0.5, value, -value).astype(np.float32)

        return DataInstance(
            inputs=[training, BinsparseFormat.from_numpy(R)],
            meta={
                "k": dataset.k,
                "eps": dataset.eps,
                "c": dataset.c,
                "max_iter": dataset.max_iter,
            },
        )


class RPKMeansBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "rp_kmeans_clustering"

    @property
    def pretty_name(self) -> str:
        return "Random Projections for k-means Clustering"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Maksim Krylykov", "mkrylykov3@gatech.edu")]

    @property
    def description(self) -> str:
        return "Labels points into k clusters."

    @property
    def motivation(self) -> str:
        return (
            "Random Projections reduce dimensionality for k-means clustering. Input"
            " points can be high-dimensional and sparse, which are then projected on a"
            " random matrix."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                authors=[
                    Author("Christos Boutsidis"),
                    Author("Anastasios Zouzias"),
                    Author("Petros Drineas"),
                ],
                title="Random Projections for $k$-means Clustering",
                journal="Arxiv",
                volume="arXiv:1011.4632",
                year=2010,
                url="https://arxiv.org/abs/1011.4632",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "No generative AI was used to implement benchmark functions."

    @property
    def suites(self) -> list[str]:
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
</ccs2012>
"""
        )

    @property
    def generators(self):
        return [
            RPKMeansGenerator(),
            RPKMeansMNISTGenerator(),
            RPKMeansCIFAR10Generator(),
            RPKMeansNetflixGenerator(),
        ]

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        """
                Labels points into k clusters.

        Args:
        ----
        xp : array_api
            The array API module to utilize
        A_benchmark : BinsparseTensor
            Sparse input matrix
        k : int
            Number of clusters
        eps : float
            Error parameter in (0, 1/3)
        c : float
            Constant factor for new dimensionality t
        max_iter : int
            Number of iterations for k-means

                Returns:
                -------
                Returns xp.array of size n: labels of input points.
        """
        k = meta["k"]
        eps = meta["eps"]
        c = meta.get("c", 1)
        max_iter = meta.get("max_iter", 100)
        assert c > 0
        assert eps > 0 and eps < 1 / 3
        assert k > 0
        A, R = data
        A_prime = xp.matmul(A, R)

        n, t = A_prime.shape
        centroids = A_prime[:k, :]
        labels = xp.zeros((n,), dtype=xp.int64)
        ks = xp.arange(k, dtype=xp.int64)
        one = xp.asarray(1, dtype=A_prime.dtype)
        for _ in range(max_iter):
            old_labels = labels
            dists = xp.sum((A_prime[:, None] - centroids[None, :]) ** 2, axis=2)
            labels = xp.argmin(dists, axis=1)
            H = xp.equal(labels[:, None], ks[None, :]).astype(A_prime.dtype)
            counts = xp.sum(H, axis=0)
            sums = xp.matmul(xp.transpose(H), A_prime)
            counts_nonz = xp.where(counts > 0, counts, one)
            new_centroids = sums / counts_nonz[:, None]
            new_centroids = xp.where((counts == 0)[:, None], centroids, new_centroids)
            centroids = new_centroids
            if xp.all(labels == old_labels).item():
                break

        return [labels]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_meta is None:
            return

        labels = to_numpy(self._output[0])

        for left, right in self._ref_meta.get("same", []):
            assert labels[left] == labels[right]
        for left, right in self._ref_meta.get("different", []):
            assert labels[left] != labels[right]
        if "cluster_count" in self._ref_meta:
            assert len(set(labels.tolist())) == self._ref_meta["cluster_count"]
