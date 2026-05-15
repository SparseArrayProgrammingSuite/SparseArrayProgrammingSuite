import math
from typing import Any

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


class RPKMeansDataset(Dataset):
    def __init__(self, source_name: str, points, k: int, eps: float, c=1, max_iter=100):
        self.source_name = source_name
        self.points = points
        self.k = k
        self.eps = eps
        self.c = c
        self.max_iter = max_iter

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
    def tags(self) -> list[str]:
        return ["clustering", "random-projection"]

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["k"] = self.k
        data["eps"] = self.eps
        data["c"] = self.c
        data["max_iter"] = self.max_iter
        return data


class RPKMeansGenerator(Generator[RPKMeansDataset]):
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
    def tags(self) -> list[str]:
        return ["clustering", "random-projection"]

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
    def datasets(self) -> list[RPKMeansDataset]:
        return [
            RPKMeansDataset(
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
            ),
            RPKMeansDataset(
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
            ),
        ]

    def generate(
        self, dataset: RPKMeansDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        A_bin = BinsparseFormat.from_numpy(dataset.points)
        return [A_bin], {
            "k": dataset.k,
            "eps": dataset.eps,
            "c": dataset.c,
            "max_iter": dataset.max_iter,
        }


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
                    Author("C. Boutsidis"),
                    Author("A. Zouzias"),
                    Author("P. Drineas"),
                ],
                title="Random Projection for k-Means Clustering",
                url="https://arxiv.org/abs/1011.4632",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "No generative AI was used to implement benchmark functions."

    @property
    def tags(self) -> list[str]:
        return ["clustering", "random-projection", "sparse"]

    @property
    def generators(self):
        return [RPKMeansGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]):
        """
                Labels points into k clusters.

        Args:
        ----
        xp : array_api
            The array API module to utilize
        A_benchmark : BinsparseFormat
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
        A = data[0]
        n, d = A.shape
        t = int(c * math.ceil(k / eps**2))
        value = 1 / (t**0.5)
        R = xp.random.rand(d, t)
        R = R < 0.5
        R = xp.where(R, value, -value)
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
