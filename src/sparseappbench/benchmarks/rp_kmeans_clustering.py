import math

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Ref,
)

xp = sparseappbench.xp


<<<<<<< HEAD
class RPKMeansDataset(Dataset):
    def __int__(
        self,
        name,
        pretty_name,
        description,
        tags,
        k,
        eps,
        c,
        A,
        max_iter,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
        self.k = k
        self.eps = eps
        self.c = c
        self.A = A
        self.max_iter = max_iter
=======
def rp_kmeans_clustering(xp, A_benchmark, k, eps, c=1, max_iter=100):
    assert c > 0
    assert eps > 0 and eps < 1 / 3
    assert k > 0
    A = xp.from_binsparse(A_benchmark)
    A = A
    n, d = A.shape
    t = int(c * math.ceil(k / eps**2))
    value = 1 / (t**0.5)
    R = xp.random.rand(d, t)
    R = R < 0.5
    R = xp.where(R, value, -value)
    A_prime = xp.matmul(A, R)
    return kmeans(xp, A_prime, k, max_iter)
>>>>>>> 9b908f6c0a4ffbba951e801b1264e5af1828c9f2

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

    # From what I have seen from this benchmark, there is no Generator.
    # # Just two hardcoded examples.

    class RPKmeansClustering(Benchmark):
        @property
        def tag(self):
            return "rp_kmeans_clustering"

        @property
        def name(self):
            return "Random Numerical Linear Algebra"

        @property
        def pretty_name(self):
            return "Johnson-Lindenstrauss Projections for K-means Clustering"

        @property
        def description(self):
            return (
                "Benchmarks Johnson-Lindenstrauss projection followed by "
                "k_means"
                "cluster assignment for each point"
            )

        @property
        def tags(self):
            return ["rnla", "approximate-k-means", "projection", "sparse"]

        @property
        def authors(self):
            return [Contributor("Maksim Krylykov", "mkrylykov3@gatech.edu")]

        @property
        def references(self):
            return [
                Ref(
                    title="Random projections for k-means clustering",
                    authors=[
                        Author("Boutsidis, C."),
                        Author("Zouzias, A."),
                        Author("Drineas, P."),
                    ],
                    year="2010",
                    url="https://dl.acm.org/doi/10.5555/2997189.2997223",
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
                "Random Projections reduce dimensionality for k-means clustering."
                "Role of Sparsity: Input points can be high-dimensional and sparse"
                "which are then projected on a random matrix."
            )

        # Little rewriting, putting the two functions they defined for readablity
        # into one function, while still keeping their naming convention.
        @property
        def benchmark(self, dataset: "RPKMeansDataset"):
            A = xp.from_benchmark(dataset.A)
            A = xp.lazy(A)
            n, d = A.shape
            t = int(dataset.c * math.ceil(dataset.k / dataset.eps**2))
            value = 1 / (t**0.5)
            R = xp.random.rand(d, t)
            R = R < 0.5
            R = xp.where(R, value, -value)
            A_prime = xp.matmul(A, R)

            n, t = A.shape

            centroids = A_prime[: dataset.k, :]
            labels = xp.zeros((n,), dtype=xp.int64)
            ks = xp.arange(dataset.k, dtype=xp.int64)
            one = xp.asarray(1, dtype=A_prime.dtype)
            for _ in range(dataset.max_iter):
                old_labels = labels
                dists = xp.sum((A_prime[:, None] - centroids[None, :]) ** 2, axis=2)
                labels = xp.argmin(dists, axis=1)
                H = xp.equal(labels[:, None], ks[None, :]).astype(A_prime.dtype)
                counts = xp.sum(H, axis=0)
                sums = xp.matmul(xp.transpose(H), A_prime)
                counts_nonz = xp.where(counts > 0, counts, one)
                new_centroids = sums / counts_nonz[:, None]
                new_centroids = xp.where(
                    (counts == 0)[:, None], centroids, new_centroids
                )
                centroids = new_centroids
                if xp.all(labels == old_labels).item():
                    break

            return labels
