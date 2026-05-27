from typing import Any

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

from saps.downloaders.snap import download_snap_dataset
from saps_framework.binsparse_format import BinsparseFormat

xp = saps.xp


class FastSVDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"FastSV input {name}."
        self._tags = tags or ["graph", "sparse"]

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


class FastSVGenerator(Generator[FastSVDataset]):
    @property
    def name(self) -> str:
        return "fastsv_inputs"

    @property
    def pretty_name(self) -> str:
        return "FastSV Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for FastSV connected-components benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["graph", "sparse", "connected-components"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Richard Wan", "rwan41@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for FastSV."

    @property
    def datasets(self) -> list[FastSVDataset]:
        return [
            FastSVDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                tags=["graph", "sparse", "connected-components", "snap", "directed"],
            ),
            FastSVDataset(
                name="snap-facebook_combined",
                pretty_name="SNAP facebook_combined",
                description=(
                    "Combined Facebook social-circle network, with 4,039 nodes and"
                    " 88,234 edges."
                ),
                tags=[
                    "graph",
                    "sparse",
                    "connected-components",
                    "snap",
                    "social-network",
                ],
            ),
            FastSVDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                tags=[
                    "graph",
                    "sparse",
                    "connected-components",
                    "snap",
                    "collaboration-network",
                ],
            ),
        ]

    def generate(self, dataset: FastSVDataset) -> tuple[list[BinsparseFormat], Any]:
        if dataset.name.startswith("snap"):
            return download_snap_dataset(dataset.name)
        raise ValueError(f"Unsupported FastSV dataset: {dataset.name}")


class FastSVBenchmark(Benchmark):
    @property
    def name(self):
        return "fastsv"

    @property
    def pretty_name(self):
        return "FastSV Algorithm"

    @property
    def description(self):
        return (
            "The FastSV algorithm is a graph algorithm used to find the connected"
            " components for a simple graph. This algorithm introduces several"
            " optimizations that allow for faster convergence to a solution compared to"
            " the SV algorithm it is based on, specifically through modifications to"
            " the tree hooking and termination condition."
        )

    @property
    def tags(self):
        return ["graph", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Richard Wan", "rwan41@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=(
                    "FastSV: A distributed-memory connected component"
                    " algorithm with fast convergence."
                ),
                authors=[
                    Author("Zhang, Y."),
                    Author("Azad, A."),
                    Author("Hu, Z."),
                ],
                journal=(
                    "Proceedings of the 2020 SIAM Conference on Parallel"
                    " Processing for Scientific Computing"
                ),
                pages="46-57",
                publisher="Society for Industrial and Applied Mathematics",
                year=2020,
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
        return ""

    @property
    def generators(self) -> list[Generator[FastSVDataset]]:
        return [FastSVGenerator()]

    def benchmark(self, data, meta):
        A = data[0]
        A = A != 0

        (n, m) = A.shape
        assert n == m

        f = xp.arange(n)
        gf = xp.asarray(f, copy=True)

        int_max = xp.iinfo(f.dtype).max

        while True:
            dup = gf

            # step 1: stochastic hooking
            mngf = xp.min(xp.where(A, xp.expand_dims(gf, 0), int_max), axis=1)
            B = xp.zeros((n, n), dtype=bool)
            B[f, xp.arange(n)] = True
            f = xp.min(xp.where(B, xp.expand_dims(mngf, 0), int_max), axis=1)

            # step 2: aggressive hooking
            f = xp.minimum(f, mngf)

            # step 3: shortcutting
            f = xp.minimum(f, gf)

            # step 4: calculate grandparents
            gf = xp.take(f, f)

            # step 5: check termination
            stop = xp.all(dup == gf)

            if stop:
                break

        return [f]
