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

from saps.downloaders.gcare import load_gcare_dataset

xp = saps.xp


class SubgraphGCareDataset(Dataset):
    def __init__(
        self,
        name,
        pretty_name,
        description,
        tags,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags

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


class SubgraphGCareGenerator(Generator[SubgraphGCareDataset]):
    @property
    def name(self) -> str:
        return "subgraph_gcare_inputs"

    @property
    def pretty_name(self) -> str:
        return "Subgraph G-CARE Input Generator"

    @property
    def description(self) -> str:
        return (
            "Transforms the G-CARE dataset to the input of subgraph matching"
            " algorithms."
        )

    @property
    def tags(self) -> list[str]:
        return ["subgraph matching", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Taishan Chen", "utallow@bu.edu"),
            Contributor("Kyle Deeds", "kdeeds@bu.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "G-CARE: A Framework for Performance Benchmarking of "
                    "Cardinality Estimation Techniques for Subgraph Matching"
                ),
                authors=[
                    Author("Yeonsu Park"),
                    Author("Seongyun Ko"),
                    Author("Sourav S Bhowmick"),
                    Author("Kyoungmin Kim"),
                    Author("Kijae Hong"),
                    Author("Wook-Shin Han"),
                ],
                year=2020,
                url="https://dl.acm.org/doi/10.1145/3318464.3389702",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the algorithms for the "
            "benchmark function. Generative AI might have been used to "
            "construct the framework, comments and helper functions."
        )

    @property
    def motivation(self) -> str:
        return (
            "Subgraph matching and counting are classic problems and widely "
            "used in query evaluations in database systems."
        )

    @property
    def datasets(self) -> list[SubgraphGCareDataset]:
        # Note: NumpyFramework will fail to run even for the smallest data. 
        return [
            SubgraphGCareDataset(
                name="human",
                pretty_name="G-CARE Human Subset (Small)",
                description=("G-CARE Human Subset (Small)"),
                tags=["small", "sparse"],
            ),
            # SubgraphGCareDataset(
            #     name="aids",
            #     pretty_name="G-CARE AIDS Subset (Medium)",
            #     description=("G-CARE AIDS Subset (Medium)"),
            #     tags=["medium", "sparse"],
            # ),
            # SubgraphGCareDataset(
            #     name="lubm80",
            #     pretty_name="G-CARE LUBM80 Subset (Large)",
            #     description=("G-CARE LUBM80 Subset (Large)"),
            #     tags=["large", "sparse"],
            # ),
            # SubgraphGCareDataset(
            #     name="yago",
            #     pretty_name="G-CARE YAGO Subset (Huge)",
            #     description=("G-CARE YAGO Subset (Huge)"),
            #     tags=["huge", "sparse"],
            # ),
        ]

    def generate(self, dataset: SubgraphGCareDataset):
        return load_gcare_dataset(dataset.name)


class SubgraphMatching(Benchmark):
    @property
    def tag(self):
        return "subgraph_matching"

    @property
    def name(self):
        return "Subgraph Matching Algorithm using einsum"

    @property
    def pretty_name(self):
        return "Subgraph Matching Algorithm using einsum"

    @property
    def description(self):
        return "Benchmarks subgraph matching algorithms using einsum operations."

    @property
    def tags(self):
        return ["subgraph-matching", "sparse"]

    @property
    def authors(self):
        return [
            Contributor("Taishan Chen", "utallow@bu.edu"),
            Contributor("Kyle Deeds", "kdeeds@bu.edu"),
        ]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the algorithms for the "
            "benchmark function. Generative AI might have been used to "
            "construct the definition of the framework."
        )

    @property
    def motivation(self):
        return (
            "Subgraph matching and counting are classic problems and widely "
            "used in query evaluations in database systems."
        )

    @property
    def generators(self) -> list[Generator[SubgraphGCareDataset]]:
        return [SubgraphGCareGenerator()]

    def benchmark(self, data, meta):
        # data is a flat list of already-converted arrays (one per matrix across
        # all queries).  Reconstruct per-query dicts using the grouping info
        # stored in meta by load_gcare_dataset().
        exprs = meta["exprs"]
        query_sizes = meta["query_sizes"]
        matrix_names = meta["matrix_names"]

        counts = np.zeros((len(exprs),), dtype=np.int64)
        offset = 0
        for i, (expr, size) in enumerate(zip(exprs, query_sizes)):
            sp_mats = dict(zip(matrix_names[offset : offset + size], data[offset : offset + size]))
            counts[i] = xp.einsum(expr, **sp_mats)
            offset += size
        return [xp.asarray(counts)]
