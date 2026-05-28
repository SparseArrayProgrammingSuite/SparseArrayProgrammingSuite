import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

from saps.downloaders.gcare import load_gcare_graph, load_gcare_query, list_gcare_queries

xp = saps.xp


class SubgraphGCareGraphDataset(Dataset):
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


class SubgraphGCareDataset(Dataset):
    def __init__(
        self,
        subset_name,
        query_name,
        pretty_name,
        description,
        tags,
    ):
        self._subset_name = subset_name
        self._query_name = query_name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags

    @property
    def name(self) -> str:
        return f"{self._subset_name}/{self._query_name}"

    @property
    def subset_name(self) -> str:
        return self._subset_name

    @property
    def query_name(self) -> str:
        return self._query_name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> list[str]:
        return self._tags


class SubgraphGCareGraphGenerator(Generator[SubgraphGCareGraphDataset]):
    @property
    def name(self) -> str:
        return "subgraph_gcare_graph"

    @property
    def pretty_name(self) -> str:
        return "Subgraph G-CARE Graph Generator"

    @property
    def description(self) -> str:
        return "Converts a G-CARE graph subset into BinsparseFormat matrices."

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
    def datasets(self) -> list[SubgraphGCareGraphDataset]:
        # Note: NumpyFramework will fail to run even for the smallest data.
        return [
            SubgraphGCareGraphDataset(
                name="human",
                pretty_name="G-CARE Human Subset (Small)",
                description=("G-CARE Human Subset (Small)"),
                tags=["small", "sparse"],
            ),
            SubgraphGCareGraphDataset(
                name="aids",
                pretty_name="G-CARE AIDS Subset (Medium)",
                description=("G-CARE AIDS Subset (Medium)"),
                tags=["medium", "sparse"],
            ),
            SubgraphGCareGraphDataset(
                name="lubm80",
                pretty_name="G-CARE LUBM80 Subset (Large)",
                description=("G-CARE LUBM80 Subset (Large)"),
                tags=["large", "sparse"],
            ),
            SubgraphGCareGraphDataset(
                name="yago",
                pretty_name="G-CARE YAGO Subset (Huge)",
                description=("G-CARE YAGO Subset (Huge)"),
                tags=["huge", "sparse"],
            ),
        ]

    def generate(self, dataset: SubgraphGCareGraphDataset):
        return load_gcare_graph(dataset.name)


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
        subsets = [
            ("human",  "G-CARE Human Subset",  "small"),
            ("aids",   "G-CARE AIDS Subset",   "medium"),
            ("lubm80", "G-CARE LUBM80 Subset", "large"),
            ("yago",   "G-CARE YAGO Subset",   "huge"),
        ]
        datasets = []
        for subset_name, subset_label, size_tag in subsets:
            for query_name in list_gcare_queries(subset_name):
                datasets.append(
                    SubgraphGCareDataset(
                        subset_name=subset_name,
                        query_name=query_name,
                        pretty_name=f"{subset_label}, Query {query_name}",
                        description=f"{subset_label} with query {query_name}",
                        tags=[size_tag, "sparse"],
                    )
                )
        return datasets

    def generate(self, dataset: SubgraphGCareDataset):
        raw_generator = SubgraphGCareGraphGenerator()
        raw_dataset = next(
            ds for ds in raw_generator.datasets if ds.name == dataset.subset_name
        )
        flat_matrices, graph_meta = raw_generator.cached_generate(raw_dataset)
        return load_gcare_query(
            dataset.subset_name, dataset.query_name,
            flat_matrices, graph_meta,
        )


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
        sp_mats = dict(zip(meta["matrix_names"], data))
        return [xp.einsum(meta["expr"], **sp_mats)]
