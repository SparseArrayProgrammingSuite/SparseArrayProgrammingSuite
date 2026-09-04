from __future__ import annotations

from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.ogb import OGBNodePropData, load_ogb_nodeprop_dataset


class OGBNodePropDataset(Dataset):
    """Base Dataset for benchmarks backed by a homogeneous OGB node dataset."""

    def __init__(
        self,
        name: str,
        *,
        source_name: str | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self.source_name = source_name if source_name is not None else name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self.source_name

    @property
    def description(self) -> str:
        return self._description or f"Open Graph Benchmark dataset {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_name"] = self.source_name
        return data


class OGBNodePropGenerator(Generator[OGBNodePropDataset]):
    """Downloads and caches OGB node-property data, shared across benchmarks."""

    @property
    def name(self) -> str:
        return "ogb_nodeprop"

    @property
    def pretty_name(self) -> str:
        return "Open Graph Benchmark Node Property Datasets"

    @property
    def description(self) -> str:
        return (
            "Downloads, prepares, and caches homogeneous Open Graph Benchmark "
            "node-property datasets. Benchmark-specific generators compose this "
            "generator so each OGB graph is downloaded and cached once."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Open Graph Benchmark: Datasets for Machine Learning on Graphs",
                authors=[
                    Author("Weihua Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal="Arxiv",
                volume="arXiv:2005.00687",
                year=2020,
                url="https://arxiv.org/abs/2005.00687",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the source datasets. Generative "
            "AI was used to help implement and audit the OGB downloader, input "
            "generator, tests, documentation, and debugging."
        )

    @property
    def motivation(self) -> str:
        return (
            "Large OGB graphs can be reused by several graph-learning benchmarks. "
            "Sharing a cacheable generator for the prepared source graph avoids "
            "redundant downloads while allowing child benchmark generators to keep "
            "their benchmark-specific inputs uncached."
        )

    @property
    def datasets(self) -> list[OGBNodePropDataset]:
        return [
            OGBNodePropDataset(
                "ogbn_arxiv",
                source_name="ogbn-arxiv",
                pretty_name="ogbn-arxiv",
                description=(
                    "Citation network of arXiv Computer Science papers for "
                    "node-property prediction."
                ),
                suites=["standard"],
            ),
            OGBNodePropDataset(
                "ogbn_products",
                source_name="ogbn-products",
                pretty_name="ogbn-products",
                description=(
                    "Amazon product co-purchasing network for large-scale "
                    "node-property prediction."
                ),
                suites=["standard"],
            ),
            OGBNodePropDataset(
                "ogbn_proteins",
                source_name="ogbn-proteins",
                pretty_name="ogbn-proteins",
                description=(
                    "Protein-protein association network with species labels and "
                    "averaged edge-feature node inputs."
                ),
                suites=["standard"],
            ),
        ]

    def generate(self, dataset: OGBNodePropDataset) -> DataInstance:
        graph = load_ogb_nodeprop_dataset(dataset.source_name)
        split_names = list(graph.split_indices)
        meta = {
            **graph.metadata,
            "num_nodes": graph.num_nodes,
            "num_raw_edges": graph.num_raw_edges,
            "num_features": graph.num_features,
            "num_outputs": graph.num_outputs,
            "split_names": split_names,
        }
        return DataInstance(
            inputs=[
                graph.adjacency,
                from_numpy(graph.features),
                from_numpy(graph.labels),
                *(from_numpy(graph.split_indices[name]) for name in split_names),
            ],
            meta=meta,
        )


class OGBNodePropBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return OGBNodePropGenerator()


def fetch_ogb_nodeprop_dataset(source_name: str) -> OGBNodePropData:
    """Fetch (and cache) a prepared OGB node-property dataset via the shared shell."""
    raw_generator = OGBNodePropGenerator()
    raw_dataset = next(
        dataset
        for dataset in raw_generator.datasets
        if dataset.source_name == source_name
    )
    raw = raw_generator.cached_generate(raw_dataset)
    split_names = raw.meta.get("split_names") or ["train", "valid", "test"]
    split_indices = {
        name: np.asarray(to_numpy(raw.inputs[index]), dtype=np.int64)
        for index, name in enumerate(split_names, start=3)
    }
    return OGBNodePropData(
        name=raw.meta["dataset_name"],
        adjacency=raw.inputs[0],
        features=np.asarray(to_numpy(raw.inputs[1]), dtype=np.float32),
        labels=np.asarray(to_numpy(raw.inputs[2])),
        split_indices=split_indices,
        num_nodes=int(raw.meta["num_nodes"]),
        num_raw_edges=int(raw.meta["num_raw_edges"]),
        num_features=int(raw.meta["num_features"]),
        num_tasks=int(raw.meta["num_tasks"]),
        num_classes=int(raw.meta["num_classes"]),
        num_outputs=int(raw.meta["num_outputs"]),
        metadata=raw.meta,
    )


def fetch_ogb_gcn_inputs(source_name: str) -> tuple[BinsparseTensor, np.ndarray, dict]:
    """Fetch the normalized adjacency and feature matrix for OGB GCN benchmarks."""
    graph = fetch_ogb_nodeprop_dataset(source_name)
    return graph.adjacency, graph.features, graph.metadata
