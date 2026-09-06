from __future__ import annotations

from typing import Any

import numpy as np

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


class OpenMLDataset(Dataset):
    """Base Dataset for benchmarks backed by an OpenML dense feature matrix."""

    def __init__(
        self,
        name: str,
        *,
        data_id: int,
        openml_name: str,
        version: int,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        scale: float = 1.0,
    ):
        self._name = name
        self.data_id = data_id
        self.openml_name = openml_name
        self.version = version
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.scale = scale

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self.openml_name

    @property
    def description(self) -> str:
        return self._description or f"OpenML dataset {self.openml_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data.update(
            {
                "data_id": self.data_id,
                "openml_name": self.openml_name,
                "version": self.version,
                "scale": self.scale,
            }
        )
        return data


class OpenMLDatasetGenerator(Generator[OpenMLDataset]):
    """Downloads and caches OpenML dense datasets shared across benchmarks."""

    @property
    def name(self) -> str:
        return "openml_dataset"

    @property
    def pretty_name(self) -> str:
        return "OpenML Datasets"

    @property
    def description(self) -> str:
        return (
            "Downloads, prepares, and caches dense OpenML datasets. "
            "Benchmark-specific generators compose this generator so each OpenML "
            "dataset is downloaded and cached once."
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
                title="OpenML",
                authors=[
                    Author("Joaquin Vanschoren"),
                    Author("Jan N. van Rijn"),
                    Author("Bernd Bischl"),
                    Author("Luis Torgo"),
                ],
                journal="ACM SIGKDD Explorations Newsletter",
                volume="15",
                number="2",
                pages="49-60",
                year=2014,
                url="https://doi.org/10.1145/2641190.2641198",
                doi="10.1145/2641190.2641198",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "Several benchmarks use the same OpenML image datasets. Sharing a "
            "cacheable generator for the prepared feature matrix avoids redundant "
            "downloads and redundant cached copies."
        )

    @property
    def datasets(self) -> list[OpenMLDataset]:
        return [
            OpenMLDataset(
                "mnist",
                data_id=554,
                openml_name="mnist_784",
                version=1,
                pretty_name="MNIST",
                description=(
                    "OpenML copy of the MNIST 28x28 handwritten digit image dataset."
                ),
                scale=255.0,
            ),
            OpenMLDataset(
                "cifar10",
                data_id=40927,
                openml_name="CIFAR_10",
                version=1,
                pretty_name="CIFAR-10",
                description=("OpenML copy of the CIFAR-10 32x32 color image dataset."),
                scale=255.0,
            ),
        ]

    def generate(self, dataset: OpenMLDataset) -> DataInstance:
        openml = _fetch_openml(dataset.data_id)
        features = np.asarray(openml.data, dtype=np.float32)
        if dataset.scale != 1.0:
            features = features / dataset.scale

        details = getattr(openml, "details", {})
        meta = {
            "data_id": dataset.data_id,
            "openml_name": dataset.openml_name,
            "version": dataset.version,
            "fetched_data_id": int(details.get("id", dataset.data_id)),
            "fetched_version": int(details.get("version", dataset.version)),
            "num_rows": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "scale": dataset.scale,
        }
        return DataInstance(inputs=[from_numpy(features)], meta=meta)


def _fetch_openml(data_id: int):
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise RuntimeError(
            "OpenML-backed benchmarks require scikit-learn to fetch datasets."
        ) from exc

    return fetch_openml(
        data_id=data_id,
        as_frame=False,
        parser="auto",
    )


class OpenMLDatasetBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return OpenMLDatasetGenerator()


def fetch_openml_dataset(source_name: str) -> DataInstance:
    """Fetch (and cache) a prepared OpenML dataset via the shared shell."""
    raw_generator = OpenMLDatasetGenerator()
    raw_dataset = next(
        dataset for dataset in raw_generator.datasets if dataset.name == source_name
    )
    return raw_generator.cached_generate(raw_dataset)


def fetch_openml_features(source_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch the prepared dense feature matrix for an OpenML dataset."""
    raw = fetch_openml_dataset(source_name)
    return np.asarray(to_numpy(raw.inputs[0]), dtype=np.float32), raw.meta
