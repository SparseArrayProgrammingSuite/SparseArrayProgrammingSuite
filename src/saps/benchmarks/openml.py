from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

import numpy as np

from binsparse.conversions import from_numpy, to_numpy
from filelock import FileLock

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.storage import DEFAULT_CACHE_DIR

# OpenML source downloads also persist under `scikit_learn_data/` inside the
# shared cache. A file lock serializes scikit-learn fetches so concurrent runners
# reuse completed downloads.


class OpenMLDataset(Dataset):
    """Base Dataset for benchmarks backed by an OpenML dense feature matrix."""

    def __init__(
        self,
        name: str,
        *,
        data_id: int,
        openml_name: str,
        version: int,
        task_id: int,
        repeat: int = 0,
        fold: int = 0,
        sample: int = 0,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        scale: float = 1.0,
    ):
        self._name = name
        self.data_id = data_id
        self.openml_name = openml_name
        self.version = version
        self.task_id = task_id
        self.repeat = repeat
        self.fold = fold
        self.sample = sample
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
                "task_id": self.task_id,
                "repeat": self.repeat,
                "fold": self.fold,
                "sample": self.sample,
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
            "Downloads and caches dense OpenML datasets and their task splits. "
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
                task_id=3573,
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
                task_id=167124,
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
        train, test = _fetch_openml_task_split(dataset, num_rows=features.shape[0])

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
            "task_id": dataset.task_id,
            "repeat": dataset.repeat,
            "fold": dataset.fold,
            "sample": dataset.sample,
            "num_train": len(train),
            "num_test": len(test),
        }
        return DataInstance(
            inputs=[from_numpy(features), from_numpy(train), from_numpy(test)],
            meta=meta,
        )


def _fetch_openml_task_split(
    dataset: OpenMLDataset, *, num_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    task_url = f"https://www.openml.org/api/v1/json/task/{dataset.task_id}"
    task = json.loads(_download_text(task_url))["task"]
    inputs = {item["name"]: item for item in task["input"]}
    data_id = int(inputs["source_data"]["data_set"]["data_set_id"])
    if int(task["task_id"]) != dataset.task_id or data_id != dataset.data_id:
        raise ValueError(f"OpenML task {dataset.task_id} has an unexpected source")
    split_url = inputs["estimation_procedure"]["estimation_procedure"][
        "data_splits_url"
    ]
    return _parse_split(_download_text(split_url), dataset, num_rows=num_rows)


def _download_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _parse_split(
    text: str, dataset: OpenMLDataset, *, num_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import arff

    rows, _ = arff.loadarff(StringIO(text))
    selected = (rows["repeat"] == dataset.repeat) & (rows["fold"] == dataset.fold)
    if "sample" in rows.dtype.names:
        selected &= rows["sample"] == dataset.sample
    elif dataset.sample != 0:
        raise ValueError(
            f"OpenML task {dataset.task_id} has no sample {dataset.sample}"
        )
    rows = rows[selected]
    if not np.all(np.isin(rows["type"], [b"TRAIN", b"TEST"])):
        raise ValueError("OpenML split contains an unknown partition type")
    train = rows["rowid"][rows["type"] == b"TRAIN"]
    test = rows["rowid"][rows["type"] == b"TEST"]
    # This also rejects duplicates, overlap, fractional IDs, and out-of-range IDs.
    if (
        not len(train)
        or not len(test)
        or not np.array_equal(
            np.sort(np.concatenate([train, test])), np.arange(num_rows)
        )
    ):
        raise ValueError(
            f"OpenML task {dataset.task_id}, repeat {dataset.repeat}, "
            f"fold {dataset.fold}, sample {dataset.sample} must partition "
            f"all {num_rows} rows into disjoint train/test sets"
        )
    return train.astype(np.int64), test.astype(np.int64)


def _fetch_openml(data_id: int):
    try:
        from sklearn.datasets import _openml
    except ImportError as exc:
        raise RuntimeError(
            "OpenML-backed benchmarks require scikit-learn to fetch datasets."
        ) from exc

    data_home = (
        Path(os.environ.get("SAPS_CACHE_DIR") or DEFAULT_CACHE_DIR)
        / "scikit_learn_data"
    )
    data_home.mkdir(parents=True, exist_ok=True)
    with FileLock(data_home / ".lock"):
        original_download = _openml._download_data_to_bunch
        original_urlopen = _openml.urlopen

        def download_with_cache_buster(url: str, *args: Any, **kwargs: Any):
            separator = "&" if "?" in url else "?"
            return original_download(
                f"{url}{separator}nocache={uuid4().hex}", *args, **kwargs
            )

        def urlopen_without_compression(request: Any, *args: Any, **kwargs: Any):
            if "nocache=" in request.full_url:
                request.remove_header("Accept-encoding")
                request.add_header("Accept-encoding", "identity")
            return original_urlopen(request, *args, **kwargs)

        _openml._download_data_to_bunch = download_with_cache_buster
        _openml.urlopen = urlopen_without_compression
        try:
            return _openml.fetch_openml(
                data_id=data_id,
                data_home=str(data_home),
                as_frame=False,
                parser="auto",
            )
        finally:
            _openml._download_data_to_bunch = original_download
            _openml.urlopen = original_urlopen


class OpenMLDatasetBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return OpenMLDatasetGenerator()


def fetch_openml_dataset(source_name: str) -> DataInstance:
    """Fetch (and cache) a prepared OpenML dataset via the shared shell."""
    raw_generator = OpenMLDatasetGenerator()
    raw_dataset = next(
        (dataset for dataset in raw_generator.datasets if dataset.name == source_name),
        None,
    )
    if raw_dataset is None:
        raise ValueError(
            f"Dataset {source_name!r} is not listed in OpenMLDatasetGenerator.datasets."
        )
    return raw_generator.cached_generate(raw_dataset)


def fetch_openml_features(source_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch the prepared dense feature matrix for an OpenML dataset."""
    raw = fetch_openml_dataset(source_name)
    return np.asarray(to_numpy(raw.inputs[0]), dtype=np.float32), raw.meta


def fetch_openml_train_test_features(
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fetch the shared feature matrix partitioned by its cached OpenML task indices."""
    raw = fetch_openml_dataset(source_name)
    if len(raw.inputs) != 3:
        raise RuntimeError(
            f"Cached OpenML dataset {source_name!r} lacks task splits. "
            "Refresh openml_dataset with --cache-datasets before benchmarking."
        )
    features, train, test = (to_numpy(value) for value in raw.inputs)
    return features[train], features[test], raw.meta
