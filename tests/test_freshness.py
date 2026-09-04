from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

import saps.benchmarks
from saps.benchmark import Benchmark, Generator
from saps.metadata import metadata_document
from saps.storage import build_storage_backend

ROOT = Path(__file__).parents[1]
FRESHNESS_KEYS = ("file", "freshness")
pytestmark = pytest.mark.freshness


def _record_key(record: dict) -> str:
    return record["name"]


def _benchmark_instances() -> Iterator[Benchmark]:
    for module_info in pkgutil.iter_modules(saps.benchmarks.__path__):
        module = importlib.import_module(f"saps.benchmarks.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if (
                issubclass(cls, Benchmark)
                and cls is not Benchmark
                and not inspect.isabstract(cls)
            ):
                yield cls()


def _generator_classes() -> Iterator[type[Generator]]:
    for module_info in pkgutil.iter_modules(saps.benchmarks.__path__):
        module = importlib.import_module(f"saps.benchmarks.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if (
                issubclass(cls, Generator)
                and cls is not Generator
                and not inspect.isabstract(cls)
            ):
                yield cls


@cache
def _fresh_metadata_document() -> dict:
    return metadata_document([ROOT / "statistics.json"])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _freshness_record(record: dict) -> dict:
    return {key: record.get(key) for key in FRESHNESS_KEYS}


def _dataset_lookups(
    metadata: dict,
) -> tuple[dict[tuple[str, str, str], dict], dict[str, list[dict]]]:
    by_full_key: dict[tuple[str, str, str], dict] = {}
    by_manifest_key: dict[str, list[dict]] = {}
    for benchmark in metadata["benchmarks"]:
        benchmark_name = benchmark["name"]
        for generator in benchmark["generators"]:
            for dataset in generator["datasets"]:
                by_full_key[(benchmark_name, generator["name"], dataset["name"])] = (
                    dataset
                )
                manifest_key = f"{generator['name']}.{dataset['name']}"
                by_manifest_key.setdefault(manifest_key, []).append(dataset)
    return by_full_key, by_manifest_key


def _trace_dataset_lookup(metadata: dict) -> dict[tuple[str, str, str], dict]:
    datasets = {}
    for benchmark in metadata["benchmarks"]:
        benchmark_name = benchmark["name"]
        for generator in benchmark["generators"]:
            for dataset in generator["datasets"]:
                if "trace" in dataset.get("tags", []):
                    datasets[(benchmark_name, generator["name"], dataset["name"])] = (
                        dataset
                    )
    return datasets


def _statistics_dataset_lookup(statistics: dict) -> dict[tuple[str, str, str], dict]:
    datasets = {}
    for benchmark in statistics.get("benchmarks", []):
        benchmark_name = benchmark["name"]
        for generator in benchmark.get("generators", []):
            for dataset in generator.get("datasets", []):
                datasets[(benchmark_name, generator["name"], dataset["name"])] = dataset
    return datasets


def test_metadata_json_is_fresh():
    expected = _fresh_metadata_document()
    actual = _read_json(ROOT / "metadata.json")
    assert actual == expected


def test_all_concrete_generators_have_parent_benchmarks():
    parented = {
        type(generator)
        for benchmark in _benchmark_instances()
        for generator in benchmark.generators
    }
    floating = sorted(
        {
            f"{cls.__module__}.{cls.__qualname__}"
            for cls in _generator_classes()
            if cls not in parented
        }
    )
    assert not floating, "concrete generators without parent benchmarks: " + ", ".join(
        floating
    )


def test_metadata_tags_are_inherited():
    metadata = _fresh_metadata_document()
    for benchmark in metadata["benchmarks"]:
        assert set(benchmark["asv_ids"]) == {"peakmem", "time"}
        benchmark_tags = set(benchmark["tags"])
        assert benchmark_tags >= {
            *benchmark.get("suites", []),
            *benchmark.get("topics", []),
        }
        for generator in benchmark["generators"]:
            generator_tags = set(generator["tags"])
            assert generator_tags >= benchmark_tags
            assert generator_tags >= {
                *generator.get("suites", []),
                *generator.get("topics", []),
            }
            for dataset in generator["datasets"]:
                assert dataset["asv_param"] == f"{generator['name']}.{dataset['name']}"
                dataset_tags = set(dataset["tags"])
                assert dataset_tags >= generator_tags
                assert dataset_tags >= {
                    *dataset.get("suites", []),
                    *dataset.get("topics", []),
                }


def test_trace_suite_has_fresh_statistics():
    metadata = _fresh_metadata_document()
    trace_datasets = _trace_dataset_lookup(metadata)
    statistics = _read_json(ROOT / "statistics.json")
    statistic_datasets = _statistics_dataset_lookup(statistics)

    assert trace_datasets, "no trace suite datasets found in metadata"
    for key, dataset in trace_datasets.items():
        assert key in statistic_datasets, f"missing statistics dataset {key}"
        statistic_dataset = statistic_datasets[key]
        assert _freshness_record(statistic_dataset) == _freshness_record(dataset), (
            f"stale statistics dataset {key}"
        )


def test_manifest_datasets_are_known_to_benchmark_metadata():
    metadata = _fresh_metadata_document()
    _, datasets = _dataset_lookups(metadata)
    manifest = _read_json(ROOT / "manifest.json")

    for key, record in manifest.items():
        assert key in datasets, f"unknown manifest dataset {key}"
        assert record["file"] in {dataset["file"] for dataset in datasets[key]}


def test_manifest_datasets_exist_in_remote_storage():
    manifest = _read_json(ROOT / "manifest.json")
    backend = build_storage_backend(
        manifest_path=ROOT / "manifest.json",
        cache_dir=ROOT / ".saps" / "outputs" / "cache",
    )
    missing = []

    for key, record in manifest.items():
        if not backend.manifest_record_exists(key, record):
            prefix = backend.manifest_record_prefix(key, record)
            missing.append(f"{key}: {backend.uri_for_prefix(prefix)}")

    assert not missing, (
        "manifest datasets missing or inaccessible in remote storage:\n"
        + "\n".join(missing)
    )
