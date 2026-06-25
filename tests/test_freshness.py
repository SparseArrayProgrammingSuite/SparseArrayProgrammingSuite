from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import tomllib
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import saps
import saps.benchmarks
from saps.benchmark import Benchmark
from saps.dependencies import dependency_versions

ROOT = Path(__file__).parents[1]
FRESHNESS_KEYS = ("file", "freshness", "dependencies")


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


@cache
def _fresh_metadata_document() -> dict:
    records: dict[str, dict] = {}
    for benchmark in _benchmark_instances():
        record = benchmark.metadata
        records.setdefault(_record_key(record), record)
    document = {"benchmarks": sorted(records.values(), key=_record_key)}
    return json.loads(json.dumps(document, sort_keys=True))


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
        benchmark_is_trace = "trace" in benchmark.get("suites", [])
        for generator in benchmark["generators"]:
            generator_is_trace = (
                benchmark_is_trace or "trace" in generator.get("suites", [])
            )
            for dataset in generator["datasets"]:
                if generator_is_trace or "trace" in dataset.get("suites", []):
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
                datasets[(benchmark_name, generator["name"], dataset["name"])] = (
                    dataset
                )
    return datasets


@cache
def _pyproject_requirements() -> dict[str, Requirement]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement_strings = [
        *pyproject["project"].get("dependencies", []),
        *pyproject["project"]["optional-dependencies"].get("test", []),
    ]
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, requirement_strings)
    }


def _assert_dependency_versions_current(record: dict, artifact_name: str) -> None:
    expected_versions = dependency_versions(record.get("dependencies", []))
    assert record.get("dependency_versions") == expected_versions, (
        f"{artifact_name} dependency versions are stale"
    )

    requirements = _pyproject_requirements()
    for version_record in expected_versions:
        package_name = canonicalize_name(version_record["name"])
        requirement = requirements.get(package_name)
        assert requirement is not None, (
            f"{artifact_name} records dependency {version_record['name']!r}, "
            "but pyproject.toml does not pin it for test runs"
        )
        assert Version(version_record["version"]) in requirement.specifier, (
            f"{artifact_name} records {version_record['name']}=="
            f"{version_record['version']}, which does not satisfy "
            f"{requirement.specifier}"
        )


def test_metadata_json_is_fresh():
    expected = _fresh_metadata_document()
    actual = _read_json(ROOT / "metadata.json")
    assert actual == expected


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
        _assert_dependency_versions_current(
            statistic_dataset, f"statistics.json:{'.'.join(key)}"
        )


def test_manifest_freshness_matches_benchmark_metadata():
    metadata = _fresh_metadata_document()
    _, datasets = _dataset_lookups(metadata)
    manifest = _read_json(ROOT / "manifest.json")

    for key, record in manifest.items():
        assert key in datasets, f"unknown manifest dataset {key}"
        expected_records = [_freshness_record(dataset) for dataset in datasets[key]]
        assert _freshness_record(record) in expected_records
        _assert_dependency_versions_current(record, f"manifest.json:{key}")
