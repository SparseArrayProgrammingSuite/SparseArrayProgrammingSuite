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

import saps.benchmarks
from saps.benchmark import Benchmark, Generator
from saps.storage import build_storage_backend

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
            generator_is_trace = benchmark_is_trace or "trace" in generator.get(
                "suites", []
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
                datasets[(benchmark_name, generator["name"], dataset["name"])] = dataset
    return datasets


@cache
def _pyproject_requirements() -> dict[str, Requirement]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    test_dependencies = (
        pyproject.get("tool", {})
        .get("poetry", {})
        .get("group", {})
        .get("test", {})
        .get("dependencies", {})
    )
    test_requirement_strings = []
    for name, constraint in test_dependencies.items():
        if isinstance(constraint, str):
            test_requirement_strings.append(f"{name}{constraint}")
        else:
            extras = ",".join(constraint.get("extras", []))
            extra_suffix = f"[{extras}]" if extras else ""
            test_requirement_strings.append(
                f"{name}{extra_suffix}{constraint['version']}"
            )
    requirement_strings = [
        *pyproject["project"].get("dependencies", []),
        *test_requirement_strings,
    ]
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, requirement_strings)
    }


def _assert_dependency_versions_declared(record: dict, artifact_name: str) -> None:
    requirements = _pyproject_requirements()
    version_records = record.get("dependency_versions")
    assert version_records is not None, (
        f"{artifact_name} does not declare dependency versions"
    )

    seen: set[str] = set()
    for version_record in version_records:
        package_name = canonicalize_name(version_record["name"])
        assert package_name not in seen, (
            f"{artifact_name} records dependency "
            f"{version_record['name']!r} more than once"
        )
        seen.add(package_name)

        requirement = requirements.get(package_name)
        assert requirement is not None, (
            f"{artifact_name} records dependency {version_record['name']!r}, "
            "but pyproject.toml does not declare it for test runs"
        )
        assert Version(version_record["version"]) in requirement.specifier, (
            f"{artifact_name} records {version_record['name']}=="
            f"{version_record['version']}, which does not match pyproject.toml "
            f"requirement {requirement}"
        )


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
        _assert_dependency_versions_declared(
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
        _assert_dependency_versions_declared(record, f"manifest.json:{key}")


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
