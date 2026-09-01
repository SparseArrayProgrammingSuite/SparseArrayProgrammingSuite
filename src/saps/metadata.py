from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import saps.benchmarks
from saps.benchmark import Benchmark


def _record_key(record: dict[str, Any]) -> str:
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


def _statistics_records(
    statistics_paths: Iterable[Path],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    records: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for statistics_path in statistics_paths:
        document = json.loads(statistics_path.read_text(encoding="utf-8"))
        for benchmark in document.get("benchmarks", []):
            benchmark_key = (benchmark["name"],)
            records.setdefault(benchmark_key, []).append(benchmark)

            for generator in benchmark.get("generators", []):
                generator_key = (*benchmark_key, generator["name"])
                records.setdefault(generator_key, []).append(generator)

                for dataset in generator.get("datasets", []):
                    dataset_key = (*generator_key, dataset["name"])
                    records.setdefault(dataset_key, []).append(dataset)
    return records


def _statistics_tags(
    records: dict[tuple[str, ...], list[dict[str, Any]]],
    key: tuple[str, ...],
    *,
    freshness: str | None = None,
) -> set[str]:
    tags: set[str] = set()
    for record in records.get(key, []):
        if freshness is not None and record.get("freshness") not in (None, freshness):
            continue
        tags.update(record.get("statistics", []))
    return tags


def _record_tags(
    record: dict[str, Any],
    statistics_records: dict[tuple[str, ...], list[dict[str, Any]]],
    key: tuple[str, ...],
    inherited: Iterable[str] = (),
    *,
    freshness: str | None = None,
) -> list[str]:
    return sorted(
        {
            *inherited,
            *record.get("suites", []),
            *record.get("topics", []),
            *_statistics_tags(statistics_records, key, freshness=freshness),
        }
    )


def metadata_document(statistics_paths: Iterable[Path] = ()) -> dict[str, Any]:
    statistics = _statistics_records(statistics_paths)
    records: dict[str, dict[str, Any]] = {}

    for benchmark in _benchmark_instances():
        record = benchmark.metadata
        benchmark_name = record["name"]
        benchmark_class = type(benchmark)
        asv_module = benchmark_class.__module__.removeprefix("saps.benchmarks.")
        record["asv_ids"] = {
            metric: f"{asv_module}.{benchmark_class.__name__}.{metric}_{benchmark_name}"
            for metric in ("peakmem", "time")
        }
        source_generators = {
            generator.name: generator for generator in benchmark.generators
        }
        record["tags"] = _record_tags(record, statistics, (benchmark_name,))

        for generator in record["generators"]:
            generator_name = generator["name"]
            generator["cacheable"] = source_generators[generator_name].cacheable
            generator["tags"] = _record_tags(
                generator,
                statistics,
                (benchmark_name, generator_name),
                record["tags"],
            )
            for dataset in generator["datasets"]:
                dataset["asv_param"] = f"{generator_name}.{dataset['name']}"
                dataset["tags"] = _record_tags(
                    dataset,
                    statistics,
                    (benchmark_name, generator_name, dataset["name"]),
                    generator["tags"],
                    freshness=dataset["freshness"],
                )

        records.setdefault(_record_key(record), record)

    document = {"benchmarks": sorted(records.values(), key=_record_key)}
    return json.loads(json.dumps(document, sort_keys=True))


def write_metadata_document(
    metadata_path: Path, statistics_paths: Iterable[Path] = ()
) -> dict[str, Any]:
    document = metadata_document(statistics_paths)
    metadata_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document
