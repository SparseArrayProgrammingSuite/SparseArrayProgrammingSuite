#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_or_add(records, name, defaults):
    for record in records:
        if record["name"] == name:
            return record
    record = {"name": name, **defaults}
    records.append(record)
    return record


def add_statistics(target, source):
    target["statistics"] = sorted(
        {*target.get("statistics", []), *source.get("statistics", [])}
    )


def add_dataset_metadata(target, source):
    for key in ("file", "freshness"):
        if key not in source:
            continue
        if key in target and target[key] != source[key]:
            raise ValueError(
                f"conflicting dataset {key} for {target['name']}: "
                f"{target[key]} != {source[key]}"
            )
        target[key] = source[key]


def merge_statistics(paths):
    merged = {"benchmarks": []}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for source_benchmark in document.get("benchmarks", []):
            benchmark = get_or_add(
                merged["benchmarks"],
                source_benchmark["name"],
                {"statistics": [], "generators": []},
            )
            add_statistics(benchmark, source_benchmark)

            for source_generator in source_benchmark.get("generators", []):
                generator = get_or_add(
                    benchmark["generators"],
                    source_generator["name"],
                    {"statistics": [], "datasets": []},
                )
                add_statistics(generator, source_generator)

                for source_dataset in source_generator.get("datasets", []):
                    dataset = get_or_add(
                        generator["datasets"],
                        source_dataset["name"],
                        {"statistics": []},
                    )
                    add_dataset_metadata(dataset, source_dataset)
                    add_statistics(dataset, source_dataset)

    for benchmark in merged["benchmarks"]:
        benchmark["generators"].sort(key=lambda record: record["name"])
        for generator in benchmark["generators"]:
            generator["datasets"].sort(key=lambda record: record["name"])
    merged["benchmarks"].sort(key=lambda record: record["name"])
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge SAPS statistics JSON files")
    parser.add_argument(
        "-o",
        "--output",
        default=Path("statistics.json"),
        type=Path,
        help="Merged statistics output path (default: statistics.json).",
    )
    parser.add_argument("statistics", nargs="+", type=Path)
    args = parser.parse_args()

    document = merge_statistics(args.statistics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(args.statistics)} statistics files into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
