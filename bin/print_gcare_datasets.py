#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _gcare_datasets(metadata: dict):
    for benchmark in metadata.get("benchmarks", []):
        for generator in benchmark.get("generators", []):
            if generator.get("name") != "subgraph_gcare_inputs":
                continue
            yield from generator.get("datasets", [])


def _constructor_call(dataset: dict) -> str:
    subset_name, query_name = dataset["name"].split("/", 1)
    return (
        "SubgraphGCareDataset(\n"
        f"    subset_name={subset_name!r},\n"
        f"    query_name={query_name!r},\n"
        f"    pretty_name={dataset['pretty_name']!r},\n"
        f"    description={dataset['description']!r},\n"
        f"    tags={dataset.get('tags', [])!r},\n"
        ")"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print G-CARE benchmark_metadata.json datasets as "
            "SubgraphGCareDataset constructor calls."
        )
    )
    parser.add_argument(
        "metadata",
        nargs="?",
        default="benchmark_metadata.json",
        help="Path to benchmark metadata JSON.",
    )
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    calls = [_constructor_call(dataset) for dataset in _gcare_datasets(metadata)]

    print("[")
    for index, call in enumerate(calls):
        suffix = "," if index < len(calls) - 1 else ""
        print(f"{call}{suffix}")
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
