#!/usr/bin/env python3
"""List benchmarks with recorded results but no success on any dataset/framework."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def summarize_results(paths: list[Path]) -> dict:
    frameworks: dict[str, Counter[str]] = {}
    benchmarks: dict[str, Counter[str]] = {}
    datasets: dict[str, set[tuple[str, str]]] = {}
    successful_datasets: set[tuple[str, str, str]] = set()

    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for benchmark in document["benchmarks"]:
            name = benchmark["name"]
            for generator in benchmark["generators"]:
                for dataset in generator["datasets"]:
                    for result in dataset["results"]:
                        framework = document["frameworks"][result["framework"]]["name"]
                        status = result["status"]
                        frameworks.setdefault(framework, Counter())[status] += 1
                        benchmarks.setdefault(name, Counter())[status] += 1
                        datasets.setdefault(name, set()).add(
                            (generator["name"], dataset["name"])
                        )
                        if status == "ok":
                            successful_datasets.add(
                                (name, generator["name"], dataset["name"])
                            )

    dataset_count = sum(len(items) for items in datasets.values())
    return {
        "result_files": [str(path) for path in paths],
        "frameworks": {
            name: dict(counts) for name, counts in sorted(frameworks.items())
        },
        "benchmark_count": len(benchmarks),
        "dataset_count": dataset_count,
        "datasets_without_success": dataset_count - len(successful_datasets),
        "benchmarks_without_success": [
            {
                "benchmark": name,
                "datasets": len(datasets[name]),
                "results": dict(counts),
            }
            for name, counts in sorted(benchmarks.items())
            if not counts["ok"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List benchmarks with recorded results but no status='ok' on any "
            "dataset/framework across the input files. Skips and nonfinite results "
            "are not successes; benchmarks without recorded results are omitted."
        )
    )
    parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="Combined results files (default: all competition/run_*/results.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Include framework status counts and dataset totals as JSON",
    )
    args = parser.parse_args()
    paths = args.results or sorted(
        (REPO_ROOT / "competition").glob("run_*/results.json")
    )
    if not paths:
        parser.error("No competition/run_*/results.json files found.")
    try:
        summary = summarize_results(paths)
    except (OSError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for benchmark in summary["benchmarks_without_success"]:
            print(benchmark["benchmark"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
