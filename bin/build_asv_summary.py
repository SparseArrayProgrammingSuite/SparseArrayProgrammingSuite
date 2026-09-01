#!/usr/bin/env python3
"""Turn ASV result JSONs into the CSV used by bin/build_plotly.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _framework_name(data: dict) -> str:
    path = data.get("env_vars", {}).get("SAPS_FRAMEWORK")
    if not path:
        return "unknown"
    return Path(path).stem.removeprefix("saps_")


def _metric_name(key: str) -> str | None:
    method = key.rsplit(".", 1)[-1]
    if method.startswith("time_"):
        return "time"
    if method.startswith("peakmem_"):
        return "memory"
    return None


def _benchmark_name(key: str) -> str:
    module, cls, _method = key.rsplit(".", 2)
    return f"{module}.{cls}"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _record_tags(record: dict) -> set[str]:
    # Same three fields run_benchmark.py's --tag/--no-tag filters look at.
    fields = ("suites", "statistics", "topics")
    return {tag for field in fields for tag in record.get(field) or []}


def _load_tags(paths: list[Path]) -> dict[tuple[str, str], set[str]]:
    """Map (benchmark module, "<generator>.<dataset>") to its inherited tags.

    A dataset inherits the tags of its generator and benchmark. Only
    metadata.json carries the benchmark file, so it must be read first for
    statistics.json's benchmark names to resolve to the same module.
    """
    modules: dict[str, str] = {}
    tags: dict[tuple[str, str], set[str]] = {}
    for path in (path for path in paths if path.exists()):
        for bench in json.loads(path.read_text(encoding="utf-8"))["benchmarks"]:
            name = bench.get("name", "")
            if bench.get("file"):
                modules[name] = Path(bench["file"]).stem
            for gen in bench.get("generators", []):
                for data in gen.get("datasets", []):
                    key = (modules.get(name, name), f"{gen['name']}.{data['name']}")
                    inherited = _record_tags(bench) | _record_tags(gen)
                    tags.setdefault(key, set()).update(inherited | _record_tags(data))
    return tags


def _read_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    framework = _framework_name(data)
    rows: list[dict] = []

    for key, payload in data.get("results", {}).items():
        metric = _metric_name(key)
        if metric is None:
            continue
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        values, params = payload[0], payload[1]
        if (
            not isinstance(values, list)
            or not params
            or not isinstance(params[0], list)
        ):
            continue

        for dataset, value in zip(params[0], values, strict=False):
            if not _is_number(value):
                continue
            rows.append(
                {
                    "benchmark": _benchmark_name(key),
                    "dataset": str(dataset),
                    "framework": framework,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build asv_summary.csv for bin/build_plotly.py."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="ASV result JSON file(s). Defaults to all JSONs under --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".saps/outputs/report_sources/warmup_full"),
        help="Directory to search for ASV result JSONs when --input is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".saps/outputs/warmup_full_visualizations/asv_summary.csv"),
        help="CSV path read by bin/build_plotly.py.",
    )
    args = parser.parse_args()

    sources = args.input or sorted(
        path
        for path in args.results_dir.rglob("*.json")
        if path.name != "benchmarks_meta.json"
    )
    if not sources:
        raise SystemExit(f"No ASV result JSON files found under {args.results_dir}")

    rows: list[dict] = []
    for source in sources:
        rows.extend(_read_json(source))
    if not rows:
        raise SystemExit("No successful result rows found")

    tags = _load_tags([Path("metadata.json"), Path("statistics.json")])
    for row in rows:
        key = (row["benchmark"].split(".")[0], row["dataset"])
        row["tags"] = ";".join(sorted(tags.get(key, ())))

    fieldnames = ["benchmark", "dataset", "framework", "metric", "value", "tags"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
