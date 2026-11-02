#!/usr/bin/env python3
"""Stamp cached manifest entries and trace statistics with the current freshness
from metadata.

Use this when a source change does not affect generated data or traced behavior
(or the data can no longer be regenerated), so the existing uploads and trace
statistics should count as current.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock


def selected_datasets(
    metadata_path: Path,
    *,
    tags: list[str] = (),
    no_tags: list[str] = (),
    patterns: list[str] = (),
    no_patterns: list[str] = (),
    cacheable_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Datasets matching the filters, keyed by manifest key.

    The filters behave like run_benchmark.py's --tag, --no-tag, --re, and --no-re.
    Uncached generators are skipped unless *cacheable_only* is false.
    """
    include_tags = {tag.strip() for tag in tags if tag.strip()}
    exclude_tags = {tag.strip() for tag in no_tags if tag.strip()}
    include_patterns = [re.compile(pattern) for pattern in patterns]
    exclude_patterns = [re.compile(pattern) for pattern in no_patterns]

    def name_matches(compiled, names) -> bool:
        return any(pattern.search(name) for pattern in compiled for name in names)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = {}
    for benchmark in metadata["benchmarks"]:
        for generator in benchmark["generators"]:
            if cacheable_only and not generator["cacheable"]:
                continue
            for dataset in generator["datasets"]:
                dataset_tags = set(dataset.get("tags", []))
                names = (benchmark["name"], generator["name"], dataset["name"])
                if include_tags and dataset_tags.isdisjoint(include_tags):
                    continue
                if include_patterns and not name_matches(include_patterns, names):
                    continue
                if exclude_tags and not dataset_tags.isdisjoint(exclude_tags):
                    continue
                if exclude_patterns and name_matches(exclude_patterns, names):
                    continue
                selected[f"{generator['name']}.{dataset['name']}"] = dataset
    return selected


def mark_fresh(
    manifest_path: Path, datasets: dict[str, dict[str, Any]], *, dry_run: bool = False
) -> tuple[list[str], list[str], int]:
    """Return (marked, uncached, already_fresh) for the selected datasets.

    Only entries with an uploaded digest for the same source file are marked;
    entries that were never cached still need generating.
    """
    with FileLock(manifest_path.with_suffix(".lock")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marked, uncached, already_fresh = [], [], 0
        for key, dataset in sorted(datasets.items()):
            record = manifest.get(key)
            if (
                not record
                or "digest" not in record
                or record.get("file") != dataset["file"]
            ):
                uncached.append(key)
            elif record.get("freshness") == dataset["freshness"]:
                already_fresh += 1
            else:
                record["freshness"] = dataset["freshness"]
                marked.append(key)
        if marked and not dry_run:
            _replace_json(manifest_path, manifest)
    return marked, uncached, already_fresh


def mark_statistics_fresh(
    statistics_path: Path, datasets: dict[str, dict[str, Any]], *, dry_run: bool = False
) -> list[str]:
    """Return the trace statistics records restamped for the selected datasets.

    Only records traced from the same source file are marked.
    """
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    marked = []
    for benchmark in statistics.get("benchmarks", []):
        for generator in benchmark.get("generators", []):
            for record in generator.get("datasets", []):
                key = f"{generator['name']}.{record['name']}"
                dataset = datasets.get(key)
                if (
                    dataset is None
                    or record.get("file") != dataset["file"]
                    or record.get("freshness") in (None, dataset["freshness"])
                ):
                    continue
                record["freshness"] = dataset["freshness"]
                marked.append(f"{benchmark['name']}/{key}")
    if marked and not dry_run:
        _replace_json(statistics_path, statistics)
    return sorted(marked)


def _replace_json(path: Path, document: Any) -> None:
    with tempfile.TemporaryDirectory(prefix=".saps-", dir=path.parent) as staging:
        staging_path = Path(staging) / path.name
        staging_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mark cached manifest entries and trace statistics as fresh for the "
            "current source, without regenerating them. Regenerate metadata.json "
            "first."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata.json"),
        help="Metadata input path (default: metadata.json).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("SAPS_MANIFEST_PATH") or "manifest.json"),
        help="Manifest to update (default: SAPS_MANIFEST_PATH or manifest.json).",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("statistics.json"),
        help="Trace statistics to update, if present (default: statistics.json).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only datasets with any of these tags (can be passed more than once).",
    )
    parser.add_argument(
        "--no-tag",
        action="append",
        default=[],
        help="Skip datasets with any of these tags (can be passed more than once).",
    )
    parser.add_argument(
        "--re",
        action="append",
        default=[],
        help=(
            "Regex filter on benchmark, generator, or dataset names "
            "(can be passed more than once)."
        ),
    )
    parser.add_argument(
        "--no-re",
        action="append",
        default=[],
        help="Regex exclusion on the same names (can be passed more than once).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List entries that would be marked without changing the manifest or "
            "statistics."
        ),
    )
    args = parser.parse_args(argv)
    if not (args.tag or args.re):
        parser.error(
            "pass --tag or --re so the whole manifest is not marked by accident"
        )
    datasets = selected_datasets(
        args.metadata.expanduser(),
        tags=args.tag,
        no_tags=args.no_tag,
        patterns=args.re,
        no_patterns=args.no_re,
    )
    marked, uncached, already_fresh = mark_fresh(
        args.manifest.expanduser(), datasets, dry_run=args.dry_run
    )
    for key in marked:
        print(key)
    action = "Would mark" if args.dry_run else "Marked"
    print(
        f"{action} {len(marked)} of {len(datasets)} selected entries fresh; "
        f"{already_fresh} already fresh, {len(uncached)} not cached"
    )
    for key in uncached:
        print(f"  not cached: {key}")

    statistics_path = args.statistics.expanduser()
    if statistics_path.exists():
        # Trace statistics cover uncached generators too.
        traced = selected_datasets(
            args.metadata.expanduser(),
            tags=args.tag,
            no_tags=args.no_tag,
            patterns=args.re,
            no_patterns=args.no_re,
            cacheable_only=False,
        )
        restamped = mark_statistics_fresh(statistics_path, traced, dry_run=args.dry_run)
        for key in restamped:
            print(key)
        print(f"{action} {len(restamped)} trace statistics records fresh")
        if restamped and not args.dry_run:
            print(
                "Rerun generate_metadata.py --statistics "
                f"{statistics_path} to fold in their tags."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
