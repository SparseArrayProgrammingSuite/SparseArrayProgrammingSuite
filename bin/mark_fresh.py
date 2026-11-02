#!/usr/bin/env python3
"""Stamp cached manifest entries with the current freshness from metadata.

Use this when a source change does not affect generated data (or the data can
no longer be regenerated), so the existing uploads should count as current.
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
) -> dict[str, dict[str, Any]]:
    """Cacheable datasets matching the filters, keyed by manifest key.

    The filters behave like run_benchmark.py's --tag, --no-tag, --re, and --no-re.
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
            if not generator["cacheable"]:
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
            with tempfile.TemporaryDirectory(
                prefix=".saps-", dir=manifest_path.parent
            ) as staging:
                staging_path = Path(staging) / "manifest.json"
                staging_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                staging_path.replace(manifest_path)
    return marked, uncached, already_fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mark cached manifest entries as fresh for the current source, "
            "without regenerating them. Regenerate metadata.json first."
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
        help="List entries that would be marked without changing the manifest.",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
