#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock


def _records(document: dict[str, Any], field: str) -> list[dict[str, Any]]:
    records = document[field]
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"metadata {field} must be a list of objects")
    return records


def cacheable_keys(metadata_path: Path) -> set[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    keys = set()
    for benchmark in _records(metadata, "benchmarks"):
        for generator in _records(benchmark, "generators"):
            if not isinstance(generator["cacheable"], bool):
                raise ValueError("metadata cacheable must be a boolean")
            if not isinstance(generator["name"], str):
                raise ValueError("metadata generator name must be a string")
            for dataset in _records(generator, "datasets"):
                if not isinstance(dataset["name"], str):
                    raise ValueError("metadata dataset name must be a string")
                if generator["cacheable"]:
                    keys.add(f"{generator['name']}.{dataset['name']}")
    return keys


def prune_manifest(
    manifest_path: Path, metadata_path: Path, *, dry_run: bool = False
) -> tuple[list[str], int]:
    """Remove entries absent from cacheable generators in the supplied metadata."""
    active_keys = cacheable_keys(metadata_path)
    with FileLock(manifest_path.with_suffix(".lock")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        removed = sorted(manifest.keys() - active_keys)
        retained = {key: value for key, value in manifest.items() if key in active_keys}
        if removed and not dry_run:
            with tempfile.TemporaryDirectory(
                prefix=".saps-", dir=manifest_path.parent
            ) as staging:
                staging_path = Path(staging) / "manifest.json"
                staging_path.write_text(
                    json.dumps(retained, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                staging_path.replace(manifest_path)
    return removed, len(retained)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prune manifest entries unused by cacheable datasets in metadata."
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
        help="Manifest to prune (default: SAPS_MANIFEST_PATH or manifest.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List unused entries without deleting them.",
    )
    args = parser.parse_args(argv)
    removed, retained = prune_manifest(
        args.manifest.expanduser(), args.metadata.expanduser(), dry_run=args.dry_run
    )
    for key in removed:
        print(key)
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(removed)} entries; retained {retained} in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
