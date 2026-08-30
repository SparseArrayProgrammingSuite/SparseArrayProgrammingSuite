from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path


@cache
def _metadata_document(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def committed_dataset_metadata(
    generator_name: str,
    dataset_name: str,
    *,
    benchmark_name: str | None = None,
) -> dict:
    """Return dataset provenance recorded by the metadata refresh step."""
    metadata_path = os.environ.get("SAPS_METADATA_PATH")
    if not metadata_path:
        repo_root = Path(os.environ.get("SAPS_REPO_ROOT", "."))
        metadata_path = str((repo_root / "metadata.json").resolve())

    matches: list[dict] = []
    for benchmark in _metadata_document(metadata_path).get("benchmarks", []):
        if benchmark_name is not None and benchmark["name"] != benchmark_name:
            continue
        for generator in benchmark.get("generators", []):
            if generator["name"] != generator_name:
                continue
            matches.extend(
                dataset
                for dataset in generator.get("datasets", [])
                if dataset["name"] == dataset_name
            )

    if not matches:
        key = f"{generator_name}.{dataset_name}"
        raise RuntimeError(f"Dataset {key} is missing from {metadata_path}")

    provenance_keys = ("file", "freshness", "dependencies")
    provenance = {key: matches[0].get(key) for key in provenance_keys}
    if any(
        {key: match.get(key) for key in provenance_keys} != provenance
        for match in matches[1:]
    ):
        key = f"{generator_name}.{dataset_name}"
        raise RuntimeError(f"Dataset {key} has conflicting metadata records")
    return provenance
