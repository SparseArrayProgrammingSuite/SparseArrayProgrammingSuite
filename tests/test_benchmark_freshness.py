from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import saps.benchmark as saps_benchmark


def test_file_signature_contains_content_hash_without_file_path():
    root = Path(__file__).parents[1]
    path = (root / "src" / "saps" / "benchmark.py").resolve()

    _, _, content_hash = saps_benchmark._file_signature(path)

    assert content_hash == hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


def test_source_freshness_for_files_hashes_content_hashes_only():
    content_hashes = ("def456", "abc123")

    freshness = saps_benchmark._source_freshness_for_files(
        tuple((0, 0, content_hash) for content_hash in content_hashes)
    )

    expected_input = "".join(
        f"{content_hash}\0" for content_hash in sorted(content_hashes)
    )
    assert freshness == hashlib.sha256(expected_input.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "source_file",
    [
        str((Path(__file__).parents[1] / "src" / "saps" / "benchmark.py").resolve()),
        "C:\\repo\\src\\saps\\benchmark.py",
    ],
)
def test_source_file_content_hash_rejects_absolute_file_paths(source_file: str):
    with pytest.raises(ValueError, match="Freshness source file must be relative"):
        saps_benchmark._source_file_content_hash(source_file, 0, 0)
