from __future__ import annotations

import hashlib

import saps.benchmark as saps_benchmark


def test_source_freshness_for_files_hashes_content_hashes_only():
    content_hashes = ("def456", "abc123")

    freshness = saps_benchmark._source_freshness_for_files(
        content_hashes,
        (),
    )

    expected_input = "".join(
        f"{content_hash}\0" for content_hash in sorted(content_hashes)
    )
    assert freshness == hashlib.sha256(expected_input.encode("utf-8")).hexdigest()


def test_source_freshness_for_files_hashes_dependencies(monkeypatch):
    def fake_dependency_versions(dependencies: list[str]) -> list[dict[str, str]]:
        assert dependencies == ["numpy"]
        return [{"name": "numpy", "version": "2.3.5"}]

    monkeypatch.setattr(saps_benchmark, "dependency_versions", fake_dependency_versions)

    freshness = saps_benchmark._source_freshness_for_files(
        ("source-hash",),
        ("numpy",),
    )

    expected_input = "source-hash\0numpy\0numpy==2.3.5\0"
    assert freshness == hashlib.sha256(expected_input.encode("utf-8")).hexdigest()
