from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def marker():
    script = Path(__file__).resolve().parents[1] / "bin/mark_fresh.py"
    spec = importlib.util.spec_from_file_location("mark_fresh", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def documents(tmp_path):
    def dataset(name, tags):
        return {"name": name, "file": "src/gen.py", "freshness": "new", "tags": tags}

    metadata = {
        "benchmarks": [
            {
                "name": "matrix_shell",
                "generators": [
                    {
                        "name": "matrix",
                        "cacheable": True,
                        "datasets": [
                            dataset("HB/a", ["small"]),
                            dataset("HB/b", ["large"]),
                            dataset("HB/fresh", ["small"]),
                            dataset("HB/uncached", ["small"]),
                        ],
                    }
                ],
            },
            {
                "name": "spmv",
                "generators": [
                    {"name": "spmv", "cacheable": True, "datasets": [dataset("x", [])]},
                    {
                        "name": "adapter",
                        "cacheable": False,
                        "datasets": [dataset("y", [])],
                    },
                ],
            },
        ]
    }
    manifest = {
        "matrix.HB/a": {"digest": "1", "file": "src/gen.py", "freshness": "old"},
        "matrix.HB/b": {"digest": "2", "file": "src/gen.py", "freshness": "old"},
        "matrix.HB/fresh": {"digest": "3", "file": "src/gen.py", "freshness": "new"},
        "matrix.HB/uncached": {"file": "src/gen.py", "freshness": "old"},
        "spmv.x": {"digest": "4", "file": "src/gen.py", "freshness": "old"},
    }
    metadata_path = tmp_path / "metadata.json"
    manifest_path = tmp_path / "manifest.json"
    metadata_path.write_text(json.dumps(metadata))
    manifest_path.write_text(json.dumps(manifest))
    return metadata_path, manifest_path


def test_marks_only_selected_cached_entries(marker, documents):
    metadata_path, manifest_path = documents
    datasets = marker.selected_datasets(metadata_path, patterns=["^matrix$"])
    marked, uncached, already_fresh = marker.mark_fresh(manifest_path, datasets)
    assert marked == ["matrix.HB/a", "matrix.HB/b"]
    assert uncached == ["matrix.HB/uncached"]
    assert already_fresh == 1
    manifest = json.loads(manifest_path.read_text())
    assert manifest["matrix.HB/a"] == {
        "digest": "1",
        "file": "src/gen.py",
        "freshness": "new",
    }
    assert manifest["spmv.x"]["freshness"] == "old"
    assert "digest" not in manifest["matrix.HB/uncached"]


def test_tag_filters_and_dry_run(marker, documents):
    metadata_path, manifest_path = documents
    before = manifest_path.read_text()
    datasets = marker.selected_datasets(
        metadata_path, patterns=["matrix"], no_tags=["large"]
    )
    assert set(datasets) == {"matrix.HB/a", "matrix.HB/fresh", "matrix.HB/uncached"}
    assert set(marker.selected_datasets(metadata_path, tags=["large"])) == {
        "matrix.HB/b"
    }
    marked, _, _ = marker.mark_fresh(manifest_path, datasets, dry_run=True)
    assert marked == ["matrix.HB/a"]
    assert manifest_path.read_text() == before


def test_requires_a_filter(marker, documents):
    metadata_path, manifest_path = documents
    with pytest.raises(SystemExit):
        marker.main(
            ["--metadata", str(metadata_path), "--manifest", str(manifest_path)]
        )
