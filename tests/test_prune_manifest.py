from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from filelock import FileLock


@pytest.fixture
def pruner():
    script = Path(__file__).resolve().parents[1] / "bin/prune_manifest.py"
    spec = importlib.util.spec_from_file_location("prune_manifest", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def documents(tmp_path):
    generator = {
        "name": "source.with.dots",
        "cacheable": True,
        "datasets": [{"name": "SNAP/matrix"}, {"name": "new"}],
    }
    metadata = {
        "benchmarks": [
            {"generators": [generator]},
            {
                "generators": [
                    generator,
                    {
                        "name": "adapter",
                        "cacheable": False,
                        "datasets": [{"name": "x"}],
                    },
                ]
            },
        ]
    }
    manifest = {
        "source.with.dots.SNAP/matrix": {
            "digest": "keep",
            "freshness": "old",
            "extra": 1,
        },
        "source.with.dots.removed": {"digest": "removed-dataset"},
        "adapter.x": {"digest": "uncached"},
        "removed.x": {"digest": "removed-generator"},
    }
    metadata_path = tmp_path / "metadata.json"
    manifest_path = tmp_path / "manifest.json"
    metadata_path.write_text(json.dumps(metadata))
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, metadata_path


def test_prune_retains_cacheable_records_unchanged(pruner, documents):
    manifest_path, metadata_path = documents
    original = json.loads(manifest_path.read_text())

    removed, retained = pruner.prune_manifest(manifest_path, metadata_path)

    assert removed == ["adapter.x", "removed.x", "source.with.dots.removed"]
    assert retained == 1
    assert json.loads(manifest_path.read_text()) == {
        "source.with.dots.SNAP/matrix": original["source.with.dots.SNAP/matrix"]
    }


def test_dry_run_and_no_op_preserve_bytes(pruner, documents):
    manifest_path, metadata_path = documents
    original = manifest_path.read_bytes()

    assert pruner.prune_manifest(manifest_path, metadata_path, dry_run=True)[1] == 1
    assert manifest_path.read_bytes() == original

    manifest_path.write_text('{"source.with.dots.new": {"digest":"keep"}}')
    original = manifest_path.read_bytes()
    assert pruner.prune_manifest(manifest_path, metadata_path) == ([], 1)
    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize(
    "document",
    [
        "invalid JSON",
        "{}",
        '{"benchmarks": {}}',
        '{"benchmarks": [{"generators": []}, {}]}',
        '{"benchmarks": [{"generators": [{"name": "x", "datasets": []}]}]}',
        '{"benchmarks": [{"generators": '
        '[{"name": "x", "cacheable": "false", "datasets": []}]}]}',
    ],
)
def test_invalid_metadata_leaves_manifest_intact(pruner, documents, document):
    manifest_path, metadata_path = documents
    original = manifest_path.read_bytes()
    metadata_path.write_text(document)

    with pytest.raises((ValueError, KeyError)):
        pruner.prune_manifest(manifest_path, metadata_path)

    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize("document", ["invalid JSON", "[]"])
def test_invalid_manifest_is_not_replaced(pruner, documents, document):
    manifest_path, metadata_path = documents
    manifest_path.write_text(document)

    with pytest.raises(ValueError):
        pruner.prune_manifest(manifest_path, metadata_path)

    assert manifest_path.read_text() == document


def test_failed_publish_preserves_manifest(pruner, documents, monkeypatch):
    manifest_path, metadata_path = documents
    original = manifest_path.read_bytes()

    def failed_replace(source, destination):
        raise OSError("Publish interrupted")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="Publish interrupted"):
        pruner.prune_manifest(manifest_path, metadata_path)

    assert manifest_path.read_bytes() == original
    assert not list(manifest_path.parent.glob(".saps-*"))


def test_prune_reads_under_uploader_lock(pruner, documents, monkeypatch):
    manifest_path, metadata_path = documents
    waiter_blocked = Event()

    class ObservedFileLock(FileLock):
        def _acquire(self):
            super()._acquire()
            if not self.is_locked:
                waiter_blocked.set()

    monkeypatch.setattr(pruner, "FileLock", ObservedFileLock)
    with ThreadPoolExecutor(max_workers=1) as workers:
        with FileLock(manifest_path.with_suffix(".lock")):
            future = workers.submit(pruner.prune_manifest, manifest_path, metadata_path)
            assert waiter_blocked.wait(timeout=10)
            manifest = json.loads(manifest_path.read_text())
            manifest["source.with.dots.new"] = {"digest": "concurrent-upload"}
            manifest_path.write_text(json.dumps(manifest))
        assert future.result()[1] == 2

    assert json.loads(manifest_path.read_text())["source.with.dots.new"] == {
        "digest": "concurrent-upload"
    }


def test_cli_uses_metadata_and_manifest_options(pruner, documents, capsys):
    manifest_path, metadata_path = documents
    original = manifest_path.read_bytes()
    args = ["--manifest", str(manifest_path), "--metadata", str(metadata_path)]

    assert pruner.main([*args, "--dry-run"]) == 0
    assert "Would remove 3 entries; retained 1" in capsys.readouterr().out
    assert manifest_path.read_bytes() == original
    assert pruner.main(args) == 0
    assert "Removed 3 entries; retained 1" in capsys.readouterr().out


def test_cli_defaults_and_manifest_environment(pruner, documents, monkeypatch):
    manifest_path, _ = documents
    monkeypatch.chdir(manifest_path.parent)
    monkeypatch.delenv("SAPS_MANIFEST_PATH", raising=False)
    assert pruner.main(["--dry-run"]) == 0

    alternate = manifest_path.with_name("alternate.json")
    manifest_path.rename(alternate)
    monkeypatch.setenv("SAPS_MANIFEST_PATH", str(alternate))
    assert pruner.main([]) == 0
    assert list(json.loads(alternate.read_text())) == ["source.with.dots.SNAP/matrix"]
