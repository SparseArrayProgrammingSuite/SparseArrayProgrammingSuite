import gzip
import importlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from filelock import FileLock

from saps.downloaders import (
    cache,
    ewap,
    frostt,
    gcare,
    kaggle,
    mccomp,
    nemo,
    ogb,
    slicot,
    snap,
)
from saps.storage import DEFAULT_CACHE_DIR


@pytest.mark.parametrize(
    "source", ["ewap", "frostt", "gcare", "mccomp", "nemo", "ogb", "slicot", "snap"]
)
@pytest.mark.parametrize("override", [None, "", "shared-cache", "~/shared-cache"])
def test_source_cache_location(monkeypatch, tmp_path, source, override):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SAPS_CACHE_DIR", raising=False)
    if override is not None:
        monkeypatch.setenv("SAPS_CACHE_DIR", override)
    module = importlib.import_module(f"saps.downloaders.{source}")
    assert (
        module._default_data_dir()
        == Path(override or DEFAULT_CACHE_DIR).expanduser() / source
    )


@pytest.fixture
def blocked_waiter(monkeypatch):
    blocked = Event()

    class ObservedFileLock(FileLock):
        def _acquire(self):
            super()._acquire()
            if not self.is_locked:
                blocked.set()

    monkeypatch.setattr(cache, "FileLock", ObservedFileLock)
    return blocked


def _concurrent(call):
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(call) for _ in range(2)]
        return [future.result(timeout=15) for future in futures]


@pytest.mark.parametrize(
    "source",
    ["ewap", "frostt", "mccomp", "nemo", "snap", "slicot", "slicot_collection"],
)
def test_concurrent_source_downloads_reuse_completed_file(
    monkeypatch, tmp_path, blocked_waiter, source
):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path / "shared"))
    root = tmp_path / "explicit"
    calls = {
        "ewap": lambda: ewap._ensure_downloaded(root, "seq_eth"),
        "frostt": lambda: frostt.download_frostt_tensor("test.gz", data_dir=root),
        "mccomp": lambda: mccomp._ensure_downloaded("test.cnf", root),
        "nemo": lambda: nemo._ensure_downloaded("test.gz", root),
        "snap": lambda: snap._ensure_downloaded("test", root),
        "slicot": lambda: slicot.download_slicot_problem("eady", data_dir=root),
        "slicot_collection": lambda: slicot.download_slicot_collection(data_dir=root),
    }

    def download(url, path):
        assert blocked_waiter.wait(timeout=10)
        if source == "snap":
            Path(path).write_bytes(gzip.compress(b"1 2\n"))
        elif source == "slicot":
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("eady.mat", b"complete")
        else:
            Path(path).write_bytes(b"complete")

    download_mock = Mock(side_effect=download)
    monkeypatch.setattr(urllib.request, "urlretrieve", download_mock)
    first, second = _concurrent(calls[source])
    assert first == second
    assert first.is_relative_to(root)
    assert first.read_bytes() == (b"1 2\n" if source == "snap" else b"complete")
    download_mock.assert_called_once()
    assert not (tmp_path / "shared").exists()


def test_failed_download_retries_without_publishing_partial_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path))

    def fail(url, path):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)
    with pytest.raises(OSError, match="interrupted"):
        frostt.download_frostt_tensor("test.gz")
    assert not (tmp_path / "frostt/test.gz").exists()
    assert not list(tmp_path.rglob("*.tmp"))
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda url, path: Path(path).write_bytes(b"complete"),
    )
    assert frostt.download_frostt_tensor("test.gz").read_bytes() == b"complete"


def test_concurrent_kaggle_downloads_publish_once(
    monkeypatch, tmp_path, blocked_waiter
):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path))

    def download(handle, *, output_dir):
        assert blocked_waiter.wait(timeout=10)
        output = Path(output_dir)
        output.mkdir()
        (output / "data").write_text("complete")
        assert not (tmp_path / "kaggle/owner/dataset").exists()
        return str(output)

    download_mock = Mock(side_effect=download)
    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=download_mock)
    )
    first, second = _concurrent(lambda: kaggle.download_kaggle_dataset("owner/dataset"))
    assert first == second == tmp_path / "kaggle/owner/dataset"
    assert (first / "data").read_text() == "complete"
    download_mock.assert_called_once()


def test_concurrent_gcare_downloads_and_extractions_run_once(
    monkeypatch, tmp_path, blocked_waiter
):
    downloads = []

    def download(url, path, *, hash):
        path = Path(path)
        if path.exists():
            return
        assert blocked_waiter.wait(timeout=10)
        downloads.append(path)
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo("complete.txt")
            item.size = 4
            archive.addfile(item, io.BytesIO(b"data"))

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(cached_download=download))
    extract = Mock(wraps=tarfile.TarFile.extractall)

    # Record each extraction while retaining the method's instance binding.
    def extract_all(self, *args, **kwargs):
        extract(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", extract_all)
    _concurrent(lambda: gcare._ensure_downloaded(tmp_path))
    assert len(downloads) == 3
    assert extract.call_count == 3
    for name in ("dataset", "queryset", "ground_truth"):
        assert (tmp_path / name / "complete.txt").read_bytes() == b"data"


def test_concurrent_ogb_downloads_reuse_processed_cache(
    monkeypatch, tmp_path, blocked_waiter
):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path))
    downloads = []

    def dataset(name, root):
        processed = Path(root) / name.replace("-", "_") / "processed/data_processed"
        if not processed.exists():
            assert blocked_waiter.wait(timeout=10)
            downloads.append(name)
            processed.parent.mkdir(parents=True)
            processed.write_bytes(b"complete")
        return "dataset"

    module = SimpleNamespace(
        decide_download=lambda url: False, download_url=lambda url, folder: None
    )
    monkeypatch.setitem(
        sys.modules,
        "ogb.nodeproppred",
        SimpleNamespace(NodePropPredDataset=dataset, dataset=module),
    )
    monkeypatch.setattr(ogb, "_prepare_ogb_nodeprop_dataset", lambda name, data: data)
    assert _concurrent(lambda: ogb.load_ogb_nodeprop_dataset("ogbn-test")) == [
        "dataset",
        "dataset",
    ]
    assert downloads == ["ogbn-test"]
    assert (tmp_path / "ogb/ogbn_test/processed/data_processed").exists()


def test_kaggle_failure_is_not_cached_and_explicit_root_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path / "shared"))
    root = tmp_path / "explicit"
    attempts = []

    def download(handle, *, output_dir):
        output = Path(output_dir)
        output.mkdir()
        (output / "data").write_text("complete")
        attempts.append(handle)
        if len(attempts) == 1:
            raise OSError("interrupted")
        return str(output)

    monkeypatch.setitem(
        sys.modules, "kagglehub", SimpleNamespace(dataset_download=download)
    )
    with pytest.raises(OSError, match="interrupted"):
        kaggle.download_kaggle_dataset("owner/dataset", data_dir=root)
    assert not (root / "owner/dataset").exists()
    result = kaggle.download_kaggle_dataset("owner/dataset", data_dir=root)
    assert result == root / "owner/dataset"
    assert (result / "data").read_text() == "complete"
    assert not (tmp_path / "shared").exists()
    assert not list(root.rglob(".saps-*"))
