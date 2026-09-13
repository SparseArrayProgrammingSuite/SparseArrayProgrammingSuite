import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import from_scipy
from filelock import FileLock

from saps.downloaders import suitesparse


class _FakeSuiteSparseMatrix(SimpleNamespace):
    def download(self, *, destpath, extract):
        self.download_args = {"destpath": destpath, "extract": extract}
        path = Path(destpath) / self.name
        path.mkdir()
        (path / f"{self.name}.mtx").write_text(f"{self.group}/{self.name}")
        return path, None


def test_download_and_read_matrix_returns_canonical_coo(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "duplicate"
    matrix_dir.mkdir()
    (matrix_dir / "duplicate.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 2.0\n"
        "1 1 3.0\n"
        "2 3 4.0\n",
        encoding="utf-8",
    )

    matrix = SimpleNamespace(name="duplicate")
    monkeypatch.setattr(
        suitesparse,
        "download_suitesparse_matrix",
        lambda name, data_dir=None: (matrix_dir, matrix),
    )

    _, _, A = suitesparse._download_and_read_matrix("test/duplicate", tmp_path)
    tensor = from_scipy(A)

    assert A.has_canonical_format
    assert A.nnz == 2
    assert np.array_equal(A.row, np.array([0, 1], dtype=A.row.dtype))
    assert np.array_equal(A.col, np.array([0, 2], dtype=A.col.dtype))
    assert np.array_equal(A.data, np.array([5.0, 4.0]))
    assert tensor.indices_0 is A.row
    assert tensor.indices_1 is A.col
    assert tensor.values is A.data


def test_download_suitesparse_matrix_requires_group_name(monkeypatch, tmp_path):
    fake_ssgetpy = SimpleNamespace(
        search=lambda **kwargs: pytest.fail("bare names should fail before search")
    )
    monkeypatch.setitem(sys.modules, "ssgetpy", fake_ssgetpy)

    with pytest.raises(ValueError, match="group/name"):
        suitesparse.download_suitesparse_matrix("m_t1", data_dir=tmp_path)


def test_download_suitesparse_matrix_uses_exact_source_name(monkeypatch, tmp_path):
    wrong = _FakeSuiteSparseMatrix(
        group="HB",
        name="gemat1",
        path=tmp_path / "wrong",
    )
    right = _FakeSuiteSparseMatrix(
        group="DNVS",
        name="m_t1",
        path=tmp_path / "right",
    )
    fake_ssgetpy = SimpleNamespace(
        search=lambda **kwargs: (
            [wrong, right] if kwargs == {"group": "DNVS", "limit": -1} else []
        )
    )
    monkeypatch.setitem(sys.modules, "ssgetpy", fake_ssgetpy)

    matrix_dir, matrix = suitesparse.download_suitesparse_matrix(
        "DNVS/m_t1", data_dir=tmp_path
    )

    assert matrix is right
    assert matrix_dir == tmp_path / "DNVS/m_t1"
    assert (matrix_dir / "m_t1.mtx").read_text() == "DNVS/m_t1"


def test_suitesparse_downloads_share_cache_and_separate_groups(monkeypatch, tmp_path):
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path / "cache"))
    matrices = [
        _FakeSuiteSparseMatrix(group=group, name="same") for group in ("A", "B")
    ]
    monkeypatch.setitem(
        sys.modules, "ssgetpy", SimpleNamespace(search=lambda **kwargs: matrices)
    )
    for matrix in matrices:
        matrix.download = Mock(wraps=matrix.download)
        for _ in range(2):
            matrix_dir, _ = suitesparse.download_suitesparse_matrix(
                f"{matrix.group}/same"
            )
            assert matrix_dir == tmp_path / "cache/suitesparse" / matrix.group / "same"
            assert (matrix_dir / "same.mtx").read_text() == f"{matrix.group}/same"
        matrix.download.assert_called_once()


def test_concurrent_suitesparse_downloads_publish_once(monkeypatch, tmp_path):
    matrix = _FakeSuiteSparseMatrix(group="test", name="small")
    monkeypatch.setitem(
        sys.modules, "ssgetpy", SimpleNamespace(search=lambda **kwargs: [matrix])
    )
    waiter_blocked = Event()
    download = matrix.download
    matrix_dir = tmp_path / "test/small"

    class ObservedFileLock(FileLock):
        def _acquire(self):
            super()._acquire()
            if not self.is_locked:
                waiter_blocked.set()

    def slow_download(**kwargs):
        result = download(**kwargs)
        assert not matrix_dir.exists()
        assert waiter_blocked.wait(timeout=10)
        return result

    monkeypatch.setattr(suitesparse, "FileLock", ObservedFileLock)
    matrix.download = Mock(side_effect=slow_download)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(
                suitesparse.download_suitesparse_matrix, "test/small", data_dir=tmp_path
            )
            for _ in range(2)
        ]
        for future in futures:
            assert future.result()[0] == matrix_dir
    matrix.download.assert_called_once()
    assert (matrix_dir / "small.mtx").read_text() == "test/small"
    assert not list(tmp_path.rglob(".saps-*"))


def test_failed_suitesparse_extraction_can_be_retried(monkeypatch, tmp_path):
    matrix = _FakeSuiteSparseMatrix(group="test", name="small")
    monkeypatch.setitem(
        sys.modules, "ssgetpy", SimpleNamespace(search=lambda **kwargs: [matrix])
    )
    download = matrix.download

    def interrupted_download(**kwargs):
        download(**kwargs)
        raise OSError("Extraction interrupted")

    monkeypatch.setattr(matrix, "download", interrupted_download)
    with pytest.raises(OSError, match="Extraction interrupted"):
        suitesparse.download_suitesparse_matrix("test/small", data_dir=tmp_path)
    assert not (tmp_path / "test/small").exists()
    assert not list(tmp_path.rglob(".saps-*"))

    monkeypatch.setattr(matrix, "download", download)
    matrix_dir, _ = suitesparse.download_suitesparse_matrix(
        "test/small", data_dir=tmp_path
    )
    assert (matrix_dir / "small.mtx").read_text() == "test/small"


def test_load_suitesparse_rhs_requires_index_for_multiple_rhs(tmp_path):
    matrix_dir = tmp_path / "multi"
    matrix_dir.mkdir()
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n3 2\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains 2 RHS vectors"):
        suitesparse.load_suitesparse_rhs(matrix_dir, "multi", expected_length=3)

    b = suitesparse.load_suitesparse_rhs(
        matrix_dir,
        "multi",
        expected_length=3,
        rhs_index=1,
    )

    assert np.array_equal(b, np.array([4.0, 5.0, 6.0]))


def test_load_suitesparse_matrix_ignores_unindexed_multiple_rhs(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "multi"
    matrix_dir.mkdir()
    (matrix_dir / "multi.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n3 2\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n",
        encoding="utf-8",
    )

    matrix = SimpleNamespace(name="multi", group="test")
    monkeypatch.setattr(
        suitesparse,
        "download_suitesparse_matrix",
        lambda name, data_dir=None: (matrix_dir, matrix),
    )

    _, b, meta = suitesparse.load_suitesparse_matrix("test/multi", data_dir=tmp_path)

    assert b is None
    assert meta["has_b_file"] is False
    assert meta["ignored_b_file"] is True
    assert "select one with rhs_index" in meta["rhs_error"]


def test_load_suitesparse_matrix_selects_rhs_index(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "multi"
    matrix_dir.mkdir()
    (matrix_dir / "multi.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n3 2\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n",
        encoding="utf-8",
    )

    matrix = SimpleNamespace(name="multi", group="test")
    monkeypatch.setattr(
        suitesparse,
        "download_suitesparse_matrix",
        lambda name, data_dir=None: (matrix_dir, matrix),
    )

    _, b, meta = suitesparse.load_suitesparse_matrix(
        "test/multi",
        data_dir=tmp_path,
        rhs_index=1,
    )

    assert np.array_equal(b, np.array([4.0, 5.0, 6.0]))
    assert meta["has_b_file"] is True
    assert "ignored_b_file" not in meta


def test_load_suitesparse_matrix_ignores_mismatched_rhs(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "bad_rhs"
    matrix_dir.mkdir()
    (matrix_dir / "bad_rhs.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n3 3 1\n1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "bad_rhs_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n2 1\n1.0\n2.0\n",
        encoding="utf-8",
    )

    matrix = SimpleNamespace(name="bad_rhs", group="test")
    monkeypatch.setattr(
        suitesparse,
        "download_suitesparse_matrix",
        lambda name, data_dir=None: (matrix_dir, matrix),
    )

    _, b, meta = suitesparse.load_suitesparse_matrix("test/bad_rhs", data_dir=tmp_path)

    assert b is None
    assert meta["has_b_file"] is False
    assert meta["ignored_b_file"] is True
    assert "expected a vector of length 3" in meta["rhs_error"]
