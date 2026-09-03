import sys
from types import SimpleNamespace

import numpy as np
import pytest

from binsparse.conversions import from_scipy

from saps.downloaders import suitesparse


class _FakeSuiteSparseMatrix(SimpleNamespace):
    def download(self, *, destpath, extract):
        self.download_args = {"destpath": destpath, "extract": extract}
        return self.path, None


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
        search=lambda **kwargs: [wrong, right]
        if kwargs == {"group": "DNVS", "limit": -1}
        else []
    )
    monkeypatch.setitem(sys.modules, "ssgetpy", fake_ssgetpy)

    matrix_dir, matrix = suitesparse.download_suitesparse_matrix(
        "DNVS/m_t1", data_dir=tmp_path
    )

    assert matrix is right
    assert matrix_dir == right.path


def test_load_suitesparse_rhs_requires_index_for_multiple_rhs(tmp_path):
    matrix_dir = tmp_path / "multi"
    matrix_dir.mkdir()
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n"
        "3 2\n"
        "1.0\n"
        "2.0\n"
        "3.0\n"
        "4.0\n"
        "5.0\n"
        "6.0\n",
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


def test_load_suitesparse_matrix_ignores_unindexed_multiple_rhs(
    monkeypatch, tmp_path
):
    matrix_dir = tmp_path / "multi"
    matrix_dir.mkdir()
    (matrix_dir / "multi.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 1\n"
        "1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n"
        "3 2\n"
        "1.0\n"
        "2.0\n"
        "3.0\n"
        "4.0\n"
        "5.0\n"
        "6.0\n",
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
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 1\n"
        "1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "multi_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n"
        "3 2\n"
        "1.0\n"
        "2.0\n"
        "3.0\n"
        "4.0\n"
        "5.0\n"
        "6.0\n",
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
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 1\n"
        "1 1 2.0\n",
        encoding="utf-8",
    )
    (matrix_dir / "bad_rhs_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n"
        "2 1\n"
        "1.0\n"
        "2.0\n",
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
