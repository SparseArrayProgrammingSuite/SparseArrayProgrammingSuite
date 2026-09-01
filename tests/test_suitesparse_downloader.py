from types import SimpleNamespace

import numpy as np

from binsparse.conversions import from_scipy

from saps.downloaders import suitesparse


def test_download_and_read_matrix_returns_canonical_coo(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "duplicate"
    matrix_dir.mkdir()
    (matrix_dir / "duplicate.mtx").write_text(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate real general",
                "3 3 3",
                "1 1 2.0",
                "1 1 3.0",
                "2 3 4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    matrix = SimpleNamespace(name="duplicate")
    monkeypatch.setattr(
        suitesparse,
        "download_suitesparse_matrix",
        lambda name, data_dir=None: (matrix_dir, matrix),
    )

    _, _, A = suitesparse._download_and_read_matrix("duplicate", tmp_path)
    tensor = from_scipy(A)

    assert A.has_canonical_format
    assert A.nnz == 2
    assert np.array_equal(A.row, np.array([0, 1], dtype=A.row.dtype))
    assert np.array_equal(A.col, np.array([0, 2], dtype=A.col.dtype))
    assert np.array_equal(A.data, np.array([5.0, 4.0]))
    assert tensor.indices_0 is A.row
    assert tensor.indices_1 is A.col
    assert tensor.values is A.data
