import io

import pytest

import numpy as np
import scipy.sparse

from binsparse.conversions import to_scipy

from saps.benchmarks import goodbooks
from saps.benchmarks.goodbooks import (
    GoodbooksRatingsGenerator,
    GoodbooksRatingsShellBenchmark,
    _load_goodbooks_matrix,
    _validate_goodbooks_matrix,
    fetch_goodbooks_matrix,
)


def _serve(monkeypatch, text):
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(text.encode())

    monkeypatch.setattr(goodbooks, "urlopen", fake_urlopen)
    return urls


def test_goodbooks_loader_parses_ratings_csv(monkeypatch):
    urls = _serve(monkeypatch, "user_id,book_id,rating\n1,3,5\n2,1,3\n1,1,4\n")

    matrix = _load_goodbooks_matrix()

    assert urls == [goodbooks._GOODBOOKS_RATINGS_URL]
    assert goodbooks._GOODBOOKS_COMMIT in urls[0]
    assert matrix.shape == (2, 3)
    assert matrix.nnz == 3
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix.toarray(), [[4.0, 0.0, 5.0], [3.0, 0.0, 0.0]])


def test_goodbooks_loader_parses_a_single_rating(monkeypatch):
    _serve(monkeypatch, "user_id,book_id,rating\n2,2,1\n")

    matrix = _load_goodbooks_matrix()

    np.testing.assert_array_equal(matrix.toarray(), [[0.0, 0.0], [0.0, 1.0]])


def test_goodbooks_shell_generator_caches_prepared_matrix(monkeypatch):
    source = scipy.sparse.csr_matrix(np.array([[5, 0, 4], [0, 3, 0]], dtype=np.float32))
    monkeypatch.setattr(goodbooks, "_load_goodbooks_matrix", lambda: source)
    monkeypatch.setattr(goodbooks, "_validate_goodbooks_matrix", lambda _: None)
    generator = GoodbooksRatingsGenerator()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert generator.cacheable
    assert dataset.suites == []
    assert GoodbooksRatingsShellBenchmark().generator.name == "goodbooks_ratings"
    np.testing.assert_array_equal(
        to_scipy(instance.inputs[0]).toarray(), source.toarray()
    )
    assert instance.meta["num_users"] == 2
    assert instance.meta["num_books"] == 3
    assert instance.meta["num_ratings"] == 3
    assert instance.meta["github_commit"] == goodbooks._GOODBOOKS_COMMIT


def test_fetch_goodbooks_matrix_uses_shared_cache(monkeypatch):
    source = scipy.sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.name))
        return GoodbooksRatingsGenerator().generate(dataset)

    monkeypatch.setattr(goodbooks, "_load_goodbooks_matrix", lambda: source)
    monkeypatch.setattr(goodbooks, "_validate_goodbooks_matrix", lambda _: None)
    monkeypatch.setattr(
        GoodbooksRatingsGenerator, "cached_generate", fake_cached_generate
    )

    matrix, meta = fetch_goodbooks_matrix()

    assert calls == [("goodbooks_ratings", "goodbooks")]
    np.testing.assert_array_equal(matrix.toarray(), source.toarray())
    assert meta["num_users"] == 2


def test_goodbooks_shell_rejects_partial_matrix():
    partial = scipy.sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError) as exc_info:
        _validate_goodbooks_matrix(partial)

    assert "full goodbooks-10k ratings matrix" in str(exc_info.value)
    assert "5976479" in str(exc_info.value)
