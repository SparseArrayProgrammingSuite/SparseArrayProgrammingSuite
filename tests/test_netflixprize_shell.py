import pytest

import numpy as np
import scipy.sparse

from binsparse.conversions import to_scipy

from saps.benchmarks.netflixprize import (
    NetflixPrizeBenchmark,
    NetflixPrizeGenerator,
    _load_netflixprize_matrix,
    _validate_netflixprize_matrix,
    fetch_netflixprize_matrix,
)


def test_netflixprize_loader_parses_combined_rating_files(tmp_path, monkeypatch):
    import kagglehub

    (tmp_path / "combined_data_1.txt").write_text(
        "1:\n10,5,2005-01-01\n20,3,2005-01-02\n2:\n10,4,2005-01-03\n"
    )
    monkeypatch.setattr(kagglehub, "dataset_download", lambda _: str(tmp_path))

    matrix = _load_netflixprize_matrix()

    assert matrix.shape == (2, 17770)
    assert matrix.nnz == 3
    np.testing.assert_array_equal(matrix.toarray()[:, :2], [[5.0, 4.0], [3.0, 0.0]])


def test_netflixprize_shell_generator_caches_prepared_matrix(monkeypatch):
    source = scipy.sparse.csr_matrix(np.array([[5, 0, 4], [0, 3, 0]], dtype=np.float32))
    monkeypatch.setattr(
        "saps.benchmarks.netflixprize._load_netflixprize_matrix", lambda: source
    )
    monkeypatch.setattr(
        "saps.benchmarks.netflixprize._validate_netflixprize_matrix", lambda _: None
    )
    generator = NetflixPrizeGenerator()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert generator.cacheable
    assert dataset.suites == []
    assert NetflixPrizeBenchmark().generator.name == "netflixprize"
    np.testing.assert_array_equal(
        to_scipy(instance.inputs[0]).toarray(), source.toarray()
    )
    assert instance.meta["num_users"] == 2
    assert instance.meta["num_movies"] == 3
    assert instance.meta["num_ratings"] == 3


def test_fetch_netflixprize_matrix_uses_shared_cache(monkeypatch):
    source = scipy.sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))

    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.name))
        return NetflixPrizeGenerator().generate(dataset)

    monkeypatch.setattr(
        "saps.benchmarks.netflixprize._load_netflixprize_matrix", lambda: source
    )
    monkeypatch.setattr(
        "saps.benchmarks.netflixprize._validate_netflixprize_matrix", lambda _: None
    )
    monkeypatch.setattr(NetflixPrizeGenerator, "cached_generate", fake_cached_generate)

    matrix, meta = fetch_netflixprize_matrix()

    assert calls == [("netflixprize", "netflix")]
    np.testing.assert_array_equal(matrix.toarray(), source.toarray())
    assert meta["num_users"] == 2


def test_netflixprize_shell_rejects_partial_matrix():
    partial = scipy.sparse.csr_matrix(np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError) as exc_info:
        _validate_netflixprize_matrix(partial)

    assert "full Netflix Prize training matrix" in str(exc_info.value)
    assert "100480507" in str(exc_info.value)
