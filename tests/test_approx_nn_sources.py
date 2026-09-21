import pytest

import numpy as np
import scipy.sparse

from binsparse.conversions import to_numpy, to_scipy

from saps.benchmarks.approx_nn import (
    JLApproxNearestNeighbor,
    JLApproxNNDataset,
    JLApproxNNDenseGenerator,
    JLApproxNNDenseNetflixGenerator,
    JLApproxNNDenseOpenMLGenerator,
    JLApproxNNRandomDataset,
    JLApproxNNSparseGenerator,
    JLApproxNNSparseNetflixGenerator,
    JLApproxNNSparseOpenMLGenerator,
)


@pytest.mark.parametrize(
    "generator_cls", [JLApproxNNDenseOpenMLGenerator, JLApproxNNSparseOpenMLGenerator]
)
def test_jl_approx_nn_openml_generator_uses_shared_shell(monkeypatch, generator_cls):
    features = np.arange(48, dtype=np.float32).reshape(12, 4)

    def fake_fetch_openml_features(source_name):
        return features, {
            "data_id": 554,
            "openml_name": "mnist_784",
            "version": 1,
            "num_rows": features.shape[0],
            "num_features": features.shape[1],
        }

    monkeypatch.setattr(
        "saps.benchmarks.approx_nn.fetch_openml_features",
        fake_fetch_openml_features,
    )
    generator = generator_cls()
    dataset = JLApproxNNDataset("mnist", k=2, eps=0.3, seed=0)

    instance = generator.generate(dataset)

    assert not generator.cacheable
    assert {tuple(row) for row in to_numpy(instance.inputs[0])} == {
        tuple(row) for row in features
    }
    assert {tuple(row) for row in to_numpy(instance.inputs[1])} == {
        tuple(row) for row in features
    }
    assert instance.inputs[2].shape == (4, 3100)
    assert instance.meta["hash_bits"] == 31
    assert instance.meta["n_tables"] == 100
    assert instance.meta["num_train"] == 12
    assert instance.meta["num_query"] == 12
    assert instance.meta["num_features"] == 4
    assert instance.meta["openml_data_id"] == 554


@pytest.mark.parametrize(
    "generator_cls", [JLApproxNNDenseNetflixGenerator, JLApproxNNSparseNetflixGenerator]
)
def test_jl_approx_nn_netflix_generator_uses_shared_shell(monkeypatch, generator_cls):
    source = scipy.sparse.csr_matrix(np.arange(30, dtype=np.float32).reshape(6, 5))

    def fake_fetch_netflixprize_matrix():
        return source, {
            "num_users": source.shape[0],
            "num_movies": source.shape[1],
            "num_ratings": source.nnz,
        }

    monkeypatch.setattr(
        "saps.benchmarks.approx_nn.fetch_netflixprize_matrix",
        fake_fetch_netflixprize_matrix,
    )
    generator = generator_cls()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert not generator.cacheable
    assert dataset.suites == ["standard"]
    assert to_scipy(instance.inputs[0]).shape == (6, 5)
    assert to_scipy(instance.inputs[1]).shape == (6, 5)
    assert instance.inputs[2].shape == (5, 3100)
    assert instance.meta["hash_bits"] == 31
    assert instance.meta["n_tables"] == 100
    assert instance.meta["num_train"] == 6
    assert instance.meta["num_query"] == 6
    assert instance.meta["source_num_ratings"] == source.nnz


def test_jl_approx_nn_benchmark_uses_openml_and_netflix_shell_generators():
    generators = JLApproxNearestNeighbor().generators
    generator_names = {generator.name for generator in generators}
    for kind in ("dense", "sparse"):
        openml_generator = next(
            generator
            for generator in generators
            if generator.name == f"jl_approx_nn_openml_{kind}"
        )
        assert [dataset.name for dataset in openml_generator.datasets] == [
            "mnist",
            "cifar10",
        ]
        assert all(
            dataset.suites == ["standard"] for dataset in openml_generator.datasets
        )
        assert f"jl_approx_nn_netflix_{kind}" in generator_names
        assert f"jl_projection_inputs_{kind}" in generator_names
        assert f"jl_projection_test_inputs_{kind}" in generator_names
    assert len(generator_names) == len(generators) == 8


def test_jl_projection_variants_share_random_inputs_and_are_reproducible():
    dataset = JLApproxNNRandomDataset(
        "custom",
        "Custom",
        "Custom",
        [],
        9,
        12,
        3,
        2,
        0.1,
        42,
        hash_bits=5,
        n_tables=7,
        candidate_target=4,
    )
    dense_generator = JLApproxNNDenseGenerator()
    sparse_generator = JLApproxNNSparseGenerator()
    dense = dense_generator.generate(dataset)
    sparse = sparse_generator.generate(dataset)
    for i in (0, 1):
        np.testing.assert_array_equal(
            to_numpy(dense.inputs[i]), to_numpy(sparse.inputs[i])
        )
    dense_projection = to_numpy(dense.inputs[2])
    sparse_projection = to_scipy(sparse.inputs[2])
    assert dense_projection.shape == sparse_projection.shape == (12, 35)
    assert np.count_nonzero(dense_projection) == dense_projection.size
    assert 0 < sparse_projection.nnz < dense_projection.size
    np.testing.assert_array_equal(
        dense_projection, to_numpy(dense_generator.generate(dataset).inputs[2])
    )
    np.testing.assert_array_equal(
        sparse_projection.toarray(),
        to_scipy(sparse_generator.generate(dataset).inputs[2]).toarray(),
    )
    assert dense.meta == {**sparse.meta, "projection_kind": "dense"}
    assert sparse.meta["projection_kind"] == "sparse"
    assert dense.meta["hash_bits"] == 5
    assert dense.meta["n_tables"] == 7
    assert dense.meta["candidate_target"] == 4
