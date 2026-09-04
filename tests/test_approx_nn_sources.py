import numpy as np
import scipy.sparse

from binsparse.conversions import to_numpy, to_scipy

from saps.benchmarks.approx_nn import (
    JLApproxNearestNeighbor,
    JLApproxNNDataset,
    JLApproxNNNetflixGenerator,
    JLApproxNNOpenMLGenerator,
)


def test_jl_approx_nn_openml_generator_uses_shared_shell(monkeypatch):
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
    generator = JLApproxNNOpenMLGenerator()
    dataset = JLApproxNNDataset("mnist", k=2, eps=0.3, seed=0)

    instance = generator.generate(dataset)

    assert not generator.cacheable
    assert {tuple(row) for row in to_numpy(instance.inputs[0])} == {
        tuple(row) for row in features
    }
    assert {tuple(row) for row in to_numpy(instance.inputs[1])} == {
        tuple(row) for row in features
    }
    assert instance.inputs[2].shape == (4, 28)
    assert instance.meta["num_train"] == 12
    assert instance.meta["num_query"] == 12
    assert instance.meta["num_features"] == 4
    assert instance.meta["openml_data_id"] == 554


def test_jl_approx_nn_netflix_generator_uses_shared_shell(monkeypatch):
    source = scipy.sparse.csr_matrix(
        np.arange(30, dtype=np.float32).reshape(6, 5)
    )

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
    generator = JLApproxNNNetflixGenerator()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert not generator.cacheable
    assert dataset.suites == ["standard"]
    assert to_scipy(instance.inputs[0]).shape == (6, 5)
    assert to_scipy(instance.inputs[1]).shape == (6, 5)
    assert instance.inputs[2].shape == (5, 20)
    assert instance.meta["num_train"] == 6
    assert instance.meta["num_query"] == 6
    assert instance.meta["source_num_ratings"] == source.nnz


def test_jl_approx_nn_benchmark_uses_openml_and_netflix_shell_generators():
    generators = JLApproxNearestNeighbor().generators
    generator_names = {generator.name for generator in generators}
    openml_generator = next(
        generator for generator in generators if generator.name == "jl_approx_nn_openml"
    )

    assert [dataset.name for dataset in openml_generator.datasets] == [
        "mnist",
        "cifar10",
    ]
    assert all(dataset.suites == ["standard"] for dataset in openml_generator.datasets)
    assert "jl_approx_nn_mnist" not in generator_names
    assert "jl_approx_nn_cifar10" not in generator_names
    assert "jl_approx_nn_netflix" in generator_names
