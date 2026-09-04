from types import SimpleNamespace

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import DataInstance
from saps.benchmarks.openml import (
    OpenMLDatasetBenchmark,
    OpenMLDatasetGenerator,
    fetch_openml_features,
)
from saps.benchmarks.rp_kmeans_clustering import (
    RPKMeansBenchmark,
    RPKMeansDataset,
    RPKMeansOpenMLGenerator,
)


def test_openml_shell_generator_scales_features_and_records_shape(monkeypatch):
    def fake_fetch_openml(data_id):
        return SimpleNamespace(
            data=np.array([[0, 255], [128, 64]], dtype=np.uint8),
            details={"id": str(data_id), "version": "1"},
        )

    monkeypatch.setattr("saps.benchmarks.openml._fetch_openml", fake_fetch_openml)
    generator = OpenMLDatasetGenerator()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert generator.cacheable
    assert OpenMLDatasetBenchmark().generator.name == "openml_dataset"
    np.testing.assert_allclose(
        to_numpy(instance.inputs[0]),
        np.array([[0.0, 1.0], [128.0 / 255.0, 64.0 / 255.0]], dtype=np.float32),
    )
    assert instance.meta["data_id"] == 554
    assert instance.meta["openml_name"] == "mnist_784"
    assert instance.meta["version"] == 1
    assert instance.meta["fetched_data_id"] == 554
    assert instance.meta["fetched_version"] == 1
    assert instance.meta["num_rows"] == 2
    assert instance.meta["num_features"] == 2


def test_fetch_openml_features_uses_shared_cache(monkeypatch):
    raw = DataInstance(
        inputs=[from_numpy(np.ones((2, 3), dtype=np.float32))],
        meta={
            "data_id": 40927,
            "openml_name": "CIFAR_10",
            "version": 1,
            "num_rows": 2,
            "num_features": 3,
        },
    )
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.name))
        return raw

    monkeypatch.setattr(
        OpenMLDatasetGenerator, "cached_generate", fake_cached_generate
    )

    features, meta = fetch_openml_features("cifar10")

    assert calls == [("openml_dataset", "cifar10")]
    np.testing.assert_array_equal(features, np.ones((2, 3), dtype=np.float32))
    assert meta["openml_name"] == "CIFAR_10"


def test_rp_kmeans_openml_generator_derives_inputs_from_cached_source(monkeypatch):
    source_features = np.arange(20, dtype=np.float32).reshape(5, 4)

    def fake_fetch_openml_features(source_name):
        return source_features, {
            "data_id": 554,
            "openml_name": "mnist_784",
            "version": 1,
            "num_rows": source_features.shape[0],
            "num_features": source_features.shape[1],
        }

    monkeypatch.setattr(
        "saps.benchmarks.rp_kmeans_clustering.fetch_openml_features",
        fake_fetch_openml_features,
    )
    generator = RPKMeansOpenMLGenerator()
    dataset = RPKMeansDataset("mnist", k=2, eps=0.3, c=0.5, max_iter=5)

    instance = generator.generate(dataset)

    assert not generator.cacheable
    np.testing.assert_array_equal(to_numpy(instance.inputs[0]), source_features)
    assert instance.inputs[1].shape == (4, 11)
    assert instance.meta["k"] == 2
    assert instance.meta["num_rows"] == 5
    assert instance.meta["num_features"] == 4
    assert instance.meta["openml_data_id"] == 554
    assert instance.meta["source_num_rows"] == 5
    assert instance.meta["source_num_features"] == 4


def test_rp_kmeans_benchmark_uses_one_openml_generator_for_standard_datasets():
    generators = RPKMeansBenchmark().generators
    openml_generator = next(
        generator for generator in generators if generator.name == "rp_kmeans_openml"
    )

    assert [dataset.name for dataset in openml_generator.datasets] == [
        "mnist",
        "cifar10",
    ]
    assert all(dataset.suites == ["standard"] for dataset in openml_generator.datasets)
    assert "rp_kmeans_mnist" not in {generator.name for generator in generators}
    assert "rp_kmeans_cifar10" not in {generator.name for generator in generators}
