import pytest

import numpy as np
import scipy.sparse

from binsparse.conversions import from_numpy, to_numpy, to_scipy

from saps.benchmark import DataInstance
from saps.benchmarks.approx_nn import (
    SimHashApproxNearestNeighbor,
    SimHashApproxNNDenseGenerator,
    SimHashApproxNNDenseNetflixGenerator,
    SimHashApproxNNDenseOpenMLGenerator,
    SimHashApproxNNRandomDataset,
    SimHashApproxNNSparseGenerator,
    SimHashApproxNNSparseNetflixGenerator,
    SimHashApproxNNSparseOpenMLGenerator,
)
from saps.benchmarks.openml import OpenMLDatasetGenerator


@pytest.mark.parametrize(
    "generator_cls",
    [SimHashApproxNNDenseOpenMLGenerator, SimHashApproxNNSparseOpenMLGenerator],
)
@pytest.mark.parametrize(
    "source_name,data_id,openml_name,task_id",
    [("mnist", 554, "mnist_784", 3573), ("cifar10", 40927, "CIFAR_10", 167124)],
)
def test_simhash_approx_nn_openml_generator_uses_cached_task_split(
    monkeypatch, generator_cls, source_name, data_id, openml_name, task_id
):
    features = np.arange(48, dtype=np.float32).reshape(12, 4)
    train = np.array([8, 3, 1, 7, 6, 0, 10, 11, 4])
    query = np.array([9, 2, 5])
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.name, dataset.task_id, dataset.fold))
        return DataInstance(
            inputs=[from_numpy(value) for value in (features, train, query)],
            meta={
                "data_id": data_id,
                "openml_name": openml_name,
                "version": 1,
                "num_rows": len(features),
                "num_features": 4,
                "task_id": task_id,
                "repeat": 0,
                "fold": 0,
                "sample": 0,
            },
        )

    monkeypatch.setattr(OpenMLDatasetGenerator, "cached_generate", fake_cached_generate)
    generator = generator_cls()
    dataset = next(d for d in generator.datasets if d.name == source_name)
    instance = generator.generate(dataset)

    assert calls == [("openml_dataset", source_name, task_id, 0)]
    assert not generator.cacheable
    np.testing.assert_array_equal(to_numpy(instance.inputs[0]), features[train])
    np.testing.assert_array_equal(to_numpy(instance.inputs[1]), features[query])
    assert instance.inputs[2].shape == (
        4,
        instance.meta["n_projections"] * instance.meta["n_tables"],
    )
    assert instance.meta["n_tables"] == dataset.max_tables
    assert 0 <= instance.meta["estimated_retrieval_probability"] <= 1
    assert instance.meta["num_train"] == 9
    assert instance.meta["num_query"] == 3
    assert instance.meta["num_features"] == 4
    assert instance.meta["openml_data_id"] == data_id
    assert instance.meta["split"] == "openml_task"
    assert instance.meta["openml_task_id"] == task_id
    assert instance.meta["openml_task_repeat"] == 0
    assert instance.meta["openml_task_fold"] == 0
    assert instance.meta["openml_task_sample"] == 0


@pytest.mark.parametrize(
    "generator_cls",
    [SimHashApproxNNDenseNetflixGenerator, SimHashApproxNNSparseNetflixGenerator],
)
def test_simhash_approx_nn_netflix_generator_uses_shared_shell(
    monkeypatch, generator_cls
):
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
    assert instance.inputs[2].shape == (
        5,
        instance.meta["n_projections"] * instance.meta["n_tables"],
    )
    assert instance.meta["n_tables"] == dataset.max_tables
    assert 0 <= instance.meta["estimated_retrieval_probability"] <= 1
    assert instance.meta["num_train"] == 6
    assert instance.meta["num_query"] == 6
    assert instance.meta["source_num_ratings"] == source.nnz


def test_simhash_approx_nn_benchmark_uses_openml_and_netflix_shell_generators():
    generators = SimHashApproxNearestNeighbor().generators
    generator_names = {generator.name for generator in generators}
    for kind in ("dense", "sparse"):
        openml_generator = next(
            generator
            for generator in generators
            if generator.name == f"simhash_approx_nn_openml_{kind}"
        )
        assert [dataset.name for dataset in openml_generator.datasets] == [
            "mnist",
            "cifar10",
        ]
        assert all(
            dataset.suites == ["standard"] for dataset in openml_generator.datasets
        )
        assert f"simhash_approx_nn_netflix_{kind}" in generator_names
        assert f"simhash_projection_inputs_{kind}" in generator_names
        assert f"simhash_projection_test_inputs_{kind}" in generator_names
    assert len(generator_names) == len(generators) == 8


def test_simhash_projection_variants_share_random_inputs_and_are_reproducible():
    dataset = SimHashApproxNNRandomDataset(
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
        max_tables=7,
        max_projections=9,
        candidate_target=4,
        target_probability=0.9,
    )
    dense_generator = SimHashApproxNNDenseGenerator()
    sparse_generator = SimHashApproxNNSparseGenerator()
    dense = dense_generator.generate(dataset)
    sparse = sparse_generator.generate(dataset)
    for i in (0, 1):
        np.testing.assert_array_equal(
            to_numpy(dense.inputs[i]), to_numpy(sparse.inputs[i])
        )
    dense_projection = to_numpy(dense.inputs[2])
    sparse_projection = to_scipy(sparse.inputs[2])
    assert (
        dense_projection.shape
        == sparse_projection.shape
        == (
            12,
            dense.meta["n_projections"] * dense.meta["n_tables"],
        )
    )
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
    assert dense.meta["n_tables"] == 7
    assert 1 <= dense.meta["n_projections"] <= 9
    assert 0 <= dense.meta["estimated_retrieval_probability"] <= 1
    assert dense.meta["candidate_target"] == 4


def test_sparse_projection_has_paper_density_and_standard_gaussian_values():
    projection = SimHashApproxNNSparseGenerator().projection(512, 256, 42)
    projection = to_scipy(projection)
    assert projection.nnz / (512 * 256) == pytest.approx(1 / np.sqrt(512), abs=0.003)
    assert abs(projection.data.mean()) < 0.1
    assert projection.data.std() == pytest.approx(1, abs=0.1)
