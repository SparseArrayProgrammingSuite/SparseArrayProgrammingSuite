import json

import pytest

import numpy as np

from binsparse.conversions import to_numpy, to_scipy

from saps.benchmark import Param
from saps.benchmarks.gcn import (
    GCNBenchmark,
    OGBGCNDataset,
    OGBGCNGenerator,
    gcn_reference_np,
)
from saps.benchmarks.ogb import OGBNodePropGenerator, fetch_ogb_nodeprop_dataset
from saps.downloaders.ogb import OGBNodePropData, normalized_undirected_adjacency


def test_ogb_gcn_generator_derives_dimensions_from_real_features(monkeypatch):
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    graph = OGBNodePropData(
        name="ogbn-arxiv",
        adjacency=normalized_undirected_adjacency(
            np.array([[0, 1], [1, 2]]), num_nodes=3
        ),
        features=features,
        labels=np.array([[0], [1], [0]]),
        split_indices={
            "train": np.array([0]),
            "valid": np.array([1]),
            "test": np.array([2]),
        },
        num_nodes=3,
        num_raw_edges=2,
        num_features=4,
        num_tasks=1,
        num_classes=2,
        num_outputs=2,
        metadata={"dataset_name": "ogbn-arxiv", "split_sizes": {"train": 1}},
    )
    generator = OGBGCNGenerator()
    dataset = generator.datasets[0]
    dataset.hidden_dim = 5
    monkeypatch.setattr(
        "saps.benchmarks.gcn.fetch_ogb_nodeprop_dataset", lambda _: graph
    )

    instance = generator.generate(dataset)

    np.testing.assert_array_equal(to_numpy(instance.inputs[1]), features)
    assert instance.inputs[2].shape == (4, 5)
    assert instance.inputs[4].shape == (5, 2)
    assert instance.meta["num_features"] == 4
    assert instance.meta["num_outputs"] == 2
    json.dumps(instance.meta)


def test_ogb_gcn_dataset_uses_shared_source_descriptor():
    dataset = OGBGCNGenerator().datasets[0]

    assert (dataset.source_name, dataset.hidden_dim) == ("ogbn-arxiv", 256)
    assert "feature_dim" not in dataset.metadata
    assert "out_dim" not in dataset.metadata


def test_ogb_gcn_generator_includes_supported_homogeneous_ogb_workloads():
    datasets = {dataset.source_name: dataset for dataset in OGBGCNGenerator().datasets}

    assert set(datasets) == {"ogbn-arxiv", "ogbn-products", "ogbn-proteins"}
    assert datasets["ogbn-arxiv"].suites == ["standard"]
    assert datasets["ogbn-products"].suites == ["standard"]
    assert datasets["ogbn-proteins"].suites == ["standard"]


def test_ogb_gcn_proteins_uses_task_count_for_output_width(monkeypatch):
    generator = OGBGCNGenerator()
    dataset = next(
        dataset
        for dataset in generator.datasets
        if dataset.source_name == "ogbn-proteins"
    )
    graph = OGBNodePropData(
        name="ogbn-proteins",
        adjacency=normalized_undirected_adjacency(
            np.array([[0, 1], [1, 0]]),
            num_nodes=2,
            edges_are_bidirectional=True,
        ),
        features=np.ones((2, 8), dtype=np.float32),
        labels=np.zeros((2, 112), dtype=np.float32),
        split_indices={
            "train": np.array([0]),
            "valid": np.array([]),
            "test": np.array([1]),
        },
        num_nodes=2,
        num_raw_edges=2,
        num_features=8,
        num_tasks=112,
        num_classes=2,
        num_outputs=112,
        metadata={"dataset_name": "ogbn-proteins"},
    )
    monkeypatch.setattr(
        "saps.benchmarks.gcn.fetch_ogb_nodeprop_dataset", lambda _: graph
    )

    instance = generator.generate(dataset)

    assert instance.inputs[4].shape == (256, 112)
    assert instance.inputs[5].shape == (112,)


def test_ogb_generator_runs_through_gcn_with_sparse_framework(monkeypatch):
    pytest.importorskip("array_api_compat")
    from frameworks.saps_sparse import PyDataSparseFramework

    adjacency = normalized_undirected_adjacency(np.array([[0, 1], [1, 2]]), num_nodes=3)
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    graph = OGBNodePropData(
        name="fake-ogb",
        adjacency=adjacency,
        features=features,
        labels=np.array([[0], [1], [0]]),
        split_indices={
            "train": np.array([0]),
            "valid": np.array([1]),
            "test": np.array([2]),
        },
        num_nodes=3,
        num_raw_edges=2,
        num_features=2,
        num_tasks=1,
        num_classes=2,
        num_outputs=2,
        metadata={"dataset_name": "fake-ogb"},
    )
    dataset = OGBGCNDataset(
        "fake_ogb",
        source_name="fake-ogb",
        hidden_dim=3,
        description="Tiny integration graph.",
    )
    generator = OGBGCNGenerator()
    monkeypatch.setattr(
        "saps.benchmarks.gcn.fetch_ogb_nodeprop_dataset", lambda _: graph
    )
    instance = generator.generate(dataset)
    param = Param(generator, dataset)
    benchmark = GCNBenchmark()

    benchmark.setup(param, use_cache=False, xp=PyDataSparseFramework())
    benchmark.run(param)

    output = to_numpy(benchmark._output[0])
    dense_adjacency = to_scipy(adjacency).toarray()
    arrays = [to_numpy(value) for value in instance.inputs[1:]]
    expected = gcn_reference_np(dense_adjacency, *arrays)
    np.testing.assert_allclose(output, expected, rtol=1e-5)


def test_ogb_shell_generator_round_trips_prepared_nodeprop_data(monkeypatch):
    adjacency = normalized_undirected_adjacency(np.array([[0, 1], [1, 2]]), num_nodes=3)
    graph = OGBNodePropData(
        name="ogbn-arxiv",
        adjacency=adjacency,
        features=np.arange(6, dtype=np.float32).reshape(3, 2),
        labels=np.array([[0], [1], [0]]),
        split_indices={
            "train": np.array([0]),
            "valid": np.array([1]),
            "test": np.array([2]),
        },
        num_nodes=3,
        num_raw_edges=2,
        num_features=2,
        num_tasks=1,
        num_classes=2,
        num_outputs=2,
        metadata={
            "dataset_name": "ogbn-arxiv",
            "num_tasks": 1,
            "num_classes": 2,
        },
    )
    monkeypatch.setattr(
        "saps.benchmarks.ogb.load_ogb_nodeprop_dataset", lambda _: graph
    )
    generator = OGBNodePropGenerator()
    dataset = generator.datasets[0]

    raw = generator.generate(dataset)

    assert generator.cacheable
    assert dataset.suites == []
    assert dataset.metadata["source_name"] == "ogbn-arxiv"
    assert "num_nodes" not in dataset.metadata
    assert raw.meta["num_nodes"] == 3
    assert raw.meta["num_raw_edges"] == 2
    assert raw.meta["num_features"] == 2
    assert raw.meta["num_outputs"] == 2
    assert raw.meta["split_names"] == ["train", "valid", "test"]
    np.testing.assert_array_equal(to_numpy(raw.inputs[1]), graph.features)
    np.testing.assert_array_equal(to_numpy(raw.inputs[3]), graph.split_indices["train"])


def test_fetch_ogb_nodeprop_dataset_uses_shared_cache(monkeypatch):
    adjacency = normalized_undirected_adjacency(np.array([[0], [1]]), num_nodes=2)
    raw = OGBNodePropData(
        name="ogbn-arxiv",
        adjacency=adjacency,
        features=np.ones((2, 2), dtype=np.float32),
        labels=np.array([[0], [1]]),
        split_indices={
            "train": np.array([0]),
            "valid": np.array([], dtype=np.int64),
            "test": np.array([1]),
        },
        num_nodes=2,
        num_raw_edges=1,
        num_features=2,
        num_tasks=1,
        num_classes=2,
        num_outputs=2,
        metadata={
            "dataset_name": "ogbn-arxiv",
            "num_tasks": 1,
            "num_classes": 2,
        },
    )
    calls = []
    monkeypatch.setattr(
        "saps.benchmarks.ogb.load_ogb_nodeprop_dataset",
        lambda _: raw,
    )
    shell_generator = OGBNodePropGenerator()
    instance = shell_generator.generate(
        next(
            dataset
            for dataset in shell_generator.datasets
            if dataset.source_name == "ogbn-arxiv"
        )
    )

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.source_name))
        return instance

    monkeypatch.setattr(OGBNodePropGenerator, "cached_generate", fake_cached_generate)

    graph = fetch_ogb_nodeprop_dataset("ogbn-arxiv")

    assert calls == [("ogb_nodeprop", "ogbn-arxiv")]
    assert graph.name == "ogbn-arxiv"
    assert graph.num_raw_edges == 1
    np.testing.assert_array_equal(graph.features, raw.features)
    np.testing.assert_array_equal(
        graph.split_indices["test"], raw.split_indices["test"]
    )
