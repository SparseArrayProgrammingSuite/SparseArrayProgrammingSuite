import numpy as np

from binsparse.conversions import to_numpy

from saps.benchmarks.gcn_backward import (
    OGBGCNTrainingDataset,
    OGBGCNTrainingGenerator,
    _targets_from_ogb_labels,
)
from saps.downloaders.ogb import OGBNodePropData, normalized_undirected_adjacency


def test_ogb_gcn_backward_generator_derives_dimensions_and_targets(monkeypatch):
    adjacency = normalized_undirected_adjacency(np.array([[0, 1], [1, 2]]), num_nodes=3)
    graph = OGBNodePropData(
        name="fake-ogb",
        adjacency=adjacency,
        features=np.arange(12, dtype=np.float32).reshape(3, 4),
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
        metadata={"dataset_name": "fake-ogb", "num_normalized_edges": 7},
    )
    monkeypatch.setattr(
        "saps.benchmarks.gcn_backward.fetch_ogb_nodeprop_dataset", lambda _: graph
    )
    generator = OGBGCNTrainingGenerator()
    dataset = OGBGCNTrainingDataset(
        "fake_ogb",
        source_name="fake-ogb",
        hidden_dim=5,
        num_iterations=3,
        learning_rate=0.2,
        description="Tiny OGB training graph.",
    )

    instance = generator.generate(dataset)

    assert not generator.cacheable
    assert len(instance.inputs) == 7
    assert instance.inputs[2].shape == (4, 5)
    assert instance.inputs[4].shape == (5, 2)
    assert instance.inputs[6].shape == (3, 2)
    np.testing.assert_array_equal(
        to_numpy(instance.inputs[6]),
        np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
    )
    assert instance.meta["num_nodes"] == 3
    assert instance.meta["num_raw_edges"] == 2
    assert instance.meta["num_features"] == 4
    assert instance.meta["num_outputs"] == 2
    assert instance.meta["num_iterations"] == 3
    assert instance.meta["learning_rate"] == 0.2


def test_ogb_gcn_backward_generator_includes_supported_workloads():
    datasets = {
        dataset.source_name: dataset for dataset in OGBGCNTrainingGenerator().datasets
    }

    assert set(datasets) == {"ogbn-arxiv", "ogbn-products", "ogbn-proteins"}
    assert datasets["ogbn-arxiv"].suites == ["standard"]
    assert datasets["ogbn-products"].suites == ["standard"]
    assert datasets["ogbn-proteins"].suites == ["standard"]
    assert datasets["ogbn-products"].hidden_dim == 256


def test_ogb_gcn_backward_multitask_targets_replace_nan():
    targets = _targets_from_ogb_labels(
        np.array([[1.0, np.nan], [0.0, 1.0]]), num_outputs=2
    )

    np.testing.assert_array_equal(
        targets,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
