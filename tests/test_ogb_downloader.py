import numpy as np

from binsparse.conversions import to_scipy

from saps.downloaders.ogb import (
    _allow_large_download,
    _prepare_ogb_nodeprop_dataset,
    normalized_undirected_adjacency,
)


def _dense(binsparse):
    return to_scipy(binsparse).toarray()


class _Dataset:
    def __init__(
        self, graph, labels, *, num_tasks, num_classes, inverse_edges=False
    ) -> None:
        self.graph = graph
        self.labels = labels
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.meta_info = {"add_inverse_edge": inverse_edges}

    def __getitem__(self, index):
        assert index == 0
        return self.graph, self.labels

    def get_idx_split(self):
        return {"train": [0], "valid": [1], "test": [2]}


def _dataset(graph, labels, *, num_tasks, num_classes, inverse_edges=False):
    return _Dataset(
        graph,
        labels,
        num_tasks=num_tasks,
        num_classes=num_classes,
        inverse_edges=inverse_edges,
    )


def test_normalized_undirected_adjacency_symmetrizes_deduplicates_and_adds_loops():
    adjacency = normalized_undirected_adjacency(
        np.array([[0, 0, 1], [1, 1, 2]]), num_nodes=3
    )

    expected = np.array(
        [
            [0.5, 1 / np.sqrt(6), 0.0],
            [1 / np.sqrt(6), 1 / 3, 1 / np.sqrt(6)],
            [0.0, 1 / np.sqrt(6), 0.5],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_dense(adjacency), expected)


def test_prepare_ogb_nodeprop_dataset_uses_ogb_fields():
    graph = _prepare_ogb_nodeprop_dataset(
        "fake-graph",
        _dataset(
            {
                "edge_index": np.array([[0, 1], [1, 2]]),
                "node_feat": np.array([[1, 2], [3, 4], [5, 6]]),
                "num_nodes": 3,
            },
            np.array([[0], [1], [0]]),
            num_tasks=1,
            num_classes=2,
        ),
    )

    assert graph.features.dtype == np.float32
    assert graph.features.shape == (3, 2)
    assert graph.labels.shape == (3, 1)
    assert graph.num_tasks == 1
    assert graph.num_classes == 2
    assert graph.num_outputs == 2
    assert graph.metadata["split_sizes"] == {"train": 1, "valid": 1, "test": 1}
    assert graph.metadata["num_normalized_edges"] == 7


def test_prepare_ogb_nodeprop_dataset_derives_protein_features_from_edges():
    raw_graph = {
        "edge_index": np.array([[0, 0, 1], [1, 2, 2]]),
        "edge_feat": np.array([[2, 4], [4, 8], [6, 10]]),
        "num_nodes": 3,
    }
    graph = _prepare_ogb_nodeprop_dataset(
        "fake-proteins",
        _dataset(
            raw_graph,
            np.array([[0], [1], [0]]),
            num_tasks=112,
            num_classes=2,
        ),
    )

    np.testing.assert_array_equal(
        graph.features, np.array([[0, 0], [2, 4], [5, 9]], dtype=np.float32)
    )
    assert graph.num_outputs == 112
    assert graph.metadata["feature_source"] == "mean_edge_feat"
    assert raw_graph["edge_feat"] is None


def test_normalization_accepts_edges_ogb_already_made_bidirectional():
    adjacency = normalized_undirected_adjacency(
        np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        num_nodes=3,
        edges_are_bidirectional=True,
    )

    expected = normalized_undirected_adjacency(np.array([[0, 1], [1, 2]]), num_nodes=3)
    np.testing.assert_allclose(_dense(adjacency), _dense(expected))


def test_prepare_ogb_nodeprop_dataset_accepts_boolean_inverse_edge_metadata():
    graph = _prepare_ogb_nodeprop_dataset(
        "fake-graph",
        _dataset(
            {
                "edge_index": np.array([[0, 1], [1, 0]]),
                "node_feat": np.ones((2, 1)),
                "num_nodes": 2,
            },
            np.array([[0], [1]]),
            num_tasks=1,
            num_classes=2,
            inverse_edges=True,
        ),
    )

    assert graph.metadata["num_normalized_edges"] == 4


def test_products_download_policy_requires_explicit_permission(monkeypatch):
    monkeypatch.delenv("SAPS_ALLOW_LARGE_DOWNLOADS", raising=False)
    assert not _allow_large_download(None)
    assert not _allow_large_download(False)

    monkeypatch.setenv("SAPS_ALLOW_LARGE_DOWNLOADS", "true")
    assert _allow_large_download(None)
