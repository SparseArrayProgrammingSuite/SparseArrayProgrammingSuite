"""Loader utilities for homogeneous Open Graph Benchmark node datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from binsparse import BinsparseTensor, COORMatrix


@dataclass(frozen=True)
class OGBNodePropData:
    """The full graph data needed by a node-property-prediction workload."""

    name: str
    adjacency: BinsparseTensor
    features: np.ndarray
    labels: np.ndarray
    split_indices: dict[str, np.ndarray]
    num_nodes: int
    num_features: int
    num_tasks: int
    num_classes: int
    num_outputs: int
    metadata: dict[str, Any]


def load_ogb_nodeprop_dataset(
    name: str,
    *,
    data_dir: str | Path | None = None,
    allow_large_download: bool | None = None,
) -> OGBNodePropData:
    """Download (if needed) and prepare a homogeneous OGB node dataset.

    The OGB package owns the raw download cache.  This adapter converts its
    framework-agnostic ``edge_index`` and ``node_feat`` fields into the
    normalized adjacency expected by a Kipf--Welling GCN.
    """
    try:
        from ogb.nodeproppred import NodePropPredDataset
        from ogb.nodeproppred import dataset as ogb_dataset_module
    except ImportError as exc:
        raise RuntimeError(
            "Loading OGB GCN datasets requires the optional 'ogb' dependency. "
            "Install the SAPS project dependencies first."
        ) from exc

    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dataset_dir = root / name.replace("-", "_")
    processed_path = dataset_dir / "processed" / "data_processed"
    raw_graph_path = dataset_dir / "raw" / "edge.csv.gz"
    needs_download = not processed_path.exists() and not raw_graph_path.exists()
    if (
        name == "ogbn-products"
        and needs_download
        and not _allow_large_download(allow_large_download)
    ):
        raise RuntimeError(
            "Downloading ogbn-products requires more than 1 GB and OGB normally "
            "prompts for confirmation. Set SAPS_ALLOW_LARGE_DOWNLOADS=1 or pass "
            "allow_large_download=True to permit the noninteractive download."
        )

    auto_accept_download = name == "ogbn-products" and needs_download
    # PyTorch 2.6 rejects OGB's NumPy-based processed cache by default.
    torch_cache_variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous_torch_cache_value = os.environ.get(torch_cache_variable)
    using_processed_cache = processed_path.exists()
    if using_processed_cache:
        os.environ[torch_cache_variable] = "1"
    # OGB prompts before downloading Products.
    original_decide_download = ogb_dataset_module.decide_download
    if auto_accept_download:
        ogb_dataset_module.decide_download = lambda _url: True
    try:
        dataset = NodePropPredDataset(name=name, root=str(root))
    finally:
        ogb_dataset_module.decide_download = original_decide_download
        if using_processed_cache:
            if previous_torch_cache_value is None:
                os.environ.pop(torch_cache_variable, None)
            else:
                os.environ[torch_cache_variable] = previous_torch_cache_value
    return _prepare_ogb_nodeprop_dataset(name, dataset)


def _prepare_ogb_nodeprop_dataset(name: str, dataset: Any) -> OGBNodePropData:
    """Convert an already-loaded OGB node-property dataset into SAPS inputs."""
    graph, labels = dataset[0]

    if "edge_index" not in graph:
        raise ValueError(f"OGB dataset {name!r} is not a homogeneous graph.")

    edge_index = np.asarray(graph["edge_index"])
    num_nodes = int(graph["num_nodes"])
    features, feature_source = _node_features(graph, edge_index, num_nodes, name)
    if features.ndim != 2 or features.shape[0] != num_nodes:
        raise ValueError(f"OGB dataset {name!r} has invalid node features.")
    if feature_source == "mean_edge_feat":
        # Release large Proteins edge features before adjacency normalization.
        graph["edge_feat"] = None

    inverse_edges = getattr(dataset, "meta_info", {}).get("add_inverse_edge", False)
    edges_are_bidirectional = inverse_edges is True or str(inverse_edges) == "True"
    adjacency = normalized_undirected_adjacency(
        edge_index,
        num_nodes,
        edges_are_bidirectional=edges_are_bidirectional,
    )
    split_indices = {
        split: np.asarray(indices, dtype=np.int64)
        for split, indices in dataset.get_idx_split().items()
    }
    normalized_edges = int(adjacency.number_of_stored_values)
    task_type = str(getattr(dataset, "task_type", ""))
    num_tasks = int(dataset.num_tasks)
    num_classes = int(dataset.num_classes)
    num_outputs = (
        num_tasks
        if "binary" in task_type or "regression" in task_type or num_tasks > 1
        else num_classes
    )
    metadata = {
        "dataset_name": name,
        "source": "Open Graph Benchmark",
        "source_url": "https://ogb.stanford.edu/docs/nodeprop/",
        "num_nodes": num_nodes,
        "num_raw_edges": int(edge_index.shape[1]),
        "num_normalized_edges": normalized_edges,
        "num_features": int(features.shape[1]),
        "num_tasks": num_tasks,
        "num_classes": num_classes,
        "num_outputs": num_outputs,
        "task_type": task_type,
        "feature_source": feature_source,
        "split_sizes": {
            split: int(indices.size) for split, indices in split_indices.items()
        },
        "normalization": "symmetric; bidirectional edges; self-loops",
        "dtype": "float32",
    }
    return OGBNodePropData(
        name=name,
        adjacency=adjacency,
        features=features,
        labels=np.asarray(labels),
        split_indices=split_indices,
        num_nodes=num_nodes,
        num_features=int(features.shape[1]),
        num_tasks=num_tasks,
        num_classes=num_classes,
        num_outputs=num_outputs,
        metadata=metadata,
    )


def normalized_undirected_adjacency(
    edge_index: np.ndarray,
    num_nodes: int,
    *,
    edges_are_bidirectional: bool = False,
) -> BinsparseTensor:
    """Return ``D^-1/2 (A + I) D^-1/2`` as a COO binsparse matrix.

    OGB citation graphs are directed.  Scorch's GCN benchmark first makes them
    bidirectional and removes duplicates, so SAPS uses the same preprocessing.
    """
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    raw_rows = edge_index[0]
    raw_cols = edge_index[1]
    if (
        np.any(raw_rows < 0)
        or np.any(raw_cols < 0)
        or np.any(raw_rows >= num_nodes)
        or np.any(raw_cols >= num_nodes)
    ):
        raise ValueError("edge_index contains node IDs outside num_nodes.")

    from scipy.sparse import coo_matrix

    adjacency = coo_matrix(
        (np.ones(raw_rows.size, dtype=np.bool_), (raw_rows, raw_cols)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    if not edges_are_bidirectional:
        adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(True)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()

    degree = np.diff(adjacency.indptr).astype(np.float32)
    inv_sqrt_degree = np.zeros(num_nodes, dtype=np.float32)
    np.divide(1.0, np.sqrt(degree), out=inv_sqrt_degree, where=degree != 0)
    coo = adjacency.tocoo(copy=False)
    rows, cols = coo.row, coo.col
    values = inv_sqrt_degree[rows] * inv_sqrt_degree[cols]
    return COORMatrix(
        (num_nodes, num_nodes),
        len(values),
        indices_0=rows,
        indices_1=cols,
        values=values,
    )


# Proteins derives node features from edge features.
def _node_features(
    graph: dict[str, Any], edge_index: np.ndarray, num_nodes: int, name: str
) -> tuple[np.ndarray, str]:
    """Return provided node features or OGB Proteins-style aggregated edge features."""
    node_features = graph.get("node_feat")
    if node_features is not None:
        features = np.asarray(node_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != num_nodes:
            raise ValueError(f"OGB dataset {name!r} has invalid node features.")
        return features, "node_feat"

    edge_features = graph.get("edge_feat")
    if edge_features is None:
        raise ValueError(
            f"OGB dataset {name!r} has neither node features nor edge features."
        )
    edge_features = np.asarray(edge_features, dtype=np.float32)
    if edge_features.ndim != 2 or edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError(f"OGB dataset {name!r} has invalid edge features.")

    targets = edge_index[1]
    features = np.zeros((num_nodes, edge_features.shape[1]), dtype=np.float32)
    for feature_index in range(edge_features.shape[1]):
        features[:, feature_index] = np.bincount(
            targets,
            weights=edge_features[:, feature_index],
            minlength=num_nodes,
        )
    degree = np.bincount(targets, minlength=num_nodes).astype(np.float32)
    np.divide(features, degree[:, None], out=features, where=degree[:, None] != 0)
    return features, "mean_edge_feat"


def _allow_large_download(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("SAPS_ALLOW_LARGE_DOWNLOADS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "ogb"
