"""Loader utilities for homogeneous Open Graph Benchmark node datasets."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
from binsparse import BinsparseTensor, COORMatrix

from saps.downloaders.cache import download_lock, source_cache_dir


@dataclass(frozen=True)
class OGBNodePropData:
    """The full graph data needed by a node-property-prediction workload."""

    name: str
    adjacency: BinsparseTensor
    features: np.ndarray
    labels: np.ndarray
    split_indices: dict[str, np.ndarray]
    num_nodes: int
    num_raw_edges: int
    num_features: int
    num_tasks: int
    num_classes: int
    num_outputs: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _OGBSource:
    """A dataset's entry in OGB's ``nodeproppred/master.csv`` (ogb 1.3.6)."""

    download_name: str
    add_inverse_edge: bool
    split: str
    num_tasks: int
    num_classes: int
    task_type: str


_OGB_SOURCES = {
    "ogbn-arxiv": _OGBSource(
        "arxiv", False, "time", 1, 40, "multiclass classification"
    ),
    "ogbn-products": _OGBSource(
        "products", True, "sales_ranking", 1, 47, "multiclass classification"
    ),
    "ogbn-proteins": _OGBSource(
        "proteins", True, "species", 112, 2, "binary classification"
    ),
}
_OGB_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/{}.zip"


def load_ogb_nodeprop_dataset(
    name: str,
    *,
    data_dir: str | Path | None = None,
) -> OGBNodePropData:
    """Download (if needed) and prepare a homogeneous OGB node dataset.

    Reads OGB's raw CSV release as the ``ogb`` package does, without it: ``ogb``
    imports PyTorch to cache its parsed copy. Downloads land where ``ogb`` puts
    them (``<root>/ogbn_arxiv/raw/...``), so either can reuse the other's. This
    adapter converts the framework-agnostic ``edge_index`` and ``node_feat``
    fields into the normalized adjacency expected by a Kipf--Welling GCN.
    """
    source = _OGB_SOURCES.get(name)
    if source is None:
        raise ValueError(
            f"Unknown OGB node dataset {name!r}; known: {sorted(_OGB_SOURCES)}."
        )
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dataset_dir = root / name.replace("-", "_")
    with download_lock(dataset_dir):
        if not (dataset_dir / "raw" / "edge.csv.gz").exists():
            _download(source, root, dataset_dir)
    return _prepare_ogb_nodeprop_dataset(name, _OGBDataset(dataset_dir, source))


def _download(source: _OGBSource, root: Path, dataset_dir: Path) -> None:
    """Fetch and unpack a release, which holds one ``download_name`` folder."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{source.download_name}.zip.tmp"
    staging = root / f"{source.download_name}.extract.tmp"
    try:
        urllib.request.urlretrieve(  # noqa: S310
            _OGB_URL.format(source.download_name), archive
        )
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as release:
            release.extractall(staging)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        (staging / source.download_name).replace(dataset_dir)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _read_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, compression="gzip", header=None).values


def _read_features(path: Path) -> np.ndarray | None:
    """Integer features as int64, others as float32, as ``ogb`` reads them."""
    if not path.exists():
        return None
    values = _read_csv(path)
    return values.astype(np.int64 if "int" in str(values.dtype) else np.float32)


class _OGBDataset:
    """The parts of ``ogb.nodeproppred.NodePropPredDataset`` SAPS reads, for
    homogeneous CSV releases (``ogb.io.read_graph_raw.read_csv_graph_raw``)."""

    def __init__(self, dataset_dir: Path, source: _OGBSource) -> None:
        self.dataset_dir = dataset_dir
        self.num_tasks = source.num_tasks
        self.num_classes = source.num_classes
        self.task_type = source.task_type
        self.meta_info = {"add_inverse_edge": source.add_inverse_edge}
        self.split = source.split

        raw = dataset_dir / "raw"
        edge = _read_csv(raw / "edge.csv.gz").T.astype(np.int64)
        (num_nodes,) = _read_csv(raw / "num-node-list.csv.gz").astype(np.int64)[:, 0]
        edge_feat = _read_features(raw / "edge-feat.csv.gz")
        if source.add_inverse_edge:
            # Each edge followed by its reverse.
            reversed_edge = edge[::-1]
            edge = np.repeat(edge, 2, axis=1)
            edge[:, 1::2] = reversed_edge
            if edge_feat is not None:
                edge_feat = np.repeat(edge_feat, 2, axis=0)
        self.graph = {
            "edge_index": edge,
            "edge_feat": edge_feat,
            "node_feat": _read_features(raw / "node-feat.csv.gz"),
            "num_nodes": int(num_nodes),
        }
        self.labels = _read_csv(raw / "node-label.csv.gz")

    def __getitem__(self, index: int) -> tuple[dict[str, Any], np.ndarray]:
        assert index == 0, "OGB node datasets hold one graph."
        return self.graph, self.labels

    def get_idx_split(self) -> dict[str, np.ndarray]:
        path = self.dataset_dir / "split" / self.split
        return {
            split: _read_csv(path / f"{split}.csv.gz").T[0]
            for split in ("train", "valid", "test")
        }


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
        num_raw_edges=int(edge_index.shape[1]),
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


def _default_data_dir() -> Path:
    return source_cache_dir("ogb")
