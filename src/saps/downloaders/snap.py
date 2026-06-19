from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path
from typing import IO

import numpy as np

from saps_framework.binsparse_format import BinsparseFormat

SNAP_DATA_BASE_URL = "https://snap.stanford.edu/data"


def download_snap_dataset(
    dataset_name: str,
    *,
    data_dir: str | Path | None = None,
    directed: bool = True,
    remap_nodes: bool = True,
) -> tuple[list[BinsparseFormat], dict]:
    """Download a SNAP edge-list dataset and return a sparse adjacency matrix.

    ``dataset_name`` may be a plain SNAP slug such as ``facebook_combined`` or a
    benchmark-facing name such as ``snap-facebook_combined``.
    """
    slug = _snap_slug(dataset_name)
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dataset_dir = root / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)

    edge_list_path = _ensure_downloaded(slug, dataset_dir)
    adjacency, meta = parse_snap_edge_list(
        edge_list_path, directed=directed, remap_nodes=remap_nodes
    )
    meta.update(
        {
            "dataset_name": dataset_name,
            "snap_slug": slug,
            "source_url": f"{SNAP_DATA_BASE_URL}/{slug}.txt.gz",
            "path": str(edge_list_path),
        }
    )
    return [adjacency], meta


def load_toy_dataset(
    data_dir: str | Path | None = None,
) -> tuple[list[BinsparseFormat], dict]:
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dataset_dir = root / "toy"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    gz_path = dataset_dir / "toy.txt.gz"
    txt_path = dataset_dir / "toy.txt"

    if not txt_path.exists():
        if not gz_path.exists():
            with gzip.open(gz_path, "wt", encoding="utf-8") as file:
                file.write("# SNAP edge list\n10 20\n20 40\n")

        with gzip.open(gz_path, "rb") as source, txt_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    edge_list_path = txt_path
    adjacency, meta = parse_snap_edge_list(
        edge_list_path, directed=True, remap_nodes=True
    )
    meta.update(
        {
            "dataset_name": "snap-toy",
            "snap_slug": "toy",
            "source_url": "N/A",
            "path": str(edge_list_path),
        }
    )
    return [adjacency], meta


def parse_snap_edge_list(
    path: str | Path,
    *,
    directed: bool = True,
    remap_nodes: bool = True,
) -> tuple[BinsparseFormat, dict]:
    """Parse a SNAP ``.txt`` or ``.txt.gz`` edge list into binsparse COO."""
    path = Path(path)
    edges = _read_edges(path)
    if not edges:
        raise ValueError(f"No edges found in SNAP edge list: {path}")

    rows = np.fromiter((src for src, _ in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((dst for _, dst in edges), dtype=np.int64, count=len(edges))

    if not directed:
        rows, cols = (
            np.concatenate((rows, cols)),
            np.concatenate((cols, rows)),
        )

    if remap_nodes:
        original_nodes, inverse = np.unique(
            np.concatenate((rows, cols)), return_inverse=True
        )
        edge_count = len(rows)
        rows = inverse[:edge_count]
        cols = inverse[edge_count:]
        node_count = len(original_nodes)
        raw_node_ids = original_nodes
    else:
        raw_node_ids = np.unique(np.concatenate((rows, cols)))
        node_count = int(max(rows.max(), cols.max())) + 1

    values = np.ones(rows.shape, dtype=bool)
    adjacency = BinsparseFormat.from_coo((rows, cols), values, (node_count, node_count))
    meta = {
        "directed": directed,
        "num_nodes": node_count,
        "num_edges": len(edges),
        "num_matrix_entries": len(rows),
        "remap_nodes": remap_nodes,
        "raw_node_ids": raw_node_ids,
        "src": 0,
    }
    return adjacency, meta


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "snap"


def _snap_slug(dataset_name: str) -> str:
    name = dataset_name.strip()
    for prefix in ("snap://", "snap:", "snap/", "snap-", "snap_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.removesuffix(".txt").removesuffix(".gz")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Invalid SNAP dataset name: {dataset_name!r}")
    return name


def _ensure_downloaded(slug: str, dataset_dir: Path) -> Path:
    gz_path = dataset_dir / f"{slug}.txt.gz"
    txt_path = dataset_dir / f"{slug}.txt"

    if txt_path.exists():
        return txt_path
    if not gz_path.exists():
        url = f"{SNAP_DATA_BASE_URL}/{slug}.txt.gz"
        tmp_path = gz_path.with_suffix(gz_path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(gz_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    with gzip.open(gz_path, "rb") as source, txt_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return txt_path


def _read_edges(path: Path) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with _open_text(path) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid edge at {path}:{line_number}: {line!r}")
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer node id at {path}:{line_number}: {line!r}"
                ) from exc
    return edges


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")
