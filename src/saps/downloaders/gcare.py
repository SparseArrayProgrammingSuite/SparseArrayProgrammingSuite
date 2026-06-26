"""Downloader and parser for the G-CARE subgraph benchmark dataset.

Reference:
    G-CARE: A Framework for Performance Benchmarking of Cardinality Estimation
    Techniques for Subgraph Matching — Park et al., SIGMOD 2020.
    https://dl.acm.org/doi/10.1145/3318464.3389702
"""

import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from saps_framework.binsparse_format import BinsparseFormat


def list_gcare_queries(
    dataset_name: str, data_dir: str | Path | None = None
) -> list[str]:
    """Return query identifiers for *dataset_name* as POSIX paths relative to
    ``queryset/<dataset_name>/``, without the ``.txt`` extension.

    Example: ``"p0/query001"``
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    _ensure_downloaded(root)
    _, queryset_dir, _ = _get_dirs(root)
    base = queryset_dir / dataset_name
    return [p.relative_to(base).with_suffix("").as_posix() for p in base.rglob("*.txt")]


def load_gcare_graph(
    dataset_name: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[BinsparseFormat], dict[str, Any]]:
    """Download (if needed) and parse the G-CARE data-graph for *dataset_name*.

    Returns ``(bin_mats, graph_meta)`` where *bin_mats* is the list of graph
    matrices as :class:`BinsparseFormat` and *graph_meta* contains
    ``"matrix_names"``, ``"max_vid"``, and ``"continous_label"``.
    Pass both directly to :func:`load_gcare_query` to build per-query inputs.
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    _ensure_downloaded(root)
    dataset_dir, _, _ = _get_dirs(root)
    max_vid, continous_label, raw_sp_mats = _parse_graph(
        dataset_dir / dataset_name / f"{dataset_name}.txt"
    )
    matrix_names: list[str] = []
    bin_mats: list[BinsparseFormat] = []
    for name, raw in raw_sp_mats.items():
        matrix_names.append(name)
        bin_mats.append(
            BinsparseFormat.from_coo(raw["I_tuple"], raw["V"], raw["shape"])
        )
    meta = {
        "matrix_names": matrix_names,
        "max_vid": max_vid,
        "continous_label": continous_label,
    }
    return bin_mats, meta


def load_gcare_query(
    dataset_name: str,
    query_rel_path: str,
    bin_mats: list[BinsparseFormat],
    graph_meta: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[BinsparseFormat], dict]:
    """Build the benchmark input for a single query.

    *query_rel_path* is a POSIX path relative to ``queryset/<dataset_name>/``
    without the ``.txt`` extension, as returned by :func:`list_gcare_queries`.

    *bin_mats* and *graph_meta* must come from :func:`load_gcare_graph` for
    the same *dataset_name* (*graph_meta* must contain ``"matrix_names"``,
    ``"max_vid"``, and ``"continous_label"``).

    Returns ``(bin_mats, meta)`` where *meta* contains scalar fields ``"expr"``,
    ``"gt"``, ``"name"``, and a list ``"matrix_names"`` parallel to *bin_mats*.
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    _ensure_downloaded(root)
    _, queryset_dir, ground_truth_dir = _get_dirs(root)

    all_sp_mats: dict[str, BinsparseFormat] = dict(
        zip(graph_meta["matrix_names"], bin_mats, strict=True)
    )
    max_vid: int = graph_meta["max_vid"]
    continous_label: bool = graph_meta["continous_label"]

    query_path = queryset_dir / dataset_name / (query_rel_path + ".txt")
    sp_mats_needed, expr = _build_query_matrices(
        query_path, all_sp_mats, max_vid, continous_label
    )

    query_stem = Path(query_rel_path).name
    gt_matches = list((ground_truth_dir / dataset_name).rglob(f"{query_stem}.txt"))
    ground_truth = (
        int(gt_matches[0].read_text().strip().split()[0]) if gt_matches else 0
    )

    bin_mats = list(sp_mats_needed.values())
    meta: dict[str, Any] = {
        "expr": expr,
        "gt": ground_truth,
        "name": query_rel_path,
        "matrix_names": list(sp_mats_needed.keys()),
    }
    return bin_mats, meta


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _get_dirs(root: Path) -> tuple[Path, Path, Path]:
    return root / "dataset", root / "queryset", root / "ground_truth"


def _ensure_downloaded(root: Path) -> None:
    """Download and extract the three G-CARE tarballs into *root* if not present."""
    import gdown

    dataset_dir, queryset_dir, ground_truth_dir = _get_dirs(root)

    dataset_link = "https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU/view?usp=sharing"
    queryset_link = "https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN/view?usp=sharing"
    ground_truth_link = "https://drive.google.com/file/d/1Bc6Q2RZQTcIB8IfOw5KafNYwPhq2BO94/view?usp=sharing"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        dataset_link,
        str(dataset_dir / "dataset.tar.gz"),
        hash="sha256:78B86CDA06115C4554CDFCFB93A7FBC8ECB759DF39927510DD02CED4228A95E4".lower(),
    )
    with tarfile.open(dataset_dir / "dataset.tar.gz", "r:gz") as tar:
        tar.extractall(path=dataset_dir, filter="data")

    queryset_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        queryset_link,
        str(queryset_dir / "queryset.tar.gz"),
        hash="sha256:C8DC9F978296559E9E55335A989CE16E7B5BCBA7AA9D43E25FBD9E588D00EBC7".lower(),
    )
    with tarfile.open(queryset_dir / "queryset.tar.gz", "r:gz") as tar:
        tar.extractall(path=queryset_dir, filter="data")

    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        ground_truth_link,
        str(ground_truth_dir / "ground_truth.tar.gz"),
        hash="sha256:22E59F4FC06FFB79711D582513C6422CA555422C9947FDE64A52F0A9292D382C".lower(),
    )
    with tarfile.open(ground_truth_dir / "ground_truth.tar.gz", "r:gz") as tar:
        tar.extractall(path=ground_truth_dir, filter="data")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_graph(p: Path):
    """Parse a G-CARE graph file into raw COO dicts.

    Returns ``(max_vid, continous_label, sp_mats)`` where *sp_mats* maps matrix
    names (``"V{label}"``, ``"E{label}"``, optionally ``"C"``) to COO dicts
    with keys ``"V"``, ``"I_tuple"``, ``"shape"``.
    """
    with p.open("r", encoding="utf-8") as f:
        max_vid = 0
        num_nodes = 0
        all_verts = []

        V_dict: dict[int, list[int]] = {}
        E_dict: dict[int, tuple[list[int], list[int]]] = {}

        for line in f.readlines():
            if line.startswith("t"):
                pass
            elif line.startswith("v"):
                vals = line.strip().split(" ")
                num_nodes += 1

                v_id = int(vals[1])
                if v_id > max_vid:
                    max_vid = v_id
                all_verts.append(v_id)

                v_labels = [int(x) for x in vals[2:]]
                if len(v_labels) == 0:
                    v_labels = [0]

                for label in v_labels:
                    if label in V_dict:
                        V_dict[label].append(v_id)
                    else:
                        V_dict[label] = [v_id]
            elif line.startswith("e"):
                vals = line.strip().split(" ")

                v_id1 = int(vals[1])
                v_id2 = int(vals[2])

                e_labels = [int(x) for x in vals[3:]]
                if len(e_labels) == 0:
                    e_labels = [0]

                for label in e_labels:
                    if label in E_dict:
                        E_dict[label][0].append(v_id1)
                        E_dict[label][1].append(v_id2)
                    else:
                        E_dict[label] = ([v_id1], [v_id2])

        # V[label]: vector of all vertices with this label
        V: dict[str, Any] = {}
        for label, verts in V_dict.items():
            V[f"V{label}"] = {
                "V": np.ones((len(verts),), dtype=np.int64),
                "I_tuple": (np.array(verts),),
                "shape": (max_vid + 1,),
            }

        # E[label]: sparse adjacency matrix of all edges with this label
        E: dict[str, Any] = {}
        for label, edges in E_dict.items():
            assert len(edges[0]) == len(edges[1])
            E[f"E{label}"] = {
                "V": np.ones((len(edges[0]),), dtype=np.int64),
                "I_tuple": (np.array(edges[0]), np.array(edges[1])),
                "shape": (max_vid + 1, max_vid + 1),
            }

        sp_mats = V | E

        if max_vid + 1 == num_nodes:
            continous_label = True
        else:
            continous_label = False
            sp_mats["C"] = {
                "V": np.ones((num_nodes,), dtype=np.int64),
                "I_tuple": (all_verts,),
                "shape": (max_vid + 1,),
            }

        return max_vid, continous_label, sp_mats


def _parse_query(p: Path, continous_label: bool = True):
    """Parse a G-CARE query file into a framework einsum expression.

    Returns ``(expr, qvs, sp_mats_name)``: the einsum string, query-variable
    names, and the set of sparse-matrix names required.
    """
    with p.open("r", encoding="utf-8") as f:
        exprs = []
        qvs = []
        sp_mats_name: set[str] = set()
        for line in f.readlines():
            if line.startswith("t"):
                pass
            elif line.startswith("v"):
                vals = line.strip().split(" ")
                qv_id = int(vals[1])
                v_label = int(vals[2])
                v_id = int(vals[3])
                qvs.append(f"v_{qv_id}")
                if v_label != -1:
                    exprs.append(f"V{v_label}[v_{qv_id}]")
                    sp_mats_name.add(f"V{v_label}")
                if v_id == -1:
                    if not continous_label:
                        exprs.append(f"C[v_{qv_id}]")
                        sp_mats_name.add("C")
                else:
                    exprs.append(f"P{v_id}[v_{qv_id}]")
                    sp_mats_name.add(f"P{v_id}")
            elif line.startswith("e"):
                vals = line.strip().split(" ")
                qv_id1 = int(vals[1])
                qv_id2 = int(vals[2])
                e_label = int(vals[3])
                exprs.append(f"E{e_label}[v_{qv_id1},v_{qv_id2}]")
                sp_mats_name.add(f"E{e_label}")

        final_expr = "S[] += " + " * ".join(exprs)
        return final_expr, qvs, sp_mats_name


def _build_query_matrices(
    query_path: Path,
    all_sp_mats: dict[str, BinsparseFormat],
    max_vid: int,
    continous_label: bool,
) -> tuple[dict[str, BinsparseFormat], str]:
    """Select matrices needed by a query from the pre-converted graph matrices.

    Missing matrix names (not present in *all_sp_mats*) get a zero or one-hot
    placeholder sized to *max_vid*.  Returns ``(sp_mats_needed, expr)``.
    """
    expr, _, sp_mats_name = _parse_query(query_path, continous_label=continous_label)

    sp_mats_needed: dict[str, BinsparseFormat] = {}
    for sp_name in sp_mats_name:
        if sp_name not in all_sp_mats:
            if sp_name.startswith("P"):  # one-hot node-id vector
                sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                    (np.array([0]), np.array([int(sp_name[1:])])),
                    np.array([1]),
                    (max_vid + 1,),
                )
            elif sp_name.startswith("V"):  # missing vertex label → all-zero vector
                sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                    (np.array([0]),),
                    np.array([0]),
                    (max_vid + 1,),
                )
            elif sp_name.startswith("E"):  # missing edge label → all-zero matrix
                sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                    (np.array([0]), np.array([0])),
                    np.array([0]),
                    (max_vid + 1, max_vid + 1),
                )
        else:
            sp_mats_needed[sp_name] = all_sp_mats[sp_name]

    return sp_mats_needed, expr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    # src/saps/downloaders/gcare.py → parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "gcare"
