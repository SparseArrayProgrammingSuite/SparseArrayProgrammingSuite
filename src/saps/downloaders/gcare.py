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

import gdown

from saps_framework.binsparse_format import BinsparseFormat


def load_gcare_dataset(
    dataset_name: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[BinsparseFormat], dict]:
    """Download (if needed) and parse a G-CARE dataset.

    Follows the same convention as ``download_snap_dataset``: returns a flat
    ``list[BinsparseFormat]`` so the harness can convert each matrix uniformly.

    Because each query needs several named matrices, the flat list concatenates
    all queries' matrices in order.  The ``meta`` dict carries the grouping
    information needed to reconstruct per-query dicts in ``benchmark()``:

    * ``"exprs"``        – einsum string, one per query
    * ``"gts"``         – ground-truth count, one per query
    * ``"names"``       – query stem name, one per query
    * ``"query_sizes"`` – number of matrices belonging to each query
    * ``"matrix_names"``– name (e.g. ``"V0"``, ``"E1"``) for every flat entry,
                          in the same order as the flat list
    """
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    dataset_dir, queryset_dir, ground_truth_dir = download_gcare_data(root)

    max_vid, continous_label, all_sp_mats = read_gcare_data(
        dataset_dir / dataset_name / f"{dataset_name}.txt"
    )

    queries: dict[str, dict] = {}

    for query_path in (queryset_dir / dataset_name).rglob("*.txt"):
        sp_mats_needed, expr = process_one_query(
            query_path, all_sp_mats, max_vid, continous_label
        )
        queries[query_path.stem] = {"matrices": sp_mats_needed, "expr": expr}

    for gt_path in (ground_truth_dir / dataset_name).rglob("*.txt"):
        with open(gt_path) as f:
            queries[gt_path.stem]["ground_truth"] = int(f.readline())

    # Flatten: one BinsparseFormat per matrix, across all queries in order.
    flat_matrices: list[BinsparseFormat] = []
    meta: dict[str, list[Any]] = {
        "exprs": [],
        "gts": [],
        "names": [],
        "query_sizes": [],
        "matrix_names": [],
    }
    for query_name, query_data in queries.items():
        named_mats: dict[str, BinsparseFormat] = query_data["matrices"]
        meta["exprs"].append(query_data["expr"])
        meta["gts"].append(query_data["ground_truth"])
        meta["names"].append(query_name)
        meta["query_sizes"].append(len(named_mats))
        for mat_name, mat in named_mats.items():
            flat_matrices.append(mat)
            meta["matrix_names"].append(mat_name)

    return flat_matrices, meta


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_gcare_data(root_dir: Path):
    """Download and extract the three G-CARE tarballs into *root_dir*.

    Returns ``(dataset_dir, queryset_dir, ground_truth_dir)``.
    """
    dataset_dir = root_dir / "dataset"
    queryset_dir = root_dir / "queryset"
    ground_truth_dir = root_dir / "ground_truth"

    dataset_link = "https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU/view?usp=sharing"
    queryset_link = "https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN/view?usp=sharing"
    ground_truth_link = "https://drive.google.com/file/d/1Bc6Q2RZQTcIB8IfOw5KafNYwPhq2BO94/view?usp=sharing"

    # Download dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        dataset_link,
        str(dataset_dir / "dataset.tar.gz"),
        hash="sha256:78B86CDA06115C4554CDFCFB93A7FBC8ECB759DF39927510DD02CED4228A95E4".lower(),
    )
    with tarfile.open(dataset_dir / "dataset.tar.gz", "r:gz") as tar:
        tar.extractall(path=dataset_dir, filter="data")

    # Download queryset
    queryset_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        queryset_link,
        str(queryset_dir / "queryset.tar.gz"),
        hash="sha256:C8DC9F978296559E9E55335A989CE16E7B5BCBA7AA9D43E25FBD9E588D00EBC7".lower(),
    )
    with tarfile.open(queryset_dir / "queryset.tar.gz", "r:gz") as tar:
        tar.extractall(path=queryset_dir, filter="data")

    # Download ground truth
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    gdown.cached_download(  # type: ignore[attr-defined]
        ground_truth_link,
        str(ground_truth_dir / "ground_truth.tar.gz"),
        hash="sha256:22E59F4FC06FFB79711D582513C6422CA555422C9947FDE64A52F0A9292D382C".lower(),
    )
    with tarfile.open(ground_truth_dir / "ground_truth.tar.gz", "r:gz") as tar:
        tar.extractall(path=ground_truth_dir, filter="data")

    return dataset_dir, queryset_dir, ground_truth_dir


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def read_gcare_data(p: Path):
    """Parse a G-CARE data-graph file.

    Returns ``(max_vid, continous_label, sp_mats)`` where *sp_mats* is a dict
    mapping matrix names (``"V{label}"``, ``"E{label}"``, optionally ``"C"``)
    to raw COO dicts (keys: ``"V"``, ``"I_tuple"``, ``"shape"``).
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


def read_gcare_query(p: Path, continous_label: bool = True):
    """Parse a G-CARE query file into a framework einsum expression.

    Returns ``(expr, qvs, sp_mats_name)`` where *expr* is a string suitable
    for ``xp.einsum()``, *qvs* is the list of query-variable names, and
    *sp_mats_name* is the set of sparse-matrix names required.
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


def process_one_query(
    query_path: Path,
    all_sp_mats: dict[str, Any],
    max_vid: int,
    continous_label: bool,
) -> tuple[dict[str, BinsparseFormat], str]:
    """Build the ``BinsparseFormat`` matrices needed for a single query.

    Returns ``(sp_mats_needed, expr)`` ready for the benchmark.
    """
    expr, _, sp_mats_name = read_gcare_query(
        query_path, continous_label=continous_label
    )

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
            sp_mat = all_sp_mats[sp_name]
            sp_mats_needed[sp_name] = BinsparseFormat.from_coo(
                sp_mat["I_tuple"], sp_mat["V"], sp_mat["shape"]
            )

    return sp_mats_needed, expr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_data_dir() -> Path:
    # src/saps/downloaders/gcare.py → parents[3] = repo root
    return Path(__file__).resolve().parents[3] / "data" / "gcare"
