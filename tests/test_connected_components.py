import gzip

import numpy as np
import pytest

import saps.benchmarks.connected_components as cc
from frameworks.saps_numpy import NumpyFramework


def _run_cc(A):
    xp = NumpyFramework()
    cc.xp = xp
    (labels,) = cc.SimplyConnectedComponentsBenchmark().benchmark((A,), {})
    return labels.ravel()


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


def test_cc_fully_connected():
    """All nodes in a clique should end up with the same label."""
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    labels = _run_cc(A)
    assert len(set(labels.tolist())) == 1, "all nodes should share one label"


def test_cc_two_disconnected_components():
    """Nodes 0-1 and nodes 2-3 are in separate undirected components."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    )
    labels = _run_cc(A)
    # nodes in the same component share a label; different components differ
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cc_isolated_nodes():
    """With no edges, every node is its own component."""
    A = np.zeros((4, 4), dtype=bool)
    labels = _run_cc(A)
    assert len(set(labels.tolist())) == 4, "each isolated node should have a unique label"


def test_cc_directed_star_pointing_inward():
    """
    Edges 1→0, 2→0, 3→0.  Each node's outgoing neighbourhood includes node 0,
    so all nodes propagate to node 0's label.
    """
    A = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    )
    labels = _run_cc(A)
    assert len(set(labels.tolist())) == 1, "all nodes should converge to the same label"


def test_cc_single_node():
    """Trivial one-node graph."""
    A = np.zeros((1, 1), dtype=bool)
    labels = _run_cc(A)
    assert labels.shape == (1,)


# ---------------------------------------------------------------------------
# Generator / downloader wiring
# ---------------------------------------------------------------------------


def test_cc_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = cc.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(cc, "download_snap_dataset", download_from_tmp)

    dataset = cc.ConnectedComponentsDataset("snap-toy")
    data, meta = cc.ConnectedComponentsGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert np.array_equal(data[0].data["indices_0"], np.array([0, 1]))
    assert np.array_equal(data[0].data["indices_1"], np.array([1, 2]))
    assert np.array_equal(data[0].data["values"], np.array([True, True]))
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0
