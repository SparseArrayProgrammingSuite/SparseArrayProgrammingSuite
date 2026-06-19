import numpy as np

import saps.benchmarks.connected_components as cc
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.snap import load_toy_dataset
from saps_framework import BinsparseFormat


def _run_cc(A):
    xp = NumpyFramework()
    cc.xp = xp
    A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
    (labels,) = cc.SimplyConnectedComponentsBenchmark().benchmark([A_bin], {})
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
    assert len(set(labels.tolist())) == 4, (
        "each isolated node should have a unique label"
    )


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


def test_cc_snap_toy():
    data, _ = load_toy_dataset()
    labels = _run_cc(data[0])
    assert len(set(labels.tolist())) == 1, "all nodes should converge to the same label"
