import gzip

import numpy as np

from saps.downloaders.snap import download_snap_dataset, parse_snap_edge_list


def test_parse_snap_edge_list_remaps_nodes(tmp_path):
    path = tmp_path / "toy.txt"
    path.write_text("# comment\n10 20\n20 40\n", encoding="utf-8")

    adjacency, meta = parse_snap_edge_list(path)

    assert adjacency.data["shape"] == (3, 3)
    assert np.array_equal(adjacency.data["indices_0"], np.array([0, 1]))
    assert np.array_equal(adjacency.data["indices_1"], np.array([1, 2]))
    assert np.array_equal(adjacency.data["values"], np.array([True, True]))
    assert np.array_equal(meta["raw_node_ids"], np.array([10, 20, 40]))
    assert meta["num_edges"] == 2
    assert meta["src"] == 0


def test_parse_snap_edge_list_can_preserve_raw_node_ids(tmp_path):
    path = tmp_path / "toy.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("# Nodes: 3 Edges: 2\n10 20\n20 40\n")

    adjacency, meta = parse_snap_edge_list(path, remap_nodes=False)

    assert adjacency.data["shape"] == (41, 41)
    assert np.array_equal(adjacency.data["indices_0"], np.array([10, 20]))
    assert np.array_equal(adjacency.data["indices_1"], np.array([20, 40]))
    assert meta["remap_nodes"] is False


def test_download_snap_dataset_uses_cached_gzip(tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as file:
        file.write("# SNAP edge list\n1 2\n2 1\n")

    data, meta = download_snap_dataset("snap-toy", data_dir=tmp_path)

    assert len(data) == 1
    assert (dataset_dir / "toy.txt").exists()
    assert data[0].data["shape"] == (2, 2)
    assert meta["snap_slug"] == "toy"
    assert meta["path"].endswith("toy.txt")
