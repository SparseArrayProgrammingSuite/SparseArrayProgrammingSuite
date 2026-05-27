import gzip

import numpy as np

import saps.benchmarks.transitive_closure as tc
from frameworks.saps_numpy import NumpyFramework


def test_transitive_closure():
    # 6-node DAG.
    xp = NumpyFramework()

    input_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )

    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=bool,
    )

    tc.xp = xp
    (res,) = tc.TransitiveClosureBenchmark().benchmark((input_matrix,), {})
    assert np.array_equal(res, expected)


def test_stc():
    # 8 node graph with 4 Stc.
    xp = NumpyFramework()
    input_matrix = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )

    expected = 4

    tc.xp = xp
    (res,) = tc.TransitiveClosureBenchmark().benchmark((input_matrix,), {})

    # count stc.
    visited_set = set()
    count = 0
    for i in range(res.shape[0]):
        comp = tuple(res[i, :])
        if comp not in visited_set:
            count += 1
            visited_set.add(comp)

    assert count == expected


def test_stc_cycle():
    # one stc. one cycle
    xp = NumpyFramework()

    input_matrix = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    tc.xp = xp
    (res,) = tc.TransitiveClosureBenchmark().benchmark((input_matrix,), {})
    # clique matrix
    expected = np.ones((3, 3), dtype=bool)
    assert np.array_equal(res, expected)


def test_stc_one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    tc.xp = xp
    (res,) = tc.TransitiveClosureBenchmark().benchmark((input_matrix,), {})

    # simple 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_closure_one_node():
    # one node
    xp = NumpyFramework()
    input_matrix = np.array([[0]], dtype=bool)

    tc.xp = xp
    (res,) = tc.TransitiveClosureBenchmark().benchmark((input_matrix,), {})

    # should be self loop, 1x1 matrix with 1
    expected = np.array([[1]], dtype=bool)
    assert np.array_equal(res, expected)


def test_transitive_closure_generator_loads_snap_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    with gzip.open(dataset_dir / "toy.txt.gz", "wt", encoding="utf-8") as f:
        f.write("# SNAP edge list\n10 20\n20 40\n")

    original_download = tc.download_snap_dataset

    def download_from_tmp(dataset_name):
        return original_download(dataset_name, data_dir=tmp_path)

    monkeypatch.setattr(tc, "download_snap_dataset", download_from_tmp)

    dataset = tc.TransitiveClosureDataset("snap-toy")
    data, meta = tc.TransitiveClosureGenerator().generate(dataset)

    assert data[0].data["shape"] == (3, 3)
    assert meta["snap_slug"] == "toy"
    assert meta["src"] == 0
