import json
import os
import subprocess
from itertools import combinations
from pathlib import Path
from unittest.mock import Mock

import pytest

import numpy as np
from scipy.sparse import coo_array, csr_array

from binsparse import COORMatrix
from binsparse.conversions import from_scipy

from saps.benchmark import DataInstance
from scripts import run_graphminer_gap as runner


def noisy_k4():
    # One-way, signed/weighted K4, opposite-signed duplicates, loop, stored zero,
    # and an isolated vertex. Boolean OR must happen before duplicate reduction.
    edges = list(combinations(range(4), 2)) + [(0, 1), (2, 2), (3, 4)]
    row, col = zip(*edges, strict=True)
    return COORMatrix(
        (5, 5),
        len(edges),
        indices_0=np.array(row),
        indices_1=np.array(col),
        values=np.array([-2] * 6 + [2, 5, 0]),
    )


def test_export_preserves_simple_undirected_graph(tmp_path):
    prefix = tmp_path / "graph"
    metadata = runner.export_graph(noisy_k4(), prefix)
    rowptr = np.fromfile(prefix.with_suffix(".vertex.bin"), dtype=np.int64)
    columns = np.fromfile(prefix.with_suffix(".edge.bin"), dtype=np.int32)
    actual = csr_array((np.ones(len(columns)), columns, rowptr), shape=(5, 5))
    expected = np.zeros((5, 5))
    expected[:4, :4] = 1 - np.eye(4)
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert metadata["undirected_edges"] == 6
    assert prefix.with_suffix(".meta.txt").read_text().split() == [
        "5",
        "12",
        "4",
        "8",
        "1",
        "2",
        "3",
        "0",
        "0",
        "0",
    ]


@pytest.mark.parametrize("output", ["num_4-cliques = 1234567890123\n", "no count\n"])
def test_counter_requires_count_and_sets_threads(tmp_path, monkeypatch, output):
    def run(command, **kwargs):
        assert command[-1] == "4"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "7"
        assert kwargs["check"]
        kwargs["stdout"].write(output)

    monkeypatch.setattr(subprocess, "run", run)
    args = (tmp_path / "binary", tmp_path / "graph", tmp_path / "log", 7)
    if output.startswith("num_"):
        assert runner.run_counter(*args)[0] == 1234567890123
    else:
        with pytest.raises(ValueError, match="Expected one"):
            runner.run_counter(*args)


def test_main_resume_and_failed_run_do_not_publish_counts(tmp_path, monkeypatch):
    binary = tmp_path / "counter"
    binary.write_text("placeholder")
    binary.chmod(0o755)
    load = Mock(return_value=DataInstance(inputs=[noisy_k4()], meta={}))
    monkeypatch.setattr(runner, "fetch_gap_graph", load)
    counter = Mock(return_value=(1, 0.1, [str(binary), "graph", "4"]))
    monkeypatch.setattr(runner, "run_counter", counter)
    args = [
        "--graphminer-bin",
        str(binary),
        "--datasets",
        "GAP-web",
        "--output-dir",
        str(tmp_path / "results"),
    ]
    assert runner.main(args) == 0
    result = tmp_path / "results/GAP-web.json"
    assert json.loads(result.read_text())["four_cliques"] == 1
    assert runner.main([*args, "--resume"]) == 0
    assert load.call_count == counter.call_count == 1
    counter.side_effect = subprocess.CalledProcessError(1, "counter")
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(args)
    assert not result.exists()


def test_known_count_mismatch_is_not_published(tmp_path, monkeypatch):
    binary = tmp_path / "counter"
    binary.touch(mode=0o755)
    monkeypatch.setattr(
        runner, "fetch_gap_graph", lambda _: DataInstance(inputs=[noisy_k4()], meta={})
    )
    monkeypatch.setattr(runner, "run_counter", lambda *args: (1, 0.1, []))
    with pytest.raises(ValueError, match="expected 90"):
        runner.main(
            [
                "--graphminer-bin",
                str(binary),
                "--datasets",
                "GAP-road",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )
    assert not (tmp_path / "results/GAP-road.json").exists()


@pytest.mark.skipif(
    not os.environ.get("GRAPHMINER_BIN"), reason="Set GRAPHMINER_BIN for integration"
)
def test_real_graphminer_matches_brute_force(tmp_path):
    binary = Path(os.environ["GRAPHMINER_BIN"]).resolve()
    rng = np.random.default_rng(42)
    graphs = [noisy_k4()]
    for size in range(4, 12):
        matrix = rng.random((size, size)) < 0.55
        graphs.append(from_scipy(coo_array(matrix)))
    for index, adjacency in enumerate(graphs):
        prefix = tmp_path / str(index) / "graph"
        runner.export_graph(adjacency, prefix)
        from binsparse.conversions import to_scipy

        graph = to_scipy(
            runner.zero_one_adjacency(adjacency, bool, simple_undirected=True)
        ).toarray()
        expected = sum(
            all(graph[a, b] for a, b in combinations(vertices, 2))
            for vertices in combinations(range(len(graph)), 4)
        )
        actual, _, _ = runner.run_counter(binary, prefix, tmp_path / f"{index}.log", 2)
        assert actual == expected
