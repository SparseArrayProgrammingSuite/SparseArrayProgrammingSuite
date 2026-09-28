import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import to_sparse

from saps.benchmark import DataInstance, Generator
from saps.benchmarks import subgraph_matching
from saps.benchmarks.subgraph_matching import (
    GCareDataset,
    GCareGraphGenerator,
    GCareHumanGenerator,
)
from saps.downloaders import gcare
from saps.storage import LocalStorageBackend


def test_gcare_cached_graph_contains_everything_for_query_setup(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    root = cache / "gcare"
    graph_dir = root / "dataset" / "human"
    query_dir = root / "queryset" / "human" / "Star_3"
    truth_dir = root / "ground_truth" / "human" / "Star_3"
    for directory in (graph_dir, query_dir, truth_dir):
        directory.mkdir(parents=True)
    (graph_dir / "human.txt").write_text("t # 0\nv 0 0\nv 2 1\ne 0 2 2\n")
    (query_dir / "uf_Q_2_1.txt").write_text("t # 0\nv 0 0 0\nv 1 1 -1\ne 0 1 2\n")
    (truth_dir / "uf_Q_2_1.txt").write_text("1\n")
    (query_dir / "missing_label.txt").write_text("v 0 99 -1\nv 1 0 -1\ne 0 1 99\n")
    monkeypatch.setenv("SAPS_CACHE_DIR", str(cache))
    # Supply local source fixtures in place of the downloader during preparation.
    download = Mock()
    monkeypatch.setattr(gcare, "_ensure_downloaded", download)
    monkeypatch.setitem(sys.modules, "gdown", None)
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", cache
    )
    monkeypatch.setattr(Generator, "backend", property(lambda _: backend))
    raw_generator = GCareGraphGenerator()
    dataset = raw_generator.datasets[0]
    assert backend.upload_dataset(raw_generator, dataset)
    download.assert_called_once_with(root)

    shutil.rmtree(root)
    download.side_effect = AssertionError("download")
    monkeypatch.setattr(
        GCareGraphGenerator,
        "generate",
        Mock(side_effect=AssertionError("regeneration")),
    )
    monkeypatch.setattr(
        backend, "download_file", Mock(side_effect=AssertionError("download"))
    )
    monkeypatch.setattr(
        subgraph_matching, "load_gcare_graph", Mock(side_effect=AssertionError("load"))
    )
    monkeypatch.setattr(
        gcare, "_parse_query", Mock(side_effect=AssertionError("parse"))
    )
    query_generator = GCareHumanGenerator()
    problem = query_generator.generate(query_generator.datasets[0])

    assert problem.meta["gt"] == 1
    assert (
        problem.meta["expr"]
        == "S[] += V0[v_0] * P0[v_0] * V1[v_1] * C[v_1] * E2[v_0,v_1]"
    )
    matrices = dict(zip(problem.meta["matrix_names"], problem.inputs, strict=True))
    np.testing.assert_array_equal(to_sparse(matrices["P0"]).todense(), [1, 0, 0])
    np.testing.assert_array_equal(to_sparse(matrices["C"]).todense(), [1, 0, 1])
    assert to_sparse(matrices["E2"])[0, 2] == 1
    missing = query_generator.generate(GCareDataset("human", "Star_3/missing_label"))
    matrices = dict(zip(missing.meta["matrix_names"], missing.inputs, strict=True))
    assert not np.any(to_sparse(matrices["V99"]).todense())
    assert not np.any(to_sparse(matrices["E99"]).todense())
    assert not root.exists()


def test_gcare_shell_passes_loader_metadata_through_without_file_io(
    monkeypatch, tmp_path
):
    metadata = {"queries": {"Star_3/example": {"expr": "S[] += V0[v_0]", "gt": 1}}}
    monkeypatch.setattr(
        subgraph_matching, "load_gcare_graph", Mock(return_value=([], metadata))
    )
    generator = GCareGraphGenerator()
    generator._backend = SimpleNamespace(cache_dir=tmp_path / "cache")
    for name in ("rglob", "open", "read_text"):
        monkeypatch.setattr(Path, name, Mock(side_effect=AssertionError("file I/O")))

    problem = generator.generate(generator.datasets[0])

    assert problem.meta is metadata
    assert problem.meta["queries"]["Star_3/example"]["gt"] == 1


@pytest.mark.parametrize(
    "metadata, exception, message",
    [
        ({}, RuntimeError, "Refresh it with --cache-datasets"),
        ({"queries": {}}, ValueError, "not in the cached G-CARE queries"),
    ],
)
def test_gcare_incomplete_cache_fails_without_downloading(
    monkeypatch, metadata, exception, message
):
    download = Mock(side_effect=AssertionError("download"))
    monkeypatch.setattr(subgraph_matching, "load_gcare_graph", download)
    monkeypatch.setattr(
        GCareGraphGenerator,
        "cached_generate",
        Mock(return_value=DataInstance(inputs=[], meta=metadata)),
    )

    with pytest.raises(exception, match=message):
        GCareHumanGenerator().generate(GCareDataset("human", "missing"))

    download.assert_not_called()
