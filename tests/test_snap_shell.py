import importlib
import shutil
from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import to_numpy, to_scipy, to_sparse

from saps.benchmark import Generator
from saps.benchmarks.snap import (
    SNAPGraphBenchmark,
    SNAPGraphGenerator,
    fetch_snap_graph,
)
from saps.downloaders import snap as downloader
from saps.metadata import _benchmark_instances
from saps.storage import LocalStorageBackend

_CONSUMERS = [
    ("BFS", "BreadthFirstSearchGenerator"),
    ("bellmanford", "BellmanFordGenerator"),
    ("centrality", "BetweennessCentralityGenerator"),
    ("connected_components", "ConnectedComponentsGenerator"),
    ("fastsv", "FastSVGenerator"),
    ("four_clique_counting", "FourCliqueCountGenerator"),
    ("pagerank", "PageRankGenerator"),
    ("transitive_closure", "TransitiveClosureGenerator"),
    ("triangle_counting", "TriangleCountGenerator"),
]


def test_snap_shell_inventory_covers_consumers():
    generator = SNAPGraphGenerator()
    declared = {d.name for d in generator.datasets}
    assert len(declared) == len(generator.datasets) == 8
    assert SNAPGraphBenchmark().name == "snap_graph_shell"
    assert generator.cacheable
    consumed = set()
    for benchmark in _benchmark_instances():
        for consumer in benchmark.generators:
            for dataset in consumer.datasets:
                # snap-toy is an in-memory test graph, not a downloaded source.
                if dataset.name.startswith("snap-") and dataset.name != "snap-toy":
                    assert dataset.name in declared
                    if consumer.name != generator.name:
                        assert not consumer.cacheable
                        consumed.add(dataset.name)
    assert consumed == declared


@pytest.mark.parametrize(("module_name", "class_name"), _CONSUMERS)
def test_snap_consumer_reads_shared_remote_graph_without_source_download(
    monkeypatch, tmp_path, module_name, class_name
):
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    monkeypatch.setattr(Generator, "backend", property(lambda _: backend))
    monkeypatch.delenv("SAPS_CACHE_DATASETS", raising=False)
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    consumer = getattr(module, class_name)()
    dataset = consumer.datasets[0]
    shell = SNAPGraphGenerator()
    source = next(d for d in shell.datasets if d.name == dataset.name)
    slug = dataset.name.removeprefix("snap-")
    path = backend.cache_dir / "snap" / slug / f"{slug}.txt"
    path.parent.mkdir(parents=True)
    path.write_text("# directed graph with a self-loop\n10 20\n20 40\n40 40\n")
    assert backend.upload_dataset(shell, source)
    # Simulate another worker with only the manifest and remote prepared object.
    shutil.rmtree(backend.cache_dir)
    forbidden = Mock(side_effect=AssertionError("Unexpected source download"))
    monkeypatch.setattr(downloader, "download_snap_dataset", forbidden)
    monkeypatch.setattr("saps.benchmarks.snap.download_snap_dataset", forbidden)
    monkeypatch.setattr(backend, "upload_dataset", forbidden)
    download = Mock(wraps=backend.download_file)
    monkeypatch.setattr(backend, "download_file", download)
    manifest = backend.manifest_path.read_bytes()

    problem = consumer.cached_generate(dataset)
    assert problem.meta["directed"] is True
    assert problem.meta["src"] == 0
    if module_name == "bellmanford":
        expected = np.array([[0, 1, np.inf], [np.inf, 0, 1], [np.inf, np.inf, 0]])
        np.testing.assert_array_equal(to_sparse(problem.inputs[0]).todense(), expected)
        assert problem.inputs[0].fill_value == np.inf
        assert problem.inputs[0].number_of_stored_values == 5
    else:
        np.testing.assert_array_equal(
            to_scipy(problem.inputs[0]).toarray(), [[0, 1, 0], [0, 0, 1], [0, 0, 1]]
        )
        np.testing.assert_array_equal(to_numpy(problem.inputs[1]), [10, 20, 40])
    # Subsequent users read the local shared shell object, not per-consumer caches.
    raw = fetch_snap_graph(dataset.name)
    assert to_scipy(raw.inputs[0]).toarray()[2, 2] == 1
    download.assert_called_once()
    assert download.call_args.args[0].startswith(f"snap_graph/{dataset.name}/")
    forbidden.assert_not_called()
    assert backend.manifest_path.read_bytes() == manifest
    assert len(list(backend.cache_dir.rglob("*.bsp.h5"))) == 1
