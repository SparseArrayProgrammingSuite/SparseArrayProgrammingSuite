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
    ("BFS", "BreadthFirstSearchSNAPGenerator"),
    ("bellmanford", "BellmanFordSNAPGenerator"),
    ("centrality", "BetweennessCentralitySNAPGenerator"),
    ("connected_components", "ConnectedComponentsSNAPGenerator"),
    ("fastsv", "FastSVSNAPGenerator"),
    ("four_clique_counting", "FourCliqueCountSNAPGenerator"),
    ("pagerank", "PageRankSNAPGenerator"),
    ("transitive_closure", "TransitiveClosureSNAPGenerator"),
    ("triangle_counting", "TriangleCountSNAPGenerator"),
    ("transitive_reduction", "TransitiveReductionSNAPGenerator"),
]


def test_snap_shell_inventory_covers_consumers():
    generator = SNAPGraphGenerator()
    declared = {d.name for d in generator.datasets}
    assert len(declared) == len(generator.datasets) == 113
    assert SNAPGraphBenchmark().name == "snap_graph_shell"
    assert generator.cacheable
    consumed = set()
    for benchmark in _benchmark_instances():
        for consumer in benchmark.generators:
            for dataset in consumer.datasets:
                if (
                    type(consumer).__name__.endswith("SNAPGenerator")
                    or consumer.name == "snap_graph"
                ):
                    assert dataset.name in declared
                    if consumer.name != generator.name:
                        assert not consumer.cacheable
                        consumed.add(dataset.name)
    assert consumed <= declared
    assert len(consumed) == 8


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
    slug = dataset.name
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
    elif module_name == "transitive_reduction":
        expected = np.array(
            [[np.inf, 1, np.inf], [np.inf, np.inf, 1], [np.inf, np.inf, np.inf]]
        )
        np.testing.assert_array_equal(to_sparse(problem.inputs[0]).todense(), expected)
        assert problem.inputs[0].number_of_stored_values == 2
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


def test_each_gap_graph_problem_has_an_explicit_snap_generator():
    problems = []
    for benchmark in _benchmark_instances():
        generators = benchmark.generators
        if not any(type(g).__name__.endswith("GAPGenerator") for g in generators):
            continue
        problems.append(benchmark.name)
        snap = [g for g in generators if type(g).__name__.endswith("SNAPGenerator")]
        assert len(snap) == 1, benchmark.name
        assert "SNAP" in snap[0].pretty_name
        assert snap[0].name.endswith("_snap_inputs")
        assert not snap[0].cacheable
    assert len(problems) == 10


def test_snap_transitive_reduction_removes_redundant_edge(monkeypatch):
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from frameworks.saps_numpy import NumpyFramework
    from saps.benchmark import DataInstance
    from saps.benchmarks import transitive_reduction as reduction

    adjacency = from_scipy(coo_array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
    raw = DataInstance(inputs=[adjacency], meta={})
    monkeypatch.setattr(reduction, "fetch_snap_graph", lambda _: raw)
    generator = reduction.TransitiveReductionSNAPGenerator()
    problem = generator.generate(generator.datasets[0])
    xp = NumpyFramework()
    actual = reduction.TransitiveReductionBenchmark().benchmark(
        xp, [xp.from_binsparse(problem.inputs[0])], problem.meta
    )[0]
    np.testing.assert_array_equal(
        actual, [[np.inf, 1, np.inf], [np.inf, np.inf, 1], [np.inf, np.inf, np.inf]]
    )
    np.testing.assert_array_equal(
        to_scipy(adjacency).toarray(), [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    )


def test_snap_catalog_metadata_and_group_concepts():
    from xml.etree import ElementTree as ET

    datasets = SNAPGraphGenerator().datasets
    assert len({group for d in datasets for group in d.groups}) == 23
    for dataset in datasets:
        metadata = dataset.metadata
        assert metadata["types"] == dataset.types
        assert metadata["description"] == dataset.description
        assert metadata["nodes"] == dataset.nodes
        assert metadata["edges"] == dataset.edges
        assert metadata["groups"] == dataset.groups
        assert ET.fromstring(dataset.concepts).findtext("concept/concept_id")
        assert dataset.topics
    reddit = [d for d in datasets if d.name == "soc-RedditHyperlinks"]
    assert len(reddit) == 1
    assert len(reddit[0].groups) == 4
    assert reddit[0].static_edges == 858490
    assert reddit[0].items == "858,490 links between 55,863 subreddits"
    assert "Subreddit hyperlinks" in reddit[0].types
    concepts = ET.fromstring(reddit[0].concepts).findall("concept/concept_id")
    assert len(concepts) == len({c.text for c in concepts}) == 3
    by_name = {d.name: d for d in datasets}
    assert by_name["as-733"].nodes == "103-6,474"
    assert by_name["wiki-hoaxes"].edges is None
    assert by_name["Deezer Ego-nets"].nodes is None
    assert by_name["Deezer Ego-nets"].graphs == 9629
    assert by_name["web-BeerAdvocate"].items == "1,586,259 beer reviews"


def test_source_selection_samples_nonzero_edge_starts_reproducibly():
    from scipy.sparse import coo_array

    from binsparse import COORMatrix

    from saps.benchmarks.snap import select_source_vertices

    # Row 0 has an explicit zero; row 3 cancels to zero; row 5 is isolated.
    adjacency = coo_array(
        ([0, 1, 1, 1, 1, -1], ([0, 1, 1, 4, 3, 3], [2, 2, 4, 4, 2, 2])),
        shape=(6, 6),
    )
    graph = COORMatrix(
        adjacency.shape,
        adjacency.nnz,
        indices_0=adjacency.row,
        indices_1=adjacency.col,
        values=adjacency.data,
    )
    expected = np.array([1, 1, 4])[np.random.default_rng(42).integers(3, size=100)]
    actual = select_source_vertices(graph, 100, seed=42)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(select_source_vertices(graph, 100, seed=42), actual)
    assert set(actual) == {1, 4}
    # Coalescing takes place on a copy, leaving the input unchanged.
    assert to_scipy(graph).nnz == 6


def test_source_selection_stays_sparse_and_preserves_global_rng():
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from saps.benchmarks.snap import select_source_vertices

    graph = from_scipy(coo_array(([1], ([999999], [2])), shape=(1000000, 1000000)))
    state = np.random.get_state()  # noqa: NPY002 - verify legacy global state is untouched
    np.testing.assert_array_equal(select_source_vertices(graph, 3), [999999] * 3)
    after = np.random.get_state()  # noqa: NPY002 - verify legacy global state is untouched
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize(
    ("shape", "values", "count", "message"),
    [
        ((3, 3), [], 1, "without nonzero edges"),
        ((3, 3), [0], 1, "without nonzero edges"),
        ((3, 2), [1], 1, "square"),
        ((3, 3), [1], 0, "positive"),
    ],
)
def test_source_selection_rejects_invalid_inputs(shape, values, count, message):
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from saps.benchmarks.snap import select_source_vertices

    indices = np.zeros(len(values), dtype=int)
    graph = from_scipy(coo_array((values, (indices, indices)), shape=shape))
    with pytest.raises(ValueError, match=message):
        select_source_vertices(graph, count)
