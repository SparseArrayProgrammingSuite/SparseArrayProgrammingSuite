import importlib
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import to_numpy, to_scipy, to_sparse

from saps.benchmark import Generator
from saps.benchmarks.bfs import (
    BFSDataset,
    BFSSNAPGenerator,
)
from saps.benchmarks.snap import (
    _MAX_DEGREES,
    NUM_SNAP_SOURCES,
    SNAPGraphGenerator,
    SNAPGraphShellBenchmark,
    fetch_snap_graph,
    seeded_source_vertices,
    select_source_vertices,
)
from saps.benchmarks.suitesparse import (
    SuiteSparseMatrixGenerator,
    fetch_suitesparse_matrix,
)
from saps.downloaders import suitesparse as downloader
from saps.metadata import _benchmark_instances
from saps.storage import LocalStorageBackend

_CONSUMERS = [
    ("bfs", "BFSSNAPGenerator"),
    ("bellman_ford", "BellmanFordSNAPGenerator"),
    ("betweenness_centrality", "BetweennessCentralitySNAPGenerator"),
    ("connected_components", "ConnectedComponentsSNAPGenerator"),
    ("fastsv", "FastSVSNAPGenerator"),
    ("four_clique_counting", "FourCliqueCountingSNAPGenerator"),
    ("pagerank", "PageRankSNAPGenerator"),
    ("transitive_closure", "TransitiveClosureSNAPGenerator"),
    ("floyd_warshall", "FloydWarshallSNAPGenerator"),
    ("triangle_counting", "TriangleCountingSNAPGenerator"),
    ("transitive_reduction", "TransitiveReductionSNAPGenerator"),
    ("mssp", "MSSPSNAPGenerator"),
    ("msbfs", "MSBFSSNAPGenerator"),
    ("mcl", "MCLSNAPGenerator"),
]
_MULTI_SOURCE = {"mssp", "msbfs"}


def _seeded_sources(graph):
    """Source k of a SNAP shell graph: the first draw of seed k."""
    return [
        int(select_source_vertices(graph, seed=seed)[0])
        for seed in range(NUM_SNAP_SOURCES)
    ]


def test_snap_max_degrees_cover_exactly_the_declared_graphs():
    datasets = SNAPGraphGenerator().datasets
    assert set(_MAX_DEGREES) == {d.source_name for d in datasets}
    assert all(d.max_degree == _MAX_DEGREES[d.source_name] > 0 for d in datasets)


def test_snap_shell_inventory_covers_consumers():
    generator = SNAPGraphGenerator()
    declared = {d.name for d in generator.datasets}
    assert len(declared) == len(generator.datasets) == 68
    assert {"soc-Slashdot0902", "as-735", "as-caida"} <= declared
    assert (
        not {
            "facebook_combined",
            "soc-Slashdot0922",
            "as-733",
            "as-Caida",
            "Deezer Ego-nets",
            "GitHub Stargazers",
            "Reddit Threads",
            "Ego-Nets",
            "ERC20-stablecoins",
        }
        & declared
    )
    assert SNAPGraphShellBenchmark().name == "snap_graph_shell"
    assert not generator.cacheable
    consumed = set()
    for benchmark in _benchmark_instances():
        for consumer in benchmark.generators:
            for dataset in consumer.datasets:
                if (
                    type(consumer).__name__.endswith("SNAPGenerator")
                    or consumer.name == "snap_graph"
                ):
                    name = getattr(dataset, "source_name", dataset.name).removeprefix(
                        "SNAP/"
                    )
                    assert name in declared
                    if consumer.name != generator.name:
                        assert not consumer.cacheable
                        consumed.add(name)
    assert consumed <= declared
    assert consumed == declared


@pytest.mark.parametrize("name", ["soc-Slashdot0902", "as-735", "as-caida"])
def test_snap_shell_preserves_suitesparse_matrix_and_discards_extras(
    monkeypatch, tmp_path, name
):
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    monkeypatch.setattr(Generator, "backend", property(lambda _: backend))
    matrix_dir = tmp_path / name
    matrix_dir.mkdir()
    # Preserve signed values, Matrix Market symmetry, and isolated vertices.
    (matrix_dir / f"{name}.mtx").write_text(
        "%%MatrixMarket matrix coordinate real symmetric\n4 4 2\n2 1 -2\n4 2 3\n"
    )
    (matrix_dir / f"{name}_b.mtx").write_text(
        "%%MatrixMarket matrix array real general\n4 1\n1\n2\n3\n4\n"
    )
    download = Mock(return_value=(matrix_dir, SimpleNamespace(group="SNAP", name=name)))
    monkeypatch.setattr(downloader, "download_suitesparse_matrix", download)
    generator = SNAPGraphGenerator()
    dataset = next(d for d in generator.datasets if d.name == name)
    shell = SuiteSparseMatrixGenerator()
    source = next(d for d in shell.datasets if d.source_name == dataset.source_name)
    assert backend.upload_dataset(shell, source)
    download.assert_called_once_with(f"SNAP/{name}", data_dir=None)
    download.side_effect = AssertionError("Unexpected source download")

    problem = generator.cached_generate(dataset)

    assert len(problem.inputs) == 1
    np.testing.assert_array_equal(
        to_scipy(problem.inputs[0]).toarray(),
        [[0, -2, 0, 0], [-2, 0, 0, 3], [0, 0, 0, 0], [0, 3, 0, 0]],
    )
    assert problem.meta == {
        "max_degree": _MAX_DEGREES[f"SNAP/{name}"],
        "sources": _seeded_sources(problem.inputs[0]),
    }
    assert problem.ref_outputs is None
    assert problem.ref_meta is None
    # Dropping the extras for graph benchmarks leaves the shared source intact.
    shared = fetch_suitesparse_matrix(dataset.source_name)
    assert len(shared.inputs) == 2
    assert shared.meta["has_b_file"]
    np.testing.assert_array_equal(to_numpy(shared.inputs[1]), [1, 2, 3, 4])


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
    shell = SuiteSparseMatrixGenerator()
    slug = getattr(dataset, "source_name", dataset.name).removeprefix("SNAP/")
    source = next(d for d in shell.datasets if d.source_name == f"SNAP/{slug}")
    path = backend.cache_dir / "suitesparse" / "SNAP" / slug / f"{slug}.mtx"
    path.parent.mkdir(parents=True)
    path.write_text(
        "%%MatrixMarket matrix coordinate pattern general\n3 3 3\n1 2\n2 3\n3 3\n"
    )
    source_download = Mock(
        return_value=(path.parent, SimpleNamespace(group="SNAP", name=slug))
    )
    monkeypatch.setattr(downloader, "download_suitesparse_matrix", source_download)
    assert backend.upload_dataset(shell, source)
    source_download.assert_called_once_with(f"SNAP/{slug}", data_dir=None)
    # Simulate another worker with only the manifest and remote prepared object.
    shutil.rmtree(backend.cache_dir)
    forbidden = Mock(side_effect=AssertionError("Unexpected source download"))
    monkeypatch.setattr(downloader, "download_suitesparse_matrix", forbidden)
    monkeypatch.setattr(
        "saps.benchmarks.suitesparse.load_suitesparse_matrix", forbidden
    )
    monkeypatch.setattr(backend, "upload_dataset", forbidden)
    download = Mock(wraps=backend.download_file)
    monkeypatch.setattr(backend, "download_file", download)
    manifest = backend.manifest_path.read_bytes()

    problem = consumer.cached_generate(dataset)
    assert len(problem.inputs) == (2 if module_name in _MULTI_SOURCE else 1)
    if getattr(dataset, "source_seed", None) is not None:
        assert problem.meta["seed"] == 0
        assert problem.meta["src"] == int(
            select_source_vertices(fetch_snap_graph(slug).inputs[0], seed=0)[0]
        )
    elif module_name in _MULTI_SOURCE:
        assert problem.meta == {
            "sources": sorted(set(_seeded_sources(fetch_snap_graph(slug).inputs[0])))
        }
    elif module_name == "mcl":
        assert problem.meta == {}
    else:
        expected_meta = {
            "max_degree": _MAX_DEGREES[f"SNAP/{slug}"],
            "sources": _seeded_sources(fetch_snap_graph(slug).inputs[0]),
        }
        if module_name in {"transitive_closure", "floyd_warshall"}:
            expected_meta["max_squarings"] = 0
        assert problem.meta == expected_meta
    if module_name in {"bellman_ford", "mssp", "floyd_warshall"}:
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
    # Subsequent users read the local shared shell object, not per-consumer caches.
    raw = fetch_snap_graph(slug)
    assert to_scipy(raw.inputs[0]).toarray()[2, 2] == 1
    if getattr(dataset, "source_seed", None) is not None:
        seeded = [d for d in consumer.datasets if d.source_name == slug]
        assert [d.source_seed for d in seeded] == list(range(10))
        from scipy.sparse.csgraph import shortest_path

        from frameworks.saps_numpy import NumpyFramework

        xp = NumpyFramework()
        benchmark = (
            module.BFSBenchmark()
            if module_name == "bfs"
            else module.BellmanFordBenchmark()
        )
        for variant in seeded:
            actual = consumer.cached_generate(variant)
            distances = shortest_path(
                to_scipy(raw.inputs[0]).toarray(),
                directed=True,
                unweighted=True,
                indices=actual.meta["src"],
            )
            expected_output = (
                np.where(np.isfinite(distances), distances + 1, 0)
                if module_name == "bfs"
                else distances
            )
            output = benchmark.benchmark(
                xp, actual.meta, xp.from_binsparse(actual.inputs[0])
            )
            np.testing.assert_array_equal(output, expected_output)
            assert actual.meta["src"] == int(
                select_source_vertices(raw.inputs[0], seed=variant.source_seed)[0]
            )
    assert len(raw.inputs) == 1
    assert raw.meta == {
        "max_degree": _MAX_DEGREES[f"SNAP/{slug}"],
        "sources": _seeded_sources(raw.inputs[0]),
    }
    download.assert_called_once()
    assert download.call_args.args[0].startswith(f"suitesparse_matrix/SNAP/{slug}/")
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
        assert snap[0].name.endswith("_snap")
        assert not snap[0].cacheable
    assert len(problems) == 14


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
        xp, problem.meta, xp.from_binsparse(problem.inputs[0])
    )
    np.testing.assert_array_equal(
        actual, [[np.inf, 1, np.inf], [np.inf, np.inf, 1], [np.inf, np.inf, np.inf]]
    )
    np.testing.assert_array_equal(
        to_scipy(adjacency).toarray(), [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    )


def test_snap_catalog_metadata_and_group_concepts():
    from xml.etree import ElementTree as ET

    datasets = SNAPGraphGenerator().datasets
    for dataset in datasets:
        metadata = dataset.metadata
        assert metadata["source_name"] == f"SNAP/{dataset.name}"
        assert metadata["max_degree"] == _MAX_DEGREES[dataset.source_name]
        assert metadata["types"] == dataset.types
        assert metadata["description"] == dataset.description
        assert metadata["nodes"] == dataset.nodes
        assert metadata["edges"] == dataset.edges
        assert metadata["groups"] == dataset.groups
        assert ET.fromstring(dataset.concepts).findtext("concept/concept_id")
        assert dataset.topics
    by_name = {d.name: d for d in datasets}
    rfa = by_name["wiki-RfA"]
    assert len(rfa.groups) == 3
    concepts = ET.fromstring(rfa.concepts).findall("concept/concept_id")
    assert len(concepts) == len({c.text for c in concepts}) == 3
    assert by_name["as-735"].nodes == "103-6,474"
    assert by_name["com-Amazon"].communities == 75149
    assert by_name["email-Eu-core-temporal"].static_edges == 24929


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


@pytest.mark.parametrize(
    ("module_name", "test_class", "benchmark_class"),
    [
        ("bfs", "BFSTestGenerator", "BFSBenchmark"),
        ("bellman_ford", "BellmanFordTestGenerator", "BellmanFordBenchmark"),
    ],
)
@pytest.mark.parametrize("seed", range(10))
def test_seeded_source_test_suite_problems(
    module_name, test_class, benchmark_class, seed
):
    from frameworks.saps_numpy import NumpyFramework

    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    generator = getattr(module, test_class)()
    dataset = next(d for d in generator.datasets if d.source_seed == seed)
    assert "test" in dataset.suites
    problem = generator.generate(dataset)
    src = problem.meta["src"]
    assert src in (1, 2)  # Never the isolated vertex or the sink.
    assert problem.meta["seed"] == seed
    xp = NumpyFramework()
    result = getattr(module, benchmark_class)().benchmark(
        xp, problem.meta, xp.from_binsparse(problem.inputs[0])
    )
    expected = (
        {1: [0, 1, 2, 3], 2: [0, 0, 1, 2]}
        if module_name == "bfs"
        else {1: [np.inf, 0, 1, 2], 2: [np.inf, np.inf, 0, 1]}
    )[src]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(to_numpy(problem.ref_outputs[0]), expected)


def test_all_snap_sources_have_ten_seeded_cases():
    from saps.benchmarks.bellman_ford import BellmanFordSNAPGenerator
    from saps.benchmarks.bfs import BFSSNAPGenerator

    graphs = SNAPGraphGenerator().datasets
    for generator in (BFSSNAPGenerator(), BellmanFordSNAPGenerator()):
        datasets = generator.datasets
        assert len(datasets) == len({d.name for d in datasets}) == len(graphs) * 10
        assert {(d.source_name, d.source_seed) for d in datasets} == {
            (g.name, seed) for g in graphs for seed in range(10)
        }
        assert all("standard" in d.suites for d in datasets)


def test_snap_with_suites_does_not_mutate_shared_graphs():
    graph = SNAPGraphGenerator().datasets[0]
    selected = graph.with_suites(["standard", "trace"])
    assert selected.suites == ["standard", "trace"]
    assert graph.suites == []
    assert all(dataset.suites == [] for dataset in SNAPGraphGenerator().datasets)
    selected.suites.append("test")
    assert graph.with_suites(["standard", "trace"]).suites == ["standard", "trace"]


def test_snap_source_trace_selection_is_seed_specific():
    graph = SNAPGraphGenerator().datasets[0]
    selected = BFSDataset(
        "selected", source_name=graph.name, source_seed=0, suites=["trace"]
    )
    other = BFSDataset("other", source_name=graph.name, source_seed=1)
    assert selected.suites == ["trace"]
    assert other.suites == []
    assert graph.suites == []


@pytest.mark.parametrize("seed", range(5))
def test_seeded_sources_match_one_selection_per_seed(seed):
    from scipy.sparse import random as sparse_random

    from binsparse.conversions import from_scipy

    graph = from_scipy(
        sparse_random(50, 50, density=0.05, format="coo", random_state=seed)
    )
    assert seeded_source_vertices(graph, 20) == [
        int(select_source_vertices(graph, seed=k)[0]) for k in range(20)
    ]


@pytest.mark.parametrize("seed", [-1, NUM_SNAP_SOURCES])
def test_snap_generator_rejects_seeds_without_a_source(monkeypatch, seed):
    from saps.benchmark import DataInstance
    from saps.benchmarks import bfs

    raw = DataInstance(
        inputs=[object()], meta={"sources": list(range(NUM_SNAP_SOURCES))}
    )
    monkeypatch.setattr(bfs, "fetch_snap_graph", lambda _: raw)
    dataset = BFSDataset("invalid", source_name="soc-Epinions1", source_seed=seed)
    with pytest.raises(ValueError, match="Source seed"):
        BFSSNAPGenerator().generate(dataset)


def test_snap_source_graph_picks_source_from_shell_sources(monkeypatch):
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from saps.benchmark import DataInstance
    from saps.benchmarks import bfs

    sources = list(range(100, 100 + NUM_SNAP_SOURCES))
    adjacency = from_scipy(coo_array([[0, 1], [0, 0]]))
    raw = DataInstance(inputs=[adjacency], meta={"max_degree": 3, "sources": sources})
    monkeypatch.setattr(bfs, "fetch_snap_graph", lambda _: raw)
    dataset = BFSDataset("seed-seven", source_name="soc-Epinions1", source_seed=7)
    problem = BFSSNAPGenerator().generate(dataset)
    assert problem.meta == {**raw.meta, "src": 107, "seed": 7}
    assert raw.meta == {"max_degree": 3, "sources": sources}


def test_multi_source_snap_problems_use_deduplicated_shell_sources(monkeypatch):
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from saps.benchmark import DataInstance
    from saps.benchmarks import mssp

    adjacency = from_scipy(coo_array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    raw = DataInstance(inputs=[adjacency], meta={"sources": [2, 0, 2, 0, 1]})
    monkeypatch.setattr(mssp, "fetch_snap_graph", lambda _: raw)
    generator = mssp.MSSPSNAPGenerator()
    problem = generator.generate(generator.datasets[0])
    assert problem.meta == {"sources": [0, 1, 2]}
    assert problem.inputs[1].shape == (3, 3)


def test_msbfs_snap_problems_use_deduplicated_shell_sources_and_edge_pattern(
    monkeypatch,
):
    from scipy.sparse import coo_array

    from binsparse.conversions import from_scipy

    from saps.benchmark import DataInstance
    from saps.benchmarks import msbfs

    adjacency = from_scipy(coo_array([[0, 2, 0], [0, 0, -1], [1, 0, 0]]))
    raw = DataInstance(inputs=[adjacency], meta={"sources": [2, 0, 2, 0, 1]})
    monkeypatch.setattr(msbfs, "fetch_snap_graph", lambda _: raw)
    generator = msbfs.MSBFSSNAPGenerator()
    problem = generator.generate(generator.datasets[0])
    assert problem.meta == {"sources": [0, 1, 2]}
    edges = to_sparse(problem.inputs[0]).todense()
    assert edges.dtype == bool
    np.testing.assert_array_equal(edges, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(to_sparse(problem.inputs[1]).todense(), np.eye(3))
