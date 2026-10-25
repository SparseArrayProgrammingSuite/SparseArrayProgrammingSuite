import inspect
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_scipy, to_scipy

from saps.benchmark import (
    DataInstance,
    Generator,
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.benchmarks import matrix_multiplication
from saps.metadata import _benchmark_instances
from saps.storage import LocalStorageBackend


def test_reference_is_cached_apart_from_inputs(monkeypatch, tmp_path):
    A = from_scipy(sps.random(20, 20, density=0.2, format="coo", random_state=0))
    monkeypatch.setattr(
        matrix_multiplication,
        "fetch_suitesparse_matrix",
        lambda _: DataInstance(inputs=[A], meta={}),
    )
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    monkeypatch.setattr(Generator, "backend", property(lambda self: backend))
    generator = matrix_multiplication.MatrixMultiplicationSuiteSparseGenerator()
    dataset = generator.datasets[0]
    reference = ReferenceGenerator(generator)

    # Dataset caching prepares the reference through its shell's datasets.
    assert backend.upload_dataset(reference, reference.datasets[0])
    stored = backend.retrieve_dataset(reference, dataset)
    assert stored.inputs == []

    monkeypatch.setattr(
        generator, "generate_reference", Mock(side_effect=AssertionError("recomputed"))
    )
    problem = generator.generate(dataset)
    assert problem.inputs == [A, A]
    expected = to_scipy(A) @ to_scipy(A)
    np.testing.assert_allclose(
        to_scipy(problem.ref_outputs[0]).toarray(), expected.toarray()
    )
    assert problem.ref_meta == stored.ref_meta


def test_reference_datasets_keep_cache_keys_but_join_no_suites():
    generator = matrix_multiplication.MatrixMultiplicationSuiteSparseGenerator()
    reference = ReferenceGenerator(generator)
    assert reference.name == f"{generator.name}_reference"
    assert reference.freshness == generator.freshness
    for dataset, wrapped in zip(generator.datasets, reference.datasets, strict=True):
        assert isinstance(wrapped, ReferenceDataset)
        assert (wrapped.name, wrapped.file, wrapped.freshness) == (
            dataset.name,
            dataset.file,
            dataset.freshness,
        )
        assert wrapped.suites == []


def _reads_cached_reference(generator):
    return "ReferenceGenerator(" in inspect.getsource(type(generator).generate)


def test_every_generator_with_a_reference_has_an_uncached_shell():
    benchmarks = list(_benchmark_instances())
    wrapped = {
        type(benchmark.generator.generator)
        for benchmark in benchmarks
        if isinstance(benchmark, ShellBenchmark)
        and isinstance(benchmark.generator, ReferenceGenerator)
    }
    referenced = {
        generator
        for benchmark in benchmarks
        for generator in benchmark.generators
        if _reads_cached_reference(generator)
    }
    assert {type(generator) for generator in referenced} == wrapped
    assert not any(generator.cacheable for generator in referenced)


def test_generators_with_cached_references_have_no_test_datasets():
    # Test runs prepare no references, so test cases belong in test generators.
    for benchmark in _benchmark_instances():
        for generator in benchmark.generators:
            if _reads_cached_reference(generator):
                assert not any(
                    "test" in dataset.suites for dataset in generator.datasets
                ), generator.name
