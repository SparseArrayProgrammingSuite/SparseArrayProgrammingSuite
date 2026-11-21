"""Suite rules that dataset caching relies on.

`--cache-datasets` prepares only datasets in a cached suite, by their own tag or
one inherited from their generator or benchmark.
"""

import importlib.util
from pathlib import Path

import pytest

from saps.benchmark import ShellBenchmark
from saps.metadata import _benchmark_instances

CACHED_SUITES = {"standard", "standard-long", "shell", "test"}

# Generators kept in the code but de-listed for their data's license, so that
# nothing runs, caches or uploads their datasets.
DELISTED_GENERATORS = {
    "dae_bdf_slicot",
    "ode_slicot",
    "netflix_prize_ratings",
    "jl_kmeans_netflix_prize",
    "jl_kmeans_netflix_prize_reference",
    "simhash_ann_netflix_prize_dense",
    "simhash_ann_netflix_prize_dense_reference",
    "simhash_ann_netflix_prize_sparse",
    "simhash_ann_netflix_prize_sparse_reference",
    "mri_masked_edge_detection_kaggle",
    "mri_sobel_edge_detection_kaggle",
}


def _datasets():
    for benchmark in _benchmark_instances():
        for generator in benchmark.generators:
            for dataset in generator.datasets:
                suites = {*benchmark.suites, *generator.suites, *dataset.suites}
                yield benchmark, generator, dataset, suites


DATASETS = list(_datasets())


def test_cached_suites_match_the_runner():
    path = Path(__file__).parents[1] / "bin/run_benchmark.py"
    spec = importlib.util.spec_from_file_location("saps_run_benchmark", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert {f"suite-{suite}" for suite in CACHED_SUITES} == runner.CACHED_SUITE_TAGS


def test_trace_and_train_datasets_are_standard():
    # Tracing and training read cached data, which only cached suites get.
    offenders = [
        f"{benchmark.name}/{generator.name}.{dataset.name}"
        for benchmark, generator, dataset, suites in DATASETS
        if suites & {"trace", "train"} and "standard" not in suites
    ]
    assert not offenders


def test_shell_generators_are_in_the_shell_suite():
    outside = {
        generator.name
        for benchmark, generator, _, _ in DATASETS
        if isinstance(benchmark, ShellBenchmark) and "shell" not in generator.suites
    }
    assert outside == {
        "netflix_prize_ratings",
        "jl_kmeans_netflix_prize_reference",
        "simhash_ann_netflix_prize_dense_reference",
        "simhash_ann_netflix_prize_sparse_reference",
    }


@pytest.mark.parametrize("name", sorted(DELISTED_GENERATORS))
def test_delisted_generators_keep_their_datasets_out_of_cached_suites(name):
    datasets = [
        (dataset, suites)
        for _, generator, dataset, suites in DATASETS
        if generator.name == name
    ]
    assert datasets, f"{name} lists no datasets"
    for dataset, suites in datasets:
        assert not suites & CACHED_SUITES, f"{name}.{dataset.name} is in {suites}"
