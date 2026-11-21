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
    "particle_simulation_ewap",
    "mccomp_instance",
    "model_counting_mccomp",
    "weighted_model_counting_mccomp",
}

# Datasets de-listed for their data's license from generators that keep others
# listed.
DELISTED_DATASETS = {
    "frostt_tensor": {
        "nell_2",
        "lbnl_network",
        "toy",
        "nips",
        "vast_2015_mc1_3d",
        "nell_1",
        "vast_2015_mc1_5d",
        "enron",
        "flickr_3d",
        "flickr_4d",
        "delicious_3d",
        "delicious_4d",
        "reddit_2015",
        "darpa",
    },
    "ogb_graph": {"ogbn-products"},
    "gcn_ogb": {"ogbn-products"},
    "gcn_backward_ogb": {"ogbn-products"},
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
        "mccomp_instance",
        # These put each listed dataset in the shell suite on its own.
        "frostt_tensor",
        "ogb_graph",
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


@pytest.mark.parametrize("name", sorted(DELISTED_DATASETS))
def test_delisted_datasets_stay_out_of_cached_suites(name):
    suites_by_dataset = {
        dataset.name: suites
        for _, generator, dataset, suites in DATASETS
        if generator.name == name
    }
    assert DELISTED_DATASETS[name] <= set(suites_by_dataset)
    for dataset_name, suites in suites_by_dataset.items():
        cached = suites & CACHED_SUITES
        if dataset_name in DELISTED_DATASETS[name]:
            assert not cached, f"{name}.{dataset_name} is in {suites}"
        else:
            assert cached, f"{name}.{dataset_name} is in no cached suite"


@pytest.mark.parametrize("name", ["gcn_ogb_reference", "gcn_backward_ogb_reference"])
def test_references_of_delisted_datasets_are_not_listed(name):
    names = {
        dataset.name for _, generator, dataset, _ in DATASETS if generator.name == name
    }
    assert names == {"ogbn-arxiv", "ogbn-proteins"}
