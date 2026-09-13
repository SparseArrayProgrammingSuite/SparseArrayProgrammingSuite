from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from saps.benchmark import Generator
from saps.benchmarks.frostt import fetch_frostt_tensor
from saps.benchmarks.model_counting import fetch_mccomp_instance
from saps.benchmarks.ogb import fetch_ogb_nodeprop_dataset
from saps.benchmarks.openml import fetch_openml_dataset
from saps.benchmarks.subgraph_matching import GCareHumanGenerator
from saps.benchmarks.suitesparse import (
    SuiteSparseDataset,
    SuiteSparseMatrixGenerator,
    fetch_suitesparse_matrix,
    suite_sparse_rhs_dataset_name,
)
from saps.metadata import _benchmark_instances


@pytest.mark.parametrize(
    "fetch",
    [
        fetch_frostt_tensor,
        fetch_mccomp_instance,
        fetch_ogb_nodeprop_dataset,
        fetch_openml_dataset,
        fetch_suitesparse_matrix,
    ],
)
def test_shell_fetch_rejects_unlisted_dataset_before_storage(monkeypatch, fetch):
    cached_generate = Mock(side_effect=AssertionError("Unexpected storage access"))
    monkeypatch.setattr(Generator, "cached_generate", cached_generate)

    with pytest.raises(ValueError, match="unlisted-dataset"):
        fetch("unlisted-dataset")

    cached_generate.assert_not_called()


def test_gcare_rejects_unlisted_shell_dataset_before_storage(monkeypatch):
    cached_generate = Mock(side_effect=AssertionError("Unexpected storage access"))
    monkeypatch.setattr(Generator, "cached_generate", cached_generate)

    with pytest.raises(ValueError, match="unlisted-dataset"):
        GCareHumanGenerator().generate(SimpleNamespace(subset_name="unlisted-dataset"))

    cached_generate.assert_not_called()


@pytest.mark.parametrize("rhs_index", [None, 999])
def test_suitesparse_rejects_unlisted_rhs_of_listed_matrix(monkeypatch, rhs_index):
    cached_generate = Mock(side_effect=AssertionError("Unexpected storage access"))
    monkeypatch.setattr(Generator, "cached_generate", cached_generate)

    with pytest.raises(ValueError, match="NYPA/Maragal_5"):
        fetch_suitesparse_matrix("NYPA/Maragal_5", rhs_index=rhs_index)

    cached_generate.assert_not_called()


@pytest.mark.parametrize(
    ("source_name", "rhs_index"),
    [
        ("HB/bcsstk01", None),
        ("NYPA/Maragal_5", 0),
        ("HB/orani678", 1),
        ("HB/orani678", 62),
    ],
)
def test_suitesparse_fetch_uses_the_declared_dataset(
    monkeypatch, source_name, rhs_index
):
    cached_generate = Mock()
    monkeypatch.setattr(Generator, "cached_generate", cached_generate)
    declared = next(
        d
        for d in SuiteSparseMatrixGenerator().datasets
        if (d.source_name, d.rhs_index) == (source_name, rhs_index)
    )

    assert (
        fetch_suitesparse_matrix(source_name, rhs_index=rhs_index)
        is cached_generate.return_value
    )
    assert cached_generate.call_args.args[0] is declared


def test_suitesparse_shell_lists_all_declared_consumers():
    datasets = SuiteSparseMatrixGenerator().datasets
    declared = {(d.source_name, d.rhs_index) for d in datasets}
    assert len(declared) == len(datasets)
    assert len({d.name for d in datasets}) == len(datasets)
    for dataset in datasets:
        assert dataset.name == suite_sparse_rhs_dataset_name(
            dataset.source_name, dataset.rhs_index
        )

    missing = set()
    for benchmark in _benchmark_instances():
        for generator in benchmark.generators:
            for dataset in generator.datasets:
                if not isinstance(dataset, SuiteSparseDataset):
                    continue
                # In-memory test matrices do not fetch a SuiteSparse source.
                if any(
                    getattr(dataset, field, None) is not None
                    for field in ("A", "adjacency")
                ):
                    continue
                if (dataset.source_name, dataset.rhs_index) not in declared:
                    missing.add(f"{generator.name}.{dataset.name}")
    assert not missing, (
        f"Add these consumers' inputs to the shell list: {sorted(missing)}"
    )

    # GAP graph datasets use their own Dataset classes and share these raw inputs.
    for name in ("road", "twitter", "web", "kron", "urand"):
        assert (f"GAP/GAP-{name}", None) in declared
